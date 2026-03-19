"""
AgenticATS - Embedding Service
Jina v5 embeddings via llama-server HTTP API, CV section parsing, sub-chunking, and ingestion pipeline.
"""

import json
import logging
import os
import re
import uuid

import requests

from db import insert_chunks_batch, delete_by_file


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jina v5 embedding endpoint (llama-server)
# ---------------------------------------------------------------------------
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:7999/v1/embeddings")


# ---------------------------------------------------------------------------
# Embedding generation via HTTP
# ---------------------------------------------------------------------------

def generate_embedding(text: str, prefix: str = "Document: ") -> list[float]:
    """
    Generate a single embedding vector via the Jina v5 llama-server endpoint.

    Args:
        text: The text to embed.
        prefix: Prefix to prepend — "Document: " for CV chunks, "Query: " for JD/search.

    Returns:
        Embedding vector as a list of floats.
    """
    prefixed = f"{prefix}{text}"
    response = requests.post(
        EMBEDDING_API_URL,
        headers={"Content-Type": "application/json"},
        json={"input": [prefixed]},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


def generate_embeddings(texts: list[str], prefix: str = "Document: ") -> list[list[float]]:
    """
    Batch-generate embeddings for multiple texts via the Jina v5 llama-server endpoint.

    Args:
        texts: List of texts to embed.
        prefix: Prefix to prepend — "Document: " for CV chunks, "Query: " for JD/search.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    prefixed = [f"{prefix}{t}" for t in texts]
    response = requests.post(
        EMBEDDING_API_URL,
        headers={"Content-Type": "application/json"},
        json={"input": prefixed},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    # Sort by index to guarantee order matches input
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


# ---------------------------------------------------------------------------
# Config-driven section parsing
# ---------------------------------------------------------------------------

def _load_section_headings() -> list[str]:
    """Load CV section headings from sections_config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "sections_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Section config not found: {config_path}\n"
            "Please create sections_config.json with a 'section_headings' list."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    headings = config.get("section_headings", [])
    if not headings:
        raise ValueError("sections_config.json must have a non-empty 'section_headings' list.")
    return headings


def _build_section_pattern(headings: list[str]) -> re.Pattern:
    """
    Build a regex that matches any of the configured headings at the start of a line.
    Handles common CV formatting: optional numbering, bullets, colons, dashes.
    Example matches:
        "EXPERIENCE", "Experience:", "2. Education", "• Skills", "--- Projects ---"
    """
    escaped = [re.escape(h) for h in headings]
    # Sort by length descending so longer headings match first
    escaped.sort(key=len, reverse=True)
    alternatives = "|".join(escaped)

    pattern = re.compile(
        rf"^\s*"                         # leading whitespace
        rf"(?:[\d]+[.\)]\s*)?"           # optional numbering: "1." or "2)"
        rf"(?:[•\-–—\*]\s*)?"            # optional bullet
        rf"(?:[-=]{{2,}}\s*)?"           # optional separator: "---" or "==="
        rf"({alternatives})"             # capture the heading
        rf"\s*[:|–—\-]?\s*$",            # optional trailing colon/dash
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern


def parse_cv_sections(text: str) -> list[dict]:
    """
    Parse extracted CV text into semantic sections based on configured headings.

    Returns:
        List of dicts: [{"section_name": "Experience", "text": "..."}, ...]
        If no headings are detected, returns the full text as a single "Full CV" section.
    """
    headings = _load_section_headings()
    pattern = _build_section_pattern(headings)

    matches = list(pattern.finditer(text))

    if not matches:
        logger.info("No section headings detected — treating entire text as one section.")
        return [{"section_name": "Full CV", "text": text.strip()}]

    sections = []

    # Content before the first heading (e.g., name, contact info at the top)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"section_name": "Header", "text": preamble})

    # Each heading to the next heading
    for i, match in enumerate(matches):
        raw_match = match.group(1)
        if raw_match is None:
            # Fallback if group 1 is empty for some reason
            raw_match = match.group(0)
            
        section_name = raw_match.strip().title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        
        print(f"[DEBUG] Found section heading: '{section_name}' (start: {match.start()}, end: {match.end()})")

        if section_text:
            sections.append({"section_name": section_name, "text": section_text})

    print(f"[PARSE] Detected {len(sections)} sections: "
          f"{[s['section_name'] for s in sections]}")
    return sections


# ---------------------------------------------------------------------------
# Sub-chunking for long sections
# ---------------------------------------------------------------------------

def sub_chunk(text: str, max_chars: int = 1500) -> list[str]:
    """
    Split a long section into smaller chunks at sentence boundaries.
    Targets roughly ~512 tokens (approximated at ~3 chars/token = ~1500 chars).
    Short texts are returned as a single chunk.
    """
    if len(text) <= max_chars:
        return [text]

    # Split into sentences (period/question/exclamation followed by space or newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_cv(file_path: str, replace_existing: bool = True) -> tuple[str, int]:
    """
    Full RAG ingestion pipeline:
      1. Extract text from the PDF/DOCX (using existing extract_text)
      2. Parse into semantic sections
      3. Sub-chunk long sections
      4. Generate embeddings (batch) with "Document: " prefix
      5. Store in PostgreSQL

    Args:
        file_path: Path to the CV file.
        replace_existing: If True, delete existing chunks for this file before inserting.

    Returns:
        Tuple of (cv_id, chunk_count).
    """
    from main import extract_text  # import here to avoid circular imports

    file_name = os.path.basename(file_path)
    print(f"\n[INGEST] Starting ingestion for: {file_name}")

    # Step 1: Extract text
    text = extract_text(file_path)
    print(f"[INGEST] Extracted {len(text)} characters of text.")

    # Step 2: Parse sections
    sections = parse_cv_sections(text)

    # Step 3: Sub-chunk and prepare records
    records = []
    cv_id = str(uuid.uuid4())
    for section in sections:
        chunks = sub_chunk(section["text"])
        for idx, chunk_text in enumerate(chunks):
            records.append({
                "cv_id": cv_id,
                "file_name": file_name,
                "section_name": section["section_name"],
                "chunk_index": idx,
                "chunk_text": chunk_text,
            })

    print(f"[INGEST] Prepared {len(records)} chunks across {len(sections)} sections.")

    # Step 4: Generate embeddings (batch) with "Document: " prefix for CV chunks
    texts_to_embed = [r["chunk_text"] for r in records]
    embeddings = generate_embeddings(texts_to_embed, prefix="Document: ")
    for record, emb in zip(records, embeddings):
        record["embedding"] = emb

    # Step 5: Store in database
    if replace_existing:
        deleted = delete_by_file(file_name)
        if deleted:
            print(f"[INGEST] Replaced {deleted} existing chunks for '{file_name}'.")

    insert_chunks_batch(records)
    print(f"[INGEST] Successfully stored {len(records)} chunks in the database.")

    return cv_id, len(records)


def ingest_cv_folder(folder_path: str, replace_existing: bool = True) -> list[tuple[str, str, int]]:
    """
    Ingest all CV files (.pdf, .docx) in a folder directory.

    Args:
        folder_path: Path to the folder containing CV files.
        replace_existing: If True, delete existing chunks for each file before inserting.

    Returns:
        List of tuples: [(file_name, cv_id, chunk_count), ...]
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    supported_extensions = {".pdf", ".docx"}
    results = []

    files = sorted(os.listdir(folder_path))
    cv_files = [
        f for f in files
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not cv_files:
        print(f"[INGEST] No .pdf or .docx files found in: {folder_path}")
        return results

    print(f"[INGEST] Found {len(cv_files)} CV files in '{folder_path}'")

    for file_name in cv_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            cv_id, chunk_count = ingest_cv(file_path, replace_existing=replace_existing)
            results.append((file_name, cv_id, chunk_count))
        except Exception as e:
            print(f"[ERROR] Failed to ingest '{file_name}': {e}")

    print(f"\n[INGEST] Folder ingestion complete: {len(results)}/{len(cv_files)} files processed.")
    return results
