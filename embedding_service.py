"""
AgenticATS - Embedding Service
Local embedding generation, CV section parsing, sub-chunking, and ingestion pipeline.
"""

import json
import logging
import os
import re
import uuid

from sentence_transformers import SentenceTransformer

from db import insert_chunks_batch, delete_by_file


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton model
# ---------------------------------------------------------------------------
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        print(f"[EMBED] Loading model '{MODEL_NAME}' (first time may download)...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[EMBED] Model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """Generate a single embedding vector for the given text."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch-generate embeddings for multiple texts."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [e.tolist() for e in embeddings]


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
        rf"(?:[-=]{2,}\s*)?"             # optional separator: "---" or "==="
        rf"({alternatives})"             # capture the heading
        rf"\s*[:|\-–—]?\s*$",            # optional trailing colon/dash
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
      4. Generate embeddings (batch)
      5. Store in PostgreSQL

    Args:
        file_path: Path to the CV file.
        replace_existing: If True, delete existing chunks for this file before inserting.

    Returns:
        Total number of chunks stored.
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

    # Step 4: Generate embeddings (batch)
    texts_to_embed = [r["chunk_text"] for r in records]
    embeddings = generate_embeddings(texts_to_embed)
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
