"""
AgenticATS - Main Entry Point
Extract text from PDF or Word documents, embed into PostgreSQL, and search via RAG.

Usage:
    python main.py <file_path>                       # Extract text only
    python main.py <file_path> --embed               # Extract + embed + store
    python main.py --search "query text"             # RAG search
    python main.py --search "query" --section Skills # Filter by section
    python main.py --init-db                         # Initialize database
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from document_service import (
    extract_text_from_pdf_sync,
    extract_text_from_word,
    pdf_has_text,
    save_text_to_docx,
)
from document_utils import normalize_text_basic, clean_extracted_text

# Load environment variables from .env file
load_dotenv()


def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF or Word document.

    Args:
        file_path: Path to the input file (.pdf or .docx).

    Returns:
        The extracted text as a string.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        if pdf_has_text(file_path):
            print("[INFO] PDF has selectable text — extracting directly.")
            text = extract_text_from_pdf_sync(file_path)
        else:
            print("[WARNING] PDF appears to be scanned/image-based. "
                  "Text extraction may be incomplete without OCR.")
            text = extract_text_from_pdf_sync(file_path)
    elif ext == ".docx":
        print("[INFO] Extracting text from Word document.")
        text = extract_text_from_word(file_path)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. "
                         "Please provide a .pdf or .docx file.")

    # Normalize extracted text
    text = normalize_text_basic(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        prog="AgenticATS",
        description="Extract text from PDF/Word documents, embed into PostgreSQL, and RAG search.",
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to the input file (.pdf or .docx)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional: save extracted text to a .docx file at this path",
        default=None,
    )
    parser.add_argument(
        "-c", "--clean",
        action="store_true",
        help="Apply additional text cleaning (list pattern detection)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Extract text, chunk by sections, embed, and store in PostgreSQL",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        metavar="QUERY",
        help="Search stored CV embeddings for a query string",
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        metavar="SECTION",
        help="Filter search results to a specific CV section (e.g., Skills, Experience)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the PostgreSQL database (create tables and indexes)",
    )

    args = parser.parse_args()

    try:
        # --- Init Database ---
        if args.init_db:
            from db import init_db
            init_db()
            return

        # --- RAG Search ---
        if args.search:
            from rag_service import rag_query
            result = rag_query(args.search, top_k=args.top_k,
                               section_filter=args.section)

            print("\n" + "=" * 60)
            print(f"RAG SEARCH: \"{result['query']}\"")
            print("=" * 60)

            if not result["chunks"]:
                print("No matching results found.")
            else:
                print(result["context"])

            print("\n" + "=" * 60)
            print(f"Retrieved {len(result['chunks'])} chunks")
            print("=" * 60)
            return

        # --- File processing (extract / embed) ---
        if not args.file_path:
            parser.error("file_path is required for extraction and embedding.")

        if args.embed:
            # Full RAG ingestion pipeline
            from embedding_service import ingest_cv
            cv_id, chunk_count = ingest_cv(args.file_path)
            print(f"\n[SUCCESS] Ingested {chunk_count} chunks into the database. (CV ID: {cv_id})")
            return

        # --- Default: extract and display text ---
        text = extract_text(args.file_path)

        if args.clean:
            text = clean_extracted_text(text)

        if args.output:
            output_path = save_text_to_docx(text, args.output)
            print(f"\n[SUCCESS] Text saved to: {output_path}")
        else:
            print("\n" + "=" * 60)
            print("EXTRACTED TEXT")
            print("=" * 60)
            print(text)

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
