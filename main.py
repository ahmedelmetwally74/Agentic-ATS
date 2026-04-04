"""
AgenticATS - Main Entry Point
Extract text from PDF or Word documents, embed into PostgreSQL, and search via RAG.
Supports three analysis modes: company (candidate ranking), applicant (CV improvement), and cv_generation (build a CV from section text files).

Usage:
    python main.py <file_path>                              # Extract text only
    python main.py <file_path> --embed                      # Extract + embed + store (single file)
    python main.py --embed <folder_path>                    # Embed all CVs in a folder
    python main.py --search "query text"                    # RAG search
    python main.py --mode company --jd-file job.txt         # Company: rank candidates + interview questions
    python main.py --mode applicant --jd-file job.txt       # Applicant: CV improvement suggestions
    python main.py --mode cv_generation --sections-dir .\cv_sections -o .\generated_cv.docx
    python main.py --init-db                                # Initialize database
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from core.document_utils import normalize_text_basic, clean_extracted_text

# Load environment variables from .env file
load_dotenv()

def _clean_text(text: str) -> str:
    """Aggressively clean text of all control characters and normalize whitespace for terminal printing."""
    if not text:
        return ""
    # Treat all whitespace including \r \n \t as space, then join back
    cleaned = " ".join(text.split())
    # Remove any non-printable characters for terminal safety
    cleaned = "".join(c for c in cleaned if c.isprintable())
    return cleaned.strip()

from core.document_service import (
    extract_text_from_pdf_sync,
    extract_text_from_word,
    pdf_has_text,
    save_text_to_docx,
)

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
        description=(
            "Extract text from PDF/Word documents, embed into PostgreSQL, "
            "and perform candidate matching with LLM-powered analysis."
        ),
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to the input file (.pdf or .docx) for text extraction",
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
        type=str,
        default=None,
        metavar="PATH",
        help="Embed a CV file or all CVs in a folder into PostgreSQL (supports .pdf and .docx)",
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
        "--match-job",
        type=str,
        default=None,
        metavar="JOB_DESCRIPTION",
        help="Rank top unique candidates for a full job description text",
    )
    parser.add_argument(
        "--jd-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Read the job description from a text file and rank candidates",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["company", "applicant", "cv_generation"],
        default=None,
        help="Analysis mode: 'company' for candidate ranking or 'applicant' for CV improvement or 'cv_generation' for building a CV from section files",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=3,
        help="Number of unique candidates to return for job matching (default: 3)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Number of raw matching chunks to retrieve before candidate grouping (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory to save PDF analysis reports (default: current directory)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the PostgreSQL database (create tables and indexes)",
    )
    parser.add_argument(
        "--cv-id",
        type=str,
        default=None,
        metavar="CV_ID",
        help="Required for applicant mode: analyze one specific CV only",
    )
    parser.add_argument(
        "--sections-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory containing CV section text files for cv_generation mode",
    )

    args = parser.parse_args()

    try:
        # --- Init Database ---
        if args.init_db:
            from core.db import init_db
            init_db()
            return

        # --- CV Generation ---
        if args.mode == "cv_generation":
            from modes.cv_generation_mode import run_cv_generation_mode

            run_cv_generation_mode(args, parser)
            return

        # --- Candidate Matching ---
        if args.match_job or args.jd_file:
            if not args.mode:
                parser.error("--mode is required for matching. Use --mode company or --mode applicant.")

            if args.mode == "company":
                from modes.company_mode import run_company_mode

                run_company_mode(args, parser, _clean_text)
            elif args.mode == "applicant":
                from modes.applicant_mode import run_applicant_mode

                run_applicant_mode(args, parser, _clean_text)
            else:
                parser.error("--mode is required for matching. Use --mode company or --mode applicant.")
            return

        # --- RAG Search ---
        if args.search:
            from core.rag_service import rag_query
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

        # --- Embedding (mode-independent) ---
        if args.embed:
            embed_path = args.embed

            if os.path.isdir(embed_path):
                # Folder mode: ingest all CVs in the directory
                from core.embedding_service import ingest_cv_folder
                results = ingest_cv_folder(embed_path)

                print("\n" + "=" * 60)
                print("FOLDER INGESTION RESULTS")
                print("=" * 60)
                for file_name, cv_id, chunk_count in results:
                    print(f"  {file_name}: {chunk_count} chunks (CV ID: {cv_id})")
                print(f"\nTotal: {len(results)} files ingested.")
                print("=" * 60)
            elif os.path.isfile(embed_path):
                # Single file mode
                from core.embedding_service import ingest_cv
                cv_id, chunk_count = ingest_cv(embed_path)
                print(f"\n[SUCCESS] Ingested {chunk_count} chunks into the database. (CV ID: {cv_id})")
            else:
                raise FileNotFoundError(f"Path not found: {embed_path}")
            return

        # --- Default: extract and display text ---
        if not args.file_path:
            parser.error("file_path is required for extraction. "
                         "Use --embed for ingestion or --search for retrieval.")

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
