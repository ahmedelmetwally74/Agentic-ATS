"""
AgenticATS - Main Entry Point
Extract text from PDF or Word documents, embed into PostgreSQL, and search via RAG.
Supports two analysis modes: company (candidate ranking) and applicant (CV improvement).

Usage:
    python main.py <file_path>                              # Extract text only
    python main.py <file_path> --embed                      # Extract + embed + store (single file)
    python main.py --embed <folder_path>                    # Embed all CVs in a folder
    python main.py --search "query text"                    # RAG search
    python main.py --mode company --jd-file job.txt         # Company: rank candidates + interview questions
    python main.py --mode applicant --jd-file job.txt       # Applicant: CV improvement suggestions
    python main.py --init-db                                # Initialize database
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from document_utils import normalize_text_basic, clean_extracted_text

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

from document_service import (
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
        choices=["company", "applicant"],
        default=None,
        help="Analysis mode: 'company' for candidate ranking or 'applicant' for CV improvement",
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

    args = parser.parse_args()

    try:
        # --- Init Database ---
        if args.init_db:
            from db import init_db
            init_db()
            return

        # --- Candidate Matching ---
        if args.match_job or args.jd_file:
            if args.match_job and args.jd_file:
                parser.error("Use either --match-job or --jd-file, not both.")

            if not args.mode:
                parser.error("--mode is required for matching. Use --mode company or --mode applicant.")

            job_description = args.match_job
            if args.jd_file:
                if not os.path.isfile(args.jd_file):
                    raise FileNotFoundError(f"Job description file not found: {args.jd_file}")
                with open(args.jd_file, "r", encoding="utf-8") as f:
                    job_description = f.read().strip()

            if not job_description:
                parser.error("A non-empty job description is required.")

            from matching_service import match_candidates, analyze_applicant_cv

            if args.mode == "company":
                result = match_candidates(
                    job_description=job_description,
                    top_candidates=args.top_candidates,
                    pool_size=args.pool_size,
                    section_filter=args.section,
                    mode="company",
                    output_dir=args.output_dir,
                )
            else:
                if not args.cv_id:
                    parser.error("--cv-id is required for applicant mode.")

                result = analyze_applicant_cv(
                    job_description=job_description,
                    cv_id=args.cv_id,
                    section_filter=args.section,
                    output_dir=args.output_dir,
                )

            print("\n" + "=" * 60)
            print(f"CANDIDATE MATCHING RESULTS ({args.mode.upper()} MODE)")
            print("=" * 60)
            if args.mode == "company":
                print(f"Raw chunk matches scanned: {result['raw_chunk_matches']}")
                print(f"Unique candidates scored: {result['candidate_count']}")
                print(f"Top candidates requested: {args.top_candidates}")
            else:
                print(f"Applicant CV ID: {result['candidates'][0]['cv_id'] if result['candidates'] else 'N/A'}")

            if result.get("requirements"):
                print("\n[Extracted Job Requirements]")
                for section, reqs in result["requirements"].items():
                    if reqs:
                        print(f"  {section}:")
                        for req in reqs:
                            print(f"    - {_clean_text(req)}")
            
            if result.get("jd_report"):
                print(f"\nJD Analysis Report: {_clean_text(result['jd_report'])}")

            if not result["candidates"]:
                print("\nNo matching candidates found.")
            else:
                for candidate in result["candidates"]:
                    print("\n" + "-" * 60)
                    print(f"File: {candidate['file_name']}")
                    print(f"CV ID: {candidate['cv_id']}")

                    if args.mode == "company":
                        print(f"Rank #{candidate['rank']}")
                        print(f"Score: {candidate['score']:.4f}")
                        print("Matched sections: " + _clean_text(", ".join(candidate["matched_sections"])))
                        print(f"\n[Summary]\n{_clean_text(candidate.get('summary', ''))}")

                        print("\n[Reasons]")
                        for reason in candidate.get("reasons", []):
                            print(f"  * {_clean_text(reason)}")

                        print("\n[Interview Questions]")
                        for q in candidate.get("questions", []):
                            print(f"  ? {_clean_text(q)}")
                    else:
                        print(f"\n[Summary]\n{_clean_text(candidate.get('summary', ''))}")

                        print("\n[Section Comparison]")
                        for sec, data in candidate.get("detailed_sections", {}).items():
                            print(f"\n  {sec}:")
                            print(f"    Comparison: {_clean_text(data.get('comparison', 'N/A'))}")

                            for item in data.get("missing_tools", []):
                                print(f"    Missing Tool: {_clean_text(item)}")

                            for item in data.get("missing_skills", []):
                                print(f"    Missing Skill: {_clean_text(item)}")

                            for item in data.get("missing_experience", []):
                                print(f"    Missing Experience: {_clean_text(item)}")

                            for item in data.get("missing_education", []):
                                print(f"    Missing Education: {_clean_text(item)}")

                        print("\n[General Improvement Suggestions]")
                        for s in candidate.get("suggestions", []):
                            print(f"  + {_clean_text(s)}")

                    print(f"\nPDF Report: {_clean_text(candidate.get('report_pdf', 'N/A'))}")

            print("\n" + "=" * 60)
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

        # --- Embedding (mode-independent) ---
        if args.embed:
            embed_path = args.embed

            if os.path.isdir(embed_path):
                # Folder mode: ingest all CVs in the directory
                from embedding_service import ingest_cv_folder
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
                from embedding_service import ingest_cv
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
