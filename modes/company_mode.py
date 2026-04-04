import os
from argparse import ArgumentParser, Namespace
from typing import Callable

from matching.company_matching_service import match_candidates


def _load_job_description(args: Namespace, parser: ArgumentParser) -> str:
    if args.match_job and args.jd_file:
        parser.error("Use either --match-job or --jd-file, not both.")

    job_description = args.match_job
    if args.jd_file:
        if not os.path.isfile(args.jd_file):
            raise FileNotFoundError(f"Job description file not found: {args.jd_file}")
        with open(args.jd_file, "r", encoding="utf-8") as f:
            job_description = f.read().strip()

    if not job_description:
        parser.error("A non-empty job description is required.")

    return job_description


def _print_requirements(requirements: dict, clean_text: Callable[[str], str]) -> None:
    if not requirements:
        return

    print("\n[Extracted Job Requirements]")
    for section, reqs in requirements.items():
        if reqs:
            print(f"  {section}:")
            for req in reqs:
                print(f"    - {clean_text(req)}")


def run_company_mode(
    args: Namespace,
    parser: ArgumentParser,
    clean_text: Callable[[str], str],
) -> None:
    job_description = _load_job_description(args, parser)
    result = match_candidates(
        job_description=job_description,
        top_candidates=args.top_candidates,
        pool_size=args.pool_size,
        section_filter=args.section,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("CANDIDATE MATCHING RESULTS (COMPANY MODE)")
    print("=" * 60)
    print(f"Raw chunk matches scanned: {result['raw_chunk_matches']}")
    print(f"Unique candidates scored: {result['candidate_count']}")
    print(f"Top candidates requested: {args.top_candidates}")

    _print_requirements(result.get("requirements", {}), clean_text)

    if result.get("jd_report"):
        print(f"\nJD Analysis Report: {clean_text(result['jd_report'])}")

    if not result["candidates"]:
        print("\nNo matching candidates found.")
    else:
        for candidate in result["candidates"]:
            print("\n" + "-" * 60)
            print(f"File: {candidate['file_name']}")
            print(f"CV ID: {candidate['cv_id']}")
            print(f"Rank #{candidate['rank']}")
            print(f"Score: {candidate['score']:.4f}")
            print("Matched sections: " + clean_text(", ".join(candidate.get("matched_sections", []))))
            print(f"\n[Summary]\n{clean_text(candidate.get('summary', ''))}")

            print("\n[Reasons]")
            for reason in candidate.get("reasons", []):
                print(f"  * {clean_text(reason)}")

            print("\n[Interview Questions]")
            for question in candidate.get("questions", []):
                print(f"  ? {clean_text(question)}")

            print(f"\nPDF Report: {clean_text(candidate.get('report_pdf', 'N/A'))}")

    print("\n" + "=" * 60)
