import os
from argparse import ArgumentParser, Namespace
from typing import Callable

from matching.applicant_matching_service import analyze_applicant_cv


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


def run_applicant_mode(
    args: Namespace,
    parser: ArgumentParser,
    clean_text: Callable[[str], str],
) -> None:
    if not args.cv_id:
        parser.error("--cv-id is required for applicant mode.")

    job_description = _load_job_description(args, parser)
    result = analyze_applicant_cv(
        job_description=job_description,
        cv_id=args.cv_id,
        section_filter=args.section,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("CANDIDATE MATCHING RESULTS (APPLICANT MODE)")
    print("=" * 60)
    print(f"Applicant CV ID: {result['candidates'][0]['cv_id'] if result['candidates'] else 'N/A'}")

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
            print(f"\n[Summary]\n{clean_text(candidate.get('summary', ''))}")

            print("\n[Section Comparison]")
            for section_name, data in candidate.get("detailed_sections", {}).items():
                print(f"\n  {section_name}:")
                print(f"    Comparison: {clean_text(data.get('comparison', 'N/A'))}")

                for item in data.get("missing_tools", []):
                    print(f"    Missing Tool: {clean_text(item)}")

                for item in data.get("missing_skills", []):
                    print(f"    Missing Skill: {clean_text(item)}")

                for item in data.get("missing_experience", []):
                    print(f"    Missing Experience: {clean_text(item)}")

                for item in data.get("missing_education", []):
                    print(f"    Missing Education: {clean_text(item)}")

            print("\n[General Improvement Suggestions]")
            for suggestion in candidate.get("suggestions", []):
                print(f"  + {clean_text(suggestion)}")

            print(f"\nPDF Report: {clean_text(candidate.get('report_pdf', 'N/A'))}")

    print("\n" + "=" * 60)
