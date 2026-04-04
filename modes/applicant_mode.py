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


def _print_requirements(requirements: list[dict], clean_text: Callable[[str], str]) -> None:
    if not requirements:
        return

    print("\n[Extracted Job Requirement Objects]")
    for requirement in requirements:
        requirement_id = requirement.get("id", "")
        text = requirement.get("text", "")
        category = requirement.get("category", "")
        importance = requirement.get("importance", "")
        strategy = requirement.get("matching_strategy", "")
        print(f"  {clean_text(requirement_id)} | {clean_text(category)} | {clean_text(importance)} | {clean_text(strategy)}")
        print(f"    - {clean_text(text)}")


def _print_requirement_results(requirement_results: list[dict], clean_text: Callable[[str], str]) -> None:
    if not requirement_results:
        print("\n[Requirement Results]\nNo requirement results were produced.")
        return

    print("\n[Requirement Results]")
    for requirement in requirement_results:
        requirement_id = requirement.get("requirement_id", requirement.get("id", ""))
        status = requirement.get("status", "N/A")
        category = requirement.get("category", "N/A")
        importance = requirement.get("importance", "N/A")

        print(f"\n  {clean_text(requirement_id)} | {clean_text(status)} | {clean_text(category)} | {clean_text(importance)}")
        print(f"    Requirement: {clean_text(requirement.get('requirement_text', requirement.get('text', '')))}")

        for evidence in requirement.get("top_evidence", []):
            print(f"    Evidence: {clean_text(evidence)}")

        print(f"    Notes: {clean_text(requirement.get('notes', ''))}")
        print(f"    Suggestion: {clean_text(requirement.get('suggestion', ''))}")


def run_applicant_mode(
    args: Namespace,
    parser: ArgumentParser,
    clean_text: Callable[[str], str],
) -> None:
    if not args.cv_id:
        parser.error("--cv-id is required for applicant mode.")
    if args.section:
        parser.error("--section is not supported in applicant mode for the new requirement-centric Phase 1 flow.")

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

    _print_requirements(result.get("requirements", []), clean_text)

    if result.get("jd_report"):
        print(f"\nJD Analysis Report: {clean_text(result['jd_report'])}")
    if result.get("jd_requirements_json"):
        print(f"JD Requirements JSON: {clean_text(result['jd_requirements_json'])}")

    if not result["candidates"]:
        print("\nNo matching candidates found.")
    else:
        for candidate in result["candidates"]:
            print("\n" + "-" * 60)
            print(f"File: {candidate['file_name']}")
            print(f"CV ID: {candidate['cv_id']}")
            print(f"\n[Summary]\n{clean_text(candidate.get('summary', ''))}")

            status_counts = candidate.get("status_counts", {})
            if status_counts:
                print("\n[Status Overview]")
                for status_name, count in status_counts.items():
                    print(f"  {clean_text(status_name)}: {count}")

            experience_summary = candidate.get("experience_summary", {})
            if experience_summary:
                print("\n[Experience Duration Summary]")
                print(f"  Runtime today: {clean_text(experience_summary.get('today', 'N/A'))}")
                print(f"  Experience section found: {experience_summary.get('experience_section_found', False)}")
                print(f"  Overlap-aware total years: {experience_summary.get('total_years', 0.0)}")

            _print_requirement_results(candidate.get("requirement_results", []), clean_text)

            print("\n[General Improvement Suggestions]")
            for suggestion in candidate.get("suggestions", []):
                print(f"  + {clean_text(suggestion)}")

            print(f"\nPDF Report: {clean_text(candidate.get('report_pdf', 'N/A'))}")

    print("\n" + "=" * 60)
