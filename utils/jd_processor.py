"""
Deterministic Job Description Processor.

This script processes a Job Description file and outputs a JSON file containing:
- Extracted requirements (literal substrings from JD)
- Classified requirements (tier: critical/important/nice_to_have + key terms)

This makes JD processing deterministic - the same JD always produces the same output.

Usage:
    python -m utils.jd_processor --input job_description.txt --output requirements.json

Or import as module:
    from utils.jd_processor import process_jd_file
    result = process_jd_file("job_description.txt")
"""

import json
import argparse
import os
import sys

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_service import decompose_job_description, classify_requirements


def process_jd_text(jd_text: str) -> dict:
    """
    Process JD text and return structured requirements.

    Args:
        jd_text: Raw job description text

    Returns:
        dict with 'requirements' (list) and 'classified_requirements' (list of dicts)
    """
    requirements = decompose_job_description(jd_text)
    classified = classify_requirements(requirements)

    return {
        "requirements": requirements,
        "classified_requirements": classified
    }


def process_jd_file(input_path: str) -> dict:
    """
    Process a JD file and return structured requirements.

    Args:
        input_path: Path to JD text file

    Returns:
        dict with 'requirements' and 'classified_requirements'
    """
    with open(input_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    return process_jd_text(jd_text)


def save_requirements(result: dict, output_path: str):
    """Save requirements to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Requirements saved to: {output_path}")


def load_requirements(input_path: str) -> dict:
    """Load pre-processed requirements from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Job Description Processor"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JD text file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for processed requirements"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Processing JD: {args.input}")
    result = process_jd_file(args.input)

    print(f"\nExtracted {len(result['requirements'])} requirements:")
    for idx, req in enumerate(result["requirements"], 1):
        print(f"  {idx}. {req[:80]}{'...' if len(req) > 80 else ''}")

    save_requirements(result, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
