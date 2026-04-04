from argparse import ArgumentParser, Namespace

from core.cv_generation_service import generate_cv_from_sections


def run_cv_generation_mode(args: Namespace, parser: ArgumentParser) -> None:
    if not args.sections_dir:
        parser.error("--sections-dir is required for cv_generation mode.")

    output_path = args.output or "generated_cv.docx"
    final_path = generate_cv_from_sections(args.sections_dir, output_path)

    print("\n" + "=" * 60)
    print("CV GENERATION RESULT")
    print("=" * 60)
    print(f"Sections directory: {args.sections_dir}")
    print(f"Generated file: {final_path}")
    print("=" * 60)
