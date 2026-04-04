import json
import os

from reporting.report_shared import AnalysisReport, ensure_report_dir, safe_report_text


def save_jd_requirements_json(requirements: list[dict], output_dir: str = ".") -> str:
    """Persist applicant-mode JD requirements for debugging and traceability."""
    ensure_report_dir(output_dir)
    json_path = os.path.join(output_dir, "jd_requirements.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(requirements, f, ensure_ascii=False, indent=2)
    print(f"[REPORT] JD requirements JSON saved: {json_path}")
    return json_path


def generate_jd_analysis_report(requirements_data, full_text: str, output_dir: str = ".") -> str:
    """Generate a standalone PDF report for the job description decomposition."""
    ensure_report_dir(output_dir)
    pdf_path = os.path.join(output_dir, "job_description_analysis.pdf")

    pdf = AnalysisReport(title="Job Description Analysis", subtitle="Extracted Requirements")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    if isinstance(requirements_data, dict):
        for section, reqs in requirements_data.items():
            if not reqs:
                continue
            pdf.add_section_title(section)
            pdf.add_bullet_list([safe_report_text(req) for req in reqs])
            pdf.ln(2)
    elif isinstance(requirements_data, list):
        pdf.add_section_title("Requirement Objects")
        for requirement in requirements_data:
            requirement_id = safe_report_text(str(requirement.get("id", "")))
            text = safe_report_text(str(requirement.get("text", "")))
            category = safe_report_text(str(requirement.get("category", "")))
            importance = safe_report_text(str(requirement.get("importance", "")))
            section_group = safe_report_text(str(requirement.get("section_group", "")))
            strategy = safe_report_text(str(requirement.get("matching_strategy", "")))

            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, f"{requirement_id} | {category} | {importance}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 5, text, new_x="LMARGIN", new_y="NEXT")
            pdf.cell(pdf.epw, 5, f"section_group: {section_group}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(pdf.epw, 5, f"matching_strategy: {strategy}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    pdf.add_page()
    pdf.add_section_title("Original Job Description Context")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(pdf.epw, 5, safe_report_text(full_text), new_x="LMARGIN", new_y="NEXT")

    pdf.output(pdf_path)
    print(f"[REPORT] JD Analysis saved: {pdf_path}")
    return pdf_path
