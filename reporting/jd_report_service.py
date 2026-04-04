import os

from reporting.report_shared import AnalysisReport, ensure_report_dir, safe_report_text


def generate_jd_analysis_report(requirements_dict: dict, full_text: str, output_dir: str = ".") -> str:
    """Generate a standalone PDF report for the job description decomposition."""
    ensure_report_dir(output_dir)
    pdf_path = os.path.join(output_dir, "job_description_analysis.pdf")

    pdf = AnalysisReport(title="Job Description Analysis", subtitle="Extracted Requirements")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    for section, reqs in requirements_dict.items():
        if not reqs:
            continue
        pdf.add_section_title(section)
        pdf.add_bullet_list([safe_report_text(req) for req in reqs])
        pdf.ln(2)

    pdf.add_page()
    pdf.add_section_title("Original Job Description Context")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(pdf.epw, 5, safe_report_text(full_text), new_x="LMARGIN", new_y="NEXT")

    pdf.output(pdf_path)
    print(f"[REPORT] JD Analysis saved: {pdf_path}")
    return pdf_path
