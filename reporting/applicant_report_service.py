from reporting.report_shared import AnalysisReport, build_candidate_report_path, ensure_report_dir, safe_report_text


def generate_applicant_report_pdf(candidate_result: dict, output_dir: str = ".") -> str:
    """Generate the applicant-mode candidate analysis PDF."""
    ensure_report_dir(output_dir)
    pdf_path = build_candidate_report_path(candidate_result, output_dir)

    pdf = AnalysisReport(
        title=f"Analysis: {candidate_result['file_name']}",
        subtitle="Applicant Mode Report",
    )
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    for section_name, data in candidate_result.get("detailed_sections", {}).items():
        pdf.add_section_title(section_name)

        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            pdf.epw,
            5,
            safe_report_text(data.get("comparison", "N/A")),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)

        missing_tools = data.get("missing_tools", [])
        missing_skills = data.get("missing_skills", [])
        missing_experience = data.get("missing_experience", [])
        missing_education = data.get("missing_education", [])
        section_suggestions = data.get("improvement_suggestions", [])

        if missing_tools:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, "Missing tools", new_x="LMARGIN", new_y="NEXT")
            pdf.add_bullet_list([safe_report_text(item) for item in missing_tools])

        if missing_skills:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, "Missing skills", new_x="LMARGIN", new_y="NEXT")
            pdf.add_bullet_list([safe_report_text(item) for item in missing_skills])

        if missing_experience:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, "Missing experience", new_x="LMARGIN", new_y="NEXT")
            pdf.add_bullet_list([safe_report_text(item) for item in missing_experience])

        if missing_education:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, "Missing education", new_x="LMARGIN", new_y="NEXT")
            pdf.add_bullet_list([safe_report_text(item) for item in missing_education])

        if section_suggestions:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, "Section improvement suggestions", new_x="LMARGIN", new_y="NEXT")
            pdf.add_bullet_list([safe_report_text(item) for item in section_suggestions])

        pdf.ln(4)

    pdf.add_page()
    pdf.add_section_title("General improvement suggestions")
    pdf.add_bullet_list([safe_report_text(item) for item in candidate_result.get("suggestions", [])])

    pdf.output(pdf_path)
    print(f"[REPORT] Report saved: {pdf_path}")
    return pdf_path
