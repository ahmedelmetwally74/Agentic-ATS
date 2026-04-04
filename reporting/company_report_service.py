from reporting.report_shared import AnalysisReport, build_candidate_report_path, ensure_report_dir, safe_report_text


def generate_company_report_pdf(candidate_result: dict, output_dir: str = ".") -> str:
    """Generate the company-mode candidate analysis PDF."""
    ensure_report_dir(output_dir)
    pdf_path = build_candidate_report_path(candidate_result, output_dir)

    pdf = AnalysisReport(
        title=f"Analysis: {candidate_result['file_name']}",
        subtitle="Company Mode Report",
    )
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_section_title("Ranking & Match Overview")
    pdf.add_key_value("Score", f"{candidate_result['score']:.4f}")
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(pdf.epw, 6, safe_report_text(candidate_result.get("summary", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.add_section_title("Why this candidate fits the position")
    pdf.add_bullet_list([safe_report_text(reason) for reason in candidate_result.get("reasons", [])])
    pdf.ln(4)

    pdf.add_section_title("Things to keep in mind")
    pdf.add_bullet_list([safe_report_text(item) for item in candidate_result.get("keep_in_mind", [])])
    pdf.ln(4)

    pdf.add_section_title("Questions to ask the candidate")
    pdf.add_bullet_list([safe_report_text(question) for question in candidate_result.get("questions", [])])

    pdf.output(pdf_path)
    print(f"[REPORT] Report saved: {pdf_path}")
    return pdf_path
