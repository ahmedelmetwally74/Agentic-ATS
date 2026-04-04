from reporting.report_shared import AnalysisReport, build_candidate_report_path, ensure_report_dir, safe_report_text


def generate_applicant_report_pdf(candidate_result: dict, output_dir: str = ".") -> str:
    """Generate the requirement-centric applicant-mode PDF report."""
    ensure_report_dir(output_dir)
    pdf_path = build_candidate_report_path(candidate_result, output_dir)

    pdf = AnalysisReport(
        title=f"Analysis: {candidate_result['file_name']}",
        subtitle="Applicant Mode Report",
    )
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_section_title("Overall Summary")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        pdf.epw,
        5,
        safe_report_text(candidate_result.get("summary", "N/A")),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    status_counts = candidate_result.get("status_counts", {})
    if status_counts:
        pdf.add_section_title("Requirement Status Overview")
        pdf.add_bullet_list(
            [
                safe_report_text(f"matched_explicitly: {status_counts.get('matched_explicitly', 0)}"),
                safe_report_text(f"partially_matched: {status_counts.get('partially_matched', 0)}"),
                safe_report_text(f"not_explicitly_stated: {status_counts.get('not_explicitly_stated', 0)}"),
                safe_report_text(f"missing_or_insufficient: {status_counts.get('missing_or_insufficient', 0)}"),
            ]
        )
        pdf.ln(2)

    experience_summary = candidate_result.get("experience_summary", {})
    if experience_summary:
        pdf.add_section_title("Experience Duration Summary")
        pdf.add_bullet_list(
            [
                safe_report_text(f"Runtime today date: {experience_summary.get('today', 'N/A')}"),
                safe_report_text(f"Experience section found: {experience_summary.get('experience_section_found', False)}"),
                safe_report_text(f"Overlap-aware total years: {experience_summary.get('total_years', 0.0)}"),
            ]
        )
        pdf.ln(2)

    for requirement_result in candidate_result.get("requirement_results", []):
        requirement_id = safe_report_text(str(requirement_result.get("requirement_id", requirement_result.get("id", ""))))
        status = safe_report_text(str(requirement_result.get("status", "")))
        pdf.add_section_title(f"{requirement_id} | {status}")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(pdf.epw, 6, "Requirement", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            pdf.epw,
            5,
            safe_report_text(requirement_result.get("requirement_text", requirement_result.get("text", ""))),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1)

        pdf.add_bullet_list(
            [
                safe_report_text(f"category: {requirement_result.get('category', '')}"),
                safe_report_text(f"importance: {requirement_result.get('importance', '')}"),
                safe_report_text(f"section_group: {requirement_result.get('section_group', '')}"),
                safe_report_text(f"matching_strategy: {requirement_result.get('matching_strategy', '')}"),
            ]
        )

        top_evidence = requirement_result.get("top_evidence", [])
        if top_evidence:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(pdf.epw, 6, "Top evidence", new_x="LMARGIN", new_y="NEXT")
            pdf.add_bullet_list([safe_report_text(item) for item in top_evidence])

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(pdf.epw, 6, "Notes", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            pdf.epw,
            5,
            safe_report_text(requirement_result.get("notes", "")),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(pdf.epw, 6, "Suggestion", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            pdf.epw,
            5,
            safe_report_text(requirement_result.get("suggestion", "")),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(4)

    pdf.add_page()
    pdf.add_section_title("Overall CV Improvement Suggestions")
    pdf.add_bullet_list([safe_report_text(item) for item in candidate_result.get("suggestions", [])])

    pdf.output(pdf_path)
    print(f"[REPORT] Report saved: {pdf_path}")
    return pdf_path
