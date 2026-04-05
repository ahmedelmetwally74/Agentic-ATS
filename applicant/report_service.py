"""
AgenticATS - Applicant Report Service
Generate PDF and Markdown reports for applicant/client mode (CV improvement suggestions).
"""

import os
import logging

from utils.report_base import AnalysisReport, _safe_text, generate_jd_analysis_report as jd_report_base

logger = logging.getLogger(__name__)


def generate_report_pdf(candidate_result: dict, mode: str, output_dir: str = ".") -> str:
    """
    Generate a mode-specific PDF report for applicant mode (CV improvement suggestions).
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(candidate_result["file_name"])[0]
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in base_name)
    pdf_path = os.path.join(output_dir, f"{safe_name}_analysis.pdf")

    pdf = AnalysisReport(title=f"Analysis: {candidate_result['file_name']}",
                         subtitle=f"{mode.capitalize()} Mode Report")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # 1. General comparison for each section
    pdf.add_section_title("CV vs Job Description: Section Comparison")
    for sec, data in candidate_result.get("detailed_sections", {}).items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 60, 140)
        # JD [Section] Requirements vs CV [Section] Context
        pdf.cell(pdf.epw, 8, f"[JD {sec} Requirements] vs. [CV {sec} Context]", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        # Show JD Requirements (the JD chunk)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(pdf.epw, 5, "JD Chunks:", new_x="LMARGIN", new_y="NEXT")
        for req in data.get("jd_requirements", []):
            pdf.cell(6, 5, "-")
            pdf.multi_cell(pdf.epw-6, 5, _safe_text(req), new_x="LMARGIN", new_y="NEXT")

        # Show comparison
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(pdf.epw, 6, "Analysis:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, _safe_text(data.get("comparison", "N/A")), new_x="LMARGIN", new_y="NEXT")

        # Show suggestions
        sug = data.get("improvement_suggestions")
        if sug:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 50, 50)
            pdf.multi_cell(pdf.epw, 5, f"Section Tip: {_safe_text(sug)}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # 2. General improvement suggestions
    pdf.add_page()
    pdf.add_section_title("General improvement suggestions")
    pdf.add_bullet_list([_safe_text(s) for s in candidate_result.get("suggestions", [])])

    pdf.output(pdf_path)
    print(f"[REPORT] Report saved: {pdf_path}")
    return pdf_path


def generate_jd_analysis_report(requirements: list | dict, full_text: str, output_dir: str = ".") -> str:
    """Delegate to base implementation."""
    return jd_report_base(requirements, full_text, output_dir)


def generate_detailed_markdown_report(candidate_result: dict, mode: str, output_dir: str = ".") -> str:
    """
    Generate a detailed markdown report for applicant mode.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(candidate_result.get("file_name", "candidate"))[0]
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in base_name)
    md_path = os.path.join(output_dir, f"{safe_name}_detailed_analysis.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# CV Analysis: {candidate_result.get('file_name', 'Unknown')}\n\n")
        f.write(f"**Mode:** {mode.capitalize()} | **Score:** {candidate_result.get('score', 0):.4f}\n\n")
        f.write("---\n\n")

        # Section-by-section comparison
        f.write("## Section Comparisons\n\n")
        for sec, data in candidate_result.get("detailed_sections", {}).items():
            f.write(f"### {sec}\n\n")
            f.write(f"**JD Requirements:** {', '.join(data.get('jd_requirements', [])[:3])}\n\n")
            f.write(f"**Analysis:** {data.get('comparison', 'N/A')}\n\n")
            if data.get("improvement_suggestions"):
                f.write(f"> **Tip:** {data['improvement_suggestions']}\n\n")
            f.write("---\n\n")

        # General suggestions
        f.write("## General Improvement Suggestions\n\n")
        for sugg in candidate_result.get("suggestions", []):
            f.write(f"- {sugg}\n")

    print(f"[REPORT] Detailed Markdown Report saved: {md_path}")
    return md_path
