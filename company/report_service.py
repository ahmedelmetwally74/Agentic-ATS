"""
AgenticATS - Company Report Service
Generate PDF and Markdown reports for employer/company mode.
"""

import os
import logging

from utils.report_base import AnalysisReport, _safe_text, generate_jd_analysis_report as jd_report_base

logger = logging.getLogger(__name__)


def generate_report_pdf(candidate_result: dict, mode: str, output_dir: str = ".") -> str:
    """
    Generate a mode-specific PDF report for company mode (interview questions).
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

    # 1. Ranking & Match Overview
    pdf.add_section_title("Ranking & Match Overview")
    pdf.add_key_value("Score", f"{candidate_result['score']:.4f}")
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(pdf.epw, 6, _safe_text(candidate_result.get("summary", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Things to keep in mind
    pdf.add_section_title("Things to keep in mind")
    for sec, data in candidate_result.get("detailed_sections", {}).items():
        keep = data.get("things_to_keep_in_mind")
        if keep:
            if isinstance(keep, list):
                keep = " ".join(str(k) for k in keep)

            if isinstance(keep, str) and "no major" not in keep.lower() and "none" not in keep.lower():
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(pdf.epw, 8, f"{sec.capitalize()} Considerations", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(pdf.epw, 5, _safe_text(keep), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

    # 3. Questions to ask the candidate
    pdf.add_section_title("Questions to ask the candidate")
    pdf.add_bullet_list([_safe_text(q) for q in candidate_result.get("questions", [])])

    pdf.output(pdf_path)
    print(f"[REPORT] Report saved: {pdf_path}")
    return pdf_path


def generate_jd_analysis_report(requirements: list | dict, full_text: str, output_dir: str = ".") -> str:
    """Delegate to base implementation."""
    return jd_report_base(requirements, full_text, output_dir)


def generate_detailed_markdown_report(candidate_result: dict, mode: str, output_dir: str = ".") -> str:
    """
    Generate a detailed markdown report with deterministic embedding scores and justifications.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(candidate_result.get("file_name", "candidate"))[0]
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in base_name)
    md_path = os.path.join(output_dir, f"{safe_name}_detailed_analysis.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Deterministic Match Analysis: {candidate_result.get('file_name', 'Unknown')}\n\n")
        f.write(f"**Mode:** {mode.capitalize()} | **Total Score:** {candidate_result.get('score', 0):.4f}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> This analysis is 100% deterministic. Scores are derived from pure embedding similarity (cosine similarity) "
                "between literal JD requirements and the candidate's CV chunks.\n\n")
        f.write("---\n\n")

        # Get match_checklist from detailed_sections
        match_checklist = []
        if "detailed_sections" in candidate_result and "Job Requirements Match" in candidate_result["detailed_sections"]:
            match_checklist = candidate_result["detailed_sections"]["Job Requirements Match"].get("match_checklist", [])

        if not match_checklist:
            f.write("## Section: Legacy Section Analyses\n\n")
            for sec, data in candidate_result.get("detailed_sections", {}).items():
                f.write(f"### Section: {sec} (Match Score: {data.get('match_score', 0.0):.4f})\n\n")
                f.write(data.get("things_to_keep_in_mind", ""))
                f.write("\n\n")
        else:
            # Things to Keep in Mind
            job_req_data = candidate_result.get("detailed_sections", {}).get("Job Requirements Match", {})
            things_to_keep = job_req_data.get("things_to_keep_in_mind", "")

            if things_to_keep:
                f.write("## Things to Keep in Mind\n\n")
                f.write(f"{things_to_keep}\n\n")
                f.write("---\n\n")

            # Requirements Match Checklist Table
            f.write("## Requirements Match Checklist\n\n")
            f.write("| Status | Score | Requirement | Reason |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for item in match_checklist:
                status = item.get("status", "No Match")
                score = item.get("score", 0.0)
                req = item.get("requirement", "")
                reason = item.get("reason", "").replace("\n", " ")
                f.write(f"| **{status}** | `{score:.4f}` | {req} | {reason} |\n")
            f.write("\n\n---\n\n")

            # Detailed Evidence & Interview Questions
            f.write("## Detailed Evidence & Interview Questions\n\n")
            for idx, item in enumerate(match_checklist, 1):
                f.write(f"### {idx}. {item.get('requirement')}\n")
                f.write(f"**Status:** {item.get('status')} (`{item.get('score'):.4f}`)\n\n")
                f.write("**Best Matching CV Evidence:**\n")
                f.write(f"> {item.get('evidence', 'No evidence found.').strip()}\n\n")
                f.write("**Suggested Interview Questions:**\n")
                for q in item.get("questions", []):
                    f.write(f"- {q}\n")
                f.write("\n---\n\n")

    print(f"[REPORT] Detailed Markdown Report saved: {md_path}")
    return md_path
