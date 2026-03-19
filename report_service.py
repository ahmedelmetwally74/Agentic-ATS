"""
AgenticATS - Report Service
Generate PDF reports for Job Descriptions and Candidate Analysis.
Layouts are mode-driven based on sections.md.
"""

import os
import logging
from datetime import datetime

from fpdf import FPDF

logger = logging.getLogger(__name__)


def _safe_text(text: str) -> str:
    """
    Ensure text only contains characters within the ISO-8859-1 range.
    """
    if not text:
        return ""
    
    replacements = {
        "\u2022": "-",      # Bullet point
        "\u2013": "-",      # En dash
        "\u2014": "--",     # Em dash
        "\u2018": "'",      # Smart single quote
        "\u2019": "'",      # Smart single quote
        "\u201c": '"',      # Smart double quote
        "\u201d": '"',      # Smart double quote
        "\u00a0": " ",      # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("latin-1", "ignore").decode("latin-1")


class AnalysisReport(FPDF):
    """Custom PDF class for AgenticATS reports."""

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.report_title = title
        self.report_subtitle = subtitle

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(self.epw, 10, self.report_title, new_x="LMARGIN", new_y="NEXT", align="C")
        if self.report_subtitle:
            self.set_font("Helvetica", "I", 9)
            self.cell(self.epw, 6, f"{self.report_subtitle} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      new_x="LMARGIN", new_y="NEXT", align="C")
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(self.epw, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_section_title(self, title: str, fill_color=(230, 235, 245)):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(*fill_color)
        self.cell(self.epw, 9, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def add_key_value(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(45, 7, f"{key}:")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.epw - 45, 7, value, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def add_bullet_list(self, items: list[str]):
        self.set_font("Helvetica", "", 10)
        for item in items:
            self.cell(6, 6, "-", align="C")
            self.multi_cell(self.epw - 6, 6, item, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)


def generate_jd_analysis_report(requirements_dict: dict, full_text: str, output_dir: str = ".") -> str:
    """
    Generate a standalone PDF report for the Job Description decomposition.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "job_description_analysis.pdf")
    
    pdf = AnalysisReport(title="Job Description Analysis", subtitle="Extracted Requirements")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    for section, reqs in requirements_dict.items():
        if not reqs: continue
        pdf.add_section_title(section)
        pdf.add_bullet_list([_safe_text(r) for r in reqs])
        pdf.ln(2)
        
    pdf.add_page()
    pdf.add_section_title("Original Job Description Context")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(pdf.epw, 5, _safe_text(full_text), new_x="LMARGIN", new_y="NEXT")
    
    pdf.output(pdf_path)
    print(f"[REPORT] JD Analysis saved: {pdf_path}")
    return pdf_path


def generate_report_pdf(candidate_result: dict, mode: str, output_dir: str = ".") -> str:
    """
    Generate a mode-specific PDF report based on sections.md.
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

    if mode == "employer":
        # --- Company Mode Headers ---
        
        # 1. Ranking & Match Overview
        pdf.add_section_title("Ranking & Match Overview")
        pdf.add_key_value("Score", f"{candidate_result['score']:.4f}")
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(pdf.epw, 6, _safe_text(candidate_result.get("summary", "")), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # 2. Why this candidate fits the position
        pdf.add_section_title("Why this candidate fits the position")
        pdf.add_bullet_list([_safe_text(r) for r in candidate_result.get("reasons", [])])
        pdf.ln(4)

        # 3. Things to keep in mind
        pdf.add_section_title("Things to keep in mind")
        pdf.add_bullet_list([_safe_text(k) for k in candidate_result.get("keep_in_mind", [])])
        pdf.ln(4)

        # 4. Questions to ask the candidate
        pdf.add_section_title("Questions to ask the candidate")
        pdf.add_bullet_list([_safe_text(q) for q in candidate_result.get("questions", [])])

    else:
        # --- Applicant Mode Headers ---
        
        # 1. General comparison for each section
        pdf.add_section_title("CV vs Job Description: Section Comparison")
        for sec, data in candidate_result.get("detailed_sections", {}).items():
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 60, 140)
            pdf.cell(pdf.epw, 8, f"Job description vs {sec} cv", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 5, _safe_text(data.get("comparison", "N/A")), new_x="LMARGIN", new_y="NEXT")
            # Also show improvement suggestions for this specific section implicitly
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
