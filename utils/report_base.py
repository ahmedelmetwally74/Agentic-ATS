"""
AgenticATS - Report Service Base
Shared utilities for PDF report generation.
"""

import os
from datetime import datetime
from fpdf import FPDF


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


def generate_jd_analysis_report(requirements: list | dict, full_text: str, output_dir: str = ".") -> str:
    """
    Generate a standalone PDF report for the Job Description decomposition.
    """
    from utils.report_base import AnalysisReport, _safe_text

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "job_description_analysis.pdf")

    pdf = AnalysisReport(title="Job Description Analysis", subtitle=f"Strict Extraction | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_section_title("Extracted Literal Requirements")

    if isinstance(requirements, list):
        pdf.add_bullet_list([_safe_text(r) for r in requirements])
    elif isinstance(requirements, dict):
        # Backward compatibility for categorized dicts
        for sec, reqs in requirements.items():
            if reqs:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(pdf.epw, 7, f"{sec}:", new_x="LMARGIN", new_y="NEXT")
                pdf.add_bullet_list([_safe_text(r) for r in reqs])

    pdf.add_page()
    pdf.add_section_title("Original Job Description Context")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(pdf.epw, 5, _safe_text(full_text), new_x="LMARGIN", new_y="NEXT")

    pdf.output(pdf_path)
    print(f"[REPORT] JD Analysis saved: {pdf_path}")
    return pdf_path
