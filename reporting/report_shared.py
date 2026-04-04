import logging
import os
from datetime import datetime

from fpdf import FPDF

logger = logging.getLogger(__name__)


def safe_report_text(text: str) -> str:
    """Ensure text only contains characters within the ISO-8859-1 range."""
    if not text:
        return ""

    replacements = {
        "\u2022": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
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
            self.cell(
                self.epw,
                6,
                f"{self.report_subtitle} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                new_x="LMARGIN",
                new_y="NEXT",
                align="C",
            )
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


def ensure_report_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def build_candidate_report_path(candidate_result: dict, output_dir: str) -> str:
    base_name = os.path.splitext(candidate_result["file_name"])[0]
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in base_name)
    return os.path.join(output_dir, f"{safe_name}_analysis.pdf")
