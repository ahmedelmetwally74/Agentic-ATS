"""
AgenticATS - Document Utilities
Text normalization, cleanup, and helper functions for document processing.
"""

import re
import unicodedata

CATEGORY_GROUPS = {
    "Summary": ["summary", "profile", "about", "objective", "professional summary", "career objective", "executive summary"],
    "Experience": ["experience", "employment", "work history", "professional experience", "professional background"],
    "Education": ["education", "academic", "qualifications", "degree", "university", "college", "studies"],
    "Skills": ["skills", "technical skills", "core competencies", "key skills", "proficiencies", "stack", "tools"],
    "Projects": ["projects", "personal projects", "technical projects", "key projects"],
    "Courses": ["courses", "training"],
    "Certifications": ["certifications", "certificates", "licenses", "awards"],
    "Languages": ["languages", "translation", "fluency"]
}

def normalize_section(name: str) -> str:
    """Normalize a section name to its primary category using CATEGORY_GROUPS."""
    name_clean = name.strip().lower()
    for category, aliases in CATEGORY_GROUPS.items():
        if any(alias in name_clean for alias in aliases):
            return category
    return name



def clean_extracted_text(text: str) -> str:
    """
    Normalize whitespace and detect common list patterns (a., (a), (i), etc.).
    """
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)

    patterns = [
        r'(?<!\w)([a-z])\.',        # a.
        r'\(([a-z])\)',             # (a)
        r'\(([ivxlcdm]+)\)',        # (i), (ii), (iv)
    ]

    for pattern in patterns:
        text = re.sub(pattern, r'\n(\1)', text, flags=re.IGNORECASE)

    return text


def normalize_text_basic(text: str) -> str:
    """
    Conservative text normalizer:
    - Remove soft hyphen and zero-width characters
    - Join hyphenated end-of-line words: "Gewähr-\nleistung" -> "Gewährleistung"
    - Normalize spaces and trim trailing spaces
    - Collapse excessive blank lines
    """
    if not text:
        return text

    # Remove invisible/break chars
    text = text.replace("\u00AD", "")   # soft hyphen
    text = text.replace("\u200B", "")   # zero-width space
    text = text.replace("\u00A0", " ")  # NBSP -> normal space

    # Join hyphenated words split across a newline
    text = re.sub(r"(\w)[-\u2011]\s*\n\s*(\w)", r"\1\2", text)

    # Trim spaces before newlines
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Collapse 3+ blank lines -> 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def normalize_quotes(text: str) -> str:
    """
    Normalize smart/curly quotes to their ASCII equivalents.
    """
    replacements = {
        "\u2019": "'",   # right single quote / apostrophe
        "\u2018": "'",   # left single quote
        "\u201C": '"',   # left double quote
        "\u201D": '"',   # right double quote
        "\u201B": "'",   # modifier apostrophe
    }

    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    text = re.sub(r"\\u2019", "'", text)
    text = re.sub(r"\\u2018", "'", text)
    text = re.sub(r"\\u201c", '"', text)
    text = re.sub(r"\\u201d", '"', text)

    return text


def remove_redundant_title_from_clause(title: str, clause: str) -> str:
    """
    Remove the clause title from the start of the clause body,
    including optional numbering and punctuation.
    """
    title_clean = re.escape(title.strip().rstrip(".:"))
    pattern = rf"^\s*(\d+\.\s*)?{title_clean}[\.:]\?\s+"
    return re.sub(pattern, "", clause.strip(), flags=re.IGNORECASE)


def title_starts_clause(title: str, clause: str) -> bool:
    """
    Check if a clause body starts with its title text.
    """
    title_clean = re.escape(title.strip().rstrip(".:"))
    pattern = rf"^(\d+\.\s*)?{title_clean}[\.:]\?\s+"
    return bool(re.match(pattern, clause.strip(), flags=re.IGNORECASE))


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace to single spaces."""
    return ' '.join(text.lower().strip().split())


def flatten(s: str) -> str:
    """Collapse all whitespace to single spaces."""
    return " ".join(s.split())
