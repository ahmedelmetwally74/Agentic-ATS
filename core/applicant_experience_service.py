from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from core.applicant_cv_parser import build_section_text_map


EXPERIENCE_SECTION_NAMES = {
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "career history",
}

MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

DATE_TOKEN_PATTERN = (
    r"(?:"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{4}"
    r"|(?:0?[1-9]|1[0-2])[/-]\d{4}"
    r"|\d{4}"
    r"|present|current|now|today"
    r")"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN_PATTERN})\s*(?:-|–|—|to|until|through)\s*(?P<end>{DATE_TOKEN_PATTERN})",
    re.IGNORECASE,
)
YEAR_REQUIREMENT_RE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*(?:\+|\-|to)?\s*(?P<high>\d+(?:\.\d+)?)?\s*years?",
    re.IGNORECASE,
)


def extract_experience_section_text(chunks: list[dict]) -> str:
    """Reconstruct only the main Experience section text from stored CV chunks."""
    section_text_map = build_section_text_map(chunks)
    matching_sections = [
        text
        for section_name, text in section_text_map.items()
        if section_name.strip().lower() in EXPERIENCE_SECTION_NAMES and text.strip()
    ]
    return "\n\n".join(matching_sections).strip()


def parse_experience_entries(experience_text: str) -> list[dict]:
    """Extract simple experience entries anchored by explicit date ranges."""
    if not experience_text.strip():
        return []

    entries: list[dict] = []
    lines = [line.strip() for line in experience_text.splitlines() if line.strip()]

    for idx, line in enumerate(lines):
        matches = list(DATE_RANGE_RE.finditer(_normalize_date_text(line)))
        for match in matches:
            entries.append(
                {
                    "entry_index": idx,
                    "text": line,
                    "date_range": match.group(0),
                }
            )

    if entries:
        return entries

    normalized_text = _normalize_date_text(experience_text)
    for idx, match in enumerate(DATE_RANGE_RE.finditer(normalized_text)):
        entries.append(
            {
                "entry_index": idx,
                "text": match.group(0),
                "date_range": match.group(0),
            }
        )

    return entries


def parse_date_range(date_range_text: str, today: date | None = None) -> tuple[date, date] | None:
    """Parse a date range like 'Feb 2025 - Present' into concrete dates."""
    today = today or date.today()
    normalized_text = _normalize_date_text(date_range_text)
    match = DATE_RANGE_RE.search(normalized_text)
    if not match:
        return None

    start_date = _parse_date_token(match.group("start"), today=today, end_of_period=False)
    end_date = _parse_date_token(match.group("end"), today=today, end_of_period=True)

    if not start_date or not end_date or end_date < start_date:
        return None

    return start_date, end_date


def merge_overlapping_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping date ranges so time is not double-counted."""
    if not ranges:
        return []

    ordered = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[date, date]] = [ordered[0]]

    for start_date, end_date in ordered[1:]:
        current_start, current_end = merged[-1]
        if start_date <= current_end:
            merged[-1] = (current_start, max(current_end, end_date))
        else:
            merged.append((start_date, end_date))

    return merged


def summarize_experience_duration(chunks: list[dict], today: date | None = None) -> dict:
    """Calculate overlap-aware experience duration from the Experience section only."""
    today = today or date.today()
    experience_text = extract_experience_section_text(chunks)
    entries = parse_experience_entries(experience_text)

    parsed_ranges: list[tuple[date, date]] = []
    valid_entries: list[dict] = []

    for entry in entries:
        parsed_range = parse_date_range(entry["date_range"], today=today)
        if not parsed_range:
            continue

        start_date, end_date = parsed_range
        parsed_ranges.append(parsed_range)
        valid_entries.append(
            {
                "entry_index": entry["entry_index"],
                "text": entry["text"],
                "date_range": entry["date_range"],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
        )

    merged_ranges = merge_overlapping_ranges(parsed_ranges)
    total_days = sum((end_date - start_date).days + 1 for start_date, end_date in merged_ranges)
    total_years = round(total_days / 365.25, 2) if total_days else 0.0

    return {
        "today": today.isoformat(),
        "experience_section_found": bool(experience_text.strip()),
        "experience_section_names": sorted(
            {
                chunk["section_name"]
                for chunk in chunks
                if chunk["section_name"].strip().lower() in EXPERIENCE_SECTION_NAMES
            }
        ),
        "parsed_entries": valid_entries,
        "merged_ranges": [
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
            for start_date, end_date in merged_ranges
        ],
        "total_days": total_days,
        "total_years": total_years,
        "top_evidence": [entry["text"] for entry in valid_entries[:3]],
    }


def extract_required_years(requirement_text: str) -> float | None:
    """Extract the minimum years required from a JD requirement when present."""
    match = YEAR_REQUIREMENT_RE.search(requirement_text)
    if not match:
        return None

    low_value = match.group("low")
    try:
        return float(low_value)
    except ValueError:
        return None


def _normalize_date_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2012": "-",
        "\u2011": "-",
        "\u00a0": " ",
        "â€“": "-",
        "â€”": "-",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_date_token(token: str, today: date, end_of_period: bool) -> date | None:
    token_clean = token.strip().lower().replace(".", "")

    if token_clean in {"present", "current", "now", "today"}:
        return today

    month_year_match = re.fullmatch(r"([a-z]+)\s+(\d{4})", token_clean)
    if month_year_match:
        month_name = month_year_match.group(1)
        year = int(month_year_match.group(2))
        month = MONTH_NAME_TO_NUMBER.get(month_name)
        if month is None:
            return None
        return _build_date(year, month, end_of_period=end_of_period)

    numeric_month_match = re.fullmatch(r"(0?[1-9]|1[0-2])[/-](\d{4})", token_clean)
    if numeric_month_match:
        month = int(numeric_month_match.group(1))
        year = int(numeric_month_match.group(2))
        return _build_date(year, month, end_of_period=end_of_period)

    year_only_match = re.fullmatch(r"(\d{4})", token_clean)
    if year_only_match:
        year = int(year_only_match.group(1))
        month = 12 if end_of_period else 1
        return _build_date(year, month, end_of_period=end_of_period)

    return None


def _build_date(year: int, month: int, end_of_period: bool) -> date:
    if end_of_period:
        next_month = datetime(year, month, 1)
        if month == 12:
            following = datetime(year + 1, 1, 1)
        else:
            following = datetime(year, month + 1, 1)
        return (following - timedelta(days=1)).date()

    return date(year, month, 1)
