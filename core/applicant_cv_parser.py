from __future__ import annotations

import re
from collections import defaultdict


BULLET_PREFIX_RE = re.compile(r"^(?:[-*•▪◦●]|[\d]+[.)])\s*")
TOKEN_RE = re.compile(r"[a-z0-9+#./-]+", re.IGNORECASE)


def build_section_text_map(chunks: list[dict]) -> dict[str, str]:
    """Reconstruct ordered section text from stored CV chunks."""
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["section_name"]].append(chunk)

    section_text_map: dict[str, str] = {}
    for section_name, section_chunks in grouped.items():
        ordered_chunks = sorted(section_chunks, key=lambda item: item["chunk_index"])
        section_text_map[section_name] = "\n".join(
            chunk["chunk_text"].strip()
            for chunk in ordered_chunks
            if chunk.get("chunk_text", "").strip()
        ).strip()

    return section_text_map


def parse_applicant_cv_items(chunks: list[dict]) -> list[dict]:
    """
    Build a minimal applicant-specific runtime structure from stored chunks.

    Phase 1 keeps this intentionally simple:
    - each non-empty line becomes one structured item
    - non-bullet lines advance entry_index
    - bullet lines attach to the current entry
    """
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["section_name"]].append(chunk)

    parsed_items: list[dict] = []

    for section_name in sorted(grouped):
        ordered_chunks = sorted(grouped[section_name], key=lambda item: item["chunk_index"])
        entry_index = -1
        bullet_index = -1

        for chunk in ordered_chunks:
            lines = chunk.get("chunk_text", "").splitlines() or [chunk.get("chunk_text", "")]
            for raw_line in lines:
                text = raw_line.strip()
                if not text:
                    continue

                is_bullet = bool(BULLET_PREFIX_RE.match(text))
                cleaned_text = BULLET_PREFIX_RE.sub("", text).strip() if is_bullet else text
                if not cleaned_text:
                    continue

                if is_bullet:
                    if entry_index < 0:
                        entry_index = 0
                    bullet_index += 1
                    current_bullet_index: int | None = bullet_index
                else:
                    entry_index += 1
                    bullet_index = -1
                    current_bullet_index = None

                parsed_items.append(
                    {
                        "section_name": section_name,
                        "entry_index": max(entry_index, 0),
                        "bullet_index": current_bullet_index,
                        "text": cleaned_text,
                        "chunk_index": chunk["chunk_index"],
                    }
                )

    return parsed_items


def get_items_for_chunk(
    parsed_items: list[dict],
    section_name: str,
    chunk_index: int,
) -> list[dict]:
    """Return parsed items that originated from a specific stored chunk."""
    return [
        item
        for item in parsed_items
        if item["section_name"] == section_name and item["chunk_index"] == chunk_index
    ]


def get_section_items(parsed_items: list[dict], section_names: list[str] | set[str]) -> list[dict]:
    """Return items that belong to one of the requested sections."""
    normalized_sections = {section.strip().lower() for section in section_names}
    return [
        item
        for item in parsed_items
        if item["section_name"].strip().lower() in normalized_sections
    ]


def find_items_by_terms(
    parsed_items: list[dict],
    terms: list[str] | set[str],
    preferred_sections: list[str] | set[str] | None = None,
    section_weight_map: dict[str, int] | None = None,
    minimum_score: int | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Return items containing any of the exact normalized search terms.

    This is intentionally lexical and conservative for applicant-mode evidence
    tracing in categories like tools, education, and languages.
    """
    normalized_terms = [_normalize_text(term) for term in terms if _normalize_text(term)]
    if not normalized_terms:
        return []

    preferred = {section.strip().lower() for section in (preferred_sections or [])}
    scored_items = []

    for item in parsed_items:
        normalized_item_text = _normalize_text(item["text"])
        if not normalized_item_text:
            continue

        matched_terms = [
            term
            for term in normalized_terms
            if _contains_term(normalized_item_text, term)
        ]
        if not matched_terms:
            continue

        score = len(matched_terms) * 10
        if item["section_name"].strip().lower() in preferred:
            score += 5
        score += _resolve_section_score(item["section_name"], section_weight_map)
        score -= item.get("entry_index", 0)
        if minimum_score is not None and score < minimum_score:
            continue
        scored_items.append((score, item))

    scored_items.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored_items[:limit]]


def rank_items_for_requirement(
    requirement_text: str,
    items: list[dict],
    preferred_sections: list[str] | set[str] | None = None,
    section_weight_map: dict[str, int] | None = None,
    limit: int = 5,
) -> list[dict]:
    """Rank candidate items for one requirement using lightweight lexical overlap."""
    preferred = {section.strip().lower() for section in (preferred_sections or [])}
    requirement_tokens = {
        token for token in TOKEN_RE.findall(_normalize_text(requirement_text))
        if len(token) > 1
    }

    scored_items = []
    for item in items:
        item_text = _normalize_text(item["text"])
        item_tokens = {token for token in TOKEN_RE.findall(item_text) if len(token) > 1}
        overlap = len(requirement_tokens & item_tokens)
        score = overlap * 10
        if item["section_name"].strip().lower() in preferred:
            score += 5
        score += _resolve_section_score(item["section_name"], section_weight_map)
        score -= item.get("entry_index", 0)
        scored_items.append((score, item))

    scored_items.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored_items[:limit]]


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _resolve_section_score(section_name: str, section_weight_map: dict[str, int] | None) -> int:
    if not section_weight_map:
        return 0
    return int(section_weight_map.get(section_name.strip().lower(), 0))


def _contains_term(text: str, term: str) -> bool:
    if " " in term or "/" in term or "+" in term or "#" in term:
        return term in text

    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return bool(re.search(pattern, text))
