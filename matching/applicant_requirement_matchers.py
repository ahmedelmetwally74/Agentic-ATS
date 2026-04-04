from __future__ import annotations

import re

from core.applicant_cv_parser import find_items_by_terms, get_section_items, rank_items_for_requirement
from llm.llm_shared import unique_keep_order


EDUCATION_SECTION_NAMES = {
    "education",
    "academic background",
    "qualifications",
}
LANGUAGE_SECTION_NAMES = {"languages", "language"}
SUMMARY_SECTION_NAMES = {
    "summary",
    "professional summary",
    "profile",
    "about",
    "about me",
    "objective",
    "career objective",
    "header",
}
TOOL_SECTION_NAMES = {
    "skills",
    "technical skills",
    "core competencies",
    "key skills",
    "experience",
    "work experience",
    "professional experience",
    "projects",
    "technical projects",
    "personal projects",
    "key projects",
    "summary",
    "professional summary",
    "profile",
}

FIELD_FAMILY_PATTERNS = {
    "engineering": [
        r"\bengineering\b",
        r"\belectronics?\b",
        r"\bcommunication\b",
        r"\belectrical\b",
        r"\bmechanical\b",
        r"\bcivil\b",
        r"\bindustrial\b",
    ],
    "computer_science": [
        r"\bcomputer science\b",
        r"\bsoftware engineering\b",
        r"\binformation systems?\b",
        r"\bcomputer engineering\b",
    ],
    "statistics": [
        r"\bstatistics?\b",
        r"\bstatistical\b",
    ],
    "mathematics": [
        r"\bmathematics?\b",
        r"\bapplied math\b",
        r"\bapplied mathematics\b",
    ],
    "data_science": [
        r"\bdata science\b",
        r"\bdata analytics?\b",
    ],
    "business": [
        r"\bbusiness\b",
        r"\bcommerce\b",
        r"\bmanagement\b",
    ],
}

LANGUAGE_TERMS = {
    "english",
    "arabic",
    "french",
    "german",
    "spanish",
    "italian",
    "turkish",
    "chinese",
    "japanese",
    "russian",
    "urdu",
}

TOOL_ALIAS_MAP = {
    "power bi": {"power bi", "powerbi"},
    "scikit-learn": {"scikit-learn", "sklearn"},
    "postgresql": {"postgresql", "postgres"},
    "google cloud platform": {"google cloud platform", "gcp"},
    "amazon web services": {"amazon web services", "aws"},
    "microsoft azure": {"microsoft azure", "azure"},
}

GENERIC_REQUIREMENT_WORDS = {
    "experience",
    "with",
    "using",
    "in",
    "knowledge",
    "of",
    "hands-on",
    "hands",
    "on",
    "ability",
    "strong",
    "good",
    "solid",
    "working",
    "background",
    "proficiency",
    "proficient",
    "familiarity",
    "expertise",
    "skills",
    "tool",
    "tools",
    "technology",
    "technologies",
}

STOPWORD_TOKENS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "for",
    "with",
    "in",
    "of",
    "using",
    "experience",
    "knowledge",
    "ability",
    "strong",
    "good",
    "solid",
}

DEGREE_LEVEL_PATTERNS = {
    "bachelor": [r"\bbachelor'?s?\b", r"\bb\.?sc\b", r"\bbsc\b", r"\bundergraduate\b"],
    "master": [r"\bmaster'?s?\b", r"\bm\.?sc\b", r"\bmsc\b", r"\bma\b", r"\bmba\b"],
    "phd": [r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b"],
    "diploma": [r"\bdiploma\b"],
}
DEGREE_LEVEL_RANK = {
    "diploma": 0,
    "bachelor": 1,
    "master": 2,
    "phd": 3,
}
PROFICIENCY_TERMS = {
    "fluent",
    "native",
    "proficient",
    "professional",
    "advanced",
    "excellent",
}
LANGUAGE_CONTEXT_TERMS = {
    "fluent",
    "native",
    "proficient",
    "proficiency",
    "mother tongue",
    "bilingual",
}
WEAK_LANGUAGE_CONTEXT_TERMS = {
    "dialect",
    "identification",
    "translation",
    "translated",
    "speech",
    "dataset",
    "nlp",
    "text",
    "classification",
    "recognition",
    "transcription",
    "ocr",
}
STRONG_SECTION_WEIGHT_MAP = {
    "languages": 14,
    "language": 14,
    "summary": 8,
    "professional summary": 8,
    "profile": 8,
    "about": 7,
    "about me": 7,
    "objective": 6,
    "career objective": 6,
    "header": 8,
    "education": 12,
    "experience": 10,
    "work experience": 10,
    "professional experience": 10,
    "employment history": 10,
    "projects": 8,
    "technical projects": 8,
    "personal projects": 7,
    "skills": 7,
    "technical skills": 7,
    "courses": -3,
    "training": -2,
    "volunteer": -4,
    "volunteer experience": -4,
    "activities": -5,
    "extracurricular": -5,
    "awards": -3,
}


def evaluate_education_requirement(requirement: dict, parsed_items: list[dict]) -> dict:
    education_items = get_section_items(parsed_items, EDUCATION_SECTION_NAMES)
    top_items = rank_items_for_requirement(
        requirement["text"],
        education_items,
        preferred_sections=EDUCATION_SECTION_NAMES,
        section_weight_map=_education_section_weight_map(),
        limit=5,
    )
    top_evidence = _format_item_evidence(top_items[:3])

    if not education_items:
        status = "missing_or_insufficient"
        notes = "No Education-section evidence was found for this requirement."
        suggestion = build_safe_requirement_suggestion(requirement, status, top_evidence)
        return _build_result(requirement, status, top_evidence, notes, suggestion)

    requirement_degree_levels = _extract_degree_levels(requirement["text"])
    requirement_field_families = _extract_field_families(requirement["text"])
    cv_text = "\n".join(item["text"] for item in education_items)
    cv_degree_levels = _extract_degree_levels(cv_text)
    cv_field_families = _extract_field_families(cv_text)

    degree_match = _matches_degree_level(requirement_degree_levels, cv_degree_levels)
    field_match = _matches_field_family(requirement_field_families, cv_field_families)
    related_field = _has_related_field(requirement_field_families, cv_field_families)

    if degree_match and field_match and top_evidence:
        status = "matched_explicitly"
        notes = "The Education section explicitly shows a degree and field that satisfy this requirement."
    elif degree_match and related_field and top_evidence:
        status = "partially_matched"
        notes = "The Education section shows a closely related field, but it does not align as cleanly as an explicit direct match."
    elif degree_match and not requirement_field_families and top_evidence:
        status = "matched_explicitly"
        notes = "The Education section explicitly shows the required degree level."
    elif top_evidence and (cv_degree_levels or cv_field_families):
        status = "missing_or_insufficient"
        notes = "The CV includes education evidence, but it does not clearly satisfy the degree and field constraints in the JD."
    else:
        status = "missing_or_insufficient"
        notes = "No clear education evidence was found that satisfies this requirement."

    suggestion = build_safe_requirement_suggestion(requirement, status, top_evidence)
    return _build_result(requirement, status, top_evidence, notes, suggestion)


def evaluate_language_requirement(requirement: dict, parsed_items: list[dict]) -> dict:
    required_languages = _extract_languages(requirement["text"])
    if not required_languages:
        status = "missing_or_insufficient"
        notes = "No explicit language term could be extracted from this requirement."
        suggestion = build_safe_requirement_suggestion(requirement, status, [])
        return _build_result(requirement, status, [], notes, suggestion)

    candidate_hits = find_items_by_terms(
        parsed_items,
        required_languages,
        preferred_sections=LANGUAGE_SECTION_NAMES | SUMMARY_SECTION_NAMES,
        section_weight_map=_language_section_weight_map(),
        limit=10,
    )
    explicit_hits = [item for item in candidate_hits if _is_explicit_language_evidence(item)]
    weak_indirect_hits = [item for item in candidate_hits if _is_weak_language_context(item)]
    top_evidence = _format_item_evidence((explicit_hits or weak_indirect_hits)[:3])

    proficiency_required = _requires_language_proficiency(requirement["text"])
    proficiency_shown = any(_contains_any(item["text"], PROFICIENCY_TERMS) for item in explicit_hits)

    if explicit_hits:
        if proficiency_required and not proficiency_shown:
            status = "partially_matched"
            notes = "The language is explicitly mentioned, but the requested proficiency level is not clearly stated."
        else:
            status = "matched_explicitly"
            notes = "The CV explicitly mentions the required language."
    elif weak_indirect_hits:
        status = "not_explicitly_stated"
        notes = "The CV references this language indirectly, but it does not explicitly state spoken-language proficiency."
    else:
        status = "missing_or_insufficient"
        notes = "The required language is not explicitly stated in the CV."

    suggestion = build_safe_requirement_suggestion(requirement, status, top_evidence)
    return _build_result(requirement, status, top_evidence, notes, suggestion)


def evaluate_tool_requirement(requirement: dict, parsed_items: list[dict], evidence_chunks: list[dict]) -> dict:
    search_terms = _extract_tool_terms(requirement["text"])
    item_hits = find_items_by_terms(
        parsed_items,
        search_terms,
        preferred_sections=TOOL_SECTION_NAMES,
        section_weight_map=_professional_section_weight_map(),
        limit=8,
    )
    strong_item_hits = [item for item in item_hits if _section_strength(item["section_name"]) >= 3]
    weak_item_hits = [item for item in item_hits if _section_strength(item["section_name"]) < 3]
    direct_chunk_hits = [
        chunk
        for chunk in evidence_chunks
        if _chunk_contains_any(chunk.get("chunk_text", ""), search_terms)
    ]

    top_evidence = _format_item_evidence((strong_item_hits or weak_item_hits)[:3])
    if not top_evidence:
        top_evidence = _format_chunk_evidence(direct_chunk_hits[:3])

    if strong_item_hits:
        status = "matched_explicitly"
        notes = "The tool is explicitly stated in strong CV evidence such as Experience, Projects, or Skills."
    elif weak_item_hits:
        status = "partially_matched"
        notes = "The tool is explicitly mentioned, but only in weaker or less directly professional CV sections."
    elif direct_chunk_hits:
        status = "partially_matched"
        notes = "Related chunk evidence was retrieved, but the tool is not stated as clearly as a direct exact or alias match in parsed CV items."
    else:
        status = "missing_or_insufficient"
        notes = "No explicit exact or alias evidence for this tool was found in the CV."

    suggestion = build_safe_requirement_suggestion(requirement, status, top_evidence)
    return _build_result(requirement, status, top_evidence, notes, suggestion)


def build_evidence_profile(
    requirement: dict,
    parsed_items: list[dict],
    evidence_chunks: list[dict],
    top_evidence: list[str],
) -> dict:
    search_terms = _extract_requirement_terms(requirement["text"])
    exact_item_hits = find_items_by_terms(
        parsed_items,
        search_terms,
        preferred_sections=None,
        section_weight_map=_professional_section_weight_map(requirement),
        limit=12,
    )

    combined_chunk_text = "\n".join(chunk.get("chunk_text", "") for chunk in evidence_chunks)
    direct_term_hits = sum(1 for term in search_terms if _contains_term(combined_chunk_text, term))
    max_similarity = max((float(chunk.get("similarity", 0.0) or 0.0) for chunk in evidence_chunks), default=0.0)
    strong_section_hit_count = sum(1 for item in exact_item_hits if _section_strength(item["section_name"]) >= 3)
    weak_section_hit_count = sum(1 for item in exact_item_hits if _section_strength(item["section_name"]) <= 1)
    strongest_section_strength = max((_section_strength(item["section_name"]) for item in exact_item_hits), default=0)

    return {
        "has_evidence": bool(top_evidence or evidence_chunks),
        "exact_item_hit_count": len(exact_item_hits),
        "direct_term_hit_count": direct_term_hits,
        "max_similarity": max_similarity,
        "top_evidence_count": len(top_evidence),
        "search_terms": search_terms,
        "strong_section_hit_count": strong_section_hit_count,
        "weak_section_hit_count": weak_section_hit_count,
        "strongest_section_strength": strongest_section_strength,
        "strong_support": strong_section_hit_count > 0 or (direct_term_hits >= 2 and max_similarity >= 0.65),
        "moderate_support": len(exact_item_hits) > 0 or direct_term_hits >= 1 or max_similarity >= 0.55,
    }


def calibrate_requirement_status(requirement: dict, raw_status: str, evidence_profile: dict) -> str:
    if not evidence_profile.get("has_evidence"):
        return "missing_or_insufficient"

    strong_support = evidence_profile.get("strong_support", False)
    moderate_support = evidence_profile.get("moderate_support", False)
    category = requirement.get("category")
    strongest_section_strength = int(evidence_profile.get("strongest_section_strength", 0) or 0)
    weak_only_support = evidence_profile.get("exact_item_hit_count", 0) > 0 and strongest_section_strength < 3

    if raw_status == "matched_explicitly":
        if strong_support and strongest_section_strength >= 3:
            return "matched_explicitly"
        if moderate_support and not weak_only_support:
            return "partially_matched"
        return "missing_or_insufficient"

    if raw_status == "partially_matched":
        if moderate_support and not weak_only_support:
            return "partially_matched"
        return "missing_or_insufficient"

    if raw_status == "not_explicitly_stated":
        if category in {"language", "soft_skill"} and moderate_support and not strong_support:
            return "not_explicitly_stated"
        return "missing_or_insufficient"

    return "missing_or_insufficient"


def build_safe_requirement_suggestion(requirement: dict, status: str, top_evidence: list[str]) -> str:
    category = requirement.get("category", "other")
    section_group = requirement.get("section_group", "cv")

    if status == "matched_explicitly":
        return f"Keep the existing {section_group} evidence clear and easy to find."

    if category == "education":
        if top_evidence:
            return "Make the degree title and field of study explicit in the Education section."
        return "If truthful, add the exact degree title and field of study in the Education section."

    if category == "language":
        if top_evidence:
            return "State the language and proficiency level clearly in the Languages section."
        return "If truthful, add the required language explicitly in a Languages section."

    if category == "tool":
        if top_evidence:
            return "Name this tool explicitly where you used it, especially in Skills or Experience."
        return "If truthful, add explicit mentions of this tool in Skills or role-specific bullets."

    if status == "not_explicitly_stated":
        return "Make the strongest existing evidence more explicit in the CV without overstating it."

    if status == "partially_matched":
        return "Clarify the strongest relevant evidence and make the requirement fit more explicit without adding unsupported claims."

    return "If truthful, add direct CV evidence for this requirement rather than implying it indirectly."


def build_calibrated_notes(
    requirement: dict,
    status: str,
    evidence_profile: dict,
    top_evidence: list[str],
    raw_notes: str,
) -> str:
    if status == "matched_explicitly":
        return raw_notes.strip() or "The CV explicitly supports this requirement."
    if status == "partially_matched":
        return "The CV shows some relevant support for this requirement, but the evidence is not strong enough to count as a clear explicit match."
    if status == "not_explicitly_stated":
        return "The CV hints at this requirement, but it does not state it clearly enough to claim an explicit match."
    return "The available CV evidence is weak, indirect, or absent for this requirement."


def _build_result(requirement: dict, status: str, top_evidence: list[str], notes: str, suggestion: str) -> dict:
    return {
        "requirement_id": requirement["id"],
        "requirement_text": requirement["text"],
        "category": requirement["category"],
        "importance": requirement["importance"],
        "section_group": requirement["section_group"],
        "matching_strategy": requirement["matching_strategy"],
        "id": requirement["id"],
        "text": requirement["text"],
        "status": status,
        "top_evidence": unique_keep_order(top_evidence, limit=3),
        "notes": notes,
        "suggestion": suggestion,
    }


def _extract_degree_levels(text: str) -> set[str]:
    lower_text = text.lower()
    detected = set()
    for level, patterns in DEGREE_LEVEL_PATTERNS.items():
        if any(re.search(pattern, lower_text) for pattern in patterns):
            detected.add(level)
    return detected


def _extract_field_families(text: str) -> set[str]:
    lower_text = text.lower()
    detected = set()
    for family, patterns in FIELD_FAMILY_PATTERNS.items():
        if any(re.search(pattern, lower_text) for pattern in patterns):
            detected.add(family)
    return detected


def _extract_languages(text: str) -> list[str]:
    lower_text = text.lower()
    return [language for language in LANGUAGE_TERMS if re.search(rf"\b{re.escape(language)}\b", lower_text)]


def _requires_language_proficiency(text: str) -> bool:
    return _contains_any(text, PROFICIENCY_TERMS)


def _matches_degree_level(required_levels: set[str], cv_levels: set[str]) -> bool:
    if not required_levels:
        return True
    if not cv_levels:
        return False

    required_rank = min(DEGREE_LEVEL_RANK[level] for level in required_levels)
    cv_rank = max(DEGREE_LEVEL_RANK[level] for level in cv_levels)
    return cv_rank >= required_rank


def _matches_field_family(required_families: set[str], cv_families: set[str]) -> bool:
    if not required_families:
        return True
    return bool(required_families & cv_families)


def _has_related_field(required_families: set[str], cv_families: set[str]) -> bool:
    if not required_families or not cv_families:
        return False

    related_pairs = {
        ("computer_science", "engineering"),
        ("computer_science", "data_science"),
        ("statistics", "mathematics"),
        ("data_science", "statistics"),
    }

    for required_family in required_families:
        for cv_family in cv_families:
            if required_family == cv_family:
                return True
            if (required_family, cv_family) in related_pairs or (cv_family, required_family) in related_pairs:
                return True
    return False


def _extract_tool_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    lowered = normalized
    for prefix in [
        "experience with ",
        "experience in ",
        "knowledge of ",
        "hands-on experience with ",
        "hands on experience with ",
        "proficiency in ",
        "familiarity with ",
        "expertise in ",
        "working knowledge of ",
    ]:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
            break

    parts = re.split(r",|/| and | or ", lowered)
    candidates = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"\b(?:tools?|technologies|frameworks?|platforms?|libraries|systems?)\b", "", cleaned).strip()
        tokens = [token for token in re.findall(r"[a-z0-9+#.-]+", cleaned) if token not in GENERIC_REQUIREMENT_WORDS]
        if cleaned:
            candidates.append(cleaned)
        candidates.extend(tokens)

    aliases = set()
    for candidate in candidates:
        aliases.add(candidate)
        if candidate in TOOL_ALIAS_MAP:
            aliases.update(TOOL_ALIAS_MAP[candidate])
        for canonical, alias_terms in TOOL_ALIAS_MAP.items():
            if candidate == canonical or candidate in alias_terms:
                aliases.add(canonical)
                aliases.update(alias_terms)

    return [alias for alias in unique_keep_order([item for item in aliases if item], limit=10)]


def _extract_requirement_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    terms = []
    for token in re.findall(r"[a-z0-9+#./-]+", normalized):
        if token in STOPWORD_TOKENS or len(token) < 2:
            continue
        terms.append(token)

    phrases = [part.strip() for part in re.split(r",|/| and | or ", normalized) if part.strip()]
    for phrase in phrases:
        if len(phrase.split()) <= 5:
            terms.append(phrase)

    return unique_keep_order(terms, limit=8)


def _format_item_evidence(items: list[dict]) -> list[str]:
    return unique_keep_order(
        [f"{item['section_name']}: {item['text']}" for item in items if item.get("text")],
        limit=3,
    )


def _format_chunk_evidence(chunks: list[dict]) -> list[str]:
    evidence = []
    for chunk in chunks:
        snippet = " ".join(chunk.get("chunk_text", "").split())
        if snippet:
            evidence.append(f"{chunk['section_name']}: {snippet[:220]}")
    return unique_keep_order(evidence, limit=3)


def _contains_any(text: str, terms: set[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_contains_term(normalized, term) for term in terms)


def _chunk_contains_any(text: str, terms: list[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_contains_term(normalized, term) for term in terms)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_term(text: str, term: str) -> bool:
    normalized_term = _normalize_text(term)
    if not normalized_term:
        return False
    if " " in normalized_term or "/" in normalized_term or "+" in normalized_term or "#" in normalized_term:
        return normalized_term in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", text))


def _is_explicit_language_evidence(item: dict) -> bool:
    section_name = item["section_name"].strip().lower()
    text = _normalize_text(item["text"])

    if section_name in LANGUAGE_SECTION_NAMES:
        return True

    if _contains_any(text, LANGUAGE_CONTEXT_TERMS):
        return True

    if re.search(r"\blanguages?\b\s*[:\-]", text):
        return True

    if section_name in SUMMARY_SECTION_NAMES and len(_extract_languages(text)) >= 2 and len(text.split()) <= 8:
        return True

    return False


def _is_weak_language_context(item: dict) -> bool:
    if _is_explicit_language_evidence(item):
        return False

    text = _normalize_text(item["text"])
    if not _extract_languages(text):
        return False

    if _contains_any(text, WEAK_LANGUAGE_CONTEXT_TERMS):
        return True

    section_name = item["section_name"].strip().lower()
    return _section_strength(section_name) <= 2


def _section_strength(section_name: str) -> int:
    normalized = section_name.strip().lower()
    if normalized in {"experience", "work experience", "professional experience", "employment history"}:
        return 4
    if normalized in {"projects", "technical projects", "personal projects", "key projects"}:
        return 3
    if normalized in {"skills", "technical skills", "core competencies", "key skills", "education", "languages", "language"}:
        return 3
    if normalized in SUMMARY_SECTION_NAMES:
        return 2
    if normalized in {"courses", "training", "certifications", "awards"}:
        return 1
    if normalized in {"activities", "extracurricular", "volunteer", "volunteer experience"}:
        return 0
    return 2


def _education_section_weight_map() -> dict[str, int]:
    return {
        "education": 18,
        "academic background": 16,
        "qualifications": 14,
    }


def _language_section_weight_map() -> dict[str, int]:
    return {
        "languages": 18,
        "language": 18,
        "summary": 10,
        "professional summary": 10,
        "profile": 10,
        "about": 8,
        "about me": 8,
        "objective": 7,
        "career objective": 7,
        "header": 9,
        "projects": -4,
        "technical projects": -4,
        "courses": -6,
        "activities": -7,
    }


def _professional_section_weight_map(requirement: dict | None = None) -> dict[str, int]:
    weight_map = dict(STRONG_SECTION_WEIGHT_MAP)
    if requirement and requirement.get("section_group") == "projects":
        weight_map["projects"] = max(weight_map.get("projects", 0), 12)
        weight_map["technical projects"] = max(weight_map.get("technical projects", 0), 12)
    return weight_map
