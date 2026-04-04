from __future__ import annotations

import json
import logging
import re

from core.applicant_experience_service import extract_required_years
from llm.llm_shared import call_llm, clean_llm_output, extract_json_payload, to_str_list, unique_keep_order

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "tool",
    "skill",
    "experience",
    "education",
    "language",
    "soft_skill",
    "domain",
    "other",
}
ALLOWED_IMPORTANCE = {"required", "preferred"}
ALLOWED_SECTION_GROUPS = {
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "languages",
    "other",
}
ALLOWED_MATCHING_STRATEGIES = {
    "semantic_evidence",
    "exact_plus_semantic",
    "experience_duration",
}
ALLOWED_STATUSES = {
    "matched_explicitly",
    "partially_matched",
    "not_explicitly_stated",
    "missing_or_insufficient",
}

GENERIC_SECTION_LABELS = {
    "summary",
    "education",
    "experience",
    "skills",
    "projects",
    "languages",
    "certifications",
    "responsibilities",
    "requirements",
    "qualifications",
}
ROLE_TITLE_RE = re.compile(r"\b(?:sr|senior|jr|junior|lead|principal|staff|head)\.?\b", re.IGNORECASE)
REQUIREMENT_SIGNAL_RE = re.compile(
    r"\b("
    r"\d+\+?\s*years?|"
    r"experience|experienced|proficien|knowledge|familiar|"
    r"ability|able to|hands-?on|degree|bachelor|master|phd|"
    r"certification|fluent|language|english|arabic|python|sql|"
    r"tensorflow|pytorch|fastapi|aws|azure|docker|kubernetes|"
    r"machine learning|deep learning|analytics|stakeholder|communication|leadership"
    r")\b",
    re.IGNORECASE,
)
TECH_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9+#./-]{2,}\b")
STOPWORD_TOKENS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "with",
    "experience",
    "knowledge",
    "ability",
    "strong",
    "good",
    "solid",
}


def decompose_job_description_to_requirements(jd_text: str) -> list[dict]:
    """Applicant-specific JD decomposition into structured requirement objects."""
    system_prompt = (
        "You are extracting structured job requirements for applicant-mode matching.\n"
        "A valid requirement is something the candidate must already have or must explicitly demonstrate.\n"
        "Examples of valid requirements: years of experience, tools, technologies, languages, degrees, fields of study, domain knowledge, soft skills, certifications.\n"
        "Examples that must be excluded: job titles, company names, section headings, labels, role purpose text, generic meta headings, and document structure words.\n"
        "Return ONLY valid JSON.\n"
        "Return a JSON array. Each object must contain exactly these keys:\n"
        '[{"id":"req_01","text":"...","category":"tool|skill|experience|education|language|soft_skill|domain|other","importance":"required|preferred","section_group":"summary|experience|education|skills|projects|certifications|languages|other","matching_strategy":"semantic_evidence|exact_plus_semantic|experience_duration"}]\n'
        "Rules:\n"
        "- Split the JD into atomic candidate-side requirements.\n"
        "- Exclude document headings, labels, role titles, company labels, and meta text.\n"
        "- Keep each requirement short, specific, and candidate-facing.\n"
        "- Use experience_duration only when the requirement explicitly asks for years of experience.\n"
        "- Use exact_plus_semantic for tools, technologies, and explicit language requirements.\n"
        "- Use semantic_evidence for other requirement types in Phase 1.\n"
        "- Infer importance from wording: must/required/need => required, preferred/nice to have/plus => preferred.\n"
        "- Do not return explanations outside the JSON array."
    )
    user_prompt = (
        f"Job Description:\n{jd_text}\n\n"
        "Extract only valid candidate requirements now.\n"
        "Return JSON only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.1, max_tokens=1400)
        parsed, extraction_status = _extract_applicant_requirement_payload(raw)
        if extraction_status == "direct_parse_failed":
            logger.warning("Applicant JD decomposition direct JSON parse failed; trying applicant-side array extraction.")
        elif extraction_status == "truncated_singleton_rejected":
            logger.warning("Applicant JD decomposition rejected a truncated single-item payload from a likely multi-item response.")

        normalized = _normalize_requirement_payload(parsed, jd_text=jd_text)
        if normalized:
            return normalized

        if parsed is None:
            logger.warning("Applicant JD decomposition JSON repair attempt started after parse failure.")
        elif _coerce_requirement_payload_to_list(parsed) is None:
            logger.warning("Applicant JD decomposition payload shape mismatch after JSON parse.")
        else:
            logger.warning("Applicant JD decomposition validation rejected all extracted items.")

        repaired = _repair_jd_requirements_json(raw, jd_text)
        repaired_normalized = _normalize_requirement_payload(repaired, jd_text=jd_text)
        if repaired_normalized:
            return repaired_normalized

        logger.warning(f"Applicant JD decomposition returned non-usable JSON after repair. Raw output preview: {raw[:1000]}")
        return fallback_decompose_job_description_to_requirements(jd_text)
    except Exception as e:
        logger.warning(f"Applicant JD decomposition failed: {e}")
        return fallback_decompose_job_description_to_requirements(jd_text)


def fallback_decompose_job_description_to_requirements(jd_text: str) -> list[dict]:
    """Deterministic fallback when structured JD decomposition fails."""
    candidate_lines: list[str] = []

    for raw_line in jd_text.splitlines():
        cleaned = _normalize_requirement_text(raw_line)
        if not cleaned:
            continue

        parts = [part.strip() for part in re.split(r"[;•]", cleaned) if part.strip()]
        for part in parts or [cleaned]:
            verdict = _requirement_shape_verdict(part)
            if verdict == "accepted":
                candidate_lines.append(part)

    normalized = []
    for index, line in enumerate(candidate_lines, start=1):
        requirement, verdict = _normalize_requirement_object({"text": line}, index)
        if requirement and verdict == "accepted":
            normalized.append(requirement)

    return _resequence_requirement_ids(normalized)


def analyze_requirement_match(
    requirement: dict,
    evidence_chunks: list[dict],
    top_evidence: list[str] | None = None,
) -> dict:
    """Evaluate one requirement against top-k evidence from the current CV only."""
    fallback = fallback_requirement_match(requirement, evidence_chunks, top_evidence=top_evidence)
    evidence_text = _format_evidence_chunks(evidence_chunks)
    evidence_summary = "\n".join(f"- {item}" for item in (top_evidence or fallback["top_evidence"]))

    system_prompt = (
        "You are evaluating one applicant CV against one job requirement.\n"
        "Use only the provided CV evidence.\n"
        "Return ONLY valid JSON using this schema:\n"
        '{'
        '"status":"matched_explicitly|partially_matched|not_explicitly_stated|missing_or_insufficient",'
        '"notes":"1-3 sentence evidence-based explanation",'
        '"suggestion":"one practical CV clarification suggestion"'
        '}\n'
        "Rules:\n"
        "- matched_explicitly only if the requirement is directly stated in the evidence.\n"
        "- partially_matched only if some relevant evidence exists but it is incomplete.\n"
        "- not_explicitly_stated is a narrow status and should be used only when the CV strongly hints at the requirement without stating it directly.\n"
        "- missing_or_insufficient when the evidence is absent, weak, or too indirect.\n"
        "- Suggestions must not imply fake years, fake seniority, fake tools, or fake qualifications.\n"
        "- Prefer phrasing like 'make existing evidence clearer' or 'if truthful, add explicit evidence'.\n"
        "- Do not include markdown or any explanation outside the JSON."
    )
    user_prompt = (
        f"Requirement object:\n{json.dumps(requirement, ensure_ascii=False)}\n\n"
        f"Top evidence summary:\n{evidence_summary or '- No evidence found'}\n\n"
        f"Retrieved CV evidence blocks:\n{evidence_text or '(No evidence retrieved)'}\n\n"
        "Evaluate the requirement now.\n"
        "Return JSON only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.1, max_tokens=600)
        parsed = extract_json_payload(raw)
        if not isinstance(parsed, dict):
            parsed = _repair_requirement_analysis_json(requirement, raw)

        if not isinstance(parsed, dict):
            logger.warning(f"Requirement analysis parse failed for {requirement['id']}. Raw output preview: {raw[:800]}")
            return fallback

        status = str(parsed.get("status", "")).strip()
        if status not in ALLOWED_STATUSES:
            status = fallback["status"]

        notes = str(parsed.get("notes", "")).strip() or fallback["notes"]
        suggestion = str(parsed.get("suggestion", "")).strip() or fallback["suggestion"]

        return {
            **_base_requirement_result(requirement),
            "status": status,
            "top_evidence": top_evidence or fallback["top_evidence"],
            "notes": notes,
            "suggestion": suggestion,
        }
    except Exception as e:
        logger.warning(f"Requirement analysis failed for {requirement['id']}: {e}")
        return fallback


def fallback_requirement_match(
    requirement: dict,
    evidence_chunks: list[dict],
    top_evidence: list[str] | None = None,
) -> dict:
    """Deterministic fallback for requirement-level applicant analysis."""
    evidence_list = top_evidence or _default_top_evidence(evidence_chunks)
    has_evidence = bool(evidence_list)

    if has_evidence:
        status = "missing_or_insufficient"
        notes = "Relevant chunks were retrieved, but the structured analysis could not confirm that the requirement is actually supported."
        suggestion = "If truthful, make the strongest existing evidence for this requirement more explicit in the CV."
    else:
        status = "missing_or_insufficient"
        notes = "No strong supporting evidence was retrieved from the CV for this requirement."
        suggestion = "If truthful, add direct CV evidence for this requirement."

    return {
        **_base_requirement_result(requirement),
        "status": status,
        "top_evidence": evidence_list,
        "notes": notes,
        "suggestion": suggestion,
    }


def synthesize_applicant_requirement_results(
    requirement_results: list[dict],
    experience_summary: dict | None = None,
) -> dict:
    """Build a grounded deterministic applicant summary from validated results."""
    return {
        "general_summary": _build_grounded_summary(requirement_results, experience_summary or {}),
        "overall_suggestions": _build_safe_overall_suggestions(requirement_results),
    }


def fallback_requirement_synthesis(requirement_results: list[dict]) -> dict:
    """Backward-compatible deterministic synthesis fallback."""
    return synthesize_applicant_requirement_results(requirement_results, experience_summary={})


def _repair_requirement_analysis_json(requirement: dict, raw_response: str):
    system_prompt = (
        "Convert the applicant-analysis response into strict JSON.\n"
        "Return ONLY valid JSON with this schema:\n"
        '{'
        '"status":"matched_explicitly|partially_matched|not_explicitly_stated|missing_or_insufficient",'
        '"notes":"...",'
        '"suggestion":"..."'
        '}\n'
        "Do not add any explanation outside the JSON."
    )
    user_prompt = (
        f"Requirement:\n{json.dumps(requirement, ensure_ascii=False)}\n\n"
        f"Raw response to repair:\n{raw_response}\n\n"
        "Return valid JSON only."
    )

    try:
        repaired_raw = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        return extract_json_payload(repaired_raw)
    except Exception as e:
        logger.warning(f"Requirement analysis repair failed for {requirement['id']}: {e}")
        return None


def _extract_applicant_requirement_payload(raw: str):
    cleaned = clean_llm_output(raw)
    if not cleaned:
        return None, "empty"

    direct_payload = _try_parse_json(cleaned)
    if direct_payload is not None:
        return direct_payload, "direct"

    largest_array = _extract_largest_balanced_json_array(cleaned)
    if largest_array:
        array_payload = _try_parse_json(largest_array)
        if array_payload is not None:
            return array_payload, "balanced_array"

    extracted_payload = extract_json_payload(raw)
    if _looks_like_requirement_object(extracted_payload):
        if _raw_looks_like_multi_requirement_payload(cleaned):
            return None, "truncated_singleton_rejected"
        return extracted_payload, "single_object"

    if extracted_payload is not None:
        return extracted_payload, "shared_fallback"

    return None, "direct_parse_failed"


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_largest_balanced_json_array(text: str) -> str | None:
    best_candidate = None
    best_length = -1

    for start_index, ch in enumerate(text):
        if ch != "[":
            continue

        depth = 0
        in_string = False
        escaped = False

        for end_index in range(start_index, len(text)):
            current = text[end_index]

            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
                continue

            if current == "[":
                depth += 1
            elif current == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start_index:end_index + 1]
                    if len(candidate) > best_length:
                        best_candidate = candidate
                        best_length = len(candidate)
                    break

    return best_candidate


def _raw_looks_like_multi_requirement_payload(text: str) -> bool:
    if not text:
        return False

    if len(re.findall(r'"id"\s*:', text)) > 1:
        return True
    if len(re.findall(r"\breq_\d+\b", text, flags=re.IGNORECASE)) > 1:
        return True
    if re.search(r"}\s*,\s*{", text):
        return True
    if text.count("{") > 1 and "[" in text:
        return True
    return False


def _normalize_requirement_payload(payload, jd_text: str) -> list[dict]:
    payload_list = _coerce_requirement_payload_to_list(payload)
    if not isinstance(payload_list, list):
        return []

    accepted: list[dict] = []
    suspicious: list[dict] = []

    for index, item in enumerate(payload_list, start=1):
        requirement, verdict = _normalize_requirement_object(item, index)
        if not requirement:
            continue
        if verdict == "accepted":
            accepted.append(requirement)
        elif verdict == "suspicious":
            suspicious.append(requirement)

    if suspicious:
        accepted.extend(_cleanup_suspicious_requirements(suspicious, jd_text))

    return _resequence_requirement_ids(accepted)


def _coerce_requirement_payload_to_list(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if _looks_like_requirement_object(payload):
            return [payload]

        for key in ("requirements", "items", "results", "data", "output"):
            value = payload.get(key)
            coerced = _coerce_requirement_payload_to_list(value)
            if coerced:
                return coerced

        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]

        object_values = [value for value in payload.values() if _looks_like_requirement_object(value)]
        if object_values:
            return object_values

    return None


def _looks_like_requirement_object(obj) -> bool:
    if not isinstance(obj, dict):
        return False

    known_fields = {
        "id",
        "text",
        "requirement",
        "description",
        "category",
        "importance",
        "section_group",
        "matching_strategy",
    }
    populated_fields = {key for key in known_fields if str(obj.get(key, "")).strip()}
    return bool({"text", "requirement", "description"} & populated_fields) or len(populated_fields) >= 3


def _repair_jd_requirements_json(raw_response: str, jd_text: str):
    system_prompt = (
        "Repair applicant JD extraction output into a strict JSON array of requirement objects.\n"
        "Return ONLY valid JSON.\n"
        "Each array item must contain exactly these keys:\n"
        '[{"id":"req_01","text":"...","category":"tool|skill|experience|education|language|soft_skill|domain|other","importance":"required|preferred","section_group":"summary|experience|education|skills|projects|certifications|languages|other","matching_strategy":"semantic_evidence|exact_plus_semantic|experience_duration"}]\n'
        "Keep only real candidate requirements. Drop headings, titles, labels, company names, and meta text."
    )
    user_prompt = (
        f"Original JD:\n{jd_text}\n\n"
        f"Raw model output to repair:\n{raw_response}\n\n"
        "Return the repaired JSON array only."
    )

    try:
        repaired_raw = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1400,
        )
        return extract_json_payload(repaired_raw)
    except Exception as e:
        logger.warning(f"Applicant JD decomposition repair failed: {e}")
        return None


def _cleanup_suspicious_requirements(requirements: list[dict], jd_text: str) -> list[dict]:
    """Optional lightweight cleanup pass for borderline extracted items."""
    if not requirements:
        return []

    system_prompt = (
        "Review borderline extracted JD items.\n"
        "Keep only entries that are real candidate requirements.\n"
        "Drop role titles, headings, labels, company references, and meta text.\n"
        "Return ONLY valid JSON as an array of requirement objects using the same schema as input."
    )
    user_prompt = (
        f"Original JD:\n{jd_text}\n\n"
        f"Borderline requirement objects:\n{json.dumps(requirements, ensure_ascii=False)}\n\n"
        "Return only the cleaned requirement objects that should be kept."
    )

    try:
        raw = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        parsed = extract_json_payload(raw)
        cleaned: list[dict] = []
        if isinstance(parsed, list):
            for index, item in enumerate(parsed, start=1):
                requirement, verdict = _normalize_requirement_object(item, index)
                if requirement and verdict == "accepted":
                    cleaned.append(requirement)
        return cleaned
    except Exception as e:
        logger.warning(f"Suspicious requirement cleanup failed: {e}")
        return []


def _normalize_requirement_object(raw_item, index: int) -> tuple[dict | None, str]:
    text = _extract_requirement_text(raw_item)

    if not text:
        return None, "rejected"

    verdict = _requirement_shape_verdict(text)
    if verdict == "rejected":
        return None, verdict

    raw_category = str(raw_item.get("category", "")).strip().lower() if isinstance(raw_item, dict) else ""
    category = raw_category if raw_category in ALLOWED_CATEGORIES else _infer_category(text)

    raw_importance = str(raw_item.get("importance", "")).strip().lower() if isinstance(raw_item, dict) else ""
    importance = raw_importance if raw_importance in ALLOWED_IMPORTANCE else _infer_importance(text)

    raw_section_group = str(raw_item.get("section_group", "")).strip().lower() if isinstance(raw_item, dict) else ""
    section_group = raw_section_group if raw_section_group in ALLOWED_SECTION_GROUPS else _infer_section_group(text, category)

    raw_strategy = str(raw_item.get("matching_strategy", "")).strip().lower() if isinstance(raw_item, dict) else ""
    matching_strategy = raw_strategy if raw_strategy in ALLOWED_MATCHING_STRATEGIES else _infer_matching_strategy(text, category)

    requirement_id = ""
    if isinstance(raw_item, dict):
        requirement_id = str(raw_item.get("id", "")).strip()
    requirement_id = requirement_id or f"req_{index:02d}"

    return {
        "id": requirement_id,
        "text": text,
        "category": category,
        "importance": importance,
        "section_group": section_group,
        "matching_strategy": matching_strategy,
    }, verdict


def _extract_requirement_text(raw_item) -> str:
    if isinstance(raw_item, dict):
        for key in ("text", "requirement", "description", "name", "value"):
            candidate = _normalize_requirement_text(str(raw_item.get(key, "")))
            if candidate:
                return candidate
        return ""
    if isinstance(raw_item, str):
        return _normalize_requirement_text(raw_item)
    return ""


def _normalize_requirement_text(text: str) -> str:
    cleaned = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    cleaned = cleaned.strip().strip("-*•")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _requirement_shape_verdict(text: str) -> str:
    if not text:
        return "rejected"

    lower_text = text.lower().rstrip(":")
    word_count = len(lower_text.split())
    has_signal = _has_requirement_signal(lower_text)

    if lower_text in GENERIC_SECTION_LABELS:
        return "rejected"

    if text.endswith(":") and not has_signal:
        return "rejected"

    if _is_heading_like_or_title_like(text) and not has_signal:
        return "rejected"

    if not has_signal:
        return "rejected"

    if word_count <= 2 and not _looks_tool_like(lower_text):
        return "suspicious"

    if _is_heading_like_or_title_like(text) and has_signal:
        return "suspicious"

    return "accepted"


def _has_requirement_signal(text: str) -> bool:
    if REQUIREMENT_SIGNAL_RE.search(text):
        return True

    tokens = [token for token in TECH_TOKEN_RE.findall(text) if token.lower() not in STOPWORD_TOKENS]
    if len(tokens) >= 2 and any(any(char.isdigit() for char in token) for token in tokens):
        return True

    if _looks_tool_like(text):
        return True

    return False


def _is_heading_like_or_title_like(text: str) -> bool:
    stripped = text.strip()
    words = stripped.split()
    if not words:
        return True

    lower_text = stripped.lower().rstrip(":")
    if lower_text in GENERIC_SECTION_LABELS:
        return True

    if len(words) <= 6 and all(word[:1].isupper() or word.isupper() for word in words if word[0].isalnum()):
        if not _has_requirement_signal(lower_text):
            return True

    if " at " in lower_text and ROLE_TITLE_RE.search(stripped):
        return True

    return False


def _looks_tool_like(text: str) -> bool:
    tokens = [token for token in re.findall(r"[a-z0-9+#./-]+", text.lower()) if token not in STOPWORD_TOKENS]
    if not tokens:
        return False

    if any(any(char.isdigit() for char in token) for token in tokens):
        return True
    if any("+" in token or "#" in token or "/" in token for token in tokens):
        return True
    if ("," in text or "/" in text) and 1 <= len(tokens) <= 5:
        return True
    return False


def _resequence_requirement_ids(requirements: list[dict]) -> list[dict]:
    seen = set()
    resequenced = []

    for requirement in requirements:
        normalized_text = requirement["text"].strip().lower()
        if not normalized_text or normalized_text in seen:
            continue
        seen.add(normalized_text)
        resequenced.append({**requirement})

    for index, requirement in enumerate(resequenced, start=1):
        requirement["id"] = f"req_{index:02d}"

    return resequenced


def _infer_category(text: str) -> str:
    lower_text = text.lower()
    if extract_required_years(text) is not None or "experience" in lower_text:
        return "experience"
    if any(term in lower_text for term in ["bachelor", "master", "degree", "phd", "university", "engineering", "statistics"]):
        return "education"
    if any(term in lower_text for term in ["english", "arabic", "french", "german", "language"]):
        return "language"
    if any(term in lower_text for term in ["leadership", "communication", "collaboration", "stakeholder"]):
        return "soft_skill"
    if any(term in lower_text for term in ["sql", "python", "pytorch", "tensorflow", "fastapi", "api", "aws", "azure", "docker", "kubernetes"]):
        return "tool"
    if any(term in lower_text for term in ["machine learning", "deep learning", "analytics", "data science", "computer vision", "nlp"]):
        return "domain"
    return "skill"


def _infer_importance(text: str) -> str:
    lower_text = text.lower()
    if any(term in lower_text for term in ["preferred", "plus", "nice to have", "good to have", "bonus"]):
        return "preferred"
    return "required"


def _infer_section_group(text: str, category: str) -> str:
    if category == "education":
        return "education"
    if category == "language":
        return "languages"
    if category in {"tool", "skill", "soft_skill", "domain"}:
        return "skills"
    if category == "experience":
        return "experience"
    return "other"


def _infer_matching_strategy(text: str, category: str) -> str:
    if category == "experience" and extract_required_years(text) is not None:
        return "experience_duration"
    if category in {"tool", "language"}:
        return "exact_plus_semantic"
    return "semantic_evidence"


def _format_evidence_chunks(evidence_chunks: list[dict]) -> str:
    if not evidence_chunks:
        return ""

    blocks = []
    for chunk in evidence_chunks:
        similarity = chunk.get("similarity")
        similarity_text = f"{similarity:.2f}" if isinstance(similarity, float) else "n/a"
        blocks.append(
            f"[Section: {chunk['section_name']} | Chunk: {chunk['chunk_index']} | Relevance: {similarity_text}]\n"
            f"{chunk['chunk_text']}"
        )
    return "\n\n".join(blocks)


def _default_top_evidence(evidence_chunks: list[dict]) -> list[str]:
    evidence = []
    for chunk in evidence_chunks:
        snippet = " ".join(chunk.get("chunk_text", "").split())
        if snippet:
            evidence.append(f"{chunk['section_name']}: {snippet[:220]}")
    return unique_keep_order(evidence, limit=3)


def _base_requirement_result(requirement: dict) -> dict:
    return {
        "requirement_id": requirement["id"],
        "requirement_text": requirement["text"],
        "category": requirement["category"],
        "importance": requirement["importance"],
        "section_group": requirement["section_group"],
        "matching_strategy": requirement["matching_strategy"],
        "id": requirement["id"],
        "text": requirement["text"],
    }


def _build_grounded_summary(requirement_results: list[dict], experience_summary: dict) -> str:
    status_counts = {
        status: sum(1 for result in requirement_results if result.get("status") == status)
        for status in ALLOWED_STATUSES
    }

    explicit_examples = _collect_requirement_examples(requirement_results, "matched_explicitly", limit=2)
    gap_examples = _collect_requirement_examples(requirement_results, "missing_or_insufficient", limit=2)

    parts = [
        "This applicant analysis was validated requirement by requirement.",
        (
            f"Explicit matches: {status_counts['matched_explicitly']}, "
            f"partial matches: {status_counts['partially_matched']}, "
            f"not explicitly stated: {status_counts['not_explicitly_stated']}, "
            f"missing or insufficient: {status_counts['missing_or_insufficient']}."
        ),
    ]

    if explicit_examples:
        parts.append("The clearest supported requirements are: " + "; ".join(explicit_examples) + ".")
    if gap_examples:
        parts.append("The main unsupported or weakly supported requirements are: " + "; ".join(gap_examples) + ".")

    total_years = experience_summary.get("total_years")
    if experience_summary.get("experience_section_found") and isinstance(total_years, (int, float)):
        parts.append(
            f"Experience-only date parsing found approximately {float(total_years):.2f} overlap-aware years in the main Experience section."
        )

    return " ".join(parts)


def _build_safe_overall_suggestions(requirement_results: list[dict]) -> list[str]:
    ordered_results = sorted(
        requirement_results,
        key=lambda result: (
            0 if result.get("status") == "missing_or_insufficient" else
            1 if result.get("status") == "partially_matched" else
            2 if result.get("status") == "not_explicitly_stated" else
            3
        ),
    )
    suggestions = [
        str(result.get("suggestion", "")).strip()
        for result in ordered_results
        if result.get("status") != "matched_explicitly" and result.get("suggestion")
    ]
    return unique_keep_order(suggestions, limit=8)


def _collect_requirement_examples(requirement_results: list[dict], status: str, limit: int = 2) -> list[str]:
    examples = []
    for result in requirement_results:
        if result.get("status") != status:
            continue
        text = str(result.get("requirement_text", result.get("text", ""))).strip()
        if text:
            examples.append(text)
        if len(examples) >= limit:
            break
    return examples
