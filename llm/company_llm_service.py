import json
import logging

from llm.llm_shared import call_llm, extract_json_payload, to_str_list, unique_keep_order

logger = logging.getLogger(__name__)


def fallback_company_section_analysis(section_name: str) -> dict:
    return {
        "why_fits": [f"Relevant evidence was retrieved for the {section_name} section."],
        "things_to_keep_in_mind": [f"Validate depth and recency of {section_name.lower()} experience during the interview."],
        "questions": [
            f"Can you walk me through your experience in {section_name.lower()}?",
            f"What was your most relevant hands-on contribution in {section_name.lower()}?",
        ],
    }


def fallback_company_synthesis(section_analyses: list[dict]) -> dict:
    covered_sections = [analysis.get("section") for analysis in section_analyses if analysis.get("section")]

    why_fits = []
    keep_in_mind = []
    questions = []

    for analysis in section_analyses:
        why_fits.extend(to_str_list(analysis.get("why_fits")))
        keep_in_mind.extend(to_str_list(analysis.get("things_to_keep_in_mind")))
        questions.extend(to_str_list(analysis.get("questions")))

    why_fits = unique_keep_order(why_fits, limit=5) or ["Relevant evidence was retrieved from the CV."]
    keep_in_mind = unique_keep_order(keep_in_mind, limit=5) or ["Validate project ownership, depth, and recency during the interview."]
    questions = unique_keep_order(questions, limit=8) or ["Can you walk me through the most relevant project for this role?"]

    summary = (
        "Candidate shows relevant evidence"
        + (f" across these sections: {', '.join(covered_sections)}." if covered_sections else ".")
        + " Review the strengths and validation points below."
    )

    return {
        "ranking_overview_summary": summary,
        "why_fits": why_fits,
        "things_to_keep_in_mind": keep_in_mind,
        "questions": questions,
    }


def analyze_company_section_match(section_name: str, jd_requirements: list[str], cv_chunk: str) -> dict:
    """Perform a company-facing section-level comparison between JD requirements and CV evidence."""
    jd_req_text = "\n".join(f"- {requirement}" for requirement in jd_requirements)

    system_prompt = (
        "You are an expert HR analyst.\n"
        "Compare job requirements against CV evidence.\n"
        "Return ONLY valid JSON.\n"
        'Schema: {"why_fits": ["reason 1", "reason 2"], "things_to_keep_in_mind": ["point 1"], "questions": ["question 1", "question 2"]}\n'
        "Do not include markdown, code fences, or explanation outside the JSON."
    )
    user_prompt = (
        f"JD Section: {section_name}\n"
        f"Requirements:\n{jd_req_text}\n\n"
        f"CV Evidence:\n{cv_chunk}\n\n"
        "Return JSON only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.1, max_tokens=700)
        parsed = extract_json_payload(raw)

        if not isinstance(parsed, dict):
            logger.warning(f"Section analysis parse failed for '{section_name}'. Raw output preview: {raw[:800]}")
            return fallback_company_section_analysis(section_name)

        return {
            "why_fits": unique_keep_order(to_str_list(parsed.get("why_fits")), limit=4)
            or [f"Relevant evidence was found for {section_name}."],
            "things_to_keep_in_mind": unique_keep_order(to_str_list(parsed.get("things_to_keep_in_mind")), limit=4)
            or [f"Validate depth of {section_name.lower()} experience."],
            "questions": unique_keep_order(to_str_list(parsed.get("questions")), limit=4)
            or [f"Can you describe your experience in {section_name.lower()}?"],
        }
    except Exception as e:
        logger.warning(f"Section analysis failed for '{section_name}': {e}")
        return fallback_company_section_analysis(section_name)


def synthesize_company_candidate_analysis(section_analyses: list[dict]) -> dict:
    """Synthesize section-level company analyses into the final headers."""
    if not section_analyses:
        return fallback_company_synthesis(section_analyses)

    combined_text = "\n".join(json.dumps(analysis, ensure_ascii=False) for analysis in section_analyses)

    system_prompt = (
        "You are an expert analyst.\n"
        "Synthesize section-level candidate analyses into one final company-facing report.\n"
        "Return ONLY valid JSON.\n"
        'Schema: {"ranking_overview_summary": "...", "why_fits": ["..."], "things_to_keep_in_mind": ["..."], "questions": ["..."]}\n'
        "Use only what is supported by the provided analyses."
    )
    user_prompt = (
        f"Individual Section Analyses for one candidate:\n{combined_text}\n\n"
        "Write the final company report now.\n"
        "Return JSON only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.1, max_tokens=900)
        parsed = extract_json_payload(raw)

        if not isinstance(parsed, dict):
            logger.warning(f"Synthesis parse failed. Raw output preview: {raw[:1200]}")
            return fallback_company_synthesis(section_analyses)

        fallback = fallback_company_synthesis(section_analyses)
        return {
            "ranking_overview_summary": str(parsed.get("ranking_overview_summary", "")).strip()
            or fallback["ranking_overview_summary"],
            "why_fits": unique_keep_order(to_str_list(parsed.get("why_fits")), limit=5)
            or fallback["why_fits"],
            "things_to_keep_in_mind": unique_keep_order(to_str_list(parsed.get("things_to_keep_in_mind")), limit=5)
            or fallback["things_to_keep_in_mind"],
            "questions": unique_keep_order(to_str_list(parsed.get("questions")), limit=8)
            or fallback["questions"],
        }
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return fallback_company_synthesis(section_analyses)
