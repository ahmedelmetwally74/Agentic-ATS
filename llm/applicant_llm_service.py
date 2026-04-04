import json
import logging

from llm.llm_shared import call_llm, extract_json_payload, to_str_list, unique_keep_order

logger = logging.getLogger(__name__)


def fallback_applicant_section_analysis(section_name: str) -> dict:
    return {
        "comparison": f"Some relevant evidence was found for {section_name}, but the structured comparison was incomplete.",
        "missing_tools": [],
        "missing_skills": [],
        "missing_experience": [],
        "missing_education": [],
        "improvement_suggestions": [
            f"Make the {section_name} section more explicit and quantified.",
            f"Add clearer evidence that matches the JD requirements for {section_name.lower()}.",
        ],
    }


def fallback_applicant_synthesis(section_analyses: list[dict]) -> dict:
    covered_sections = [analysis.get("section") for analysis in section_analyses if analysis.get("section")]

    suggestions = []
    for analysis in section_analyses:
        suggestions.extend(to_str_list(analysis.get("improvement_suggestions")))

    suggestions = unique_keep_order(suggestions, limit=8) or [
        "Align the CV more explicitly with the JD requirements.",
        "Add quantified impact, tools, and project outcomes.",
    ]

    summary = (
        "Your CV partially matches the job requirements"
        + (f" across these sections: {', '.join(covered_sections)}." if covered_sections else ".")
        + " The suggestions below focus on how to make your CV stronger for this role."
    )

    return {
        "general_comparison_summary": summary,
        "improvement_suggestions": suggestions,
    }


def analyze_applicant_section_match(section_name: str, jd_requirements: list[str], cv_chunk: str) -> dict:
    """Perform an applicant-facing section-level comparison between JD requirements and CV evidence."""
    jd_req_text = "\n".join(f"- {requirement}" for requirement in jd_requirements)

    system_prompt = (
        "You are an expert career coach helping one applicant improve their CV for one specific job.\n"
        "Analyze ONLY this applicant's CV evidence against ONLY this JD section.\n"
        "Do NOT compare the applicant to any other candidate, person, or profile.\n"
        "Return ONLY valid JSON.\n"
        "Use this schema exactly:\n"
        "{\n"
        '  "comparison": "1-3 sentence comparison between this applicant CV section and the JD section",\n'
        '  "missing_tools": ["software, frameworks, platforms, libraries, or technical tools only"],\n'
        '  "missing_skills": ["capabilities or competencies only"],\n'
        '  "missing_experience": ["missing role scope, years, leadership, delivery, domain, or hands-on experience"],\n'
        '  "missing_education": ["missing degree, field of study, or certification-related education only"],\n'
        '  "improvement_suggestions": ["specific CV rewrite or content suggestions for this section only"]\n'
        "}\n"
        "Rules:\n"
        "- missing_tools must contain only actual tools, frameworks, libraries, platforms, or technical systems.\n"
        "- missing_skills must contain only skills or competencies.\n"
        "- missing_experience must contain only experience gaps.\n"
        "- missing_education must contain only education or certification gaps.\n"
        "- If a category has no gaps, return an empty list.\n"
        "- Be specific and grounded only in the provided JD and CV evidence.\n"
        "- Do not invent experience that is not shown.\n"
        "- Do not mention any other candidate.\n"
        "- Do not include markdown, code fences, or any explanation outside the JSON."
    )
    user_prompt = (
        f"JD Section: {section_name}\n"
        f"Requirements:\n{jd_req_text}\n\n"
        f"Applicant CV Evidence:\n{cv_chunk}\n\n"
        "Evaluate this applicant only.\n"
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
            return fallback_applicant_section_analysis(section_name)

        return {
            "comparison": str(parsed.get("comparison", "")).strip()
            or f"Relevant evidence was found for {section_name}, but the comparison was incomplete.",
            "missing_tools": unique_keep_order(to_str_list(parsed.get("missing_tools")), limit=5),
            "missing_skills": unique_keep_order(to_str_list(parsed.get("missing_skills")), limit=5),
            "missing_experience": unique_keep_order(to_str_list(parsed.get("missing_experience")), limit=5),
            "missing_education": unique_keep_order(to_str_list(parsed.get("missing_education")), limit=5),
            "improvement_suggestions": unique_keep_order(to_str_list(parsed.get("improvement_suggestions")), limit=5)
            or [f"Strengthen the {section_name} section with clearer and more specific evidence."],
        }
    except Exception as e:
        logger.warning(f"Section analysis failed for '{section_name}': {e}")
        return fallback_applicant_section_analysis(section_name)


def synthesize_applicant_candidate_analysis(section_analyses: list[dict]) -> dict:
    """Synthesize section-level applicant analyses into the final headers."""
    if not section_analyses:
        return fallback_applicant_synthesis(section_analyses)

    combined_text = "\n".join(json.dumps(analysis, ensure_ascii=False) for analysis in section_analyses)

    system_prompt = (
        "You are an expert career coach writing a final report for ONE applicant.\n"
        "Use only the provided section analyses for this same applicant.\n"
        "Do NOT compare the applicant to any other candidate, benchmark candidate, or external profile.\n"
        "Return ONLY valid JSON.\n"
        "Use this schema exactly:\n"
        '{'
        '"general_comparison_summary": "2-4 sentence summary focused only on this applicant and this job", '
        '"improvement_suggestions": ["specific, practical CV improvement suggestion 1", "specific, practical CV improvement suggestion 2"]'
        '}\n'
        "Rules:\n"
        "- Mention only this applicant.\n"
        "- Do not say 'other candidate', 'another applicant', or similar phrases.\n"
        "- Summarize only the main fit level and the main gaps for this applicant.\n"
        "- Suggestions must be practical CV edits or additions.\n"
        "- No markdown, no code fences, no explanation outside JSON."
    )
    user_prompt = (
        f"Individual Section Analyses for one applicant:\n{combined_text}\n\n"
        "Write the final applicant report now.\n"
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
            return fallback_applicant_synthesis(section_analyses)

        fallback = fallback_applicant_synthesis(section_analyses)
        return {
            "general_comparison_summary": str(parsed.get("general_comparison_summary", "")).strip()
            or fallback["general_comparison_summary"],
            "improvement_suggestions": unique_keep_order(to_str_list(parsed.get("improvement_suggestions")), limit=8)
            or fallback["improvement_suggestions"],
        }
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return fallback_applicant_synthesis(section_analyses)
