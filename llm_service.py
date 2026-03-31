"""
AgenticATS - LLM Service
Qwen chat completions via llama-server HTTP API for candidate analysis.
Implements a Section-Level Map-Reduce pattern with robust JSON extraction.
"""

import json
import logging
import os
import re
import requests

logger = logging.getLogger(__name__)

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")


def call_llm(messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 1200) -> str:
    """
    Call the local chat completions endpoint.
    Lower temperature helps structured JSON output.
    """
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=None,
        )
        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]

        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        print("\n[DEBUG] Full message keys:", list(msg.keys()))
        print("[DEBUG] content repr:", repr(content[:500]))
        # print("[DEBUG] reasoning repr:", repr(reasoning[:500]))

        content = content.strip()
        if not content:
            raise ValueError("LLM returned empty content. Check reasoning_content / llama-server thinking settings.")

        return content  
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


def _clean_llm_output(raw: str) -> str:
    """
    Remove common wrappers that break JSON parsing:
    - <think>...</think>
    - ```json ... ```
    - stray backticks
    """
    if not raw:
        return ""

    cleaned = raw.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _extract_json_payload(raw: str):
    """
    Try to find and parse the first valid JSON object/array anywhere in the model output.
    This is much safer than raw.find('{') / raw.rfind('}').
    """
    cleaned = _clean_llm_output(raw)
    decoder = json.JSONDecoder()

    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            return obj
        except json.JSONDecodeError:
            continue

    return None


def _to_str_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        items = [value]

    result = []
    for item in items:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _unique_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    result = []

    for item in items:
        norm = item.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(item.strip())

    if limit is not None:
        return result[:limit]
    return result


def _normalize_requirements_dict(obj, allowed_sections: list[str]) -> dict[str, list[str]]:
    """
    Keep only known JD section keys and normalize values to list[str].
    """
    key_map = {s.lower(): s for s in allowed_sections}
    final = {s: [] for s in allowed_sections}

    if isinstance(obj, dict):
        for k, v in obj.items():
            canonical = key_map.get(str(k).strip().lower())
            if not canonical:
                continue
            final[canonical] = _unique_keep_order(_to_str_list(v), limit=5)

    final = {k: v for k, v in final.items() if v}
    return final


def _fallback_decompose_job_description(jd_text: str) -> dict[str, list[str]]:
    """
    Deterministic fallback from the raw JD text if the model fails to return JSON.
    This is much better than returning only 'General'.
    """
    sections = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]
    result = {s: [] for s in sections}

    lines = []
    for line in jd_text.splitlines():
        cleaned = line.strip(" \t-•*")
        if cleaned:
            lines.append(cleaned)

    for line in lines:
        low = line.lower()

        if any(x in low for x in [
            "about the job", "purpose of the job", "responsible for",
            "business insights", "strategic decision-making", "drive innovation"
        ]):
            result["Summary"].append(line)

        if any(x in low for x in [
            "bachelor", "master", "phd", "degree", "computer science",
            "engineering", "statistics"
        ]):
            result["Education"].append(line)

        if any(x in low for x in [
            "english", "arabic", "fluent"
        ]):
            result["Languages"].append(line)

        if any(x in low for x in [
            "years of experience", "+ years", "experience in", "solid experience",
            "leadership", "mentor", "stakeholder", "projects", "managing business stakeholders"
        ]):
            result["Experience"].append(line)

        if any(x in low for x in [
            "machine learning", "deep learning", "generative ai", "llm",
            "gan", "diffusion", "mlops", "sql", "reinforcement learning",
            "foundation models", "analytical mindset", "business acumen",
            "predictive models", "advanced analytics"
        ]):
            result["Skills"].append(line)

        if any(x in low for x in [
            "pipeline", "deployment", "monitoring", "governance",
            "open-source", "research", "patent", "model drift"
        ]):
            result["Projects"].append(line)

        if any(x in low for x in [
            "certification", "certifications", "certificate"
        ]):
            result["Certifications"].append(line)

    result = {
        k: _unique_keep_order(v, limit=5)
        for k, v in result.items()
        if v
    }

    if result:
        return result

    return {
        "Experience": ["Professional background in the role domain"],
        "Skills": ["Relevant technical skills for the role"],
    }


def _fallback_section_analysis(section_name: str, mode: str) -> dict:
    if mode == "company":
        return {
            "why_fits": [f"Relevant evidence was retrieved for the {section_name} section."],
            "things_to_keep_in_mind": [f"Validate depth and recency of {section_name.lower()} experience during the interview."],
            "questions": [
                f"Can you walk me through your experience in {section_name.lower()}?",
                f"What was your most relevant hands-on contribution in {section_name.lower()}?"
            ]
        }

    return {
        "comparison": f"Some relevant evidence was found for {section_name}, but the structured comparison was incomplete.",
        "missing_tools": [],
        "missing_skills": [],
        "missing_experience": [],
        "missing_education": [],
        "improvement_suggestions": [
            f"Make the {section_name} section more explicit and quantified.",
            f"Add clearer evidence that matches the JD requirements for {section_name.lower()}."
        ]
    }


def _fallback_synthesis(section_analyses: list[dict], mode: str) -> dict:
    covered_sections = [a.get("section") for a in section_analyses if a.get("section")]

    if mode == "company":
        why_fits = []
        keep_in_mind = []
        questions = []

        for a in section_analyses:
            why_fits.extend(_to_str_list(a.get("why_fits")))
            keep_in_mind.extend(_to_str_list(a.get("things_to_keep_in_mind")))
            questions.extend(_to_str_list(a.get("questions")))

        why_fits = _unique_keep_order(why_fits, limit=5) or ["Relevant evidence was retrieved from the CV."]
        keep_in_mind = _unique_keep_order(keep_in_mind, limit=5) or ["Validate project ownership, depth, and recency during the interview."]
        questions = _unique_keep_order(questions, limit=8) or ["Can you walk me through the most relevant project for this role?"]

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

    suggestions = []
    for a in section_analyses:
        suggestions.extend(_to_str_list(a.get("improvement_suggestions")))

    suggestions = _unique_keep_order(suggestions, limit=8) or [
        "Align the CV more explicitly with the JD requirements.",
        "Add quantified impact, tools, and project outcomes."
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


def decompose_job_description(jd_text: str) -> dict[str, list[str]]:
    """
    Decompose a full Job Description into categorized requirements.
    """
    sections = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]

    # system_prompt = (
    #     "You are an expert technical recruiter.\n"
    #     "Read the full job description and extract the main requirements.\n"
    #     "Return ONLY one valid JSON object.\n"
    #     "Use ONLY these keys exactly if relevant: "
    #     + ", ".join(sections)
    #     + ".\n"
    #     "Each value must be a list of short requirement strings.\n"
    #     "Do not add any explanation before or after the JSON.\n"
    #     'Example format: {"Experience": ["4+ years in ML"], "Skills": ["SQL", "MLOps"]}'
    # )

    system_prompt = (
        "You are an expert technical recruiter.\n"
        "Read the full job description and extract the requirements into a structured JSON object.\n"
        "Return ONLY one valid JSON object.\n"
        "Allowed keys only: Summary, Experience, Education, Skills, Projects, Certifications, Languages.\n"
        "Each value must be a list of short requirement strings.\n"
        "Do not include generic headings like 'About the job', 'Purpose of the job', 'Job specification', or 'Skills and Abilities'.\n"
        "Do not copy long paragraphs. Split requirements into short, specific bullet-style strings.\n"
        "Put each requirement under the most relevant key.\n"
        "If a key has no clear requirements, omit it.\n"
        "Prefer concrete requirements such as years of experience, tools, methods, technical areas, education, and languages.\n"
        'Example: {"Experience": ["4+ years in AI/ML"], "Skills": ["SQL", "MLOps", "Deep learning"], "Education": ["Bachelor\'s degree in Computer Science, Engineering, or Statistics"]}'
    )

    user_prompt = (
        f"Job Description:\n{jd_text}\n\n"
        "Extract the requirements now.\n"
        "Return JSON only."
    )   

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.1, max_tokens=900)
        parsed = _extract_json_payload(raw)
        normalized = _normalize_requirements_dict(parsed, sections)

        if normalized:
            return normalized

        logger.warning(f"JD decomposition returned non-usable JSON. Raw output preview: {raw[:800]}")
        return _fallback_decompose_job_description(jd_text)

    except Exception as e:
        logger.warning(f"JD decomposition failed: {e}")
        return _fallback_decompose_job_description(jd_text)


def analyze_section_match(section_name: str, jd_requirements: list[str], cv_chunk: str, mode: str) -> dict:
    """
    Perform a section-level comparison between JD requirements and CV evidence.
    """
    jd_req_text = "\n".join([f"- {r}" for r in jd_requirements])

    if mode == "company":
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
    else:
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
        parsed = _extract_json_payload(raw)

        if not isinstance(parsed, dict):
            logger.warning(f"Section analysis parse failed for '{section_name}'. Raw output preview: {raw[:800]}")
            return _fallback_section_analysis(section_name, mode)

        if mode == "company":
            return {
                "why_fits": _unique_keep_order(_to_str_list(parsed.get("why_fits")), limit=4)
                           or [f"Relevant evidence was found for {section_name}."],
                "things_to_keep_in_mind": _unique_keep_order(_to_str_list(parsed.get("things_to_keep_in_mind")), limit=4)
                           or [f"Validate depth of {section_name.lower()} experience."],
                "questions": _unique_keep_order(_to_str_list(parsed.get("questions")), limit=4)
                           or [f"Can you describe your experience in {section_name.lower()}?"],
            }

        return {
            "comparison": str(parsed.get("comparison", "")).strip()
                        or f"Relevant evidence was found for {section_name}, but the comparison was incomplete.",
            "missing_tools": _unique_keep_order(_to_str_list(parsed.get("missing_tools")), limit=5),
            "missing_skills": _unique_keep_order(_to_str_list(parsed.get("missing_skills")), limit=5),
            "missing_experience": _unique_keep_order(_to_str_list(parsed.get("missing_experience")), limit=5),
            "missing_education": _unique_keep_order(_to_str_list(parsed.get("missing_education")), limit=5),
            "improvement_suggestions": _unique_keep_order(_to_str_list(parsed.get("improvement_suggestions")), limit=5)
                        or [f"Strengthen the {section_name} section with clearer and more specific evidence."],
        }

    except Exception as e:
        logger.warning(f"Section analysis failed for '{section_name}': {e}")
        return _fallback_section_analysis(section_name, mode)


def synthesize_candidate_analysis(section_analyses: list[dict], mode: str) -> dict:
    """
    Synthesize section-level analyses into the final mode-based headers.
    """
    if not section_analyses:
        return _fallback_synthesis(section_analyses, mode)

    combined_text = "\n".join([json.dumps(a, ensure_ascii=False) for a in section_analyses])

    if mode == "company":
        system_prompt = (
            "You are an expert analyst.\n"
            "Synthesize section-level candidate analyses into one final company-facing report.\n"
            "Return ONLY valid JSON.\n"
            'Schema: {"ranking_overview_summary": "...", "why_fits": ["..."], "things_to_keep_in_mind": ["..."], "questions": ["..."]}\n'
            "Use only what is supported by the provided analyses."
        )
    else:
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
        parsed = _extract_json_payload(raw)

        if not isinstance(parsed, dict):
            logger.warning(f"Synthesis parse failed. Raw output preview: {raw[:1200]}")
            return _fallback_synthesis(section_analyses, mode)

        if mode == "company":
            return {
                "ranking_overview_summary": str(parsed.get("ranking_overview_summary", "")).strip()
                                            or _fallback_synthesis(section_analyses, mode)["ranking_overview_summary"],
                "why_fits": _unique_keep_order(_to_str_list(parsed.get("why_fits")), limit=5)
                            or _fallback_synthesis(section_analyses, mode)["why_fits"],
                "things_to_keep_in_mind": _unique_keep_order(_to_str_list(parsed.get("things_to_keep_in_mind")), limit=5)
                            or _fallback_synthesis(section_analyses, mode)["things_to_keep_in_mind"],
                "questions": _unique_keep_order(_to_str_list(parsed.get("questions")), limit=8)
                            or _fallback_synthesis(section_analyses, mode)["questions"],
            }

        return {
            "general_comparison_summary": str(parsed.get("general_comparison_summary", "")).strip()
                                          or _fallback_synthesis(section_analyses, mode)["general_comparison_summary"],
            "improvement_suggestions": _unique_keep_order(_to_str_list(parsed.get("improvement_suggestions")), limit=8)
                                       or _fallback_synthesis(section_analyses, mode)["improvement_suggestions"],
        }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return _fallback_synthesis(section_analyses, mode)

def rewrite_cv_section(section_name: str, raw_text: str) -> str:
    """
    Light cleanup for CV sections.
    Returns plain text, not JSON.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Rewrite the provided CV section into clean, professional CV-ready text.\n"
        "Keep it concise.\n"
        "Do not invent facts.\n"
        "Use only the information provided.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Section name: {section_name}\n\n"
        f"Raw section content:\n{raw_text}\n\n"
        "Rules:\n"
        "- Keep it concise and professional.\n"
        "- If the content is already good, lightly clean it.\n"
        "- For skills/courses/awards/activities, prefer short bullet-style lines.\n"
        "- Do not add explanations outside the final rewritten section.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=500).strip()
    except Exception:
        return raw_text


def rewrite_projects_for_cv(raw_text: str) -> str:
    """
    Rewrite projects into:
    Project Title
    - bullet
    - bullet

    Max 2 bullets per project.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Rewrite the projects section into a professional CV format.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Raw projects content:\n{raw_text}\n\n"
        "Rules:\n"
        "- Each project must start with its title on one line.\n"
        "- Under each project, write at most 2 short bullet points.\n"
        "- Keep the bullets concise and practical.\n"
        "- Focus on the most important contribution/result only.\n"
        "- Do not write long paragraphs.\n"
        "- Do not invent facts.\n"
        "- Return only the final project content.\n"
        "- Do not repeat the rules, examples, or instructions in the output.\n"
        "- Output format must be exactly like this:\n"
        "Project Title\n"
        "- bullet 1\n"
        "- bullet 2\n\n"
        "Another Project Title\n"
        "- bullet 1\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=700).strip()
    except Exception:
        return raw_text

def rewrite_summary_for_cv(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Rewrite the summary into a concise, professional CV summary.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Raw summary:\n{raw_text}\n\n"
        "Rules:\n"
        "- Keep the same meaning.\n"
        "- Do not invent any new facts.\n"
        "- Maximum 55 words.\n"
        "- Make it suitable for a CV.\n"
        "- Return only the final summary text.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=180).strip()
    except Exception:
        return raw_text

def rewrite_experience_for_cv(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Rewrite work experience into clean CV job blocks.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Raw experience:\n{raw_text}\n\n"
        "Rules:\n"
        "- Group each job into one block.\n"
        "- First line must be: Role at Company (Date).\n"
        "- Then write 2 to 3 short bullet points.\n"
        "- Keep the bullets concise and professional.\n"
        "- Do not invent any facts.\n"
        "- Separate jobs with one blank line.\n"
        "- Return only the final experience content.\n"
        "- Example:\n"
        "AI / ML Engineer Intern at Siemens (2025 - Present)\n"
        "- Built internal AI tools for document workflows.\n"
        "- Worked on LLM-based contract analysis tasks.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=500).strip()
    except Exception:
        return raw_text

def rewrite_education_for_cv(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Rewrite education into a clean CV education block.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Raw education:\n{raw_text}\n\n"
        "Rules:\n"
        "- First line: degree.\n"
        "- Second line: faculty/university.\n"
        "- Third line: extra academic detail if available.\n"
        "- Keep it concise and professional.\n"
        "- Do not invent any facts.\n"
        "- Return only the final education block.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=220).strip()
    except Exception:
        return raw_text

def rewrite_skills_for_cv(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Organize raw skills into grouped CV skill categories.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Raw skills:\n{raw_text}\n\n"
        "Rules:\n"
        "- Group related skills into clear categories.\n"
        "- Format each line like: Category: item1, item2, item3\n"
        "- Do not invent tools or technologies not present in the raw input.\n"
        "- Keep only categories supported by the input.\n"
        "- Return only the final grouped skills section.\n"
        "- Example:\n"
        "Programming Languages: Python, SQL\n"
        "Machine Learning & Deep Learning: scikit-learn, TensorFlow, PyTorch\n"
        "Backend & APIs: Flask, FastAPI\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=300).strip()
    except Exception:
        return raw_text

def rewrite_simple_list_for_cv(section_name: str, raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    system_prompt = (
        "You are an expert CV writer.\n"
        "Rewrite the content into short, clean CV bullet lines.\n"
        "Return plain text only.\n"
    )

    user_prompt = (
        f"Section name: {section_name}\n\n"
        f"Raw content:\n{raw_text}\n\n"
        "Rules:\n"
        "- Keep it concise and professional.\n"
        "- Prefer short bullet-style lines.\n"
        "- Do not invent any facts.\n"
        "- Return only the final text.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return call_llm(messages, temperature=0.1, max_tokens=250).strip()
    except Exception:
        return raw_text