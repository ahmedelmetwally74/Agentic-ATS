import json
import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")


def call_llm(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1200) -> str:
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


def clean_llm_output(raw: str) -> str:
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


def extract_json_payload(raw: str):
    """
    Try to find and parse the first valid JSON object/array anywhere in the model output.
    This is much safer than raw.find('{') / raw.rfind('}').
    """
    cleaned = clean_llm_output(raw)
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


def to_str_list(value) -> list[str]:
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


def unique_keep_order(items: list[str], limit: int | None = None) -> list[str]:
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


def normalize_requirements_dict(obj, allowed_sections: list[str]) -> dict[str, list[str]]:
    """
    Keep only known JD section keys and normalize values to list[str].
    """
    key_map = {section.lower(): section for section in allowed_sections}
    final = {section: [] for section in allowed_sections}

    if isinstance(obj, dict):
        for key, value in obj.items():
            canonical = key_map.get(str(key).strip().lower())
            if not canonical:
                continue
            final[canonical] = unique_keep_order(to_str_list(value), limit=5)

    return {key: value for key, value in final.items() if value}


def fallback_decompose_job_description(jd_text: str) -> dict[str, list[str]]:
    """
    Deterministic fallback from the raw JD text if the model fails to return JSON.
    This is much better than returning only 'General'.
    """
    sections = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]
    result = {section: [] for section in sections}

    lines = []
    for line in jd_text.splitlines():
        cleaned = line.strip(" \t-â€¢*")
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
        key: unique_keep_order(value, limit=5)
        for key, value in result.items()
        if value
    }

    if result:
        return result

    return {
        "Experience": ["Professional background in the role domain"],
        "Skills": ["Relevant technical skills for the role"],
    }


def decompose_job_description(jd_text: str) -> dict[str, list[str]]:
    """
    Decompose a full Job Description into categorized requirements.
    """
    sections = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]

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
        parsed = extract_json_payload(raw)
        normalized = normalize_requirements_dict(parsed, sections)

        if normalized:
            return normalized

        logger.warning(f"JD decomposition returned non-usable JSON. Raw output preview: {raw[:800]}")
        return fallback_decompose_job_description(jd_text)

    except Exception as e:
        logger.warning(f"JD decomposition failed: {e}")
        return fallback_decompose_job_description(jd_text)
