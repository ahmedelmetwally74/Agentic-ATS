from llm.llm_shared import call_llm


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
