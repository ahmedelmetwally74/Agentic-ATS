"""
AgenticATS - LLM Service
Qwen3.5-2B chat completions via llama-server HTTP API for candidate analysis.
Implements a Unified Section-to-Section matching pattern and robust JSON parsing.
Includes full transparency logging for debugging.
"""

import json
import logging
import os
import re
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3.5-2B chat completions endpoint (llama-server)
# ---------------------------------------------------------------------------
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")


def call_llm(messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 2048) -> str:
    """
    Call the Qwen3.5-2B chat completions endpoint.
    Prints all requests/responses for full transparency.
    """
    print("\n" + "="*80)
    print(">>> LLM REQUEST")
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        if len(content) > 2000:
            print(f"[{role}] {content[:1000]} ... [TRUNCATED] ... {content[-1000:]}")
        else:
            print(f"[{role}] {content}")
    print("-" * 80)

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 42,
        "top_p": 1.0,
        "top_k": 1,
        "repeat_penalty": 1.0,
    }

    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        
        print("<<< LLM RESPONSE")
        print(raw_content[:2000] + ("..." if len(raw_content) > 2000 else ""))
        print("="*80 + "\n")
        
        return raw_content
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        print(f"<<< LLM ERROR: {e}")
        print("="*80 + "\n")
        raise


def clean_llm_json(raw: str) -> any:
    """
    Extract and repair JSON from LLM output.
    Returns a dict or list depending on the content.
    Normalizes keys to lowercase if it's a dict.
    """
    try:
        # Find start and end of either { or [
        match = re.search(r'([\[{])', raw)
        if not match:
            return {}
        start = match.start()
        
        # Find the matching closing bracket/brace
        char = match.group(1)
        inverse = ']' if char == '[' else '}'
        end = raw.rfind(inverse)
        
        if end == -1:
            return {}
        
        json_str = raw[start:end+1]
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str) # Cleanup trailing commas
        data = json.loads(json_str)
        
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items()}
        return data
    except Exception as e:
        logger.warning(f"JSON cleaning failed: {e}")
        return {}


def decompose_job_description(jd_text: str) -> list[str]:
    """
    Decompose a full Job Description into a flat list of literal requirements.
    Every string must be an exact, unaltered substring from the original JD.
    """
    system_prompt = (
        "You are an expert technical recruiter specializing in high-precision Job Description analysis.\n"
        "GOAL: Identify every distinct requirement sentence or bullet point in the JD.\n\n"
        "STRICT EXTRACTION RULES:\n"
        "1. NO ALTERING TEXT: Every requirement MUST be an EXACT substring from the original JD. No paraphrasing or summarization.\n"
        "2. NO TITLES: Do not include section headers like 'Skills:' or 'Education:'. Only the requirements.\n"
        "3. GRANULARITY: Each requirement should be a single standalone point.\n"
        "4. EXHAUSTIVE: Capture all technical skills, soft skills, experience, and education requirements.\n\n"
        "Output ONLY a JSON list of strings."
    )
    
    user_prompt = f"Job Description Text:\n---\n{jd_text}\n---"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=1536)
        data = clean_llm_json(raw)
        
        if isinstance(data, list):
            return [str(i).strip() for i in data if str(i).strip()]
        elif isinstance(data, dict):
            # Flatten if it still tries to categorize
            flat = []
            for val in data.values():
                if isinstance(val, list):
                    flat.extend(val)
                elif isinstance(val, str):
                    flat.append(val)
            return [str(i).strip() for i in flat if str(i).strip()]
        return []
    except Exception as e:
        logger.error(f"JD decomposition failed: {e}")
        return []


def justify_match(requirement: str, cv_evidence: str, status: str, score: float) -> dict:
    """
    Generate a qualitative justification for a pre-calculated embedding match.
    Ensures the reason aligns with the mathematical status and score.
    """
    # Determine requirement type for question generation
    req_lower = requirement.lower()
    is_language = any(kw in req_lower for kw in ["fluent", "arabic", "english", "language"])
    is_technical = any(kw in req_lower for kw in ["sql", "ml", "ai", "python", "machine learning", "deep learning",
                         "generative", "llm", "gans", "diffusion", "mlops", "reinforcement", "neural",
                         "tensorflow", "pytorch", "keras", "sklearn", "pandas", "numpy", "aws", "azure", "gcp",
                         "docker", "kubernetes", "api", "rest", "graph"])

    if is_language:
        system_prompt = (
            "You are an expert HR Analyst. You are given a JD requirement, the best matching CV evidence, "
            "and a mathematically calculated match status and score.\n\n"
            "GOAL: Provide a natural language justification for this match.\n"
            "STRICT RULES:\n"
            "1. ALIGNMENT: Your justification MUST explain why the status was assigned based on the evidence.\n"
            "2. CONCISE: Keep the reason to 2-3 sentences.\n"
            "3. NO QUESTIONS: Language requirements do not need interview questions - return empty questions list.\n"
            "4. NO INTERNAL MONOLOGUE: Return only the final JSON.\n\n"
            "Output ONLY valid JSON with keys: 'reason', 'questions' (list)."
        )
    elif is_technical:
        system_prompt = (
            "You are an expert HR Analyst. You are given a JD requirement, the best matching CV evidence, "
            "and a mathematically calculated match status and score.\n\n"
            "GOAL: Provide a natural language justification and 1-2 targeted interview questions that probe for "
            "SPECIFIC IMPLEMENTATION DETAILS.\n"
            "STRICT RULES:\n"
            "1. TECHNICAL DEPTH: Questions must probe for concrete details - what models, what scale, "
            "what results, what architecture, what challenges overcame.\n"
            "2. ALIGNMENT: Your justification MUST explain why the status was assigned based on the evidence.\n"
            "3. BE SPECIFIC: Reference actual projects, technologies, or achievements from the CV evidence.\n"
            "4. CONCISE: Keep the reason to 2-3 sentences.\n"
            "5. NO INTERNAL MONOLOGUE: Return only the final JSON.\n\n"
            "Output ONLY valid JSON with keys: 'reason', 'questions' (list)."
        )
    else:
        system_prompt = (
            "You are an expert HR Analyst. You are given a JD requirement, the best matching CV evidence, "
            "and a mathematically calculated match status and score.\n\n"
            "GOAL: Provide a natural language justification and brief behavioral interview questions.\n"
            "STRICT RULES:\n"
            "1. BEHAVIORAL: Questions should ask about real situations demonstrating this skill.\n"
            "2. ALIGNMENT: Your justification MUST explain why the status was assigned based on the evidence.\n"
            "3. CONCISE: Keep the reason to 2-3 sentences.\n"
            "4. NO INTERNAL MONOLOGUE: Return only the final JSON.\n\n"
            "Output ONLY valid JSON with keys: 'reason', 'questions' (list)."
        )
    
    user_prompt = (
        f"Requirement: {requirement}\n"
        f"CV Evidence: {cv_evidence}\n"
        f"Calculated Status: {status} (Similarity Score: {score:.4f})\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=1024)
        result = clean_llm_json(raw)
        if isinstance(result, dict):
            return result
        return {"reason": str(raw), "questions": []}
    except Exception as e:
        logger.error(f"Error in justifying match: {e}")
        return {"reason": "Automatic justification failed.", "questions": []}


def generate_gap_analysis(technical_gaps: list, partial_gaps: list,
                         critical_requirements: list) -> str:
    """
    Generate detailed gap analysis with specific risks and interview probes.
    """
    if not technical_gaps and not partial_gaps:
        return "Strong match across all technical requirements. No significant gaps identified."

    system_prompt = (
        "You are an expert HR Analyst. For each gap, provide a brief but specific analysis:\n"
        "1. What this gap means for the role\n"
        "2. How critical this gap is\n"
        "3. What to probe in the interview\n\n"
        "FORMAT: Write 2-3 sentences per gap. Be specific about risks, not generic.\n"
        "If multiple similar gaps exist, group them together.\n"
        "Output ONLY the analysis text, no JSON."
    )

    gaps_text = []
    for g in technical_gaps:
        gaps_text.append(f"Missing requirement: {g['requirement']}")
    for g in partial_gaps:
        gaps_text.append(f"May need verification: {g['requirement']}")

    user_prompt = (
        f"Critical requirements for this role: {', '.join(critical_requirements)}\n\n"
        f"Gaps to analyze:\n" + "\n".join(gaps_text)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=768)
        return raw.strip()
    except Exception as e:
        logger.error(f"Error generating gap analysis: {e}")
        # Fallback
        parts = []
        if technical_gaps:
            parts.append(f"Missing {len(technical_gaps)} technical requirements")
        if partial_gaps:
            parts.append(f"{len(partial_gaps)} requirements may need verification")
        return ". ".join(parts) + "."


def expand_jd_requirements(jd_text: str, requirements: dict | None = None) -> dict:
    """
    Expand short JD requirements into detailed descriptions for better embedding retrieval.
    Processes one section at a time for maximum anchoring and focus.
    """
    if requirements is None:
        requirements = decompose_job_description(jd_text)
    
    expanded = {}
    
    system_prompt = (
        "You are an expert Technical Recruiter specializing in AI and Data Science. "
        "Expand the following job description into a JSON object where keys are section names. "
        "For each section, provide a single paragraph (2-3 sentences) of the ideal candidate profile. "
        "Output ONLY valid JSON."
    )
    
    for section, reqs in requirements.items():
        if not reqs: continue
        
        user_prompt = (
            f"Section: {section}\nConcise Requirements: {json.dumps(reqs)}\n\n"
            f"Task: Write a detailed, context-rich paragraph describing the ideal candidate for this section."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            raw = call_llm(messages, temperature=0.0, max_tokens=1024)
            data = clean_llm_json(raw)
            if isinstance(data, dict):
                # Pick the first string value if it returned a dict, else the raw text
                detail = next((v for v in data.values() if isinstance(v, str)), raw)
            else:
                detail = raw
            expanded[section] = str(detail).strip()
        except Exception as e:
            expanded[section] = " ".join(reqs)
            
    return expanded


def semantic_chunk_section(section_name: str, text: str) -> list[str]:
    """
    Intelligently split a CV section into isolated, atomic items (e.g., one job role, one project, one degree).
    This drastically improves deterministic matching by isolating context.
    """

    system_prompt = (
        "You are a CV Parser. Your task is to split the provided CV section text into a JSON array of strings following some rules and examples.\n"
        "CRITICAL RULES FOR CONTEXT PRESERVATION AND SPLITTING:\n"
        "1. EACH job/project/item must be its OWN string in the array. The string MUST COMBINE the title, company/name, dates, AND EVERY associated bullet or description (if any) into a single paragraph string.\n"
        "2. Some projects might contain groups of sub projects. These might include names such as key projects, major/minor projects, internship projects, ...etc.\n"
        "3. DO NOT summarize, alter, or omit ANY original text. Just concatenate all lines belonging to the same item.\n"
        "4. Try to detect patterns to know when to start a new string. Keep in mind that a single item can have multiple bullets. You MUST not separate items that feel like they should have belonged to the same string\n"
        "EXAMPLE1:\n"
        "If text is:\n"
        "Movie System - Jun 2024\n"
        "- Used Keras and ML.\n"
        "- Deployed to AWS.\n\n"
        "AI Engineer - Seimens 01/01/2001 - present\n"
        "Developed ..."
        "Worked on ...\n"
        "You output:\n"
        "[\"Movie System - Jun 2024 - Used Keras and ML. - Deployed to AWS.\", \"AI Engineer - Seimens 01/01/2001 - present. Developed ... . Worked on ... .\"]\n\n"
        "EXAMPLE 2:"
        "Major projects\n"
        "Project 1\n"
        "bullet 1\n"
        "bullet 2\n"
        "Project 2\n"
        "bullet 1\n"
        "Internship projects\n"
        "Project 3\n"
        "bullet 1\n"
        "You output:\n"
        "[\"Project 1. bullet 1. bullet 2.\", \"Project 2. bullet 1.\", \"Project 3. bullet 1.\"]\n\n"
        "5. Completely ignore any page footers like page numbers 1/2. You will no them as they will not have any association with the section name.\n"
        "Output ONLY a valid JSON array of strings: [\"Item 1 + Details\", \"Item 2 + Details\"]."
    )
    user_prompt = f"Section: {section_name}\n\nText:\n{text}\n\nReturn the JSON array of strings."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=2048)
        
        # Extract JSON array
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1:
            json_str = raw[start:end+1]
            chunks = json.loads(json_str)
            if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks):
                return [c.strip() for c in chunks if c.strip()]
                
        # Fallback if parsing fails or LLM disobeys
        logger.warning(f"Failed to parse JSON array for semantic sub_chunk. Fallback to raw text split.")
        from embedding_service import sub_chunk # Fallback to naive
        return sub_chunk(text)
        
    except Exception as e:
        logger.warning(f"Semantic chunking failed: {e}")
        from embedding_service import sub_chunk # Fallback to naive
        return sub_chunk(text)


def analyze_section_match(section_name: str, jd_requirements: str, cv_evidence: str, mode: str) -> dict:
    """
    Perform a granular holistic comparison between JD requirements and a pool of CV chunks.
    jd_requirements: JSON list of requirement strings.
    cv_evidence: JSON list of CV text chunks (the evidence pool).
    """
    if mode == "employer":
        system_prompt = (
            "You are an expert HR Analyst. Evaluate each JD requirement against the POOL of CV evidence chunks.\n"
            "DETERMINISTIC STATUS RULES:\n"
            "- 'Close Match':\n"
            "  * ALL aspects of the requirement are explicitly met.\n"
            "  * EXPLICIT RULE: Any specific Engineering degree satisfies a general 'Engineering degree' requirement.\n"
            f"  * DATE MATH RULE: Today is {datetime.now().strftime('%Y-%m-%d')}. Treat 'Present' as today.\n"
            "    1. EXCLUDE education, diplomas, and internships from 'years of experience' sums.\n"
            "    2. ONLY use the EXACT date strings found in the CV (e.g., 'Feb 2025 - Present').\n"
            "    3. DO NOT INVENT NUMBERS. If a role says 'Oct 2024 - Jan 2025', it is ~0.3 years. Mar 2025 - Mar 2026 is 1 year.\n"
            "    4. VERIFY TENURE: If the total relevant working duration is lower than the JD requirement, it IS NOT a Close Match.\n"
            "  * DO NOT penalize for irrelevant chunks in the pool.\n"
            "- 'Partial Match': Evidence is incomplete, or tenure is slightly below requirement (e.g. 3.5 years for 4).\n"
            "- 'No Match': Zero relevant evidence or tenure is significantly below requirement (e.g. 1.2 years for 4).\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "1. EXACT COUNT: Return exactly ONE checklist object for each string in the 'JD Requirements' list.\n"
            "2. UNIQUE IDS: Use 1-based index (1, 2, 3...) for 'requirement' ID.\n"
            "3. NO INTERNAL MONOLOGUE: Under 'reason', provide ONLY the final justification. NO scratchpad math.\n"
            "4. NO HALLUCINATION: If the math doesn't add up to the requirement, you MUST state the actual calculated duration in the reason.\n"
            "Output ONLY valid JSON."
        )
        user_prompt = (
            f"Section: {section_name}\n\nJD Requirements:\n{jd_requirements}\n\nCV Evidence Pool:\n{cv_evidence}\n\n"
            "Return JSON with keys:\n"
            "1. 'why_fits': Brief summary of section fit.\n"
            "2. 'match_checklist': A list of objects for each requirement number: {'requirement': int, 'status': 'Close Match'|'Partial Match'|'No Match', 'reason': 'Explain why'}]\n"
            "3. 'things_to_keep_in_mind': Potential gaps or risks.\n"
            "4. 'questions': Up to 5 targeted interview questions."
        )
        defaults = {"why_fits": "Candidate background noted.", "match_checklist": [], "things_to_keep_in_mind": "No data.", "questions": []}
    else:
        system_prompt = (
            "You are an expert Career Coach. Evaluate each JD requirement against the POOL of CV evidence chunks.\n"
            "STATUS RULES:\n"
            "- 'Close Match': Full evidence found OR semantic equivalent present (e.g. Electronics Engineering satisfies Engineering).\n"
            "- 'Partial Match': Limited or indirect evidence.\n"
            "- 'No Match': No relevant evidence.\n\n"
            "CRITICAL RULES:\n"
            "1. Ignore irrelevant chunks if a match is found elsewhere.\n"
            "2. Be generous with semantic equivalents for degrees and core tech stacks.\n"
            f"Current date: {datetime.now().strftime('%Y-%m-%d')} to be taken if experience years is to be calculated. Output ONLY valid JSON."
        )
        user_prompt = (
            f"Section: {section_name}\n\nJD Requirements:\n{jd_requirements}\n\nCV Evidence Pool:\n{cv_evidence}\n\n"
            "Return JSON with keys:\n"
            "1. 'comparison': Detailed gap analysis.\n"
            "2. 'match_checklist': A list of objects for each requirement number: {'requirement': int, 'status': 'Close Match'|'Partial Match'|'No Match', 'reason': '...'}]\n"
            "3. 'improvement_suggestions': Actionable tips."
        )
        defaults = {"comparison": "Section comparison handled.", "match_checklist": [], "improvement_suggestions": "No specific suggestions."}

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        # Use temperature 0.0 for deterministic scoring
        raw = call_llm(messages, temperature=0.0, max_tokens=2048)
        result = clean_llm_json(raw)
        
        # Unwrap if LLM nested the response under the section name (common behavior)
        sn_lower = section_name.lower()
        if sn_lower in result and isinstance(result[sn_lower], dict):
            result = {k.lower(): v for k, v in result[sn_lower].items()}

        for k, v in defaults.items():
            if k not in result: result[k] = v
            
        result["_debug_system"] = system_prompt
        result["_debug_prompt"] = user_prompt
        result["_debug_response"] = raw
        return result
    except Exception as e:
        defaults["_debug_system"] = system_prompt
        defaults["_debug_prompt"] = user_prompt
        defaults["_debug_response"] = f"ERROR: {e}"
        return defaults


def classify_requirements(requirements: list[str]) -> list[dict]:
    """
    Classify each JD requirement into tiers and extract key technical terms.
    Input: list of requirement strings from STEP 1 (decompose_job_description)
    Output: list of dicts {"requirement": str, "tier": "critical|important|nice_to_have", "key_terms": [str]}
    Uses LLM with temperature=0.0, seed=42 for deterministic output.
    """
    if not requirements:
        return []

    system_prompt = (
        "You are an expert technical recruiter. Classify each job requirement by importance tier "
        "and extract key technical terms for exact match checking.\n\n"
        "TIER DEFINITIONS:\n"
        "- 'critical': Non-negotiable skills, minimum years of experience, required degrees/certifications, "
        "or specific technologies without which the role cannot be performed.\n"
        "- 'important': Strongly preferred qualifications that significantly impact performance in the role.\n"
        "- 'nice_to_have': Bonus qualifications that add value but are not essential.\n\n"
        "KEY TERMS: Extract 2-5 essential technical terms, jargon, or exact phrases from each requirement "
        "that should trigger an exact match boost. Include tools, frameworks, methodologies, degrees, "
        "or specific certifications.\n\n"
        "CRITICAL RULES:\n"
        "1. Be conservative with 'critical' tier - only mark truly essential requirements.\n"
        "2. Key terms should be exact matches (case-insensitive) from the requirement text.\n"
        "3. Output ONLY valid JSON - a list of objects with keys: requirement, tier, key_terms.\n"
        "4. Preserve the original requirement text exactly as provided.\n"
    )

    requirements_json = json.dumps(requirements, indent=2)
    user_prompt = f"Requirements to classify:\n{requirements_json}\n\nReturn a JSON list of classification objects."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=2048)
        data = clean_llm_json(raw)

        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, dict):
                    result.append({
                        "requirement": str(item.get("requirement", "")),
                        "tier": str(item.get("tier", "important")),
                        "key_terms": [str(k) for k in item.get("key_terms", []) if k]
                    })
            return result
        return [{"requirement": r, "tier": "important", "key_terms": []} for r in requirements]
    except Exception as e:
        logger.error(f"Requirement classification failed: {e}")
        return [{"requirement": r, "tier": "important", "key_terms": []} for r in requirements]


def synthesize_candidate_analysis(section_analyses: list[dict], mode: str) -> dict:
    """
    Synthesize holistic section-to-section analyses into the final report headers.
    """
    summary_data = []
    for a in section_analyses:
        section = a.get("section", "General")
        if mode == "employer":
            summary_data.append({
                "s": section, 
                "f": a.get("why_fits", ""), 
                "g": a.get("things_to_keep_in_mind", ""),
                "q": a.get("questions", [])
            })
        else:
            summary_data.append({
                "s": section, 
                "c": a.get("comparison", ""), 
                "u": a.get("improvement_suggestions", "")
            })

    combined_text = json.dumps(summary_data)
    
    if mode == "employer":
        keys_desc = "\"ranking_overview_summary\" (string), \"why_fits\" (list), \"things_to_keep_in_mind\" (list), \"questions\" (list of max 5 most insightful/critical questions total across all sections)."
        defaults = {"ranking_overview_summary": "Summary unavailable.", "why_fits": [], "things_to_keep_in_mind": [], "questions": []}
    else:
        keys_desc = "\"general_comparison_summary\" (string), \"improvement_suggestions\" (list)."
        defaults = {"general_comparison_summary": "Summary processed.", "improvement_suggestions": []}

    system_prompt = "You are an expert analyst. Summarize these granular analyses into a CONCISE recap. For 'questions', select ONLY the 5 most impactful ones from the provided lists. Do NOT repeat or loop. Output ONLY valid JSON."
    user_prompt = f"Analyses (including section strengths, gaps, and proposed questions):\n{combined_text}\n\nGoal: Summarize according to the JD. Keys: {keys_desc}"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=2048) # Ultra low temp to avoid loops
        result = clean_llm_json(raw)
        
        # Flatten summary if it's an object instead of string (LLM quirk)
        summary = result.get("general_comparison_summary", result.get("ranking_overview_summary"))
        if isinstance(summary, dict):
            result["general_comparison_summary"] = ". ".join([str(v) for v in summary.values()])
            result["ranking_overview_summary"] = result["general_comparison_summary"]

        for k, v in defaults.items():
            if k not in result: result[k] = v
        return result
    except Exception as e:
        return defaults
