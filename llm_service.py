"""
AgenticATS - LLM Service
Qwen3.5-2B chat completions via llama-server HTTP API for candidate analysis.
Implements a Section-Level Map-Reduce pattern.
"""

import json
import logging
import os
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3.5-2B chat completions endpoint (llama-server)
# ---------------------------------------------------------------------------
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")


def call_llm(messages: list[dict], temperature: float = 0.5,
             max_tokens: int = 2048) -> str:
    """
    Call the Qwen3.5-2B chat completions endpoint with a generous timeout.
    Temperature is 0.5 for deterministic technical matching.
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
            timeout=180,  # High timeout for sequential Map-Reduce calls
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


def decompose_job_description(jd_text: str) -> dict[str, list[str]]:
    """
    Decompose a full Job Description into categorized requirements.
    """
    sections = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]
    config_path = "sections_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                all_heads = config.get("section_headings", [])
                if all_heads: 
                    sections = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]
        except: pass

    system_prompt = (
        "You are an expert technical recruiter. Break down a Job Description into core requirements. "
        "Categorize these requirements into standard CV sections: " + ", ".join(sections) + ". "
        "Extract 2-4 points per relevant section. Output ONLY a valid JSON object where keys are "
        "section names and values are lists of requirement strings."
    )
    user_prompt = f"Full Job Description:\n{jd_text}\n\nReturn Categorized JSON Requirements."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        raw = call_llm(messages, temperature=0.5, max_tokens=1536)
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1:
            return {"General": ["Core technical skills", "Professional experience"]}
        
        result = json.loads(raw[start:end+1])
        if not isinstance(result, dict):
            return {"General": [str(result)]}
            
        final_dict = {}
        for k, v in result.items():
            if isinstance(v, list):
                final_dict[str(k)] = [str(i) for i in v]
            else:
                final_dict[str(k)] = [str(v)]
        return final_dict

    except Exception as e:
        logger.warning(f"JD decomposition failed: {e}")
        return {"Experience": ["Professional background"], "Skills": ["Technical proficiency"]}


def analyze_section_match(section_name: str, jd_requirements: list[str], cv_chunk: str, mode: str) -> dict:
    """
    Perform a section-level comparison between JD requirements and a CV chunk.
    Output is tailored to the requested headers in sections.md.
    """
    jd_req_text = "\n".join([f"- {r}" for r in jd_requirements])
    
    if mode == "employer":
        # Company mode headers: Why fits, Things to keep in mind, Questions
        system_prompt = (
            "You are an expert HR Analyst. Compare JD requirements against a CV section. "
            "Determine strengths, unclear points, and specific questions. Output ONLY valid JSON."
        )
        user_prompt = (
            f"JD Section: {section_name}\n"
            f"Requirements:\n{jd_req_text}\n\n"
            f"CV Evidence:\n{cv_chunk}\n\n"
            "Return JSON with keys:\n"
            '- "why_fits": (detailed strengths for this section, max 50 words)\n'
            '- "things_to_keep_in_mind": (unclear points or gaps for this section, max 50 words)\n'
            '- "questions": (2-3 tailored questions for this section)'
        )
    else:
        # Applicant mode headers: Job description vs current cv, Improvement suggestions
        system_prompt = (
            "You are an expert Career Coach. Compare JD requirements against a CV section. "
            "Identify missing skills/experience and suggest improvements. Output ONLY valid JSON."
        )
        user_prompt = (
            f"JD Section: {section_name}\n"
            f"Requirements:\n{jd_req_text}\n\n"
            f"CV Evidence:\n{cv_chunk}\n\n"
            "Return JSON with keys:\n"
            '- "comparison": (How they match, specifically list missing tools/skills/exp, max 50 words)\n'
            '- "improvement_suggestions": (Actionable CV improvements, max 50 words)'
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        raw = call_llm(messages, temperature=0.5, max_tokens=1024)
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1:
            return {"error": "Comparison unavailable"}
        return json.loads(raw[start:end+1])
    except Exception as e:
        return {"error": f"Matching error: {str(e)}"}


def synthesize_candidate_analysis(section_analyses: list[dict], mode: str) -> dict:
    """
    Synthesize section-level analyses into the final mode-based headers.
    """
    combined_text = "\n".join([json.dumps(a) for a in section_analyses])
    
    if mode == "employer":
        goal_desc = (
            "Generate: 1) A 'Ranking & Match Overview' (score-aware summary), "
            "2) A consolidated 'Why fits' list, 3) Important 'Things to keep in mind', "
            "and 4) Tailored technical/HR 'Questions'."
        )
        keys_desc = (
            '- "ranking_overview_summary": (Paragraph summary of candidate fit)\n'
            '- "why_fits": (list of top 3-5 strengths)\n'
            '- "things_to_keep_in_mind": (list of gaps/red flags)\n'
            '- "questions": (list of 5-8 tailored questions)'
        )
    else:
        goal_desc = (
            "Generate: 1) A detailed section comparison summary and "
            "2) A consolidated list of 'General improvement suggestions' for the CV."
        )
        keys_desc = (
            '- "general_comparison_summary": (Paragraph summary of gaps across sections)\n'
            '- "improvement_suggestions": (list of 5-8 actionable CV improvements)'
        )

    system_prompt = (
        "You are an expert analyst. Synthesize section-level analyses into a FINAL report. "
        "Strict Rule: Only report what is explicitly in the evidence. Do NOT hallucinate."
    )
    user_prompt = (
        f"Individual Section Analyses:\n{combined_text}\n\n"
        f"Goal: {goal_desc}\n"
        f"Return JSON with keys:\n{keys_desc}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        raw = call_llm(messages, temperature=0.5, max_tokens=2048)
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("Synthesis failed: No JSON object.")
        return json.loads(raw[start:end+1])
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {
            "ranking_overview_summary" if mode == "employer" else "general_comparison_summary": "Analysis failed.",
            "why_fits" if mode == "employer" else "improvement_suggestions": ["Manual review recommended."]
        }
