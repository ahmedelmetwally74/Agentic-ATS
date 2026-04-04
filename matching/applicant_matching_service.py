from __future__ import annotations

from typing import Any

from core.applicant_cv_parser import get_items_for_chunk, parse_applicant_cv_items, rank_items_for_requirement
from core.applicant_experience_service import extract_required_years, summarize_experience_duration
from core.db import get_chunks_for_cv, search_similar_for_cv
from core.embedding_service import generate_embedding
from llm.applicant_llm_service import (
    analyze_requirement_match,
    decompose_job_description_to_requirements,
    synthesize_applicant_requirement_results,
)
from llm.llm_shared import unique_keep_order
from matching.applicant_requirement_matchers import (
    build_calibrated_notes,
    build_evidence_profile,
    build_safe_requirement_suggestion,
    calibrate_requirement_status,
    evaluate_education_requirement,
    evaluate_language_requirement,
    evaluate_tool_requirement,
)
from reporting.applicant_report_service import generate_applicant_report_pdf
from reporting.jd_report_service import generate_jd_analysis_report, save_jd_requirements_json


def analyze_applicant_cv(
    job_description: str,
    cv_id: str,
    section_filter: str | None = None,
    output_dir: str = ".",
) -> dict[str, Any]:
    """
    Analyze one applicant CV against one JD using requirement-centric matching.
    Phase 1.1 keeps company mode untouched while tightening applicant quality.
    """
    if section_filter:
        raise ValueError("--section is not supported in applicant mode for the requirement-centric flow.")

    print("\n[STEP 1] Decomposing Job Description into Requirement Objects...")
    requirements = decompose_job_description_to_requirements(job_description)
    jd_requirements_json_path = save_jd_requirements_json(requirements, output_dir)
    jd_report_path = generate_jd_analysis_report(requirements, job_description, output_dir)

    print(f"[STEP 2] Loading applicant CV chunks for cv_id={cv_id} ...")
    all_candidate_chunks = get_chunks_for_cv(cv_id)

    if not all_candidate_chunks:
        return {
            "candidates": [],
            "mode": "applicant",
            "requirements": requirements,
            "jd_report": jd_report_path,
            "jd_requirements_json": jd_requirements_json_path,
            "raw_chunk_matches": 0,
            "candidate_count": 0,
        }

    print("[STEP 3] Parsing applicant CV items and Experience-only duration evidence...")
    parsed_items = parse_applicant_cv_items(all_candidate_chunks)
    experience_summary = summarize_experience_duration(all_candidate_chunks)

    print("[STEP 4] Evaluating applicant requirements one by one...")
    requirement_results = []
    for requirement in requirements:
        result = _evaluate_requirement(requirement, cv_id, parsed_items, experience_summary)
        requirement_results.append(result)

    synthesis = synthesize_applicant_requirement_results(requirement_results, experience_summary=experience_summary)
    status_counts = _build_status_counts(requirement_results)

    result = {
        "cv_id": cv_id,
        "file_name": all_candidate_chunks[0]["file_name"],
        "summary": synthesis.get("general_summary", ""),
        "suggestions": synthesis.get("overall_suggestions", []),
        "requirement_results": requirement_results,
        "status_counts": status_counts,
        "experience_summary": experience_summary,
        "parsed_cv_items_count": len(parsed_items),
        "jd_report": jd_report_path,
        "jd_requirements_json": jd_requirements_json_path,
    }

    pdf_path = generate_applicant_report_pdf(result, output_dir=output_dir)
    result["report_pdf"] = pdf_path

    return {
        "candidates": [result],
        "mode": "applicant",
        "requirements": requirements,
        "jd_report": jd_report_path,
        "jd_requirements_json": jd_requirements_json_path,
        "raw_chunk_matches": len(all_candidate_chunks),
        "candidate_count": 1,
    }


def _evaluate_requirement(
    requirement: dict,
    cv_id: str,
    parsed_items: list[dict],
    experience_summary: dict,
) -> dict:
    strategy = requirement.get("matching_strategy")
    category = requirement.get("category")

    if strategy == "experience_duration":
        return _evaluate_years_requirement(requirement, experience_summary)

    if category == "education":
        return evaluate_education_requirement(requirement, parsed_items)

    if category == "language":
        return evaluate_language_requirement(requirement, parsed_items)

    evidence_chunks = _retrieve_requirement_evidence(cv_id, requirement, top_k=_evidence_pool_size(requirement))

    if category == "tool":
        return evaluate_tool_requirement(requirement, parsed_items, evidence_chunks)

    top_evidence = _build_top_evidence(requirement, evidence_chunks, parsed_items)
    raw_result = analyze_requirement_match(
        requirement,
        evidence_chunks,
        top_evidence=top_evidence,
    )
    evidence_profile = build_evidence_profile(requirement, parsed_items, evidence_chunks, top_evidence)
    calibrated_status = calibrate_requirement_status(requirement, raw_result.get("status", ""), evidence_profile)
    calibrated_notes = build_calibrated_notes(
        requirement,
        calibrated_status,
        evidence_profile,
        top_evidence,
        raw_result.get("notes", ""),
    )
    safe_suggestion = build_safe_requirement_suggestion(requirement, calibrated_status, top_evidence)

    return {
        "requirement_id": requirement["id"],
        "requirement_text": requirement["text"],
        "category": requirement["category"],
        "importance": requirement["importance"],
        "section_group": requirement["section_group"],
        "matching_strategy": requirement["matching_strategy"],
        "id": requirement["id"],
        "text": requirement["text"],
        "status": calibrated_status,
        "top_evidence": top_evidence,
        "notes": calibrated_notes,
        "suggestion": safe_suggestion,
    }


def _retrieve_requirement_evidence(cv_id: str, requirement: dict, top_k: int = 3) -> list[dict]:
    query_text = (
        f"Requirement: {requirement['text']}\n"
        f"Category: {requirement['category']}\n"
        f"Section group: {requirement['section_group']}\n"
        f"Matching strategy: {requirement['matching_strategy']}"
    )
    query_embedding = generate_embedding(query_text, prefix="Query: ")
    return search_similar_for_cv(cv_id, query_embedding, top_k=top_k)


def _evaluate_years_requirement(requirement: dict, experience_summary: dict) -> dict:
    required_years = extract_required_years(requirement["text"])
    top_evidence = experience_summary.get("top_evidence", [])
    total_years = float(experience_summary.get("total_years", 0.0) or 0.0)
    parsed_entries = experience_summary.get("parsed_entries", [])

    if not experience_summary.get("experience_section_found"):
        status = "missing_or_insufficient"
        notes = (
            "No main Experience section with parseable date evidence was found, "
            "so the applicant flow could not support this years-of-experience requirement."
        )
    elif not parsed_entries:
        status = "missing_or_insufficient"
        notes = (
            "The Experience section exists, but usable date ranges could not be parsed from it, "
            "so the years-of-experience requirement remains unsupported."
        )
    elif required_years is None:
        status = "missing_or_insufficient"
        notes = "The requirement was marked as experience_duration, but no numeric years target could be extracted from the JD text."
    else:
        status = _classify_years_requirement_status(required_years, total_years)
        if status == "matched_explicitly":
            notes = (
                f"The Experience section provides approximately {total_years:.2f} years of overlap-aware dated experience, "
                f"which meets or exceeds the stated {required_years:.2f}-year requirement."
            )
        elif status == "partially_matched":
            notes = (
                f"The Experience section shows approximately {total_years:.2f} years of overlap-aware dated experience, "
                f"which is a near miss against the stated {required_years:.2f}-year requirement."
            )
        elif total_years > 0:
            notes = (
                f"The Experience section shows approximately {total_years:.2f} years of overlap-aware dated experience, "
                f"which is materially below the stated {required_years:.2f}-year requirement."
            )
        else:
            notes = "No dated Experience coverage could be established from the main Experience section."

    suggestion = build_safe_requirement_suggestion(requirement, status, top_evidence)
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
        "top_evidence": top_evidence,
        "notes": notes,
        "suggestion": suggestion,
    }


def _classify_years_requirement_status(required_years: float, total_years: float) -> str:
    if total_years >= required_years:
        return "matched_explicitly"
    if total_years <= 0:
        return "missing_or_insufficient"

    near_miss_threshold = max(required_years * 0.75, required_years - 1.0)
    if total_years >= near_miss_threshold:
        return "partially_matched"
    return "missing_or_insufficient"


def _evidence_pool_size(requirement: dict) -> int:
    category = requirement.get("category")
    if category in {"tool", "experience", "domain"}:
        return 8
    return 6


def _build_status_counts(requirement_results: list[dict]) -> dict:
    statuses = [
        "matched_explicitly",
        "partially_matched",
        "not_explicitly_stated",
        "missing_or_insufficient",
    ]
    return {
        status: sum(1 for result in requirement_results if result.get("status") == status)
        for status in statuses
    }


def _build_top_evidence(requirement: dict, evidence_chunks: list[dict], parsed_items: list[dict]) -> list[str]:
    candidate_items = []

    for chunk in evidence_chunks:
        candidate_items.extend(get_items_for_chunk(parsed_items, chunk["section_name"], chunk["chunk_index"]))

    section_weight_map = _get_section_weight_profile(requirement)
    ranked_items = rank_items_for_requirement(
        requirement["text"],
        candidate_items,
        preferred_sections=[requirement.get("section_group", "")],
        section_weight_map=section_weight_map,
        limit=5,
    )
    evidence: list[str] = [
        f"{item['section_name']}: {item['text']}"
        for item in ranked_items
        if item.get("text")
    ]

    if not evidence:
        reranked_chunks = sorted(
            evidence_chunks,
            key=lambda chunk: (
                section_weight_map.get(chunk["section_name"].strip().lower(), 0),
                float(chunk.get("similarity", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for chunk in reranked_chunks:
            snippet = " ".join(chunk.get("chunk_text", "").split())
            if snippet:
                evidence.append(f"{chunk['section_name']}: {snippet[:220]}")

    return unique_keep_order(evidence, limit=3)


def _get_section_weight_profile(requirement: dict) -> dict[str, int]:
    category = requirement.get("category")
    section_group = requirement.get("section_group", "").strip().lower()

    if category == "language":
        return {
            "languages": 18,
            "language": 18,
            "summary": 10,
            "professional summary": 10,
            "profile": 10,
            "header": 9,
            "experience": -2,
            "work experience": -2,
            "projects": -6,
            "technical projects": -6,
            "courses": -8,
            "activities": -9,
        }

    if category == "education":
        return {
            "education": 18,
            "academic background": 16,
            "qualifications": 14,
        }

    weights = {
        "experience": 15,
        "work experience": 15,
        "professional experience": 15,
        "employment history": 15,
        "projects": 11,
        "technical projects": 11,
        "personal projects": 9,
        "skills": 7,
        "technical skills": 7,
        "summary": 4,
        "professional summary": 4,
        "courses": -4,
        "training": -3,
        "activities": -6,
        "extracurricular": -6,
        "volunteer": -5,
        "volunteer experience": -5,
        "awards": -3,
    }
    if section_group == "projects":
        weights["projects"] = 15
        weights["technical projects"] = 15
        weights["experience"] = 12
        weights["work experience"] = 12
        weights["professional experience"] = 12
    return weights
