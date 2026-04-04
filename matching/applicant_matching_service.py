from __future__ import annotations

from typing import Any

from llm.applicant_llm_service import analyze_applicant_section_match, synthesize_applicant_candidate_analysis
from llm.llm_shared import decompose_job_description
from core.db import get_chunks_for_cv, search_best_chunk_for_cv
from core.embedding_service import generate_embedding
from reporting.applicant_report_service import generate_applicant_report_pdf
from reporting.jd_report_service import generate_jd_analysis_report

from matching.matching_shared import (
    build_normalized_weights,
    build_section_embeddings,
    collect_matched_sections,
    get_section_config,
    group_chunks_by_normalized_section,
    normalize_section,
    sort_chunks_for_section,
)


def analyze_applicant_cv(
    job_description: str,
    cv_id: str,
    section_filter: str | None = None,
    output_dir: str = ".",
) -> dict[str, Any]:
    """
    Analyze one specific applicant CV against the JD.
    Applicant mode should never rank multiple candidates.
    """
    print("\n[STEP 1] Decomposing Job Description into Categorized Requirements...")
    requirements_dict = decompose_job_description(job_description)

    _, weights_config = get_section_config()
    normalized_weights = build_normalized_weights(requirements_dict, weights_config)

    print("\n[STEP 2] Generating section-level aggregated embeddings...")
    section_embeddings = build_section_embeddings(requirements_dict)

    jd_report_path = generate_jd_analysis_report(requirements_dict, job_description, output_dir)

    print(f"[STEP 3] Loading applicant CV chunks for cv_id={cv_id} ...")
    all_candidate_chunks = get_chunks_for_cv(cv_id)

    if not all_candidate_chunks:
        return {
            "candidates": [],
            "mode": "applicant",
            "requirements": requirements_dict,
            "jd_report": jd_report_path,
            "raw_chunk_matches": 0,
            "candidate_count": 0,
        }

    file_name = all_candidate_chunks[0]["file_name"]
    cv_sections_full = group_chunks_by_normalized_section(all_candidate_chunks)
    query_embedding = generate_embedding(job_description, prefix="Query: ")

    section_analyses = []
    detailed_sections = {}
    candidate_score = 0.0
    matched_chunk_refs: set[tuple[str, int]] = set()

    for jd_section, requirements in requirements_dict.items():
        if not requirements:
            continue

        canonical_jd_section = normalize_section(jd_section)
        relevant_cv_text = ""
        best_sim_for_score = 0.0

        matched_cv_chunks = cv_sections_full.get(canonical_jd_section, [])

        if matched_cv_chunks:
            matched_cv_chunks = sort_chunks_for_section(matched_cv_chunks)
            relevant_cv_text = "\n\n".join(chunk["chunk_text"] for chunk in matched_cv_chunks)

            fallback_chunk = search_best_chunk_for_cv(
                cv_id,
                section_embeddings.get(jd_section, query_embedding),
            )
            if fallback_chunk:
                best_sim_for_score = fallback_chunk["similarity"]

            for chunk in matched_cv_chunks:
                matched_chunk_refs.add((chunk["section_name"], chunk["chunk_index"]))
        else:
            fallback_chunk = search_best_chunk_for_cv(
                cv_id,
                section_embeddings.get(jd_section, query_embedding),
            )
            if fallback_chunk:
                relevant_cv_text = fallback_chunk["chunk_text"]
                best_sim_for_score = fallback_chunk["similarity"]
                matched_chunk_refs.add((fallback_chunk["section_name"], fallback_chunk["chunk_index"]))

        if relevant_cv_text:
            analysis = analyze_applicant_section_match(canonical_jd_section, requirements, relevant_cv_text)
            analysis["section"] = canonical_jd_section
            analysis["similarity"] = best_sim_for_score
            analysis["weight"] = normalized_weights.get(jd_section, 0.0)

            section_analyses.append(analysis)
            detailed_sections[canonical_jd_section] = analysis
            candidate_score += best_sim_for_score * analysis["weight"]
        else:
            detailed_sections[canonical_jd_section] = {
                "section": canonical_jd_section,
                "similarity": 0.0,
                "weight": normalized_weights.get(jd_section, 0.0),
                "comparison": "This section is missing or does not show relevant evidence for the job description.",
                "missing_tools": [],
                "missing_skills": [],
                "missing_experience": [f"No clear evidence for {canonical_jd_section} was found."],
                "missing_education": [],
                "improvement_suggestions": [
                    f"Add or strengthen the {canonical_jd_section} section so it matches the job description more clearly."
                ],
            }

    synthesis = synthesize_applicant_candidate_analysis(section_analyses)
    matched_sections = collect_matched_sections(all_candidate_chunks, matched_chunk_refs)

    result = {
        "cv_id": cv_id,
        "file_name": file_name,
        "rank": 1,
        "score": candidate_score,
        "matched_sections": matched_sections,
        "detailed_sections": detailed_sections,
        "summary": synthesis.get("general_comparison_summary", ""),
        "suggestions": synthesis.get("improvement_suggestions", []),
    }

    if section_filter:
        result["section_filter"] = section_filter

    pdf_path = generate_applicant_report_pdf(result, output_dir=output_dir)
    result["report_pdf"] = pdf_path

    return {
        "candidates": [result],
        "mode": "applicant",
        "requirements": requirements_dict,
        "jd_report": jd_report_path,
        "raw_chunk_matches": len(all_candidate_chunks),
        "candidate_count": 1,
    }
