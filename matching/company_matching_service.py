from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm.company_llm_service import analyze_company_section_match, synthesize_company_candidate_analysis
from llm.llm_shared import decompose_job_description
from core.db import get_chunks_for_cv, search_best_chunk_for_cv, search_similar_pool
from core.embedding_service import generate_embedding
from reporting.company_report_service import generate_company_report_pdf
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


def match_candidates(
    job_description: str,
    top_candidates: int = 3,
    pool_size: int = 50,
    section_filter: str | None = None,
    output_dir: str = ".",
) -> dict[str, Any]:
    """
    Rank candidates using Metadata-Driven Section-to-Section Comparison.
    Ensures JD sections are compared against the relevant CV sections.
    """
    print("\n[STEP 1] Decomposing Job Description into Categorized Requirements...")
    requirements_dict = decompose_job_description(job_description)

    _, weights_config = get_section_config()
    normalized_weights = build_normalized_weights(requirements_dict, weights_config)

    print("\n[STEP 2] Generating section-level aggregated embeddings...")
    section_embeddings = build_section_embeddings(requirements_dict)

    jd_report_path = generate_jd_analysis_report(requirements_dict, job_description, output_dir)

    print(f"[STEP 3] Identifying top candidates via global pool search (size={pool_size})...")
    query_embedding = generate_embedding(job_description, prefix="Query: ")
    raw_results = search_similar_pool(query_embedding, pool_size=pool_size, section_filter=section_filter)

    if not raw_results:
        return {
            "candidates": [],
            "mode": "company",
            "requirements": requirements_dict,
            "jd_report": jd_report_path,
            "raw_chunk_matches": 0,
            "candidate_count": 0,
        }

    grouped_pool: defaultdict[str, list[dict]] = defaultdict(list)
    for result in raw_results:
        grouped_pool[result["cv_id"]].append(result)

    candidate_ranking_pool = []
    for cv_id, chunks in grouped_pool.items():
        avg_sim = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
        candidate_ranking_pool.append({
            "cv_id": cv_id,
            "file_name": chunks[0]["file_name"],
            "broad_score": avg_sim,
        })

    candidate_ranking_pool.sort(key=lambda item: item["broad_score"], reverse=True)
    top_candidates_pool = candidate_ranking_pool[:top_candidates]

    final_results = []

    print(f"[STEP 4] Metadata-Driven Deep Analysis of top {len(top_candidates_pool)} candidates...")
    for idx, candidate in enumerate(top_candidates_pool, 1):
        cv_id = candidate["cv_id"]
        file_name = candidate["file_name"]
        print(f"  -> Explicit Section Matching #{idx}: {file_name}")

        all_candidate_chunks = get_chunks_for_cv(cv_id) or grouped_pool[cv_id]
        pool_chunks = grouped_pool[cv_id]

        cv_sections_full = group_chunks_by_normalized_section(all_candidate_chunks)
        cv_sections_scored = group_chunks_by_normalized_section(pool_chunks)

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

                scored_chunks = cv_sections_scored.get(canonical_jd_section, [])
                if scored_chunks:
                    best_sim_for_score = max(float(chunk.get("similarity", 0.0)) for chunk in scored_chunks)
                else:
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
                analysis = analyze_company_section_match(canonical_jd_section, requirements, relevant_cv_text)
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
                    "why_fits": ["No relevant section found in CV."],
                    "things_to_keep_in_mind": [f"The CV does not show clear evidence for {canonical_jd_section}."],
                    "questions": [f"Can you clarify your experience related to {canonical_jd_section.lower()}?"],
                }

        synthesis = synthesize_company_candidate_analysis(section_analyses)
        matched_sections = collect_matched_sections(all_candidate_chunks, matched_chunk_refs)

        result = {
            "cv_id": cv_id,
            "file_name": file_name,
            "rank": idx,
            "score": candidate_score,
            "matched_sections": matched_sections,
            "detailed_sections": detailed_sections,
            "summary": synthesis.get("ranking_overview_summary", ""),
            "reasons": synthesis.get("why_fits", []),
            "keep_in_mind": synthesis.get("things_to_keep_in_mind", []),
            "questions": synthesis.get("questions", []),
        }

        pdf_path = generate_company_report_pdf(result, output_dir=output_dir)
        result["report_pdf"] = pdf_path
        final_results.append(result)

    final_results.sort(key=lambda item: item["score"], reverse=True)
    for rank, result in enumerate(final_results, 1):
        result["rank"] = rank

    return {
        "candidates": final_results[:top_candidates],
        "mode": "company",
        "requirements": requirements_dict,
        "jd_report": jd_report_path,
        "raw_chunk_matches": len(raw_results),
        "candidate_count": len(candidate_ranking_pool),
    }
