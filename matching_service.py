"""
AgenticATS - Candidate Matching Service
Rank candidates using Metadata-Driven Section-to-Section Comparison.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any
from collections import defaultdict

from db import search_similar_pool, search_best_chunk_for_cv, get_chunks_for_cv
from embedding_service import generate_embedding
from llm_service import (
    decompose_job_description,
    analyze_section_match,
    synthesize_candidate_analysis
)
from report_service import generate_report_pdf, generate_jd_analysis_report

logger = logging.getLogger(__name__)


SECTION_ALIASES = {
    "Summary": [
        "summary", "profile", "professional summary", "career summary",
        "objective", "about me", "header"
    ],
    "Experience": [
        "experience", "work experience", "professional experience",
        "employment history", "career history"
    ],
    "Education": [
        "education", "academic background", "academic qualifications"
    ],
    "Skills": [
        "skills", "technical skills", "core competencies",
        "competencies", "tech stack", "technologies"
    ],
    "Projects": [
        "projects", "project experience", "research",
        "publications", "open source"
    ],
    "Certifications": [
        "certifications", "certificates", "courses", "training"
    ],
    "Languages": [
        "languages", "language"
    ],
}


def _get_section_config() -> tuple[list[str], dict[str, float]]:
    """Load section headings and weights from sections_config.json."""
    headings = []
    weights = {}
    config_path = "sections_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                headings = config.get("section_headings", [])
                weights = config.get("section_weights", {})
        except Exception as e:
            logger.warning(f"Failed to load section config: {e}")
    return headings, weights


def _normalize_section(name: str, aliases: list[str]) -> str:
    """
    Normalize a section name to a canonical category.
    """
    name_clean = name.strip().lower()

    for canonical, alias_list in SECTION_ALIASES.items():
        if canonical.lower() == name_clean:
            return canonical

        if canonical.lower() in name_clean:
            return canonical

        for alias in alias_list:
            if alias in name_clean:
                return canonical

    return name.strip().title()


def match_candidates(job_description: str, top_candidates: int = 3,
                     pool_size: int = 50, section_filter: str | None = None,
                     mode: str = "company",
                     output_dir: str = ".") -> dict[str, Any]:
    """
    Rank candidates using Metadata-Driven Section-to-Section Comparison.
    Ensures JD sections are compared against the relevant CV sections.
    """
    print(f"\n[STEP 1] Decomposing Job Description into Categorized Requirements...")
    requirements_dict = decompose_job_description(job_description)

    headings, weights_config = _get_section_config()
    sections_present = list(requirements_dict.keys())

    total_w = sum(weights_config.get(s, 1.0) for s in sections_present)
    normalized_weights = {
        s: weights_config.get(s, 1.0) / total_w if total_w > 0 else 1.0 / len(sections_present)
        for s in sections_present
    } if sections_present else {}

    print(f"[STEP 2] Generating section-level aggregated embeddings...")
    section_embeddings = {}
    for section, reqs in requirements_dict.items():
        if not reqs:
            continue
        section_text = f"Section: {section}\n" + "\n".join([f"- {r}" for r in reqs])
        section_embeddings[section] = generate_embedding(section_text, prefix="Query: ")

    jd_report_path = generate_jd_analysis_report(requirements_dict, job_description, output_dir)

    print(f"[STEP 3] Identifying top candidates via global pool search (size={pool_size})...")
    query_embedding = generate_embedding(job_description, prefix="Query: ")
    raw_results = search_similar_pool(query_embedding, pool_size=pool_size, section_filter=section_filter)

    if not raw_results:
        return {
            "candidates": [],
            "mode": mode,
            "requirements": requirements_dict,
            "jd_report": jd_report_path,
            "raw_chunk_matches": 0,
            "candidate_count": 0
        }

    grouped_pool = defaultdict(list)
    for r in raw_results:
        grouped_pool[r["cv_id"]].append(r)

    candidate_ranking_pool = []
    for cv_id, chunks in grouped_pool.items():
        avg_sim = sum(c["similarity"] for c in chunks) / len(chunks)
        candidate_ranking_pool.append({
            "cv_id": cv_id,
            "file_name": chunks[0]["file_name"],
            "broad_score": avg_sim
        })

    candidate_ranking_pool.sort(key=lambda x: x["broad_score"], reverse=True)

    top_candidates_pool = candidate_ranking_pool[:top_candidates]

    final_results = []

    print(f"[STEP 4] Metadata-Driven Deep Analysis of top {len(top_candidates_pool)} candidates...")
    for idx, cand in enumerate(top_candidates_pool, 1):
        cv_id = cand["cv_id"]
        file_name = cand["file_name"]
        print(f"  -> Explicit Section Matching #{idx}: {file_name}")

        all_cand_chunks = get_chunks_for_cv(cv_id)
        if not all_cand_chunks:
            all_cand_chunks = grouped_pool[cv_id]

        pool_chunks = grouped_pool[cv_id]

        cv_sections_full = defaultdict(list)
        cv_sections_scored = defaultdict(list)

        for chunk in all_cand_chunks:
            norm_name = _normalize_section(chunk["section_name"], headings)
            cv_sections_full[norm_name].append(chunk)

        for chunk in pool_chunks:
            norm_name = _normalize_section(chunk["section_name"], headings)
            cv_sections_scored[norm_name].append(chunk)

        section_analyses = []
        detailed_sections = {}
        candidate_score = 0.0
        unique_matched_chunk_refs = set()

        for jd_section, reqs in requirements_dict.items():
            if not reqs:
                continue

            canonical_jd_section = _normalize_section(jd_section, headings)

            relevant_cv_text = ""
            best_sim_for_score = 0.0

            matched_cv_chunks = cv_sections_full.get(canonical_jd_section, [])

            if matched_cv_chunks:
                matched_cv_chunks = sorted(
                    matched_cv_chunks,
                    key=lambda x: (x["section_name"], x["chunk_index"])
                )
                relevant_cv_text = "\n\n".join([c["chunk_text"] for c in matched_cv_chunks])

                scored_chunks = cv_sections_scored.get(canonical_jd_section, [])
                if scored_chunks:
                    best_sim_for_score = max(float(c.get("similarity", 0.0)) for c in scored_chunks)
                else:
                    fallback_chunk = search_best_chunk_for_cv(
                        cv_id,
                        section_embeddings.get(jd_section, query_embedding)
                    )
                    if fallback_chunk:
                        best_sim_for_score = fallback_chunk["similarity"]

                for c in matched_cv_chunks:
                    unique_matched_chunk_refs.add((c["section_name"], c["chunk_index"]))

            else:
                fallback_chunk = search_best_chunk_for_cv(
                    cv_id,
                    section_embeddings.get(jd_section, query_embedding)
                )
                if fallback_chunk:
                    relevant_cv_text = fallback_chunk["chunk_text"]
                    best_sim_for_score = fallback_chunk["similarity"]
                    unique_matched_chunk_refs.add((fallback_chunk["section_name"], fallback_chunk["chunk_index"]))

            if relevant_cv_text:
                analysis = analyze_section_match(canonical_jd_section, reqs, relevant_cv_text, mode)
                analysis["section"] = canonical_jd_section
                analysis["similarity"] = best_sim_for_score
                analysis["weight"] = normalized_weights.get(jd_section, 0.0)

                section_analyses.append(analysis)
                detailed_sections[canonical_jd_section] = analysis
                candidate_score += best_sim_for_score * analysis["weight"]
            else:
                if mode == "company":
                    detailed_sections[canonical_jd_section] = {
                        "section": canonical_jd_section,
                        "similarity": 0.0,
                        "weight": normalized_weights.get(jd_section, 0.0),
                        "why_fits": ["No relevant section found in CV."],
                        "things_to_keep_in_mind": [f"The CV does not show clear evidence for {canonical_jd_section}."],
                        "questions": [f"Can you clarify your experience related to {canonical_jd_section.lower()}?"],
                    }
                else:
                    detailed_sections[canonical_jd_section] = {
                        "section": canonical_jd_section,
                        "similarity": 0.0,
                        "weight": normalized_weights.get(jd_section, 0.0),
                        "comparison": "No relevant section found in CV.",
                        "improvement_suggestions": [f"Add a clearer {canonical_jd_section} section aligned with the job requirements."],
                    }

        synthesis = synthesize_candidate_analysis(section_analyses, mode)

        matched_sections = sorted({
            c["section_name"]
            for c in all_cand_chunks
            if (c["section_name"], c["chunk_index"]) in unique_matched_chunk_refs
        })

        result = {
            "cv_id": cv_id,
            "file_name": file_name,
            "rank": idx,
            "score": candidate_score,
            "matched_sections": matched_sections,
            "detailed_sections": detailed_sections
        }

        if mode == "company":
            result.update({
                "summary": synthesis.get("ranking_overview_summary", ""),
                "reasons": synthesis.get("why_fits", []),
                "keep_in_mind": synthesis.get("things_to_keep_in_mind", []),
                "questions": synthesis.get("questions", [])
            })
        else:
            result.update({
                "summary": synthesis.get("general_comparison_summary", ""),
                "suggestions": synthesis.get("improvement_suggestions", [])
            })

        pdf_path = generate_report_pdf(result, mode, output_dir)
        result["report_pdf"] = pdf_path

        final_results.append(result)

    final_results.sort(key=lambda x: x["score"], reverse=True)
    for i, res in enumerate(final_results, 1):
        res["rank"] = i

    return {
        "candidates": final_results[:top_candidates],
        "mode": mode,
        "requirements": requirements_dict,
        "jd_report": jd_report_path,
        "raw_chunk_matches": len(raw_results),
        "candidate_count": len(candidate_ranking_pool)
    }

def analyze_applicant_cv(job_description: str,
                         cv_id: str,
                         section_filter: str | None = None,
                         output_dir: str = ".") -> dict[str, Any]:
    """
    Analyze one specific applicant CV against the JD.
    Applicant mode should never rank multiple candidates.
    """
    print(f"\n[STEP 1] Decomposing Job Description into Categorized Requirements...")
    requirements_dict = decompose_job_description(job_description)

    headings, weights_config = _get_section_config()
    sections_present = list(requirements_dict.keys())

    total_w = sum(weights_config.get(s, 1.0) for s in sections_present)
    normalized_weights = {
        s: weights_config.get(s, 1.0) / total_w if total_w > 0 else 1.0 / len(sections_present)
        for s in sections_present
    } if sections_present else {}

    print(f"[STEP 2] Generating section-level aggregated embeddings...")
    section_embeddings = {}
    for section, reqs in requirements_dict.items():
        if not reqs:
            continue
        section_text = f"Section: {section}\n" + "\n".join([f"- {r}" for r in reqs])
        section_embeddings[section] = generate_embedding(section_text, prefix="Query: ")

    jd_report_path = generate_jd_analysis_report(requirements_dict, job_description, output_dir)

    print(f"[STEP 3] Loading applicant CV chunks for cv_id={cv_id} ...")
    all_cand_chunks = get_chunks_for_cv(cv_id)

    if not all_cand_chunks:
        return {
            "candidates": [],
            "mode": "applicant",
            "requirements": requirements_dict,
            "jd_report": jd_report_path,
            "raw_chunk_matches": 0,
            "candidate_count": 0,
        }

    file_name = all_cand_chunks[0]["file_name"]

    cv_sections_full = defaultdict(list)
    for chunk in all_cand_chunks:
        norm_name = _normalize_section(chunk["section_name"], headings)
        cv_sections_full[norm_name].append(chunk)

    query_embedding = generate_embedding(job_description, prefix="Query: ")

    section_analyses = []
    detailed_sections = {}
    candidate_score = 0.0
    unique_matched_chunk_refs = set()

    for jd_section, reqs in requirements_dict.items():
        if not reqs:
            continue

        canonical_jd_section = _normalize_section(jd_section, headings)
        relevant_cv_text = ""
        best_sim_for_score = 0.0

        matched_cv_chunks = cv_sections_full.get(canonical_jd_section, [])

        if matched_cv_chunks:
            matched_cv_chunks = sorted(
                matched_cv_chunks,
                key=lambda x: (x["section_name"], x["chunk_index"])
            )
            relevant_cv_text = "\n\n".join([c["chunk_text"] for c in matched_cv_chunks])

            fallback_chunk = search_best_chunk_for_cv(
                cv_id,
                section_embeddings.get(jd_section, query_embedding)
            )
            if fallback_chunk:
                best_sim_for_score = fallback_chunk["similarity"]

            for c in matched_cv_chunks:
                unique_matched_chunk_refs.add((c["section_name"], c["chunk_index"]))
        else:
            fallback_chunk = search_best_chunk_for_cv(
                cv_id,
                section_embeddings.get(jd_section, query_embedding)
            )
            if fallback_chunk:
                relevant_cv_text = fallback_chunk["chunk_text"]
                best_sim_for_score = fallback_chunk["similarity"]
                unique_matched_chunk_refs.add((fallback_chunk["section_name"], fallback_chunk["chunk_index"]))

        if relevant_cv_text:
            analysis = analyze_section_match(canonical_jd_section, reqs, relevant_cv_text, mode="applicant")
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

    synthesis = synthesize_candidate_analysis(section_analyses, mode="applicant")

    matched_sections = sorted({
        c["section_name"]
        for c in all_cand_chunks
        if (c["section_name"], c["chunk_index"]) in unique_matched_chunk_refs
    })

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

    pdf_path = generate_report_pdf(result, mode="applicant", output_dir=output_dir)
    result["report_pdf"] = pdf_path

    return {
        "candidates": [result],
        "mode": "applicant",
        "requirements": requirements_dict,
        "jd_report": jd_report_path,
        "raw_chunk_matches": len(all_cand_chunks),
        "candidate_count": 1,
    }