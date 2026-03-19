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

from db import search_similar_pool, search_best_chunk_for_cv
from embedding_service import generate_embedding
from llm_service import (
    decompose_job_description,
    analyze_section_match,
    synthesize_candidate_analysis
)
from report_service import generate_report_pdf, generate_jd_analysis_report

logger = logging.getLogger(__name__)


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
    """Normalize a section name to its primary category if it matches an alias."""
    name_clean = name.strip().lower()
    # Basic categories to map to
    categories = ["Summary", "Experience", "Education", "Skills", "Projects", "Certifications", "Languages"]
    
    # Check aliases from config
    # (Simple mapping: if alias is in categories, we use it. 
    # In a real app, we'd have a dict mapping aliases to categories.)
    for cat in categories:
        if cat.lower() in name_clean:
            return cat
    return name


def match_candidates(job_description: str, top_candidates: int = 3,
                     pool_size: int = 50, section_filter: str | None = None,
                     mode: str = "employer",
                     output_dir: str = ".") -> dict[str, Any]:
    """
    Rank candidates using Metadata-Driven Section-to-Section Comparison.
    Ensures JD sections are compared against the relevant CV sections.
    """
    print(f"\n[STEP 1] Decomposing Job Description into Categorized Requirements...")
    requirements_dict = decompose_job_description(job_description)
    
    headings, weights_config = _get_section_config()
    sections_present = list(requirements_dict.keys())
    
    # Normalize weights for the sections found in the JD
    total_w = sum(weights_config.get(s, 1.0) for s in sections_present)
    normalized_weights = {s: weights_config.get(s, 1.0) / total_w if total_w > 0 else 1.0/len(sections_present) 
                         for s in sections_present}
    
    # Pre-generate embeddings for each JD section (aggregated requirements)
    print(f"[STEP 2] Generating section-level aggregated embeddings...")
    section_embeddings = {}
    for section, reqs in requirements_dict.items():
        if not reqs: continue
        section_text = f"Section: {section}\n" + "\n".join([f"- {r}" for r in reqs])
        section_embeddings[section] = generate_embedding(section_text, prefix="Query: ")
    
    # Generate Standalone JD Analysis Report
    jd_report_path = generate_jd_analysis_report(requirements_dict, job_description, output_dir)

    # 1. Initial Global Search to identify top pool
    print(f"[STEP 3] Identifying top candidates via global pool search (size={pool_size})...")
    query_embedding = generate_embedding(job_description, prefix="Query: ")
    raw_results = search_similar_pool(query_embedding, pool_size=pool_size, section_filter=section_filter)
    
    if not raw_results:
        return {"candidates": [], "mode": mode, "requirements": requirements_dict, "jd_report": jd_report_path, "raw_chunk_matches": 0, "candidate_count": 0}

    # Group all chunks in the pool by candidate
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
    top_candidates_pool = candidate_ranking_pool[:max(top_candidates, 5)]

    final_results = []
    
    print(f"[STEP 4] Metadata-Driven Deep Analysis of top {len(top_candidates_pool)} candidates...")
    for idx, cand in enumerate(top_candidates_pool, 1):
        cv_id = cand["cv_id"]
        file_name = cand["file_name"]
        print(f"  -> Explicit Section Matching #{idx}: {file_name}")
        
        # Pull ALL chunks for this candidate from the DB to have full context per section
        # (We use the pre-fetched chunks from global search as a base, but we might need more 
        # to guarantee we have all chunks for a specific section.)
        all_cand_chunks = grouped_pool[cv_id]
        
        # Group candidate chunks by their normalized section name
        cv_sections = defaultdict(list)
        for chunk in all_cand_chunks:
            norm_name = _normalize_section(chunk["section_name"], headings)
            cv_sections[norm_name].append(chunk)

        section_analyses = []
        detailed_sections = {}
        candidate_score = 0.0
        unique_matched_chunk_indices = set()
        
        for jd_section, reqs in requirements_dict.items():
            if not reqs: continue
            
            # Find the MOST RELEVANT CV context for this JD Section
            relevant_cv_text = ""
            best_sim_for_score = 0.0
            
            # Primary: Metadata Match (by section name/alias)
            matched_cv_chunks = cv_sections.get(jd_section, [])
            if matched_cv_chunks:
                relevant_cv_text = "\n".join([c["chunk_text"] for c in matched_cv_chunks])
                # Similarity for scoring: Use the max similarity of any chunk in the matched section
                # to the JD section embedding
                best_sim_for_score = max([c["similarity"] for c in matched_cv_chunks])
                for c in matched_cv_chunks: unique_matched_chunk_indices.add(c["chunk_index"])
            else:
                # Fallback: Vector Search (find the best matching chunk if section name didn't match)
                fallback_chunk = search_best_chunk_for_cv(cv_id, section_embeddings.get(jd_section, query_embedding))
                if fallback_chunk:
                    relevant_cv_text = fallback_chunk["chunk_text"]
                    best_sim_for_score = fallback_chunk["similarity"]
                    unique_matched_chunk_indices.add(fallback_chunk["chunk_index"])
            
            if relevant_cv_text:
                # Compare JD section requirements against the relevant CV context
                analysis = analyze_section_match(jd_section, reqs, relevant_cv_text, mode)
                analysis["section"] = jd_section
                analysis["similarity"] = best_sim_for_score
                analysis["weight"] = normalized_weights.get(jd_section, 0.0)
                
                section_analyses.append(analysis)
                detailed_sections[jd_section] = analysis
                candidate_score += best_sim_for_score * analysis["weight"]
            else:
                detailed_sections[jd_section] = {
                    "section": jd_section,
                    "similarity": 0.0,
                    "weight": normalized_weights.get(jd_section, 0.0),
                    "comparison" if mode != "employer" else "why_fits": "No relevant section found in CV."
                }

        # REDUCE: Final synthesis
        synthesis = synthesize_candidate_analysis(section_analyses, mode)
        
        # Build precise result object
        result = {
            "cv_id": cv_id,
            "file_name": file_name,
            "rank": idx,
            "score": candidate_score,
            "matched_sections": sorted(list(set([c["section_name"] for c in all_cand_chunks if c["chunk_index"] in unique_matched_chunk_indices]))),
            "detailed_sections": detailed_sections
        }
        
        if mode == "employer":
            result.update({
                "summary": synthesis.get("ranking_overview_summary", ""),
                "reasons": synthesis.get("why_fits", []),
                "keep_in_mind": synthesis.get("things_to_keep_in_mind", []),
                "questions": synthesis.get("questions", [])
            })
        else: # client mode
            result.update({
                "summary": synthesis.get("general_comparison_summary", ""),
                "suggestions": synthesis.get("improvement_suggestions", [])
            })

        # Generate report
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
