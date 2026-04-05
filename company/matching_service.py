"""
AgenticATS - Candidate Matching Service
Rank candidates using Unified Section-to-Section Matching.
Pairs JD categories directly against CV categories using robust metadata mapping.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any
from collections import defaultdict

from utils.db import search_similar_pool, get_all_chunks_for_cv
from utils.embedding_service import generate_embedding
from utils.llm_service import (
    decompose_job_description,
    classify_requirements,
    generate_gap_analysis
)
from company.report_service import generate_report_pdf, generate_jd_analysis_report, generate_detailed_markdown_report

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


def match_candidates(job_description: str = None, top_candidates: int = 3,
                     pool_size: int = 50, section_filter: str | None = None,
                     mode: str = "company",
                     output_dir: str = ".",
                     preprocessed_requirements: dict = None) -> dict[str, Any]:
    """
    Rank candidates using Flat Literal Matching.
    Scores are 100% deterministic, based on max cosine similarity for each JD requirement.

    Args:
        job_description: Raw job description text (if not using preprocessed_requirements)
        preprocessed_requirements: Dict with 'requirements' and 'classified_requirements' from jd_processor
    """
    # Use pre-processed requirements if provided
    if preprocessed_requirements:
        requirements_list = preprocessed_requirements.get("requirements", [])
        classified_requirements = preprocessed_requirements.get("classified_requirements", [])
        job_description = job_description or ""
        print(f"\n[STEP 1] Using pre-processed requirements ({len(requirements_list)} requirements)")
    else:
        print(f"\n[STEP 1] Extracting Literal Requirements from Job Description...")
        requirements_list = decompose_job_description(job_description)

        if not requirements_list:
            print("Warning: No requirements extracted from JD.")
            return {"candidates": [], "mode": mode, "requirements": [], "jd_report": "", "raw_chunk_matches": 0, "candidate_count": 0}

        # Phase 1b: Classify requirements by tier
        print(f"[STEP 1b] Classifying requirements by importance tier...")
        classified_requirements = classify_requirements(requirements_list)

    # Separate critical requirements for Phase 3 pool enhancement
    critical_reqs = [r for r in classified_requirements if r["tier"] == "critical"]
    critical_req_texts = [r["requirement"] for r in critical_reqs]

    # Generate Standalone JD Analysis Report (Handle flat list)
    jd_report_path = generate_jd_analysis_report(requirements_list, job_description, output_dir)

    # Phase 3: Initial Global Search to identify top pool
    print(f"[STEP 2] Identifying top candidates via global pool search (size={pool_size})...")
    query_embedding = generate_embedding(job_description, prefix="Query: ")
    raw_results = search_similar_pool(query_embedding, pool_size=pool_size, section_filter=section_filter)

    if not raw_results:
        return {"candidates": [], "mode": mode, "requirements": requirements_list, "jd_report": jd_report_path, "raw_chunk_matches": 0, "candidate_count": 0}

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

    # Phase 3b: Per-critical-requirement boost check
    # For each candidate, check if they pass ANY critical requirement
    from utils.embedding_service import generate_embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    if critical_req_texts:
        print(f"[STEP 2b] Checking critical requirements against top pool candidates...")
        critical_embeddings = generate_embeddings(critical_req_texts, prefix="Query: ")
        for cand in candidate_ranking_pool:
            cv_id = str(cand["cv_id"])
            all_cand_chunks = get_all_chunks_for_cv(cv_id)
            if not all_cand_chunks:
                continue
            # Check each critical requirement
            critical_pass_count = 0
            for crit_req, crit_emb in zip(critical_req_texts, critical_embeddings):
                for chunk in all_cand_chunks:
                    sim = cosine_similarity([crit_emb], [chunk["embedding"]])[0][0]
                    if sim >= 0.6:  # At least partial match
                        critical_pass_count += 1
                        break
            cand["critical_pass_count"] = critical_pass_count
            # Boost broad_score if candidate passes critical requirements
            if critical_pass_count > 0:
                cand["broad_score"] = cand["broad_score"] + (0.05 * critical_pass_count)

    candidate_ranking_pool.sort(key=lambda x: x["broad_score"], reverse=True)
    top_candidates_pool = candidate_ranking_pool[:max(top_candidates, 5)]

    final_results = []

    print(f"[STEP 3] Deterministic Deep Analysis of top {len(top_candidates_pool)} candidates...")
    from utils.llm_service import justify_match

    # Pre-calculate JD embeddings for all requirements once
    jd_embeddings = generate_embeddings(requirements_list, prefix="Query: ")

    # Tier weights - critical has much higher weight to prioritize essential requirements
    tier_weights = {"critical": 4.0, "important": 1.0, "nice_to_have": 0.3}

    for idx, cand in enumerate(top_candidates_pool, 1):
        cv_id = str(cand["cv_id"])
        file_name = cand["file_name"]
        print(f"  -> Deep Matching #{idx}: {file_name}")

        # Pull ALL chunks for this candidate to guarantee full coverage
        all_cand_chunks = get_all_chunks_for_cv(cv_id)
        if not all_cand_chunks:
            continue

        match_checklist = []
        total_embedding_score = 0.0
        weighted_score_sum = 0.0
        weight_sum = 0.0
        critical_scores = []

        for req_idx, (req, req_emb) in enumerate(zip(requirements_list, jd_embeddings)):
            # Get classification for this requirement
            if req_idx < len(classified_requirements):
                classification = classified_requirements[req_idx]
                tier = classification["tier"]
                key_terms = classification["key_terms"]
            else:
                classification = {"tier": "important", "key_terms": []}
                tier = "important"
                key_terms = []

            # Calculate similarity against EVERY chunk
            max_sim = -1.0
            best_chunk_text = "N/A"

            for chunk in all_cand_chunks:
                sim = cosine_similarity([req_emb], [chunk["embedding"]])[0][0]
                # 2d. Exact Match Boost
                if key_terms:
                    chunk_text_lower = chunk["chunk_text"].lower()
                    for key_term in key_terms:
                        if key_term.lower() in chunk_text_lower:
                            sim = min(1.0, sim + 0.15)
                            break
                if sim > max_sim:
                    max_sim = sim
                    best_chunk_text = chunk["chunk_text"]

            # Map status based on fixed thresholds
            if max_sim >= 0.85:
                status = "Close Match"
            elif max_sim >= 0.60:
                status = "Partial Match"
            else:
                status = "No Match"

            # Track critical scores for floor penalty
            if tier == "critical":
                critical_scores.append(max_sim)

            # 2a. Weighted Scoring contribution
            weight = tier_weights.get(tier, 1.0)
            weighted_score_sum += max_sim * weight
            weight_sum += weight

            # Justify the match
            justification = justify_match(req, best_chunk_text, status, max_sim)

            match_checklist.append({
                "requirement": req,
                "tier": tier,
                "status": status,
                "score": round(max_sim, 4),
                "reason": justification.get("reason", ""),
                "questions": justification.get("questions", []),
                "evidence": best_chunk_text
            })
            total_embedding_score += max_sim

        # 2a. Weighted Score (primary score - rewards matching important requirements)
        weighted_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0

        # 2b. Base candidate score starts with weighted score
        candidate_score = weighted_score

        # 2c. Critical Requirement Bonus/Penalty
        # Bonus if ALL critical requirements are well matched (>= 0.8)
        # Penalty if ANY critical is weak (< 0.5)
        if critical_scores:
            if all(cs >= 0.8 for cs in critical_scores):
                candidate_score = candidate_score * 1.12  # 12% bonus for strong critical match
            elif any(cs < 0.5 for cs in critical_scores):
                candidate_score = candidate_score * 0.70  # 30% penalty for weak critical

        # Cap at 1.0
        candidate_score = min(1.0, candidate_score)

        # Generate things_to_keep_in_mind from match checklist
        partial_matches = [m for m in match_checklist if m["status"] == "Partial Match"]
        no_matches = [m for m in match_checklist if m["status"] == "No Match"]

        # Soft skills that are typically inferred from experience, not explicitly listed
        soft_skill_keywords = ["fast-paced", "dynamic environment", "detail-oriented", "process oriented",
                               "flexibility", "diverse working styles", "analytical mindset", "business acumen"]

        def is_soft_skill(req_text):
            req_lower = req_text.lower()
            return any(skill in req_lower for skill in soft_skill_keywords)

        # Filter technical gaps (exclude soft skills that can be inferred)
        technical_no = [m for m in no_matches if not is_soft_skill(m['requirement'])]
        technical_partial = [m for m in partial_matches if not is_soft_skill(m['requirement'])]

        # Generate rich gap analysis using LLM
        things_to_keep_in_mind_text = generate_gap_analysis(
            technical_gaps=technical_no + technical_partial,
            partial_gaps=partial_matches,
            critical_requirements=[r["requirement"] for r in critical_reqs] if critical_reqs else []
        )

        # For compatibility with report service, we'll put everything in a single section called "Requirements Match"
        detailed_sections = {
            "Job Requirements Match": {
                "match_checklist": match_checklist,
                "match_score": candidate_score,
                "weighted_score": round(weighted_score, 4),
                "things_to_keep_in_mind": things_to_keep_in_mind_text
            }
        }

        # Collect questions with filtering - prioritize technical, skip language
        language_keywords = ["fluent", "arabic", "english", "language"]
        technical_keywords = ["implement", "model", "algorithm", "framework", "tool", "pipeline",
                            "deployment", "architecture", "scale", "data", "training", "neural",
                            "ml", "ai", "sql", "python", "docker", "cloud", "aws", "azure"]

        def is_language_req(req_text):
            return any(kw in req_text.lower() for kw in language_keywords)

        def is_technical_question(q_text):
            return any(kw in q_text.lower() for kw in technical_keywords)

        technical_questions = []
        soft_skill_questions = []
        for m in match_checklist:
            if is_language_req(m["requirement"]):
                continue  # Skip language requirements
            for q in m["questions"]:
                if is_technical_question(q):
                    technical_questions.append(q)
                else:
                    soft_skill_questions.append(q)

        # Prioritize technical questions, fill with soft skill if needed
        final_questions = (technical_questions + soft_skill_questions)[:5]

        result = {
            "cv_id": cv_id,
            "file_name": file_name,
            "rank": idx,
            "score": candidate_score,
            "detailed_sections": detailed_sections,
            "summary": detailed_sections["Job Requirements Match"]["things_to_keep_in_mind"],
            "questions": final_questions
        }

        # Generate reports
        pdf_path = generate_report_pdf(result, mode, output_dir)
        result["report_pdf"] = pdf_path

        md_path = generate_detailed_markdown_report(result, mode, output_dir)
        result["detailed_report_md"] = md_path

        final_results.append(result)

    final_results.sort(key=lambda x: x["score"], reverse=True)
    for i, res in enumerate(final_results, 1):
        res["rank"] = i

    return {
        "candidates": final_results[:top_candidates],
        "mode": mode,
        "requirements": requirements_list,
        "classified_requirements": classified_requirements,
        "jd_report": jd_report_path,
        "raw_chunk_matches": len(raw_results),
        "candidate_count": len(candidate_ranking_pool)
    }
