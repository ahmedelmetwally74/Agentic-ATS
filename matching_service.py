"""
AgenticATS - Candidate Matching Service
Rank unique candidates for a job description using stored CV chunks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from db import search_similar_pool
from embedding_service import generate_embedding


SECTION_ALIASES = {
    "work experience": "experience",
    "experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "technical skills": "skills",
    "skills": "skills",
    "core skills": "skills",
    "technical projects": "projects",
    "projects": "projects",
    "project experience": "projects",
    "education": "education",
    "courses": "courses",
    "certifications": "courses",
    "training": "courses",
    "summary": "header",
    "profile": "header",
    "header": "header",
    "full cv": "full_cv",
}

SECTION_WEIGHTS = {
    "experience": 1.35,
    "skills": 1.25,
    "projects": 1.15,
    "courses": 1.00,
    "education": 0.85,
    "header": 0.60,
    "full_cv": 1.00,
    "other": 0.95,
}

SECTION_REASON_TEXT = {
    "experience": "Strong match in work experience.",
    "skills": "Relevant technical skills were found.",
    "projects": "Relevant project work supports the match.",
    "courses": "Related courses or certifications add support.",
    "education": "Educational background is relevant.",
    "header": "The profile/header content aligns with the role.",
    "full_cv": "The overall CV content aligns with the role.",
    "other": "Additional CV content supports the match.",
}


def normalize_section_name(section_name: str) -> str:
    """Map raw CV section names into normalized matching buckets."""
    if not section_name:
        return "other"

    key = section_name.strip().lower()
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key]

    for alias, normalized in SECTION_ALIASES.items():
        if alias in key:
            return normalized

    return "other"


def _truncate(text: str, max_len: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _group_by_candidate(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["cv_id"]].append(row)
    return grouped


def _best_chunk_per_section(candidate_chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_by_section: dict[str, dict[str, Any]] = {}
    for chunk in candidate_chunks:
        normalized = normalize_section_name(chunk["section_name"])
        current = best_by_section.get(normalized)
        if current is None or chunk["similarity"] > current["similarity"]:
            enriched = dict(chunk)
            enriched["normalized_section"] = normalized
            enriched["section_weight"] = SECTION_WEIGHTS.get(normalized, SECTION_WEIGHTS["other"])
            enriched["weighted_similarity"] = enriched["similarity"] * enriched["section_weight"]
            best_by_section[normalized] = enriched
    return best_by_section


def _calculate_candidate_score(best_sections: dict[str, dict[str, Any]], top_sections_limit: int = 3) -> tuple[float, list[dict[str, Any]]]:
    if not best_sections:
        return 0.0, []

    ranked_sections = sorted(
        best_sections.values(),
        key=lambda item: (item["weighted_similarity"], item["similarity"]),
        reverse=True,
    )
    top_sections = ranked_sections[:top_sections_limit]

    total_weight = sum(item["section_weight"] for item in top_sections)
    if total_weight == 0:
        return 0.0, top_sections

    base_score = sum(item["weighted_similarity"] for item in top_sections) / total_weight
    coverage_bonus = min(0.03, max(0, len(top_sections) - 1) * 0.01)
    final_score = min(1.0, base_score + coverage_bonus)
    return final_score, top_sections


def _build_reasons(top_sections: list[dict[str, Any]]) -> list[str]:
    reasons = []
    for item in top_sections:
        normalized = item["normalized_section"]
        section_label = item["section_name"]
        base_reason = SECTION_REASON_TEXT.get(normalized, SECTION_REASON_TEXT["other"])
        reasons.append(f"{base_reason} (Section: {section_label})")
    return reasons


def match_candidates(job_description: str, top_candidates: int = 3,
                     pool_size: int = 50, section_filter: str | None = None) -> dict[str, Any]:
    """
    Rank unique candidates for a job description.

    Steps:
      1. Embed the job description.
      2. Retrieve a larger raw pool of similar chunks.
      3. Group results by cv_id.
      4. Keep the best chunk per normalized section.
      5. Score and rank unique candidates.

    Returns:
        {
            "job_description": str,
            "pool_size": int,
            "raw_chunk_matches": int,
            "candidates": [
                {
                    "rank": int,
                    "cv_id": str,
                    "file_name": str,
                    "score": float,
                    "matched_sections": [...],
                    "reasons": [...],
                    "evidence_chunks": [...],
                }
            ]
        }
    """
    query_embedding = generate_embedding(job_description)
    raw_results = search_similar_pool(
        query_embedding=query_embedding,
        pool_size=pool_size,
        section_filter=section_filter,
    )

    grouped = _group_by_candidate(raw_results)
    ranked_candidates: list[dict[str, Any]] = []

    for cv_id, candidate_chunks in grouped.items():
        best_sections = _best_chunk_per_section(candidate_chunks)
        score, top_sections = _calculate_candidate_score(best_sections)
        if not top_sections:
            continue

        ranked_candidates.append({
            "cv_id": cv_id,
            "file_name": candidate_chunks[0]["file_name"],
            "score": score,
            "matched_sections": [item["section_name"] for item in top_sections],
            "reasons": _build_reasons(top_sections),
            "evidence_chunks": [
                {
                    "section_name": item["section_name"],
                    "normalized_section": item["normalized_section"],
                    "similarity": round(item["similarity"], 4),
                    "weighted_similarity": round(item["weighted_similarity"], 4),
                    "chunk_index": item["chunk_index"],
                    "chunk_text": _truncate(item["chunk_text"]),
                }
                for item in top_sections
            ],
        })

    ranked_candidates.sort(
        key=lambda item: (item["score"], len(item["matched_sections"])),
        reverse=True,
    )

    final_candidates = []
    for idx, candidate in enumerate(ranked_candidates[:top_candidates], start=1):
        enriched = dict(candidate)
        enriched["rank"] = idx
        final_candidates.append(enriched)

    return {
        "job_description": job_description,
        "pool_size": pool_size,
        "raw_chunk_matches": len(raw_results),
        "candidate_count": len(ranked_candidates),
        "candidates": final_candidates,
    }
