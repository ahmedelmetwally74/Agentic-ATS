from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from core.embedding_service import generate_embedding

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def get_section_config() -> tuple[list[str], dict[str, float]]:
    """Load section headings and weights from sections_config.json."""
    headings: list[str] = []
    weights: dict[str, float] = {}
    config_path = PROJECT_ROOT / "sections_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                headings = config.get("section_headings", [])
                weights = config.get("section_weights", {})
        except Exception as e:
            logger.warning(f"Failed to load section config: {e}")
    return headings, weights


def normalize_section(name: str) -> str:
    """Normalize a section name to a canonical category."""
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


def build_normalized_weights(requirements_dict: dict[str, list[str]], weights_config: dict[str, float]) -> dict[str, float]:
    """Normalize configured weights for the currently parsed JD sections."""
    sections_present = list(requirements_dict.keys())
    total_w = sum(weights_config.get(section, 1.0) for section in sections_present)

    if not sections_present:
        return {}

    if total_w <= 0:
        return {section: 1.0 / len(sections_present) for section in sections_present}

    return {
        section: weights_config.get(section, 1.0) / total_w
        for section in sections_present
    }


def build_section_embeddings(requirements_dict: dict[str, list[str]]) -> dict[str, list[float]]:
    """Generate one embedding per JD section using its extracted requirements."""
    section_embeddings: dict[str, list[float]] = {}
    for section, reqs in requirements_dict.items():
        if not reqs:
            continue
        section_text = f"Section: {section}\n" + "\n".join(f"- {req}" for req in reqs)
        section_embeddings[section] = generate_embedding(section_text, prefix="Query: ")
    return section_embeddings


def group_chunks_by_normalized_section(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group CV chunks by normalized section name."""
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[normalize_section(chunk["section_name"])].append(chunk)
    return grouped


def sort_chunks_for_section(chunks: list[dict]) -> list[dict]:
    """Sort chunks in the same way the original matching flow expected them."""
    return sorted(chunks, key=lambda chunk: (chunk["section_name"], chunk["chunk_index"]))


def collect_matched_sections(all_chunks: list[dict], matched_refs: set[tuple[str, int]]) -> list[str]:
    """Return the distinct original section names that contributed evidence."""
    return sorted({
        chunk["section_name"]
        for chunk in all_chunks
        if (chunk["section_name"], chunk["chunk_index"]) in matched_refs
    })
