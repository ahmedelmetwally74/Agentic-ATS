"""
AgenticATS - RAG Service
Retrieval and context formatting for LLM consumption.
"""

import logging

from db import search_similar
from embedding_service import generate_embedding

logger = logging.getLogger(__name__)


def retrieve_context(query: str, top_k: int = 5,
                     section_filter: str = None) -> list[dict]:
    """
    Embed the query and retrieve the top_k most similar CV chunks.

    Args:
        query: Natural language query string.
        top_k: Number of results to return.
        section_filter: Optional section name to restrict search (e.g., "Skills").

    Returns:
        List of result dicts with keys:
        id, file_name, section_name, chunk_index, chunk_text, similarity.
    """
    print(f"[RAG] Embedding query: '{query}'")
    query_embedding = generate_embedding(query, prefix="Query: ")

    results = search_similar(query_embedding, top_k=top_k,
                             section_filter=section_filter)
    print(f"[RAG] Retrieved {len(results)} chunks.")
    return results


def format_context_for_llm(results: list[dict]) -> str:
    """
    Format retrieved chunks into a structured string suitable for LLM prompt injection.

    Output format:
        [Source: filename.pdf | Section: Experience | Relevance: 0.87]
        chunk text here...

        [Source: filename.pdf | Section: Skills | Relevance: 0.82]
        chunk text here...
    """
    if not results:
        return "(No relevant context found.)"

    blocks = []
    for r in results:
        header = (
            f"[CV ID: {r['cv_id']} | Source: {r['file_name']} | "
            f"Section: {r['section_name']} | "
            f"Relevance: {r['similarity']:.2f}]"
        )
        blocks.append(f"{header}\n{r['chunk_text']}")

    return "\n\n".join(blocks)


def rag_query(query: str, top_k: int = 5,
              section_filter: str = None) -> dict:
    """
    Full RAG retrieval: embed query → search → format context.

    Returns:
        {
            "query": the original query,
            "context": formatted string ready for LLM prompt,
            "chunks": raw result list with metadata
        }
    """
    results = retrieve_context(query, top_k=top_k, section_filter=section_filter)
    context = format_context_for_llm(results)

    return {
        "query": query,
        "context": context,
        "chunks": results,
    }
