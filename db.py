"""
AgenticATS - Database Layer
PostgreSQL + pgvector connection management, schema init, and vector search.
"""

import os
import logging

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimensions


def get_connection():
    """Create a new PostgreSQL connection from environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "agenticats"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def init_db():
    """
    Initialize the database: create pgvector extension and cv_chunks table.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS cv_chunks (
                    id SERIAL PRIMARY KEY,
                    cv_id VARCHAR(36) NOT NULL,
                    file_name TEXT NOT NULL,
                    section_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR({VECTOR_DIM}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Index for fast cosine similarity search
            # IVFFlat requires at least some rows to train; we create it anyway
            # and PostgreSQL will handle it gracefully.
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_cv_chunks_embedding
                ON cv_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

        conn.commit()
        logger.info("Database initialized successfully.")
        print("[DB] Database initialized — table 'cv_chunks' is ready.")
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to initialize database: {e}") from e
    finally:
        conn.close()


def insert_chunk(cv_id: str, file_name: str, section_name: str, chunk_index: int,
                 chunk_text: str, embedding: list[float]):
    """Insert a single chunk with its embedding into the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cv_chunks (cv_id, file_name, section_name, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (cv_id, file_name, section_name, chunk_index, chunk_text, str(embedding)),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to insert chunk: {e}") from e
    finally:
        conn.close()


def insert_chunks_batch(chunks: list[dict]):
    """
    Batch-insert multiple chunks.
    Each dict must have: file_name, section_name, chunk_index, chunk_text, embedding.
    """
    if not chunks:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            values = [
                (
                    c["cv_id"],
                    c["file_name"],
                    c["section_name"],
                    c["chunk_index"],
                    c["chunk_text"],
                    str(c["embedding"]),
                )
                for c in chunks
            ]
            execute_values(
                cur,
                """
                INSERT INTO cv_chunks (cv_id, file_name, section_name, chunk_index, chunk_text, embedding)
                VALUES %s
                """,
                values,
                template="(%s, %s, %s, %s, %s, %s::vector)",
            )
        conn.commit()
        logger.info(f"Inserted {len(chunks)} chunks in batch.")
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to batch insert chunks: {e}") from e
    finally:
        conn.close()


def search_similar(query_embedding: list[float], top_k: int = 5,
                   section_filter: str = None) -> list[dict]:
    """
    Find the top_k most similar chunks using cosine distance.

    Args:
        query_embedding: The query vector (384-dim).
        top_k: Number of results to return.
        section_filter: Optional — restrict search to a specific section name.

    Returns:
        List of dicts with keys: id, file_name, section_name, chunk_index,
        chunk_text, similarity.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if section_filter:
                cur.execute(
                    """
                    SELECT id, cv_id, file_name, section_name, chunk_index, chunk_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM cv_chunks
                    WHERE LOWER(section_name) = LOWER(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (str(query_embedding), section_filter,
                     str(query_embedding), top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, cv_id, file_name, section_name, chunk_index, chunk_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM cv_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (str(query_embedding), str(query_embedding), top_k),
                )

            rows = cur.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "cv_id": row[1],
                    "file_name": row[2],
                    "section_name": row[3],
                    "chunk_index": row[4],
                    "chunk_text": row[5],
                    "similarity": float(row[6]),
                })
            return results
    finally:
        conn.close()


def delete_by_file(file_name: str) -> int:
    """
    Delete all chunks for a given file. Useful for re-ingestion.
    Returns the number of deleted rows.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM cv_chunks WHERE file_name = %s", (file_name,)
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()
