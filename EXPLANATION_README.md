# AgenticATS — Complete Code Explanation

> **A deep-dive reference document explaining every script, module, class, and function in the AgenticATS project.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Directory Structure](#3-directory-structure)
4. [Environment & Configuration](#4-environment--configuration)
5. [Entry Point — `main.py`](#5-entry-point--mainpy)
6. [Utility Layer — `utils/`](#6-utility-layer--utils)
   - [6.1 `document_service.py`](#61-document_servicepy)
   - [6.2 `document_utils.py`](#62-document_utilspy)
   - [6.3 `embedding_service.py`](#63-embedding_servicepy)
   - [6.4 `db.py`](#64-dbpy)
   - [6.5 `rag_service.py`](#65-rag_servicepy)
   - [6.6 `llm_service.py`](#66-llm_servicepy)
   - [6.7 `jd_processor.py`](#67-jd_processorpy)
   - [6.8 `report_base.py`](#68-report_basepy)
   - [6.9 `sections_config.json`](#69-sections_configjson)
7. [Company Module — `company/`](#7-company-module--company)
   - [7.1 `matching_service.py`](#71-matching_servicepy)
   - [7.2 `report_service.py`](#72-report_servicepy)
8. [Applicant Module — `applicant/`](#8-applicant-module--applicant)
   - [8.1 `report_service.py`](#81-report_servicepy)
9. [Debug Script — `debug_db.py`](#9-debug-script--debug_dbpy)
10. [Data Flow Walkthrough](#10-data-flow-walkthrough)
11. [Models & Infrastructure](#11-models--infrastructure)
12. [Quick Reference: All Functions](#12-quick-reference-all-functions)

---

## 1. Project Overview

**AgenticATS** (Agentic Applicant Tracking System) is a locally-hosted, AI-powered candidate matching and CV analysis platform. It combines:

- **Document Extraction** — High-fidelity text extraction from PDF and Word (.docx) files.
- **RAG Pipeline** — Retrieval-Augmented Generation using vector embeddings stored in PostgreSQL + pgvector.
- **LLM-Powered Analysis** — Uses a local Qwen 3.5-2B model (via `llama-server`) for semantic understanding, while keeping all **scoring 100% deterministic** through pure cosine similarity.
- **Dual-Mode Output** — Two analysis perspectives:
  - **Company Mode** — Rank candidates against a JD, generate interview questions.
  - **Applicant Mode** — Provide CV improvement suggestions for a specific role.

### Key Design Principle: Deterministic Scoring

All match scores are computed mathematically using embedding cosine similarity. The LLM is used **only** for:
1. Extracting requirements from job descriptions.
2. Semantically chunking CV sections.
3. Generating natural-language justifications **after** scores are fixed.

This means the same inputs always produce the same scores.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           main.py (CLI)                             │
│   Modes: extract | --embed | --search | --mode company/applicant    │
└────────┬──────────────┬───────────────┬───────────────┬─────────────┘
         │              │               │               │
         ▼              ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
│ document_    │ │ embedding_  │ │ rag_        │ │ matching_service │
│ service.py   │ │ service.py  │ │ service.py  │ │ (company/)       │
│ ────────── │ │ ─────────── │ │ ─────────── │ │ ──────────────── │
│ PDF/DOCX     │ │ Jina v5     │ │ Query →     │ │ Flat Literal     │
│ extraction   │ │ embeddings  │ │ Retrieve →  │ │ Matching +       │
│              │ │ + CV parsing│ │ Format      │ │ Weighted Scoring │
└──────┬───────┘ └──────┬──────┘ └──────┬──────┘ └────────┬─────────┘
       │                │               │                  │
       │                ▼               ▼                  ▼
       │         ┌─────────────────────────────────────────────────┐
       │         │               db.py (PostgreSQL + pgvector)     │
       │         │  cv_chunks table: id, cv_id, file_name,         │
       │         │  section_name, chunk_index, chunk_text, embedding│
       └────────►│                                                  │
                 └─────────────────────────────────────────────────┘
                                     ▲
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
       ┌──────┴──────┐    ┌─────────┴────────┐   ┌────────┴─────────┐
       │ llm_service │    │ report_base.py   │   │ report_service   │
       │ .py         │    │ (PDF generation) │   │ company/ or      │
       │ ─────────── │    │                  │   │ applicant/       │
       │ Qwen 3.5-2B │    └──────────────────┘   └──────────────────┘
       │ via llama-  │
       │ server HTTP │
       └─────────────┘
```

---

## 3. Directory Structure

```
AgenticATS/
├── main.py                        # CLI entry point
├── debug_db.py                    # Database debugging utility
├── requirements.txt               # Python dependencies
├── requirements.json              # Pre-processed JD requirements (output artifact)
├── sections.md                    # Design doc: report section headers per mode
├── .env.example                   # Environment variable template
├── .env                           # Actual env vars (git-ignored)
├── .gitignore                     # Git ignore rules
│
├── utils/                         # Core utility/service layer
│   ├── __init__.py
│   ├── document_service.py        # PDF/DOCX text extraction & redlining
│   ├── document_utils.py          # Text normalization helpers
│   ├── embedding_service.py       # Embedding generation, CV parsing, ingestion pipeline
│   ├── db.py                      # PostgreSQL + pgvector database operations
│   ├── rag_service.py             # RAG retrieval and context formatting
│   ├── llm_service.py             # LLM calls: JD decomposition, matching, synthesis
│   ├── jd_processor.py            # Deterministic JD pre-processing pipeline
│   ├── report_base.py             # Shared PDF report class & JD report generator
│   └── sections_config.json       # CV section headings & importance weights
│
├── company/                       # Company/Employer mode
│   ├── __init__.py
│   ├── matching_service.py        # Candidate ranking engine (the "brain")
│   └── report_service.py          # Company-specific PDF & Markdown reports
│
├── applicant/                     # Applicant/Client mode
│   ├── __init__.py
│   └── report_service.py          # Applicant-specific PDF & Markdown reports
│
├── models/                        # Local LLM model files (git-ignored)
│   ├── instructions.txt           # llama-server launch commands
│   ├── Qwen3.5-2B/               # Chat completions model (GGUF)
│   ├── Qwen3.5-0.8B/             # Smaller alternative model
│   └── jinav5/                    # Jina v5 embedding model (GGUF)
│
├── Resources/                     # Input CV files (.pdf, .docx)
├── Reports/                       # Generated analysis reports (PDF, Markdown)
└── llama.cpp/                     # llama.cpp build (git-ignored)
```

---

## 4. Environment & Configuration

### `.env` / `.env.example`

| Variable             | Default                                        | Purpose                                              |
| -------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| `POSTGRES_HOST`      | `localhost`                                    | PostgreSQL server hostname                           |
| `POSTGRES_PORT`      | `5432`                                         | PostgreSQL server port                               |
| `POSTGRES_DB`        | `agenticats`                                   | Database name                                        |
| `POSTGRES_USER`      | `postgres`                                     | Database username                                    |
| `POSTGRES_PASSWORD`  | *(required)*                                   | Database password                                    |
| `EMBEDDING_API_URL`  | `http://127.0.0.1:7999/v1/embeddings`          | Jina v5 embedding server endpoint                    |
| `LLM_API_URL`        | `http://localhost:8000/v1/chat/completions`     | Qwen 3.5-2B chat completions endpoint                |

### `requirements.txt` — Python Dependencies

| Package           | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| `PyMuPDF`         | High-fidelity PDF text extraction (fitz)           |
| `python-docx`     | Word document (.docx) reading and writing          |
| `PyPDF2`          | PDF text presence detection                        |
| `openpyxl`        | Excel support (dependency)                         |
| `lxml`            | XML processing for DOCX internals                  |
| `python-redlines` | Track-changes / redline document generation        |
| `requests`        | HTTP client for LLM and embedding API calls        |
| `fpdf2`           | PDF report generation                              |
| `psycopg2-binary` | PostgreSQL driver                                  |
| `python-dotenv`   | `.env` file loader                                 |

### `sections_config.json`

Contains two key objects:

- **`section_headings`** — A list of 40+ recognized CV section header strings (e.g., `"Summary"`, `"Professional Experience"`, `"Technical Skills"`, etc.) used by the CV section parser to split extracted text.
- **`section_weights`** — Importance weights per section category used for weighted scoring:
  - Experience: 4, Skills: 3, Education: 2, Projects: 2, Certifications: 1, Summary: 1, Languages: 0.5

---

## 5. Entry Point — `main.py`

The CLI application that orchestrates all operations.

### Functions

#### `_clean_text(text: str) -> str`
- **Purpose**: Sanitize text for terminal output by removing all control characters and collapsing whitespace.
- **How**: Uses `str.split()` + `str.join()` to normalize whitespace, then filters non-printable characters via `str.isprintable()`.
- **Used by**: All terminal print statements for match results, reasons, and questions.

#### `extract_text(file_path: str) -> str`
- **Purpose**: Unified text extraction dispatcher for PDF and DOCX files.
- **How**:
  1. Validates file existence.
  2. Checks file extension (`.pdf` or `.docx`).
  3. For PDFs: calls `pdf_has_text()` to determine if selectable text exists, then calls `extract_text_from_pdf_sync()`.
  4. For DOCX: calls `extract_text_from_word()`.
  5. Normalizes the result with `normalize_text_basic()`.
- **Returns**: Cleaned extracted text string.
- **Raises**: `FileNotFoundError`, `ValueError` for unsupported formats.

#### `main()`
- **Purpose**: Argument parser and command dispatcher — the main entry point of the application.
- **CLI Arguments**:

| Argument              | Type    | Description                                                   |
| --------------------- | ------- | ------------------------------------------------------------- |
| `file_path`           | pos.    | Path to a PDF/DOCX for text extraction                        |
| `-o` / `--output`     | str     | Save extracted text to a .docx file                           |
| `-c` / `--clean`      | flag    | Apply additional text cleaning (list patterns)                |
| `--embed PATH`        | str     | Embed a single CV file or entire folder into PostgreSQL       |
| `--search QUERY`      | str     | RAG search across stored CV embeddings                        |
| `--section`           | str     | Filter search/match to a specific CV section                  |
| `--top-k`             | int     | Number of search results (default: 5)                         |
| `--match-job`         | str     | Inline job description text for candidate matching            |
| `--jd-file PATH`      | str     | Read job description from a text file                         |
| `--requirements-file` | str     | Load pre-processed requirements JSON from `--process-jd`      |
| `--mode`              | choice  | `company` (ranking) or `applicant` (CV improvement)           |
| `--process-jd`        | flag    | Pre-process a JD file into deterministic requirements JSON    |
| `--input`             | str     | Input file for `--process-jd`                                 |
| `--top-candidates`    | int     | Number of top candidates to return (default: 3)               |
| `--pool-size`         | int     | Raw matching chunk pool size (default: 50)                    |
| `--output-dir`        | str     | Directory for PDF reports (default: `./Reports`)              |
| `--init-db`           | flag    | Initialize the PostgreSQL database                            |

- **Execution Flow** (priority order):
  1. `--process-jd` → calls `process_jd_file()` and `save_requirements()`.
  2. `--init-db` → calls `init_db()`.
  3. `--match-job` / `--jd-file` / `--requirements-file` → calls `match_candidates()` from `company/matching_service.py`.
  4. `--search` → calls `rag_query()` from `rag_service.py`.
  5. `--embed` → calls `ingest_cv()` or `ingest_cv_folder()` from `embedding_service.py`.
  6. Default: extracts text and prints or saves to DOCX.

---

## 6. Utility Layer — `utils/`

### 6.1 `document_service.py`

Core document extraction, generation, and redlining engine.

#### `extract_text_from_pdf(file_path: str) -> str` *(async)*
- **Purpose**: Async wrapper for PDF text extraction.
- **How**: Delegates to `extract_text_from_pdf_sync()`.

#### `_extract_page_text_by_y(page) -> str`
- **Purpose**: Extract text from a single PyMuPDF page while preserving the visual layout.
- **How**:
  1. Calls `page.get_text("dict")` to get all text spans with bounding boxes.
  2. Sorts spans by Y-coordinate (top-to-bottom).
  3. Groups spans into horizontal lines using **vertical overlap detection** (≥50% of span height).
  4. Within each line, sorts spans by X-coordinate (left-to-right) to preserve reading order.
  5. Joins spans with double-space separator.
- **Why**: Standard `get_text()` fails on CVs where dates are right-aligned — they appear on separate lines. This algorithm visually reconstructs the layout so `"Software Engineer"` and `"2022-Present"` stay on the same line.

#### `extract_text_from_pdf_sync(file_path: str) -> str`
- **Purpose**: Synchronous PDF text extraction using PyMuPDF.
- **How**: Opens the PDF, iterates over all pages, calls `_extract_page_text_by_y()` for each.

#### `extract_text_from_pdf_stream(file) -> str` *(async)*
- **Purpose**: Extract text from an in-memory PDF byte stream (e.g., from an upload).
- **How**: Reads bytes with `await file.read()`, opens with `fitz.open(stream=...)`.

#### `pdf_has_text(file_path: str) -> bool`
- **Purpose**: Detect whether a PDF contains selectable/searchable text (vs. scanned images).
- **How**:
  - Single-page: checks if `extract_text()` returns non-empty content.
  - Multi-page: checks if **two consecutive pages** both contain text (reduces false positives from header-only pages).
- **Used by**: `extract_text()` in `main.py` to warn users about scanned PDFs.

#### `extract_text_from_word(file_path: str) -> str`
- **Purpose**: Extract text from a Word (.docx) file, preserving numbered list formatting.
- **How**:
  1. Opens with `python-docx`.
  2. For each paragraph, checks for `numPr` (numbering properties) in the paragraph's XML.
  3. If numbered: generates labels like `"1. "`, `"2. "` (level 0) or `"a. "`, `"b. "` (level 1+).
  4. Tracks counters per `(numId, ilvl)` pair to maintain correct numbering.

#### `save_text_to_docx(text: str, output_path: str = None) -> str`
- **Purpose**: Save plain text to a new Word document.
- **How**: Creates a `Document()`, adds each non-empty line as a paragraph, saves to path.

#### `apply_corrections_for_clauses(file_path, differences, suggestions, output_path) -> str`
- **Purpose**: Apply clause-level corrections to a Word document (NDA redlining feature).
- **How**:
  1. Filters suggestions where `Change == True`.
  2. For each correction, finds the target paragraphs using start/end text hints (first 6 words, last 6 words).
  3. Uses a sliding window if the clause spans multiple paragraphs.
  4. Replaces the matched text block with the correction text.
  5. Removes absorbed paragraphs from the document.
- **Note**: This is a legacy feature from the NDA analysis origin of the project.

#### `remove_columns_from_docx(docx_path: str) -> None`
- **Purpose**: Remove column layout from a Word document's section properties.
- **How**: Unzips the DOCX, parses `word/document.xml`, removes `<w:cols>` elements, re-zips.

#### `generate_redline(author_tag, original_path, modified_path, output_path) -> str`
- **Purpose**: Generate a track-changes (redline) comparison between two Word documents.
- **How**: Uses `XmlPowerToolsEngine` from `python-redlines` to produce a redlined DOCX, then cleans columns.

---

### 6.2 `document_utils.py`

Text normalization, cleanup, and helper functions.

#### `CATEGORY_GROUPS` *(constant)*
- **Purpose**: Maps canonical section names to their common aliases.
- **Example**: `"Experience"` → `["experience", "employment", "work history", "professional experience", ...]`
- **Used by**: `normalize_section()` to consolidate varied section headings.

#### `normalize_section(name: str) -> str`
- **Purpose**: Map a raw section heading (e.g., `"Professional Experience"`) to its canonical category (`"Experience"`).
- **How**: Case-insensitive scan of `CATEGORY_GROUPS` aliases. Returns the original name if no match.

#### `clean_extracted_text(text: str) -> str`
- **Purpose**: Normalize whitespace and detect common list patterns.
- **How**:
  1. Collapses horizontal whitespace.
  2. Normalizes vertical whitespace.
  3. Detects patterns like `a.`, `(a)`, `(i)`, `(ii)` and inserts newlines before them.

#### `normalize_text_basic(text: str) -> str`
- **Purpose**: Conservative text normalizer for post-extraction cleanup.
- **How**:
  1. Removes invisible characters: soft hyphen (`\u00AD`), zero-width space (`\u200B`).
  2. Converts non-breaking spaces to regular spaces.
  3. Joins hyphenated words split across line breaks (e.g., `"Gewähr-\nleistung"` → `"Gewährleistung"`).
  4. Trims trailing whitespace from lines.
  5. Collapses 3+ consecutive blank lines to 2.

#### `normalize_quotes(text: str) -> str`
- **Purpose**: Convert smart/curly quotes to ASCII equivalents.
- **How**: Character-by-character replacement of Unicode quote characters + regex cleanup of escaped Unicode sequences.

#### `remove_redundant_title_from_clause(title: str, clause: str) -> str`
- **Purpose**: Strip the clause title from the beginning of a clause body.
- **How**: Builds a regex that matches the title (with optional numbering and punctuation) at the start.

#### `title_starts_clause(title: str, clause: str) -> bool`
- **Purpose**: Check whether a clause body starts with its title text.
- **How**: Same regex as above, but returns a boolean.

#### `normalize(text: str) -> str`
- **Purpose**: Lowercase and collapse all whitespace to single spaces.
- **Used by**: Clause matching in `document_service.py`.

#### `flatten(s: str) -> str`
- **Purpose**: Collapse all whitespace to single spaces (case-preserving).
- **Used by**: Clause literal matching in `document_service.py`.

---

### 6.3 `embedding_service.py`

Handles embedding generation, CV section parsing, sub-chunking, and the full ingestion pipeline.

#### `EMBEDDING_API_URL` *(constant)*
- **Value**: Read from `EMBEDDING_API_URL` env var (default: `http://127.0.0.1:7999/v1/embeddings`).
- **Purpose**: Endpoint for the Jina v5 embedding model running on llama-server.

#### `generate_embedding(text: str, prefix: str = "Document: ") -> list[float]`
- **Purpose**: Generate a single 1024-dimensional embedding vector.
- **How**: Prepends the prefix to the text, sends a POST to the embedding API, extracts `data[0]["embedding"]`.
- **Prefixes**:
  - `"Document: "` — Used for CV chunks (storage).
  - `"Query: "` — Used for JD requirements and search queries (retrieval).
- **Why prefixes**: Jina v5 uses asymmetric embedding — different prefixes optimize for storage vs. retrieval.

#### `generate_embeddings(texts: list[str], prefix: str = "Document: ") -> list[list[float]]`
- **Purpose**: Batch-generate embeddings for multiple texts in a single API call.
- **How**: Same as above but sends all texts in one `input` array, then sorts results by `index` to guarantee order.

#### `_load_section_headings() -> list[str]`
- **Purpose**: Load the list of recognized CV section headings from `sections_config.json`.
- **Raises**: `FileNotFoundError` if config missing, `ValueError` if `section_headings` list is empty.

#### `_build_section_pattern(headings: list[str]) -> re.Pattern`
- **Purpose**: Compile a regex pattern that matches any configured heading at the start of a line.
- **How**: 
  1. Escapes all heading strings for regex safety.
  2. Sorts by length descending (so `"Professional Experience"` matches before `"Experience"`).
  3. Builds a pattern that handles: optional numbering (`1.` or `2)`), optional bullets (`•`, `-`, `*`), optional separators (`---`, `===`), optional trailing colons/dashes.
- **Example match**: `"  2. Professional Experience:"`, `"• Skills"`, `"--- Projects ---"`

#### `parse_cv_sections(text: str) -> list[dict]`
- **Purpose**: Split extracted CV text into semantic sections.
- **How**:
  1. Loads headings from config, builds the regex.
  2. Finds all heading matches in the text.
  3. Text before the first heading becomes `"Header"` (name, contact info).
  4. Text between each heading becomes its respective section.
- **Returns**: List of `{"section_name": "Experience", "text": "..."}` dicts.
- **Fallback**: If no headings detected, returns the full text as a single `"Full CV"` section.

#### `sub_chunk(text: str, max_chars: int = 1500) -> list[str]`
- **Purpose**: Split a long section into smaller chunks at sentence boundaries.
- **How**: Targets ~512 tokens (~1500 chars). Splits at `.!?` followed by whitespace. Greedily fills chunks without exceeding `max_chars`.
- **Used as fallback**: When LLM semantic chunking fails.

#### `ingest_cv(file_path: str, replace_existing: bool = True) -> tuple[str, int]`
- **Purpose**: Full RAG ingestion pipeline for a single CV file.
- **Pipeline**:
  1. **Extract**: Calls `extract_text()` from `main.py`.
  2. **Parse**: Calls `parse_cv_sections()` to split into sections.
  3. **Chunk**: For "heavy" sections (Experience, Projects, Education, Certifications, Courses), uses LLM `semantic_chunk_section()`. For simple sections, uses `sub_chunk()`.
  4. **Embed**: Batch-generates embeddings with `"Document: "` prefix.
  5. **Store**: Optionally deletes existing chunks for this file, then batch-inserts to PostgreSQL.
- **Returns**: `(cv_id, chunk_count)` — the UUID assigned to this CV and number of chunks stored.

#### `ingest_cv_folder(folder_path: str, replace_existing: bool = True) -> list[tuple]`
- **Purpose**: Ingest all `.pdf` and `.docx` files in a folder.
- **How**: Lists directory, filters by extension, calls `ingest_cv()` for each file. Catches per-file errors to continue processing.
- **Returns**: List of `(file_name, cv_id, chunk_count)` tuples.

---

### 6.4 `db.py`

PostgreSQL + pgvector database layer.

#### `VECTOR_DIM` *(constant)*
- **Value**: `1024` — Matches the Jina v5 embedding output dimensions.

#### `get_connection() -> psycopg2.connection`
- **Purpose**: Create a new PostgreSQL connection from environment variables.
- **Used by**: Every database function (connections are not pooled — each operation opens/closes its own).

#### `init_db()`
- **Purpose**: Initialize the database schema.
- **How**: Creates the `vector` extension and the `cv_chunks` table with columns:
  - `id` (SERIAL PK), `cv_id` (VARCHAR 36), `file_name` (TEXT), `section_name` (TEXT), `chunk_index` (INT), `chunk_text` (TEXT), `embedding` (VECTOR(1024)), `created_at` (TIMESTAMP).
- **Idempotent**: Uses `IF NOT EXISTS` — safe to call multiple times.

#### `insert_chunk(cv_id, file_name, section_name, chunk_index, chunk_text, embedding)`
- **Purpose**: Insert a single chunk with its embedding vector.
- **How**: Standard INSERT with `%s::vector` cast for the embedding.

#### `insert_chunks_batch(chunks: list[dict])`
- **Purpose**: Efficiently batch-insert multiple chunks using `psycopg2.extras.execute_values`.
- **How**: Constructs a values list and uses the `%s::vector` template for embedding casting.
- **Used by**: `ingest_cv()` after all embeddings are generated.

#### `_run_similarity_search(query_embedding, limit, section_filter=None) -> list[dict]`
- **Purpose**: Shared low-level cosine similarity search.
- **How**: Uses pgvector's `<=>` operator for cosine distance. Computes similarity as `1 - distance`.
- **Returns**: List of result dicts with: `id`, `cv_id`, `file_name`, `section_name`, `chunk_index`, `chunk_text`, `similarity`.
- **Section filter**: Optional `WHERE LOWER(section_name) = LOWER(%s)` clause.

#### `search_similar(query_embedding, top_k=5, section_filter=None) -> list[dict]`
- **Purpose**: Public interface for RAG search — returns top-k most similar chunks.
- **Delegates to**: `_run_similarity_search()`.

#### `search_similar_pool(query_embedding, pool_size=50, section_filter=None) -> list[dict]`
- **Purpose**: Retrieve a larger raw pool for candidate ranking (may contain multiple chunks per candidate).
- **Delegates to**: `_run_similarity_search()` with larger limit.

#### `delete_by_file(file_name: str) -> int`
- **Purpose**: Delete all chunks for a given file name. Useful for re-ingestion.
- **Returns**: Number of deleted rows.

#### `search_best_chunk_for_cv(cv_id, query_embedding) -> dict | None`
- **Purpose**: Find the single most similar chunk for a specific candidate.
- **How**: Adds `WHERE cv_id = %s` to the similarity search, `LIMIT 1`.
- **Used in**: Map phase of the matching process.

#### `get_all_chunks_for_cv(cv_id: str) -> list[dict]`
- **Purpose**: Retrieve **all** chunks for a specific CV to provide full context for deep analysis.
- **Returns**: List of dicts with `chunk_index`, `chunk_text`, `section_name`, `embedding`, `similarity` (default 1.0).
- **Used by**: `matching_service.py` during deterministic per-requirement scoring.

---

### 6.5 `rag_service.py`

RAG (Retrieval-Augmented Generation) retrieval and context formatting.

#### `retrieve_context(query, top_k=5, section_filter=None) -> list[dict]`
- **Purpose**: Embed a natural-language query and retrieve matching chunks.
- **How**: Generates query embedding with `"Query: "` prefix, then calls `search_similar()`.

#### `format_context_for_llm(results: list[dict]) -> str`
- **Purpose**: Format retrieved chunks into a structured string for LLM prompt injection.
- **Output format**:
  ```
  [CV ID: xxx | Source: filename.pdf | Section: Experience | Relevance: 0.87]
  chunk text here...
  ```

#### `rag_query(query, top_k=5, section_filter=None) -> dict`
- **Purpose**: Full RAG pipeline: embed → search → format.
- **Returns**: `{"query": str, "context": str, "chunks": list}`.
- **Used by**: `--search` CLI command.

---

### 6.6 `llm_service.py`

All LLM interactions via the local Qwen 3.5-2B chat completions API.

#### `LLM_API_URL` *(constant)*
- **Value**: Read from `LLM_API_URL` env var (default: `http://localhost:8000/v1/chat/completions`).

#### `call_llm(messages, temperature=0.0, max_tokens=2048) -> str`
- **Purpose**: Send a chat completion request to the Qwen 3.5-2B model.
- **How**:
  1. Prints all request messages for transparency (truncates messages >2000 chars).
  2. Sends POST with: `temperature`, `max_tokens`, `seed=42`, `top_p=1.0`, `top_k=1`, `repeat_penalty=1.0`.
  3. Prints the raw response.
  4. Returns the content string from `choices[0].message.content`.
- **Determinism**: `temperature=0.0`, `seed=42`, `top_k=1` — ensures reproducible outputs.
- **Timeout**: 180 seconds.

#### `clean_llm_json(raw: str) -> any`
- **Purpose**: Extract and repair JSON from potentially noisy LLM output.
- **How**:
  1. Finds the first `{` or `[` in the raw text.
  2. Finds the matching last `}` or `]`.
  3. Cleans trailing commas (`,]` → `]`, `,}` → `}`).
  4. Parses with `json.loads()`.
  5. If result is a dict, normalizes all keys to lowercase.
- **Fallback**: Returns `{}` if parsing fails.

#### `decompose_job_description(jd_text: str) -> list[str]`
- **Purpose**: Extract every distinct requirement from a JD as exact, unaltered substrings.
- **Rules enforced via prompt**:
  - No paraphrasing — every requirement must be a literal substring of the JD.
  - No section headers — only the requirement text.
  - Granularity — each point is standalone.
  - Exhaustive — captures technical skills, soft skills, experience, education.
- **Returns**: List of requirement strings.
- **Handles**: Both flat lists and mistakenly categorized dicts (flattens them).

#### `justify_match(requirement, cv_evidence, status, score) -> dict`
- **Purpose**: Generate a natural-language justification for a pre-calculated embedding match.
- **Key design**: The score and status are **already determined** mathematically. The LLM only provides reasoning.
- **Prompt selection**: Three different system prompts based on requirement type:
  - **Language requirements** (Arabic, English, fluent) → No interview questions needed.
  - **Technical requirements** (SQL, ML, Python, etc.) → Questions probe implementation details.
  - **Other** → Behavioral interview questions.
- **Returns**: `{"reason": str, "questions": list}`.

#### `generate_gap_analysis(technical_gaps, partial_gaps, critical_requirements) -> str`
- **Purpose**: Generate a detailed gap analysis with specific risks and interview probes.
- **How**: Sends gap details to the LLM with instructions to provide 2-3 sentences per gap covering: meaning for the role, criticality, and what to probe.
- **Fallback**: Returns a simple count-based summary if LLM call fails.

#### `expand_jd_requirements(jd_text, requirements=None) -> dict`
- **Purpose**: Expand short JD requirements into detailed descriptions for better embedding retrieval.
- **How**: Processes one section at a time, asking the LLM to write a detailed paragraph of the ideal candidate profile.

#### `semantic_chunk_section(section_name: str, text: str) -> list[str]`
- **Purpose**: Intelligently split a CV section into isolated, atomic items using the LLM.
- **Why**: A standard `sub_chunk()` might split in the middle of a job role. This function understands that "Software Engineer at Google — 2020-2023 — built X, shipped Y" is a single atomic unit.
- **Prompt**: Provides detailed examples and rules:
  - Each job/project/item must be its own string.
  - Combine title, company, dates, and all bullets into one paragraph.
  - Don't summarize or alter text.
  - Detect patterns for item boundaries.
  - Ignore page footers.
- **Fallback**: Uses `sub_chunk()` if LLM parsing fails.

#### `analyze_section_match(section_name, jd_requirements, cv_evidence, mode) -> dict`
- **Purpose**: Perform granular holistic comparison between JD requirements and CV chunks.
- **Employer mode prompt rules**:
  - Close Match: ALL aspects met. Engineering degrees satisfy general "Engineering" requirement.
  - Date Math: Excludes education/internships from experience calculations. Uses exact CV dates.
  - Partial Match: Evidence incomplete or tenure slightly below.
  - No Match: Zero evidence or tenure significantly below.
- **Applicant mode**: More generous with semantic equivalents.
- **Returns**: Dict with `why_fits`, `match_checklist`, `things_to_keep_in_mind`, `questions` (employer) or `comparison`, `match_checklist`, `improvement_suggestions` (applicant).

#### `classify_requirements(requirements: list[str]) -> list[dict]`
- **Purpose**: Classify each JD requirement into importance tiers and extract key technical terms.
- **Tiers**:
  - `critical` — Non-negotiable skills, required degrees/certs, specific technologies.
  - `important` — Strongly preferred qualifications.
  - `nice_to_have` — Bonus qualifications.
- **Key terms**: 2-5 essential technical terms per requirement for exact-match boosting.
- **Returns**: List of `{"requirement": str, "tier": str, "key_terms": [str]}`.

#### `synthesize_candidate_analysis(section_analyses, mode) -> dict`
- **Purpose**: Combine multiple section-level analyses into a final summary report.
- **Employer mode output**: `ranking_overview_summary`, `why_fits`, `things_to_keep_in_mind`, `questions` (top 5 most impactful).
- **Applicant mode output**: `general_comparison_summary`, `improvement_suggestions`.

---

### 6.7 `jd_processor.py`

Deterministic Job Description pre-processing pipeline. Allows JD processing to be done once and reused.

#### `process_jd_text(jd_text: str) -> dict`
- **Purpose**: Process raw JD text into structured requirements.
- **How**: Calls `decompose_job_description()` then `classify_requirements()`.
- **Returns**: `{"requirements": [str], "classified_requirements": [dict]}`.

#### `process_jd_file(input_path: str) -> dict`
- **Purpose**: Read a JD text file and process it.
- **Delegates to**: `process_jd_text()`.

#### `save_requirements(result: dict, output_path: str)`
- **Purpose**: Save processed requirements to a JSON file.
- **Output**: Pretty-printed JSON with `indent=2`.

#### `load_requirements(input_path: str) -> dict`
- **Purpose**: Load pre-processed requirements from a JSON file.
- **Used by**: `main.py` when `--requirements-file` is provided.

#### `main()`
- **Purpose**: Standalone CLI entry point for JD processing.
- **Usage**: `python -m utils.jd_processor --input job.txt --output reqs.json`

---

### 6.8 `report_base.py`

Shared PDF report generation utilities.

#### `_safe_text(text: str) -> str`
- **Purpose**: Ensure text only contains ISO-8859-1 compatible characters for FPDF.
- **How**: Replaces common Unicode characters (bullets, dashes, smart quotes, NBSP), then encodes to `latin-1` with `ignore` error handling.

#### `class AnalysisReport(FPDF)`
- **Purpose**: Custom PDF class with AgenticATS branding.

##### `__init__(self, title: str, subtitle: str = "")`
- Stores report title and subtitle.

##### `header(self)`
- Renders centered title (Helvetica Bold 14), subtitle with timestamp (Helvetica Italic 9), and a horizontal line separator.

##### `footer(self)`
- Renders centered page number (`Page X/{nb}`).

##### `add_section_title(self, title, fill_color=(230, 235, 245))`
- Renders a section heading with light blue background fill.

##### `add_key_value(self, key, value)`
- Renders a key-value pair with the key in bold and the value in regular weight.

##### `add_bullet_list(self, items)`
- Renders a bulleted list with `-` markers.

#### `generate_jd_analysis_report(requirements, full_text, output_dir) -> str`
- **Purpose**: Generate a standalone PDF report for the Job Description decomposition.
- **Contents**:
  - Page 1: All extracted literal requirements as a bullet list.
  - Page 2: Original full JD text for reference.
- **Output**: `{output_dir}/job_description_analysis.pdf`.

---

## 7. Company Module — `company/`

### 7.1 `matching_service.py`

**The "brain" of the system** — implements the full candidate ranking engine.

#### `_get_section_config() -> tuple[list[str], dict[str, float]]`
- **Purpose**: Load section headings and weights from `sections_config.json`.
- **Returns**: `(headings_list, weights_dict)`.

#### `match_candidates(job_description, top_candidates, pool_size, section_filter, mode, output_dir, preprocessed_requirements) -> dict`
- **Purpose**: The master orchestration function for candidate ranking.
- **Full Pipeline (6 phases)**:

**Phase 1 — JD Decomposition**:
1. If `preprocessed_requirements` provided, use those directly.
2. Otherwise, call `decompose_job_description()` to extract literal requirements.
3. Call `classify_requirements()` to tier each requirement as critical/important/nice_to_have.
4. Separate critical requirements for pool enhancement.
5. Generate standalone JD analysis PDF report.

**Phase 2 — Global Pool Search**:
1. Embed the full JD text with `"Query: "` prefix.
2. Call `search_similar_pool()` to retrieve `pool_size` matching chunks.
3. Group results by `cv_id` and compute average similarity per candidate.

**Phase 2b — Critical Requirement Boost**:
1. Embed all critical requirements.
2. For each candidate in the pool, check every critical requirement against every CV chunk.
3. Count how many critical requirements have at least one chunk with similarity ≥ 0.6.
4. Boost the candidate's broad score by `0.05 × critical_pass_count`.

**Phase 3 — Rank & Select Top Candidates**:
1. Sort candidates by boosted broad score.
2. Select top `max(top_candidates, 5)` for deep analysis.

**Phase 4 — Deterministic Deep Analysis** (per candidate):
1. Retrieve **all** chunks for the candidate via `get_all_chunks_for_cv()`.
2. Pre-generate embeddings for all JD requirements.
3. For each requirement:
   - Compute cosine similarity against every CV chunk.
   - Apply **exact key term boost** (+0.15, capped at 1.0) if classified key terms appear in the chunk.
   - Determine status: ≥0.85 → "Close Match", ≥0.60 → "Partial Match", else → "No Match".
   - Apply **tier weighting**: critical=4.0×, important=1.0×, nice_to_have=0.3×.
   - Call `justify_match()` for natural-language reasoning.
4. Calculate weighted score.
5. Apply **critical requirement bonus/penalty**:
   - All critical ≥ 0.8 → 12% bonus.
   - Any critical < 0.5 → 30% penalty.
6. Generate gap analysis via LLM.
7. Filter and prioritize interview questions (technical over soft-skill, skip language).

**Phase 5 — Report Generation** (per candidate):
- `generate_report_pdf()` → Summary PDF with score, gaps, and interview questions.
- `generate_detailed_markdown_report()` → Full Markdown with per-requirement breakdown.

**Phase 6 — Final Ranking**:
- Sort all analyzed candidates by final score descending.
- Re-assign rank numbers.

- **Returns**:
  ```python
  {
      "candidates": [list of candidate result dicts],
      "mode": str,
      "requirements": [list of requirement strings],
      "classified_requirements": [list of classification dicts],
      "jd_report": str (path),
      "raw_chunk_matches": int,
      "candidate_count": int
  }
  ```

#### Helper functions within `match_candidates()`:

##### `is_soft_skill(req_text) -> bool`
- Checks if a requirement is a soft skill (fast-paced, detail-oriented, etc.) to exclude from technical gap analysis.

##### `is_language_req(req_text) -> bool`
- Checks if a requirement is about language proficiency to skip generating questions for it.

##### `is_technical_question(q_text) -> bool`
- Checks if an interview question is technically oriented for prioritization.

---

### 7.2 `report_service.py`

Company-mode specific report generation.

#### `generate_report_pdf(candidate_result, mode, output_dir) -> str`
- **Purpose**: Generate a PDF with interview questions and gap analysis.
- **Sections**:
  1. **Ranking & Match Overview** — Score and summary.
  2. **Things to keep in mind** — Considerations per section (skips if "no major" or "none").
  3. **Questions to ask the candidate** — Bulleted interview questions.
- **Output**: `{output_dir}/{SafeCandidateName}_analysis.pdf`.

#### `generate_jd_analysis_report(requirements, full_text, output_dir) -> str`
- Delegates to `report_base.generate_jd_analysis_report()`.

#### `generate_detailed_markdown_report(candidate_result, mode, output_dir) -> str`
- **Purpose**: Generate a comprehensive Markdown report with full deterministic analysis.
- **Sections**:
  1. **Deterministic Match Analysis** header with mode and total score.
  2. **Things to Keep in Mind** — Narrative gap analysis.
  3. **Requirements Match Checklist** — Table with Status, Score, Requirement, Reason columns.
  4. **Detailed Evidence & Interview Questions** — Per-requirement breakdown with:
     - Status and score
     - Best matching CV evidence (blockquoted)
     - Suggested interview questions
- **Output**: `{output_dir}/{SafeCandidateName}_detailed_analysis.md`.

---

## 8. Applicant Module — `applicant/`

### 8.1 `report_service.py`

Applicant-mode specific report generation.

#### `generate_report_pdf(candidate_result, mode, output_dir) -> str`
- **Purpose**: Generate a PDF with CV improvement suggestions.
- **Sections**:
  1. **CV vs Job Description: Section Comparison** — For each JD section:
     - JD requirements listed as bullets.
     - Analysis comparison text.
     - Section-specific improvement tips (in muted red).
  2. **General improvement suggestions** — Bulleted list.
- **Output**: `{output_dir}/{SafeCandidateName}_analysis.pdf`.

#### `generate_jd_analysis_report(requirements, full_text, output_dir) -> str`
- Delegates to `report_base.generate_jd_analysis_report()`.

#### `generate_detailed_markdown_report(candidate_result, mode, output_dir) -> str`
- **Purpose**: Generate a Markdown report for applicant CV improvement.
- **Sections**:
  1. **Section Comparisons** — Per-section: JD requirements, analysis, tips.
  2. **General Improvement Suggestions** — Bulleted list.
- **Output**: `{output_dir}/{SafeCandidateName}_detailed_analysis.md`.

---

## 9. Debug Script — `debug_db.py`

#### `debug_db()`
- **Purpose**: Quick database inspection utility.
- **How**: Connects directly to PostgreSQL, queries all chunks for a hardcoded `cv_id`, and prints the first 50 characters of each chunk with its section name.
- **Usage**: `python debug_db.py`
- **Note**: The `cv_id` is hardcoded — meant for developer debugging only.

---

## 10. Data Flow Walkthrough

### Flow 1: CV Ingestion (`--embed`)

```
Input CV file (.pdf/.docx)
    │
    ▼
extract_text()                          # main.py
    │ Uses document_service.py
    ▼
parse_cv_sections()                     # embedding_service.py
    │ Uses sections_config.json regex
    ▼
┌─────────────────────────────┐
│ For each section:           │
│   Heavy? → semantic_chunk() │  ← LLM call (llm_service.py)
│   Light? → sub_chunk()      │  ← Naive sentence splitting
└─────────────┬───────────────┘
              ▼
generate_embeddings()                   # embedding_service.py
    │ Jina v5 via HTTP → 1024-dim vectors
    ▼
insert_chunks_batch()                   # db.py
    │ PostgreSQL + pgvector
    ▼
Stored: cv_id, file_name, section_name,
        chunk_index, chunk_text, embedding
```

### Flow 2: Candidate Matching (`--mode company --jd-file`)

```
Input JD text file
    │
    ▼
decompose_job_description()             # llm_service.py → LLM
    │ Extracts literal requirement strings
    ▼
classify_requirements()                 # llm_service.py → LLM
    │ Assigns tiers: critical/important/nice_to_have
    ▼
generate_embedding(jd_text)             # embedding_service.py
    │
    ▼
search_similar_pool()                   # db.py → pgvector cosine search
    │ Returns pool_size raw matches
    ▼
Group by cv_id → Avg similarity
    │
    ▼
Critical requirement boost check        # Per candidate, per critical req
    │ cosine_similarity via sklearn
    ▼
Sort → Select top N candidates
    │
    ▼
┌───────────────────────────────────────────────────┐
│ For each top candidate:                           │
│   get_all_chunks_for_cv()     # Full CV context   │
│   For each JD requirement:                        │
│     cosine_similarity vs ALL chunks               │
│     + Key term exact match boost                  │
│     → Status (Close/Partial/No)                   │
│     → justify_match() via LLM                     │
│   Weighted score with tier weights                │
│   Critical bonus/penalty                          │
│   generate_gap_analysis() via LLM                 │
│   Filter & rank interview questions               │
└───────────────────────┬───────────────────────────┘
                        ▼
generate_report_pdf()                    # company/report_service.py
generate_detailed_markdown_report()      # company/report_service.py
    │
    ▼
Output: PDF + Markdown in Reports/
```

### Flow 3: RAG Search (`--search`)

```
Search query string
    │
    ▼
generate_embedding(query, "Query: ")    # embedding_service.py
    │
    ▼
search_similar(top_k=5)                 # db.py → pgvector
    │
    ▼
format_context_for_llm()                # rag_service.py
    │
    ▼
Print formatted results to terminal
```

---

## 11. Models & Infrastructure

### Local LLM Servers (via `llama.cpp`)

Two separate `llama-server` instances must be running:

**1. Chat Completions — Qwen 3.5-2B** (Port 8000):
```bash
./llama.cpp/build/bin/Release/llama-server \
  -m models/Qwen3.5-2B/Qwen3.5-2B-UD-Q4_K_XL.gguf \
  --mmproj models/Qwen3.5-2B/mmproj-BF16.gguf \
  -c 131072 \
  -ctk q4_0 -ctv q4_0 \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --port 8000 --host 0.0.0.0
```
- **Model**: Qwen 3.5-2B with Q4_K_XL quantization.
- **Context**: 131072 tokens with Q4_0 KV-cache quantization.
- **Thinking disabled**: Prevents chain-of-thought leaking into responses.

**2. Embeddings — Jina v5** (Port 7999):
```bash
./llama.cpp/build/bin/Release/llama-server \
  -m models/jinav5/v5-small-retrieval-Q8_0.gguf \
  --n-gpu-layers 0 \
  --embedding --pooling last \
  -c 1024 -ub 512 \
  --port 7999 --host 0.0.0.0
```
- **Model**: Jina v5 Small Retrieval with Q8_0 quantization.
- **CPU only**: `--n-gpu-layers 0`.
- **Output**: 1024-dimensional vectors.

### PostgreSQL + pgvector

- The `pgvector` extension must be installed.
- Table `cv_chunks` stores document vectors alongside text.
- Similarity searches use the `<=>` cosine distance operator with `ORDER BY` for efficient retrieval.

---

## 12. Quick Reference: All Functions

| Module | Function | Purpose |
| --- | --- | --- |
| **main.py** | `_clean_text(text)` | Sanitize text for terminal output |
| | `extract_text(file_path)` | Unified PDF/DOCX text extraction |
| | `main()` | CLI argument parser and command dispatcher |
| **document_service.py** | `extract_text_from_pdf(file_path)` | Async PDF extraction wrapper |
| | `_extract_page_text_by_y(page)` | Layout-preserving page text extraction |
| | `extract_text_from_pdf_sync(file_path)` | Sync PDF text extraction |
| | `extract_text_from_pdf_stream(file)` | In-memory PDF stream extraction |
| | `pdf_has_text(file_path)` | Detect if PDF has selectable text |
| | `extract_text_from_word(file_path)` | DOCX extraction with numbered lists |
| | `save_text_to_docx(text, output_path)` | Save text as Word document |
| | `apply_corrections_for_clauses(...)` | Apply clause-level corrections (NDA redlining) |
| | `remove_columns_from_docx(docx_path)` | Strip column layouts from DOCX XML |
| | `generate_redline(...)` | Track-changes comparison between DOCXes |
| **document_utils.py** | `normalize_section(name)` | Map section heading to canonical category |
| | `clean_extracted_text(text)` | Normalize whitespace and detect list patterns |
| | `normalize_text_basic(text)` | Remove invisible chars, join hyphenated words |
| | `normalize_quotes(text)` | Convert smart quotes to ASCII |
| | `remove_redundant_title_from_clause(...)` | Strip title from clause body |
| | `title_starts_clause(title, clause)` | Check if clause begins with title |
| | `normalize(text)` | Lowercase + collapse whitespace |
| | `flatten(s)` | Collapse whitespace (case-preserving) |
| **embedding_service.py** | `generate_embedding(text, prefix)` | Single embedding via Jina v5 |
| | `generate_embeddings(texts, prefix)` | Batch embeddings via Jina v5 |
| | `_load_section_headings()` | Load CV section headings from config |
| | `_build_section_pattern(headings)` | Compile section-matching regex |
| | `parse_cv_sections(text)` | Split CV text into semantic sections |
| | `sub_chunk(text, max_chars)` | Split text at sentence boundaries |
| | `ingest_cv(file_path, replace_existing)` | Full single-CV ingestion pipeline |
| | `ingest_cv_folder(folder_path, ...)` | Batch folder ingestion |
| **db.py** | `get_connection()` | Create PostgreSQL connection |
| | `init_db()` | Create pgvector extension & table |
| | `insert_chunk(...)` | Insert single chunk |
| | `insert_chunks_batch(chunks)` | Batch insert chunks |
| | `_run_similarity_search(...)` | Low-level cosine similarity search |
| | `search_similar(query_emb, top_k, ...)` | RAG search (top-k) |
| | `search_similar_pool(query_emb, pool, ...)` | Candidate pool search |
| | `delete_by_file(file_name)` | Delete chunks by filename |
| | `search_best_chunk_for_cv(cv_id, ...)` | Best single chunk for a CV |
| | `get_all_chunks_for_cv(cv_id)` | All chunks for a CV |
| **rag_service.py** | `retrieve_context(query, top_k, ...)` | Embed query + search |
| | `format_context_for_llm(results)` | Format chunks for LLM prompt |
| | `rag_query(query, top_k, ...)` | Full RAG pipeline |
| **llm_service.py** | `call_llm(messages, temp, max_tokens)` | Send chat completion request |
| | `clean_llm_json(raw)` | Extract JSON from LLM output |
| | `decompose_job_description(jd_text)` | Extract literal JD requirements |
| | `justify_match(req, evidence, status, ...)` | Generate match justification |
| | `generate_gap_analysis(...)` | Analyze gaps with risks |
| | `expand_jd_requirements(jd_text, ...)` | Expand short requirements |
| | `semantic_chunk_section(section, text)` | LLM-powered semantic chunking |
| | `analyze_section_match(...)` | Section-to-section comparison |
| | `classify_requirements(requirements)` | Tier classification + key terms |
| | `synthesize_candidate_analysis(...)` | Combine section analyses |
| **jd_processor.py** | `process_jd_text(jd_text)` | Process raw JD text |
| | `process_jd_file(input_path)` | Process JD from file |
| | `save_requirements(result, output_path)` | Save to JSON |
| | `load_requirements(input_path)` | Load from JSON |
| | `main()` | Standalone CLI entry point |
| **report_base.py** | `_safe_text(text)` | ISO-8859-1 safe encoding |
| | `AnalysisReport.__init__(title, subtitle)` | Custom PDF constructor |
| | `AnalysisReport.header()` | Report header with title/date |
| | `AnalysisReport.footer()` | Page number footer |
| | `AnalysisReport.add_section_title(...)` | Section heading with fill |
| | `AnalysisReport.add_key_value(key, value)` | Key-value pair |
| | `AnalysisReport.add_bullet_list(items)` | Bulleted list |
| | `generate_jd_analysis_report(...)` | JD decomposition PDF |
| **company/matching_service.py** | `_get_section_config()` | Load section config |
| | `match_candidates(...)` | Master candidate ranking engine |
| **company/report_service.py** | `generate_report_pdf(...)` | Company PDF report |
| | `generate_jd_analysis_report(...)` | Delegates to base |
| | `generate_detailed_markdown_report(...)` | Company Markdown report |
| **applicant/report_service.py** | `generate_report_pdf(...)` | Applicant PDF report |
| | `generate_jd_analysis_report(...)` | Delegates to base |
| | `generate_detailed_markdown_report(...)` | Applicant Markdown report |
| **debug_db.py** | `debug_db()` | Database inspection utility |

---

*Generated for the AgenticATS project — Last updated: April 2026*
