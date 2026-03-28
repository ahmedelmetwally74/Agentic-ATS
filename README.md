# AgenticATS

AgenticATS is a command-line project for reading CVs from PDF or DOCX files, extracting their structural sections, generating embeddings, storing those embeddings in PostgreSQL with `pgvector`, and performing LLM-powered candidate matching.

The latest version has been rebuilt to use **llama-server** for both structured embeddings (Jina v5) and intelligent reasoning (Qwen3.5-2B). It supports extracting, ranking, and generating personalized PDF reports for candidates against a job description.

---

## 🚀 Key Features

- **Jina v5 Embeddings**: Uses high-quality 1024-dimensional embeddings served locally via `llama-server`.
- **LLM Reasoning**: Uses Qwen3.5-2B to generate natural language explanations of match quality.
- **Two Analysis Modes**:
  - **Company Mode**: Upload a folder of CVs. Match against a job description to rank top candidates. Generates a PDF report per candidate with specific reasons they fit the role and custom **interview questions** to ask them.
  - **Applicant Mode**: Upload your own CV. Match against a job description to receive a PDF report analyzing your strengths and providing targeted **CV improvement suggestions** (missing skills, project ideas, gaps).
- **Automated PDF Reports**: Match results aren't just printed to the terminal; they are saved as clean, formatted PDF reports per candidate for easy sharing or review.

---

## What the project does

1. Read a CV file (`.pdf` or `.docx`) or a whole folder of them.
2. Extract text and detect CV sections (Work Experience, Education, Technical Skills, etc.).
3. Split sections into chunks and embed them via `llama-server` (Jina v5).
4. Store the vectors in PostgreSQL (`pgvector`).
5. Rank stored candidates against a full job description.
6. Use Qwen3.5-2B to analyze the match and generate insights.
7. Save the final analysis to a `.pdf` report.

---

## Prerequisites

Before running the project, make sure you have:

1. **Python 3.10+**
2. **PostgreSQL** with the **pgvector** extension installed.
3. Two running instances of **llama-server**:
   - Instance 1: Running a **Jina v5** embedding model at `http://127.0.0.1:7999/v1/embeddings`
   - Instance 2: Running a **Qwen3.5-2B** chat model at `http://localhost:8000/v1/chat/completions`

*(To install `pgvector` on Windows, you can use the included `install_pgvector.bat` script.)*

---

## Installation & Setup

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment Variables**:
Copy `.env.example` to `.env` and configure your Postgres connection and API URLs:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agenticats
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password

EMBEDDING_API_URL=http://127.0.0.1:7999/v1/embeddings
LLM_API_URL=http://localhost:8000/v1/chat/completions
```

3. **Initialize the Database**:
Run this once to create the `vector` extension and the `cv_chunks` table (configured for 1024-dimension Jina vectors).
```cmd
python main.py --init-db
```

*> Note: If you have an older version of this project using 384-dimension vectors, you must drop the old `cv_chunks` table before running `--init-db`.*

---

## CLI Usage Guide

### 1) Company Mode (Candidate Ranking)

**Step A: Ingest a folder of CVs**
You can point `--embed` directly to a directory. The system will process all `.pdf` and `.docx` files it finds.
```cmd
python main.py --embed ".\Resources\CandidatesFolder"
```

**Step B: Match Job Description & Generate PDF Reports**
Specify `--mode company` and provide the path to your Job Description text file.
```cmd
python main.py --mode company --jd-file ".\job_description.txt" --top-candidates 3 --output-dir ".\Reports"
```
*Output*: A ranked terminal list + 3 PDF reports containing reasons for selection and suggested interview questions.

---

### 2) Applicant Mode (CV Improvement)

**Step A: Ingest the Applicant's CV**
```cmd
python main.py --embed ".\Resources\My_CV.pdf"
```

**Step B: Analyze Against Job Description**
Specify `--mode applicant` and provide the target Job Description.
```cmd
python main.py --mode applicant --jd-file ".\target_role.txt" --output-dir ".\Reports"
```
*Output*: A PDF report detailing current strengths and actionable suggestions for how to improve the CV for this specific role.

---

### 3) Text Extraction & Standard Search

**View raw extracted text from a CV:**
```cmd
python main.py ".\Resources\My_CV.pdf"
```

**Save extracted text to a Word Document:**
```cmd
python main.py ".\Resources\My_CV.pdf" -o ".\Extracted_CV.docx"
```

**Run a simple RAG chunk search (no LLM, just vector retrieval):**
```cmd
python main.py --search "machine learning algorithms" --top-k 5
```

**Search specifically within technical skills:**
```cmd
python main.py --search "Python and SQL" --section "Technical Skills"
```

---

## Project Structure Overview

- `main.py`: The CLI entry point. Handles arguments and orchestrates the flow.
- `embedding_service.py`: Calls the local Jina v5 embedding API. Parses CV text into sections and chunks. Prepares the "Document: "/"Query: " patterns required by Jina.
- `llm_service.py`: Calls local Qwen3.5-2B for Company/Applicant contextual analysis.
- `report_service.py`: Generates the formatted PDF analysis reports.
- `matching_service.py`: Executes the candidate retrieval pool, scores them section by section, and prepares data for the LLM.
- `rag_service.py`: Simple chunk retrieval for debugging or standard search.
- `db.py`: Postgres / pgvector connection pool and queries.
- `document_service.py`: Core PyMuPDF & docx extraction logic.
- `sections_config.json`: The heuristic rules for identifying CV sections like "Work Experience" or "Skills."
