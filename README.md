# AgenticATS

AgenticATS is a local command-line project for reading CVs from PDF or DOCX files, extracting their sections, generating embeddings, storing them in PostgreSQL with `pgvector`, and analyzing CVs against a job description using a locally served LLM.

The project uses two local `llama-server` instances:
- one for **embeddings**
- one for **chat / reasoning**

It supports two workflows:
- **Company mode**: rank multiple stored candidates against a job description
- **Applicant mode**: analyze one specific CV against a job description and generate CV improvement suggestions

---

## Key Features

- Local **Jina v5** embeddings served through `llama-server`
- Local **Qwen** chat analysis served through `llama-server`
- PostgreSQL + `pgvector` storage for CV chunks
- Section-aware CV parsing and matching
- PDF reports for both job description analysis and final match analysis
- Two analysis modes:
  - **Company mode** for candidate ranking
  - **Applicant mode** for CV improvement

---

## What the project does

1. Read a CV file (`.pdf` or `.docx`) or a folder of CVs.
2. Extract text and split the CV into logical sections.
3. Chunk the extracted sections.
4. Generate embeddings for those chunks.
5. Store the vectors in PostgreSQL with `pgvector`.
6. Compare stored CV data against a job description.
7. Use a local LLM to generate structured match analysis.
8. Save the result as PDF reports.

---

## Analysis Modes

### Company Mode
Use this mode when you want to rank multiple candidates against one job description.

Output includes:
- ranking overview
- why a candidate fits the role
- things to keep in mind
- interview questions

### Applicant Mode
Use this mode when you want to evaluate one specific CV against one job description.

Output includes:
- section-by-section comparison
- missing skills / tools / experience / education
- practical CV improvement suggestions

Applicant mode is designed to analyze **one CV only**.

---

## Prerequisites

Before running the project, make sure you have:

1. **Python 3.10+**
2. **PostgreSQL** with the **pgvector** extension installed
3. A working Python virtual environment
4. Two running instances of **llama-server**:
   - **Embedding server** on `http://127.0.0.1:7999/v1/embeddings`
   - **Chat server** on `http://127.0.0.1:8000/v1/chat/completions`

> On Windows, you can use the provided `install_pgvector.bat` helper if needed.

---

## Environment Variables

Create a `.env` file and configure:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agenticats
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password

EMBEDDING_API_URL=http://127.0.0.1:7999/v1/embeddings
LLM_API_URL=http://127.0.0.1:8000/v1/chat/completions
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Initialize the database:

```bash
python main.py --init-db
```

> If you previously used an older vector dimension schema, recreate the `cv_chunks` table before reinitializing.

---

## Local Runtime Setup

Open **three terminals**.

### Terminal 1 - Embedding Server

```powershell
cd "C:\Users\z0053beh\Downloads"
.\llama_build\llama-server.exe -m .\models\v5-small-retrieval-Q8_0.gguf --embeddings --port 7999 --host 127.0.0.1
```

This server handles:
- embeddings
- retrieval
- vector search

---

### Terminal 2 - Chat / LLM Server

```powershell
cd "C:\Users\z0053beh\Downloads"
.\llama_build\llama-server.exe -m .\models\Qwen3.5-0.8B-Q8_0.gguf --port 8000 --host 127.0.0.1 --reasoning off
```

Important:
- make sure the log shows `thinking = 0`
- if `thinking = 1`, structured JSON output may fail in applicant or company analysis

---

### Terminal 3 - Project Terminal

```powershell
cd "D:\Agentic-ATS"
.\venv\Scripts\activate
```

Use this terminal for all project commands.

---

## Core CLI Commands

### 1) Embed a Single CV

```powershell
python main.py --embed "D:\Agentic-ATS\Resources\Ahmed El-Metwally CV 2026.pdf"
```

Use this when you want to ingest one CV into PostgreSQL.

---

### 2) Embed a Folder of CVs

```powershell
python main.py --embed ".\Resources\CandidatesFolder"
```

Use this for company mode when you want to ingest multiple CVs at once.

---

### 3) Basic Retrieval Test

```powershell
python main.py --search "Python machine learning deep learning" --top-k 5
```

This is useful for checking whether:
- embeddings are working
- vectors are stored correctly
- retrieval is returning relevant chunks

---

### 4) Company Mode

```powershell
python main.py --mode company --jd-file ".\job_description.txt" --top-candidates 3 --output-dir ".\Reports"
```

Expected output:
- ranked candidates in the terminal
- one JD analysis PDF
- one PDF report per selected candidate

---

### 5) Applicant Mode

```powershell
python main.py --mode applicant --cv-id "5a7c3e36-fae7-4fa9-b21f-01bfd2a78796" --jd-file "D:\Agentic-ATS\orange_sr_data_science_jd.txt" --output-dir "D:\Agentic-ATS\Reports-applicant"
```

Expected output:
- one applicant analysis in the terminal
- one JD analysis PDF
- one applicant PDF report

> Applicant mode should be used with a specific `cv_id` for one stored CV.

---

## Output Files

Typical output locations:

### Company mode
- `Reports/job_description_analysis.pdf`
- one analysis PDF per ranked candidate

### Applicant mode
- `Reports-applicant/job_description_analysis.pdf`
- `Reports-applicant/<CV name>_analysis.pdf`

---

## Debugging Tips

### If `--search` works but applicant/company mode fails
Check the chat server first.

The chat server must:
- be running on port `8000`
- use `--reasoning off`
- show `thinking = 0`

### If you see `LLM returned empty content`
That usually means the chat server is still returning reasoning output instead of normal `content`.

### If retrieval quality looks wrong
Check:
- embedding server is running on `7999`
- PostgreSQL is up
- the CV was actually embedded
- the correct `cv_id` is being used in applicant mode

---

## Project Structure Overview

- `main.py` - CLI entry point and orchestration
- `embedding_service.py` - CV parsing, chunking, embedding calls, ingestion helpers
- `llm_service.py` - job description decomposition, section analysis, synthesis
- `matching_service.py` - company ranking flow and applicant single-CV flow
- `report_service.py` - PDF generation
- `rag_service.py` - retrieval-only search flow
- `db.py` - PostgreSQL / `pgvector` queries and helpers
- `document_service.py` - raw PDF / DOCX extraction logic
- `sections_config.json` - section detection rules and weights

---

## Recommended Local Workflow

### For company mode
1. Start the embedding server
2. Start the chat server with `--reasoning off`
3. Initialize DB if needed
4. Embed a folder of CVs
5. Run company mode with a JD file

### For applicant mode
1. Start the embedding server
2. Start the chat server with `--reasoning off`
3. Embed the applicant CV if needed
4. Use the stored `cv_id`
5. Run applicant mode with the target JD file

---

## Notes

- The current local setup uses `Qwen3.5-0.8B-Q8_0.gguf` for chat.
- The applicant flow is functional, but output quality may still improve with prompt tuning or a larger local chat model.
- The embedding setup is currently stable and suitable for retrieval.
