# AgenticATS

AgenticATS is a command-line project for reading CVs from PDF or DOCX files, splitting them into meaningful sections, generating embeddings, storing those embeddings in PostgreSQL with `pgvector`, and retrieving relevant information through semantic search.

The latest version also supports **job-description-based candidate matching**. Instead of returning only the top matching chunks, the project can now rank the **top unique candidates** for a full job description and explain why each one was selected.

---

## What the project does

At a high level, the project works like this:

1. Read a CV file (`.pdf` or `.docx`)
2. Extract and normalize the text
3. Detect CV sections such as **Work Experience**, **Education**, **Technical Skills**, and **Projects**
4. Split long sections into smaller chunks
5. Generate embeddings using `sentence-transformers`
6. Store the chunks and embeddings in PostgreSQL using `pgvector`
7. Search the stored chunks later using a natural-language query
8. Rank the top unique candidates for a full job description using grouped chunk matches

---

## Current modes supported by the CLI

The current codebase supports four main modes:

1. **Text extraction** from PDF or DOCX
2. **CV ingestion** into PostgreSQL with embeddings
3. **RAG-style chunk search** over stored CV chunks
4. **Job-description-based candidate ranking** with unique top-N results

---

## Current project files and what each one does

### `main.py`
The main entry point for the CLI.

It handles:
- argument parsing
- database initialization
- raw text extraction
- optional text cleaning
- saving extracted text to DOCX
- embedding and ingestion
- semantic chunk search
- job-description-based candidate matching

This is the file you run from the terminal.

### `document_service.py`
Handles document-level operations.

It contains functions for:
- extracting text from PDF files
- checking whether a PDF contains selectable text
- extracting text from Word (`.docx`) files
- saving extracted text into a new DOCX file
- additional Word/redline-related helper functions that are not part of the current CV matching flow

For the current CV pipeline, the most relevant functions are:
- `pdf_has_text()`
- `extract_text_from_pdf_sync()`
- `extract_text_from_word()`
- `save_text_to_docx()`

### `document_utils.py`
Contains text cleanup and normalization helpers.

Examples:
- basic normalization after extraction
- optional cleanup of list-like patterns
- small helper functions used for text comparison or formatting

In the current CLI flow, the main active functions are:
- `normalize_text_basic()`
- `clean_extracted_text()`

### `embedding_service.py`
This is the ingestion and embedding layer.

It is responsible for:
- loading the embedding model
- generating embeddings for one text or many texts
- loading CV section headings from `sections_config.json`
- parsing extracted CV text into sections
- sub-chunking long sections
- running the full ingestion pipeline through `ingest_cv()`

This file is the core of the embedding workflow.

### `db.py`
Handles PostgreSQL and `pgvector` operations.

It is responsible for:
- opening database connections from environment variables
- initializing the database schema
- inserting chunks and embeddings
- deleting old chunks for the same file
- running vector similarity search over stored chunks

This is the storage and retrieval backend of the project.

### `rag_service.py`
Handles chunk-level retrieval logic.

It is responsible for:
- embedding the user query
- retrieving similar chunks from PostgreSQL
- formatting the retrieved chunks into readable context

This is the main search layer used by `--search`.

### `matching_service.py`
Handles candidate-level ranking for a full job description.

It is responsible for:
- embedding the full job description
- retrieving a larger pool of matching chunks
- grouping those chunks by `cv_id`
- keeping the strongest chunk per section
- applying section-aware scoring
- ranking the top unique candidates
- generating readable reasons and evidence chunks for each candidate

This is the main matching layer used by `--match-job` and `--jd-file`.

### `sections_config.json`
Contains the list of section headings the parser tries to detect in CVs.

Examples include:
- `Work Experience`
- `Education`
- `Technical Skills`
- `Courses`
- `Awards`
- `Extracurricular Activities`

If you want to support more CV heading styles, this is the first file to update.

### `orange_sr_data_science_jd.txt`
An example job description text file that can be used for testing the candidate matching flow.

This file is optional, but useful for running the CLI with `--jd-file`.

### `install_pgvector.bat`
A Windows helper script that installs `pgvector` by:
- loading the Visual Studio build tools environment
- pointing to the PostgreSQL installation folder
- cloning the `pgvector` repository
- building and installing it

This is a setup helper, not part of the runtime logic.

### `.env.example`
A template showing the PostgreSQL environment variables expected by the project.

You should copy it to a real `.env` file and update the values before running the project.

### `README.md`
The setup and usage guide for the project.

---

## Prerequisites

Before running the project, make sure you have:

- **Python 3.10+** installed
- **PostgreSQL** installed
- the **pgvector** extension installed in PostgreSQL
- a valid `.env` file with your PostgreSQL connection settings

If you are on Windows and need to build `pgvector` locally, you will also need:

- **Microsoft Visual Studio Build Tools**
- **Git**

---

## Environment variables

Create a `.env` file in the project root using `.env.example` as a guide.

Example:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agenticats
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
```

`main.py` loads these values automatically using `python-dotenv`.

---

## Python dependencies

Install the required Python packages in your virtual environment.

Example:

```bash
pip install python-dotenv psycopg2-binary sentence-transformers pymupdf PyPDF2 python-docx requests urllib3
```

If you already manage dependencies in another way, you can keep using your current setup.

---

## Installing `pgvector` on Windows

You can either run the included batch file:

```cmd
install_pgvector.bat
```

Or run the commands manually from a Visual Studio developer command prompt:

```cmd
set "PGROOT=C:\Program Files\PostgreSQL"
cd %TEMP%
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

> Note:
> The included `install_pgvector.bat` is currently hardcoded for PostgreSQL 13 and Visual Studio 2022 Build Tools. Update those paths if your local setup is different.

---

## Initialize the database

Run this once before embedding any CVs:

```cmd
python main.py --init-db
```

This creates the `vector` extension if needed and creates the `cv_chunks` table.

---

## How to run the project

### 1) Extract text only

Use this when you just want to read the CV text from a file.

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf"
```

### 2) Extract text and apply additional cleanup

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" -c
```

### 3) Extract text and save it to a Word file

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" -o ".\Ahmed_CV.docx"
```

### 4) Ingest a CV into PostgreSQL with embeddings

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" --embed
```

This does all of the following:
- extract the text
- detect sections
- split long sections into smaller chunks
- generate embeddings
- store the result in PostgreSQL

### 5) Search the stored CV chunks

```cmd
python main.py --search "FastAPI experience"
```

### 6) Search only inside one section

```cmd
python main.py --search "Python" --section "Technical Skills" --top-k 3
```

### 7) Rank top candidates from a direct job description string

```cmd
python main.py --match-job "Senior Data Scientist with strong ML, deep learning, GenAI, MLOps, SQL, and leadership experience" --top-candidates 2 --pool-size 50
```

### 8) Rank top candidates from a job description text file

```cmd
python main.py --jd-file ".\orange_sr_data_science_jd.txt" --top-candidates 2 --pool-size 50
```

### 9) Restrict candidate matching to a single section

```cmd
python main.py --jd-file ".\orange_sr_data_science_jd.txt" --top-candidates 2 --pool-size 50 --section "Technical Skills"
```

---

## Important behavior notes

### `--init-db` should be run separately

In the current code, `--init-db` exits immediately after initializing the database.

That means this is the correct sequence:

```cmd
python main.py --init-db
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" --embed
```

Do **not** expect this command to do both steps in one run:

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" --embed --init-db
```

### `-o` and `-c` apply to extraction mode, not embedding mode

In the current `main.py`, when `--embed` is used, the program runs `ingest_cv()` and then exits.

So these options are useful in extraction mode:
- `-o / --output`
- `-c / --clean`

But they are not part of the embedding flow in the current implementation.

### `--search` returns chunks, while `--match-job` and `--jd-file` return candidates

The search mode is chunk-based.
It returns the most similar stored CV chunks for a query.

The matching mode is candidate-based.
It retrieves a larger chunk pool first, then groups results by `cv_id`, scores each candidate, and returns the top unique candidates.

### First model load may download files

The embedding model is loaded through `sentence-transformers` using:

- `all-MiniLM-L6-v2`

On the first run, the model may need to download from Hugging Face.
If your network uses SSL inspection, a corporate proxy, antivirus filtering, or tools such as Zscaler, the model download may fail until that network layer is disabled or configured correctly.

---

## Runtime flow

### Extraction flow

When you run:

```cmd
python main.py <file_path>
```

The project does the following:

1. validate that the file exists
2. detect the file extension
3. if the file is a PDF, check whether it contains selectable text
4. extract the text from PDF or DOCX
5. normalize the extracted text
6. optionally clean it further with `-c`
7. either print the text or save it to DOCX with `-o`

### Embedding / ingestion flow

When you run:

```cmd
python main.py <file_path> --embed
```

The project does the following:

1. call `ingest_cv()` from `embedding_service.py`
2. extract the text through `extract_text()` in `main.py`
3. parse the text into CV sections using `sections_config.json`
4. split long sections into smaller chunks
5. generate embeddings for all chunks
6. optionally remove older chunks for the same file name
7. insert the new chunks into PostgreSQL

### Chunk search flow

When you run:

```cmd
python main.py --search "your query"
```

The project does the following:

1. embed the query text
2. search similar vectors in `cv_chunks`
3. optionally filter by section name
4. format the top results as readable context
5. print the retrieved chunks to the terminal

### Candidate matching flow

When you run:

```cmd
python main.py --jd-file "job_description.txt" --top-candidates 2 --pool-size 50
```

The project does the following:

1. read the full job description text
2. embed the job description
3. retrieve a larger pool of matching CV chunks
4. optionally filter by section name
5. group matches by `cv_id`
6. keep the best chunk per normalized section
7. apply section-aware scoring
8. rank the top unique candidates
9. print reasons and evidence chunks for each returned candidate

---

## Stored database structure

The project stores CV chunks in a table named `cv_chunks`.

The table currently includes:
- `id`
- `cv_id`
- `file_name`
- `section_name`
- `chunk_index`
- `chunk_text`
- `embedding`
- `created_at`

The embedding dimension is `384`, which matches the current model output.

---

## Quick command examples

Here are the most useful commands in one place.

### Initialize the database

```cmd
python main.py --init-db
```

### Extract text from a PDF

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf"
```

### Extract text and save it as DOCX

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" -o ".\Ahmed_CV.docx"
```

### Extract text with additional cleanup

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" -c
```

### Ingest one CV into PostgreSQL

```cmd
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" --embed
```

### Ingest multiple CVs one by one

```cmd
python main.py ".\Resources\Adel Elwan CV.pdf" --embed
python main.py ".\Resources\Ahmed El-Metwally CV 2026.pdf" --embed
python main.py ".\Resources\Amgad_Shalaby_Resume.pdf" --embed
python main.py ".\Resources\Ashraf_Mahmoud.pdf" --embed
python main.py ".\Resources\Asmaa Ali .pdf" --embed
python main.py ".\Resources\Dina Mohamed CV .pdf" --embed
```

### Search all stored CV chunks

```cmd
python main.py --search "machine learning projects"
```

### Search inside the Technical Skills section only

```cmd
python main.py --search "Python" --section "Technical Skills" --top-k 5
```

### Rank the top 2 unique candidates from a JD file

```cmd
python main.py --jd-file ".\orange_sr_data_science_jd.txt" --top-candidates 2 --pool-size 50
```

### Rank the top 3 unique candidates from a direct JD string

```cmd
python main.py --match-job "Senior Data Scientist with ML, deep learning, GenAI, SQL, MLOps, and leadership experience" --top-candidates 3 --pool-size 50
```

---

## Example output of candidate matching

The candidate matching mode prints a ranked list that includes:
- rank number
- file name
- CV ID
- final score
- matched sections
- reasons for selection
- evidence chunks with similarity values

This makes the result easier to inspect than a raw chunk list.

---

## Summary

If you only want text extraction, run `main.py` with a file path.

If you want to store CVs in PostgreSQL for retrieval, initialize the database once, then run `--embed` for each CV.

If you want chunk-level retrieval, use `--search`.

If you want candidate-level ranking for a full job description, use `--match-job` or `--jd-file` together with `--top-candidates`.
