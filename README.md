# Installation and Setup Guide

## Prerequisites

Before starting, make sure you have the following installed:

- **Microsoft Visual Studio 2019 Community Edition** or higher  
- **pgAdmin v18**  

---

## Step 1: Open Command Prompt as Administrator

Run the following:

- **x64 Native Tools Command Prompt for VS 2019**  

---

## Step 2: Build and Install `pgvector`

Run the following commands in the command prompt:

```cmd
set "PGROOT=C:\Program Files\PostgreSQL\18"
cd %TEMP%
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```
## Step 3: Initialize the Database

Start by initializing the database:

```cmd
python main.py --init-db
```
## Step 4: Run the Code

To process a CV PDF file and generate an output Word document with embeddings:
```cmd
python .\main.py '.\Resources\Ahmed El-Metwally CV.pdf' --embed --init-db -o "./Ahmed_CV.docx" -c
```

--embed : Generates embeddings from the PDF.

--init-db : Initializes the database (required if running for the first time).

-o : Specifies the output .docx file path.

-c : Optional flag for additional processing (as per script functionality).

You are now ready to run the project!