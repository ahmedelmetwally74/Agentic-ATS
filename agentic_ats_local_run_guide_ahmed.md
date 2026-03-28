# Agentic ATS - Local Run Guide (Ahmed)

## Overview
Use **3 separate terminals** when running the project locally:

1. **Terminal 1** → Embedding server  
2. **Terminal 2** → Chat / LLM server  
3. **Terminal 3** → Project terminal

---

## Terminal 1 - Embedding Server
Open PowerShell and go to:

```powershell
cd "C:\Users\z0053beh\Downloads"
```

Start the embedding server:

```powershell
.\llama_build\llama-server.exe -m .\models\v5-small-retrieval-Q8_0.gguf --embeddings --port 7999 --host 127.0.0.1
```

This server handles:
- embeddings
- retrieval
- semantic search

Keep this terminal running while working on the project.

---

## Terminal 2 - Chat / LLM Server
Open another PowerShell window and go to:

```powershell
cd "C:\Users\z0053beh\Downloads"
```

Start the chat server with reasoning disabled:

```powershell
.\llama_build\llama-server.exe -m .\models\Qwen3.5-0.8B-Q8_0.gguf --port 8000 --host 127.0.0.1 --reasoning off
```

### Important check
In the server logs, make sure you see:

```text
thinking = 0
```

If you see:

```text
thinking = 1
```

then reasoning is still enabled, and applicant mode will likely fail because the model may return empty `content`.

Keep this terminal running as well.

---

## Terminal 3 - Project Terminal
Open a third terminal and go to the project folder:

```powershell
cd "D:\Agentic-ATS"
.\venv\Scripts\activate
```

Use this terminal to run the project commands below.

---

# Main Commands

## 1) Initialize the database
Run this if you need to initialize the database tables:

```powershell
python main.py --init-db
```

---

## 2) Embed a CV
Example:

```powershell
python main.py --embed "D:\Agentic-ATS\Resources\Ahmed El-Metwally CV 2026.pdf"
```

If the project has already been updated to reuse an existing CV by file name, it should return the existing `cv_id` instead of re-embedding the same file again.

---

## 3) Test retrieval / search
Use this to confirm that retrieval is working:

```powershell
python main.py --search "Python machine learning deep learning" --top-k 5
```

If this returns CV chunks, then:
- the embedding server is working
- retrieval is working
- stored embeddings are available in the database

---

## 4) Run Applicant Mode
Use this command to analyze **one CV only** against the job description:

```powershell
python main.py --mode applicant --cv-id "5a7c3e36-fae7-4fa9-b21f-01bfd2a78796" --jd-file "D:\Agentic-ATS\orange_sr_data_science_jd.txt" --output-dir "D:\Agentic-ATS\Reports-applicant"
```

---

# Output Locations

## PDF reports
You will usually find them here:

```text
D:\Agentic-ATS\Reports-applicant
```

Typical generated files:
- `job_description_analysis.pdf`
- `Ahmed El-Metwally CV 2026_analysis.pdf`

---

## Debug output
Debug output appears in **Terminal 3**.

When things are working correctly, you should see:
- `content repr:` containing real JSON or normal model output
- no error saying:

```text
LLM returned empty content
```

---

# Quick Checklist Before Any Run

Make sure:
- **Terminal 1** is running on port `7999`
- **Terminal 2** is running on port `8000`
- **Terminal 2** shows:

```text
thinking = 0
```

- **Terminal 3** is inside:

```text
D:\Agentic-ATS
```
