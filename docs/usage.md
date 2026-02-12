# Usage Guide

## 1. Index your documents

Place documents under:
resources/data/


Then run:
```bash
python rag/index.py

## 2. CLI usage
```bash
python chat_cli.py

Session-aware
Evidence-grounded
Local only

## 3. Web UI
uvicorn server:app --reload

Open:
http://localhost:8000

Supports:
Model switching
Mobile-friendly chat
Session persistence