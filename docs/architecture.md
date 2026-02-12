# Architecture Overview

LocalRAG follows a strict pipeline:

Retrieval → Reasoning → Memory

## High-level flow

Documents → Indexer → Local Vector Store → Retrieval Tools → LLM → User

## Key principles

1. Retrieval is mandatory for factual answers
2. Memory never replaces retrieval
3. Documents never leave the local machine
4. The LLM only sees:
   - The user question
   - Retrieved excerpts
   - A small session context

## Core components

### Indexer (`rag/index.py`)
- Extracts text from documents
- Chunks content
- Generates embeddings
- Stores vectors locally (FAISS)

### Vector Store (`rag/store.py`)
- FAISS-backed
- Cosine similarity search
- No remote calls

### Retrieval Tools (`rag/tool.py`)
- `rag_search_2pass`: default retrieval
- `rag_get_page`: page-specific retrieval

### Prompt Layer (`prompts/`)
- Shared across CLI + Web
- Enforces evidence grounding

### Session Memory (`memory.py`)
- Lightweight, bounded memory
- Keeps conversational continuity

### Interfaces
- CLI (ADK agent)
- Web UI (FastAPI)


                ┌──────────────┐
                │   Documents  │
                │ (PDF, TXT…)  │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Indexer     │  ← index.py
                │  (FAISS)     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Vector DB   │  ← LocalRAGStore
                │ (local)      │
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ CLI (ADK)    │              │ FastAPI Web  │
│ chat_cli     │              │ server.py    │
└──────┬───────┘              └──────┬───────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ ResearchAgent│              │ Prompt Builder│
│ (with tools) │              │ + Memory      │
└──────┬───────┘              └──────┬───────┘
       ▼                              ▼
┌────────────────────────────────────────────┐
│         LLM (OpenAI / Gemini)               │
└────────────────────────────────────────────┘



📄 Supported Document Types
Indexing (default):
.pdf (text-based)
.txt
.csv
.xlsx
.docx
⚠️ Complex PDFs with scanned pages or multi-column tables should be handled separately (OCR / Docling pipeline can be added later).