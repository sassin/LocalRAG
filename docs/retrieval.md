# Retrieval & RAG Strategy

LocalRAG uses Retrieval-Augmented Generation (RAG) by default.

## Two-pass retrieval (default)

Every question performs two searches:

### Pass 1 — Precision
- Original user question
- Finds highly relevant chunks

### Pass 2 — Recall
- Expanded query with structural terms
- Improves recall for:
  - Tables
  - Figures
  - Results sections
  - Methods
  - Numeric data

Results are merged and deduplicated.

## Page-level retrieval

Use `rag_get_page` when:
- You know the exact page
- A table or figure is missed by semantic search

Example:
```json
{
  "mode": "get_page",
  "source_path": "paper.pdf",
  "page": 4,
  "message": "Summarize Table I"
}


Table handling

Tables are:
Indexed as separate chunks
Chunked with larger windows
Preserved verbatim (rows + columns)

If a table exists in records.json but is not found:
Use rag_get_page
Or ask explicitly for the table