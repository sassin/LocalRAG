# Session Memory

LocalRAG uses lightweight, safe session memory.

## What is remembered

- Last 2–4 conversational turns
- A short rolling topic summary
- Recently referenced sources

## What is NOT remembered

- Full document text
- Raw evidence chunks
- Long conversation history

## Why this design?

- Prevents hallucination
- Keeps prompts small
- Preserves truth guarantees

## How memory is used

- Improves follow-up questions
- Provides context hints
- Never replaces retrieval

Every question still performs fresh retrieval.
