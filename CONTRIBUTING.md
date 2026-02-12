# Contributing to LocalRAG

Thanks for your interest in contributing!

## Design principles (must-follow)

1. Retrieval always comes before reasoning
2. Memory must never override evidence
3. No document content leaves the local machine
4. Answers must remain explainable and citeable

## What we welcome

- Better document extractors
- Retrieval improvements
- UX enhancements
- Bug fixes
- Documentation improvements

## What we avoid

- Hidden prompt tricks
- Hallucination-friendly shortcuts
- Opaque abstractions
- Remote indexing by default

## How to contribute

1. Fork the repo
2. Create a feature branch
3. Add tests or examples where relevant
4. Keep changes minimal and explicit
5. Open a PR with a clear explanation

## Coding style

- Prefer clarity over cleverness
- Small, composable functions
- Explicit over implicit behavior

## Reporting issues

Please include:
- What you expected
- What happened
- Evidence from `records.json` if relevant
