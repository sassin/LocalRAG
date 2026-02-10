# rag/tool.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

from rag.store import LocalRAGStore

_BASE = Path(__file__).resolve().parents[1]
_STORE = _BASE / "resources" / ".rag_store"
_store = LocalRAGStore(_STORE)

# Defaults
DEFAULT_K1 = 6
DEFAULT_K2 = 10

MAX_TOTAL_CHARS = 9000
MAX_PER_CHUNK = 1000


# -----------------------
# Formatting
# -----------------------

def _format_hits(
    hits: List[Dict[str, Any]],
    max_total_chars: int = MAX_TOTAL_CHARS,
    max_per_chunk: int = MAX_PER_CHUNK,
) -> str:
    """
    Convert retrieval hits into a compact text blob for the LLM.
    Each excerpt is prefixed with: [source p.X c.Y]
    """
    out: List[str] = []
    total = 0

    for h in hits:
        src = h.get("source_path", "unknown")
        page = h.get("page")
        ci = h.get("chunk_index")

        header = f"[{src}"
        if page is not None and page != "":
            header += f" p.{page}"
        if ci is not None:
            header += f" c.{ci}"
        header += "]"

        txt = (h.get("text") or "").strip().replace("\r\n", "\n")
        if not txt:
            continue

        txt = txt[:max_per_chunk]
        block = f"{header}\n{txt}"

        if total + len(block) > max_total_chars:
            break

        out.append(block)
        total += len(block)

    return "\n\n".join(out) if out else "NO_HITS"


def _dedupe_key(h: Dict[str, Any]) -> Tuple[str, Any, Any]:
    return (h.get("source_path", ""), h.get("page"), h.get("chunk_index"))


# -----------------------
# Single-pass retrieval
# -----------------------

def rag_search(query: str, k: int = 8) -> str:
    """
    Single-pass semantic retrieval from the local vector store.
    """
    hits = _store.search(query, k=k)
    if not hits:
        return "NO_HITS"
    return _format_hits(hits)


# -----------------------
# Evidence-driven 2-pass retrieval (default)
# -----------------------

_STOP = {
    "the", "and", "or", "to", "of", "in", "for", "on", "with", "by", "as", "at", "from",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "we", "you", "they", "their", "our", "an", "a", "not", "no", "yes", "can", "could",
    "may", "might", "will", "would", "should", "than", "then", "also", "such", "into",
}


def _tokenize(text: str) -> List[str]:
    # Keep words and hyphenated terms; ignore short tokens
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())


def _extract_generic_cues(text: str) -> Set[str]:
    """
    Generic cues that often indicate 'structured' sections like tables, figures, numeric results.
    Domain-agnostic.
    """
    cues: Set[str] = set()
    t = text.lower()

    if "%" in text:
        cues.update({"%", "percent", "percentage"})

    if re.search(r"\bn\s*=\s*\d+", t):
        cues.update({"n=", "sample", "cohort"})

    if re.search(r"\b(table|figure|fig\.?|appendix|supplement|supplementary)\b", t):
        cues.update({"table", "figure", "appendix", "supplementary"})

    if re.search(r"\b(mean|median|range|sd|std|p[- ]?value|confidence interval|ci)\b", t):
        cues.update({"mean", "median", "range", "sd", "p-value", "confidence interval"})

    return cues


def _build_expansion_from_hits(hits: List[Dict[str, Any]], max_terms: int = 18) -> str:
    """
    Evidence-driven expansion:
    - extract frequent tokens from pass-1 hits
    - add generic numeric/table cues if present
    - add a few structure words to increase recall
    """
    if not hits:
        return ""

    freq: Dict[str, int] = {}
    cues: Set[str] = set()

    # Use only the top few hits to avoid query drift
    for h in hits[:5]:
        txt = h.get("text") or ""
        cues |= _extract_generic_cues(txt)
        for tok in _tokenize(txt):
            if tok in _STOP:
                continue
            freq[tok] = freq.get(tok, 0) + 1

    # Most frequent terms first
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    terms = [t for t, _ in top[:max_terms]]

    # Generic document structure tokens
    structure = ["results", "findings", "discussion", "conclusion"]

    # Dedup while preserving order
    seen: Set[str] = set()
    final: List[str] = []
    for t in (terms + list(cues) + structure):
        t = (t or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        final.append(t)

    return " " + " ".join(final) if final else ""


def rag_search_2pass(query: str, k1: int = DEFAULT_K1, k2: int = DEFAULT_K2) -> str:
    """
    Two-pass retrieval (default):
    Pass 1: precision (original query)
    Pass 2: recall (expanded using tokens/cues extracted from Pass 1 evidence)
    Merge + dedupe, then format.
    """
    hits1 = _store.search(query, k=k1) or []
    expansion = _build_expansion_from_hits(hits1)
    hits2 = _store.search(query + expansion, k=k2) or []

    merged: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Any, Any]] = set()

    for h in hits1:
        key = _dedupe_key(h)
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)

    for h in hits2:
        key = _dedupe_key(h)
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)

    if not merged:
        return "NO_HITS"

    return _format_hits(merged)


# -----------------------
# Deterministic page fetch (fallback for tables/figures/pages)
# -----------------------

def rag_get_page(
    source_path: str,
    page: int,
    max_total_chars: int = 12000,
    max_per_chunk: int = 1400,
) -> str:
    """
    Deterministically return all indexed chunks for a given doc + page.
    Useful for: "Table 1 on page 4", "What does page 12 say?", etc.

    NOTE:
    - source_path must match records.json 'source_path' exactly
      (e.g. "JAAD 2022 Cutaneous manifestations.pdf")
    - page is an int (as stored in records.json)
    """
    out: List[str] = []
    total = 0

    # records loaded inside LocalRAGStore
    for r in _store.records:
        if r.get("source_path") != source_path:
            continue
        if r.get("page") != page:
            continue

        ci = r.get("chunk_index")
        header = f"[{source_path} p.{page}"
        if ci is not None:
            header += f" c.{ci}"
        header += "]"

        txt = (r.get("text") or "").strip().replace("\r\n", "\n")
        if not txt:
            continue
        txt = txt[:max_per_chunk]

        block = f"{header}\n{txt}"
        if total + len(block) > max_total_chars:
            break

        out.append(block)
        total += len(block)

    return "\n\n".join(out) if out else "NO_HITS"
