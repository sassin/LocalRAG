# rag/tool.py
from pathlib import Path
from rag.store import LocalRAGStore
import re
from typing import List, Dict, Any, Tuple, Set


_BASE = Path(__file__).resolve().parents[1]
_STORE = _BASE / "resources" / ".rag_store"
_store = LocalRAGStore(_STORE)


# -----------------------
# Formatting helpers
# -----------------------

def _format_hits(
    hits: List[Dict[str, Any]],
    max_total: int = 5500,
    max_per_chunk: int = 900,
) -> str:
    """
    Convert raw hit dicts into a compact, self-identifying text blob for the LLM.
    Each chunk is prefixed with: [source p.X c.Y]
    """
    out: List[str] = []
    total = 0

    for h in hits:
        src = h.get("source_path", "unknown")
        page = h.get("page")
        chunk_index = h.get("chunk_index")

        header = f"[{src}"
        if page is not None and page != "":
            header += f" p.{page}"
        if chunk_index is not None:
            header += f" c.{chunk_index}"
        header += "]"

        txt = (h.get("text") or "").strip().replace("\r\n", "\n")
        if not txt:
            continue

        txt = txt[:max_per_chunk]
        block = f"{header}\n{txt}"

        if total + len(block) > max_total:
            break

        out.append(block)
        total += len(block)

    return "\n\n".join(out) if out else "NO_HITS"


def _dedupe_key(h: Dict[str, Any]) -> Tuple[str, Any, Any]:
    return (h.get("source_path", ""), h.get("page"), h.get("chunk_index"))


# -----------------------
# Single-pass retrieval
# -----------------------

def rag_search(query: str, k: int = 6) -> str:
    """
    Single-pass semantic retrieval from the local vector store.
    Returns top-k excerpts with source/page headers.
    """
    hits = _store.search(query, k=k)
    if not hits:
        return "NO_HITS"
    return _format_hits(hits, max_total=5500, max_per_chunk=900)


# -----------------------
# Two-pass retrieval (generic + intent-aware)
# -----------------------

def _infer_intent(query: str) -> str:
    q = query.lower()

    if re.search(r"\b(compare|versus|vs\.?|difference|different|contrast)\b", q):
        return "compare"

    if re.search(r"\b(table|tables|figure|figures|chart|charts|plot|appendix|appendices|supplement|supplementary)\b", q):
        return "table"

    if (
        re.search(
            r"\b(how many|count|counts|total|totals|percent|percentage|rate|frequency|distribution|"
            r"mean|median|average|sd|std|standard deviation|p[- ]?value|confidence|interval)\b",
            q,
        )
        or "%" in q
        or "n=" in q
        or "N=" in q
    ):
        return "quant"

    if re.search(r"\b(method|methods|methodology|approach|experiment|setup|dataset|pipeline|model|analysis|procedure|protocol)\b", q):
        return "method"

    return "general"


def _expansion_for_intent(intent: str) -> str:
    """
    Domain-agnostic expansion terms to improve recall of structured/numeric sections
    across different types of documents.
    """
    # Common structural tokens found in most papers/reports/specs
    common = (
        " section sections overview summary abstract introduction background conclusion "
        " results findings discussion key points highlights "
        " table tables fig figure figures chart charts appendix appendices "
        " supplement supplementary"
    )

    # Patterns frequently present in quantitative and structured reporting
    numeric = (
        " data dataset sample population measurements metrics values "
        " n= N= count counts total totals % percent percentage ratio rate rates "
        " frequency frequencies distribution distributions "
        " mean median average sd std standard deviation range min max "
        " p-value p value confidence interval ci significance"
    )

    compare = " compare compared versus vs difference differences contrast higher lower increase decrease"

    method = " methods methodology approach procedure protocol setup experiment evaluation implementation pipeline"

    if intent == "quant":
        return common + " " + numeric
    if intent == "table":
        # Table/figure/caption cues
        return common + " " + numeric + " table 1 table 2 fig. figure 1 figure 2 caption"
    if intent == "compare":
        return common + " " + numeric + " " + compare
    if intent == "method":
        return common + " " + method
    return common


def rag_search_2pass(query: str, k1: int = 6, k2: int = 12) -> str:
    """
    Two-pass retrieval tuned for deeper answers:
    - Pass 1: precision (original query)
    - Pass 2: recall (generic, intent-aware expansion)
    Merges results with Pass 1 priority and dedupes by (source_path, page, chunk_index).
    Returns a single merged excerpt block (no PASS labels) to keep prompts compact.
    """
    hits1 = _store.search(query, k=k1)

    intent = _infer_intent(query)
    expanded_query = query + " " + _expansion_for_intent(intent)
    hits2 = _store.search(expanded_query, k=k2)

    # Merge with Pass 1 priority + dedupe
    seen: Set[Tuple[str, Any, Any]] = set()
    merged: List[Dict[str, Any]] = []

    for h in (hits1 or []):
        key = _dedupe_key(h)
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)

    for h in (hits2 or []):
        key = _dedupe_key(h)
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)

    if not merged:
        return "NO_HITS"

    # Increased evidence bandwidth to support detailed answers
    return _format_hits(merged, max_total=9000, max_per_chunk=1000)
