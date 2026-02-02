from pathlib import Path
from rag.store import LocalRAGStore

_BASE = Path(__file__).resolve().parents[1]
_STORE = _BASE / "resources" / ".rag_store"

_store = LocalRAGStore(_STORE)

def rag_search(query: str, k: int = 6) -> str:
    hits = _store.search(query, k=k)
    if not hits:
        return "NO_HITS"

    MAX_TOTAL = 5500          # total chars returned to LLM
    MAX_PER_CHUNK = 900       # per excerpt cap

    out = []
    total = 0

    for h in hits:
        src = h["source_path"]
        page = h.get("page")
        header = f"[{src}" + (f" p.{page}]" if page else "]")

        txt = (h.get("text") or "").strip().replace("\r\n", "\n")
        if not txt:
            continue
        txt = txt[:MAX_PER_CHUNK]

        block = f"{header}\n{txt}"
        if total + len(block) > MAX_TOTAL:
            break

        out.append(block)
        total += len(block)

    return "\n\n".join(out) if out else "NO_HITS"


def rag_search_2pass(query: str, k1: int = 6, k2: int = 6) -> str:
    """
    Deterministic two-pass retrieval.
    Pass 1 = user's query
    Pass 2 = expanded query for better recall of numeric/clinical sections
    """
    # Pass 1
    r1 = rag_search(query, k=k1)

    # Pass 2 (expanded)
    expansion = (
        " clinical presentation symptoms symptomatic incidental incidence prevalence frequency "
        "abdominal pain gastrointestinal bleeding dysphagia diarrhoea nausea vomiting "
        "endoscopic subepithelial polyp nodule ampulla periampullary ampulla of Vater "
        "location distribution second portion duodenum intussusception "
        "treatment management resection follow-up recurrence outcome prognosis "
        "immunohistochemistry IHC immunostain synaptophysin chromogranin S100 cytokeratin "
        "n= % cases series cohort"
    )
    r2 = rag_search(query + expansion, k=k2)

    # Merge (keep it readable + avoid giant prompt payload)
    if r1 == "NO_HITS" and r2 == "NO_HITS":
        return "NO_HITS"

    parts = []
    if r1 != "NO_HITS":
        parts.append("PASS_1_RESULTS:\n" + r1)
    if r2 != "NO_HITS":
        parts.append("PASS_2_RESULTS:\n" + r2)

    return "\n\n".join(parts)


