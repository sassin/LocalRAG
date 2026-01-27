from pathlib import Path
from rag.store import LocalRAGStore

_BASE = Path(__file__).resolve().parents[1]
_STORE = _BASE / "resources" / ".rag_store"

_store = LocalRAGStore(_STORE)

def rag_search(query: str, k: int = 6) -> str:
    hits = _store.search(query, k=k)
    if not hits:
        return "NO_HITS"

    out = []
    for h in hits:
        if h.get("page"):
            cite = f"{h['source_path']} (page {h['page']})"
        else:
            cite = f"{h['source_path']}"

        out.append(f"[{cite}] score={h['score']:.3f}\n{h['text']}")
    return "\n\n".join(out)

