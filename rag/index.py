import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def chunk_text(text: str, chunk_size=1400, overlap=250):
    lines = text.splitlines()

    chunks = []
    buf = []
    char_count = 0

    def flush():
        nonlocal buf, char_count
        if buf:
            chunks.append("\n".join(buf).strip())
        buf = []
        char_count = 0

    for line in lines:
        # Heuristic: table-like line
        is_table_line = sum(c.isdigit() for c in line) >= 3

        if char_count > chunk_size:
            flush()

        buf.append(line)
        char_count += len(line)

        # Keep table blocks together
        if is_table_line:
            continue

    flush()
    return [c for c in chunks if c]



def discover_docs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def iter_pdf_pages(path: Path):
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            yield (i + 1, t)  # 1-based page number


def build_index(docs_root: Path, store_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    store_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(model_name)

    docs = discover_docs(docs_root)
    if not docs:
        raise RuntimeError(f"No docs found under {docs_root}")

    # ✅ IMPORTANT: initialize ONCE, OUTSIDE the loop
    records: List[Dict[str, Any]] = []
    texts: List[str] = []

    for doc in docs:
        rel = doc.relative_to(docs_root).as_posix()
        ext = doc.suffix.lower()

        if ext == ".pdf":
            for page_num, page_text in iter_pdf_pages(doc):
                for ci, chunk in enumerate(chunk_text(page_text)):
                    records.append(
                        {
                            "source_path": rel,
                            "page": page_num,
                            "chunk_index": ci,
                            "text": chunk,
                        }
                    )
                    texts.append(chunk)
        else:
            raw = doc.read_text(encoding="utf-8", errors="ignore")
            for ci, chunk in enumerate(chunk_text(raw)):
                records.append(
                    {
                        "source_path": rel,
                        "page": None,
                        "chunk_index": ci,
                        "text": chunk,
                    }
                )
                texts.append(chunk)

    if not texts:
        raise RuntimeError("Docs were discovered but no extractable text was found (are PDFs scanned?).")

    X = np.asarray(embedder.encode(texts, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # ✅ IMPORTANT: write ONCE, AFTER processing ALL docs
    faiss.write_index(index, str(store_dir / "index.faiss"))
    (store_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (store_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "docs_root": str(docs_root),
                "num_docs": len(docs),
                "num_chunks": len(records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✅ Indexed {len(records)} chunks from {len(docs)} docs into {store_dir}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    docs_root = base / "resources" / "data"
    store_dir = base / "resources" / ".rag_store"

    build_index(docs_root, store_dir)
