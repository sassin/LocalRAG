import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}



def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()

def read_text_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def iter_pdf_pages(path: Path):
    """
    Yields (page_number_1_based, page_text)
    """
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = ""
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            yield (i + 1, text)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def discover_docs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def build_index(docs_root: Path, store_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    store_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(model_name)

    docs = discover_docs(docs_root)
    if not docs:
        raise RuntimeError(f"No docs found under {docs_root}")

    records: List[Dict[str, Any]] = []
    texts: List[str] = []

    for doc in docs:
        rel = doc.relative_to(docs_root).as_posix()
        ext = doc.suffix.lower()

    if ext == ".pdf":
        # index each page separately so citations can include page numbers
        for page_num, page_text in iter_pdf_pages(doc):
            for i, chunk in enumerate(chunk_text(page_text)):
                records.append({
                    "source_path": rel,
                    "chunk_index": i,
                    "page": page_num,
                    "text": chunk,
                })
                texts.append(chunk)
    else:
        raw = read_text_doc(doc)
        for i, chunk in enumerate(chunk_text(raw)):
            records.append({
                "source_path": rel,
                "chunk_index": i,
                "page": None,
                "text": chunk,
            })
            texts.append(chunk)

    X = np.asarray(embedder.encode(texts, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, str(store_dir / "index.faiss"))
    (store_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (store_dir / "meta.json").write_text(
        json.dumps({"model_name": model_name, "docs_root": str(docs_root), "num_chunks": len(records)}, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Indexed {len(records)} chunks from {len(docs)} docs into {store_dir}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    docs_root = base / "resources" / "data"
    store_dir = base / "resources" / ".rag_store"

    build_index(docs_root, store_dir)
