import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalRAGStore:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.index = faiss.read_index(str(store_dir / "index.faiss"))
        self.records: List[Dict[str, Any]] = json.loads((store_dir / "records.json").read_text(encoding="utf-8"))
        meta = json.loads((store_dir / "meta.json").read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(meta["model_name"])

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, k)

        out = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            r = self.records[idx]
            out.append(
                {
                    "score": float(score),
                    "source_path": r["source_path"],
                    "chunk_index": r["chunk_index"],
                    "page": r.get("page"),
                    "text": r["text"],
                }
            )
        return out
