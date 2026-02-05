# rag/index.py
"""
Build a local vector index over documents under resources/data.

Supported formats (current):
- .pdf  (text-based PDFs; scanned PDFs may yield little/no text)
- .docx
- .doc  (optional conversion via LibreOffice "soffice")
- .txt
- .csv
- .xlsx

Outputs (default):
- resources/.rag_store/index.faiss
- resources/.rag_store/records.json
- resources/.rag_store/meta.json

Dependencies (recommended):
- sentence-transformers
- faiss-cpu
- numpy
- pdfplumber  (preferred for PDFs; fallback to pypdf if missing)
- pypdf       (fallback PDF extractor)
- python-docx
- pandas
- openpyxl
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "faiss is required. Install with: pip install faiss-cpu"
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from e

# PDF extractors (prefer pdfplumber for better table-ish extraction)
_PDFPLUMBER_OK = False
try:
    import pdfplumber  # type: ignore

    _PDFPLUMBER_OK = True
except Exception:
    _PDFPLUMBER_OK = False

_PYPDF_OK = False
try:
    from pypdf import PdfReader  # type: ignore

    _PYPDF_OK = True
except Exception:
    _PYPDF_OK = False

# DOCX extractor
try:
    from docx import Document  # type: ignore
except Exception as e:
    raise RuntimeError("python-docx is required. Install with: pip install python-docx") from e

# CSV/XLSX extractors (pandas preferred; fallback for csv is built-in)
_PANDAS_OK = False
try:
    import pandas as pd  # type: ignore

    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".txt", ".csv", ".xlsx"}


@dataclass
class ExtractedBlock:
    """
    A text block extracted from a document.
    `ref` is page number (int) for PDFs, sheet name (str) for XLSX, or None.
    """
    ref: Optional[Union[int, str]]
    text: str


def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 250) -> List[str]:
    """
    Chunk text with overlap. Tuned to keep dense numeric/table-like paragraphs intact.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def discover_docs(docs_root: Path) -> List[Path]:
    return sorted([p for p in docs_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


# -----------------------
# Extractors
# -----------------------

def extract_txt(path: Path) -> List[ExtractedBlock]:
    return [ExtractedBlock(ref=None, text=path.read_text(encoding="utf-8", errors="ignore"))]


def extract_pdf(path: Path) -> List[ExtractedBlock]:
    blocks: List[ExtractedBlock] = []

    if _PDFPLUMBER_OK:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = (page.extract_text() or "").strip()
                if t:
                    blocks.append(ExtractedBlock(ref=i + 1, text=t))
        return blocks

    if _PYPDF_OK:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            try:
                t = (page.extract_text() or "").strip()
            except Exception:
                t = ""
            if t:
                blocks.append(ExtractedBlock(ref=i + 1, text=t))
        return blocks

    raise RuntimeError(
        "No PDF extractor available. Install pdfplumber (recommended) or pypdf.\n"
        "pip install pdfplumber pypdf"
    )


def extract_docx(path: Path) -> List[ExtractedBlock]:
    doc = Document(str(path))
    # paragraphs
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    # simple table extraction (rows joined by tabs)
    table_lines: List[str] = []
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                table_lines.append("\t".join(cells))

    text = "\n".join(paras)
    if table_lines:
        text = (text + "\n\n" if text else "") + "TABLES:\n" + "\n".join(table_lines)

    return [ExtractedBlock(ref=None, text=text)]


def _find_soffice() -> Optional[str]:
    """
    Find LibreOffice 'soffice' binary (best effort).
    On Windows, it's often in Program Files.
    """
    # If in PATH
    for candidate in ["soffice", "soffice.exe"]:
        if shutil_which(candidate):
            return candidate

    # Common Windows installs
    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def shutil_which(cmd: str) -> Optional[str]:
    # minimal 'which' to avoid importing shutil if you prefer
    from shutil import which
    return which(cmd)


def extract_doc(path: Path) -> List[ExtractedBlock]:
    """
    .doc is not reliably readable in pure Python.
    We convert .doc -> .docx using LibreOffice (soffice) if available,
    then parse the resulting .docx.
    """
    soffice = _find_soffice()
    if not soffice:
        # Skip gracefully (keeps indexing robust)
        # You can also raise here if you prefer.
        return [ExtractedBlock(ref=None, text="")]

    out_dir = path.parent
    # LibreOffice writes output with same base name
    # Example: file.doc -> file.docx
    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "docx",
        str(path),
        "--outdir",
        str(out_dir),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return [ExtractedBlock(ref=None, text="")]

    converted = path.with_suffix(".docx")
    if converted.exists():
        return extract_docx(converted)
    return [ExtractedBlock(ref=None, text="")]


def extract_csv(path: Path, max_rows: int = 2000) -> List[ExtractedBlock]:
    """
    Convert CSV to a text table. For very large CSVs, cap rows.
    """
    if _PANDAS_OK:
        df = pd.read_csv(str(path))
        if len(df) > max_rows:
            df = df.head(max_rows)
        text = df.to_string(index=False)
        return [ExtractedBlock(ref=None, text=text)]

    # Fallback: built-in csv reader
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append([c.strip() for c in row])

    # Join rows with tabs to preserve columns somewhat
    text = "\n".join(["\t".join(r) for r in rows])
    return [ExtractedBlock(ref=None, text=text)]


def extract_xlsx(path: Path, max_rows_per_sheet: int = 2000) -> List[ExtractedBlock]:
    """
    Convert each sheet into text table. ref = sheet name.
    """
    blocks: List[ExtractedBlock] = []

    if not _PANDAS_OK:
        raise RuntimeError(
            "XLSX support requires pandas + openpyxl.\n"
            "pip install pandas openpyxl"
        )

    xls = pd.ExcelFile(str(path))
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet)
        if len(df) > max_rows_per_sheet:
            df = df.head(max_rows_per_sheet)
        text = df.to_string(index=False)
        if text.strip():
            blocks.append(ExtractedBlock(ref=sheet, text=text))

    return blocks


def extract_document(path: Path) -> List[ExtractedBlock]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext == ".doc":
        return extract_doc(path)
    if ext == ".txt":
        return extract_txt(path)
    if ext == ".csv":
        return extract_csv(path)
    if ext == ".xlsx":
        return extract_xlsx(path)
    return [ExtractedBlock(ref=None, text="")]


# -----------------------
# Index builder
# -----------------------

def build_index(
    docs_root: Path,
    store_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    docs_root = docs_root.resolve()
    store_dir = store_dir.resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    docs = discover_docs(docs_root)
    if not docs:
        raise RuntimeError(f"No supported docs found under: {docs_root}")

    embedder = SentenceTransformer(model_name)

    # IMPORTANT: aggregate across ALL docs
    records: List[Dict[str, Any]] = []
    texts: List[str] = []

    num_skipped_empty = 0
    num_extracted_blocks = 0

    for doc_path in docs:
        rel = doc_path.relative_to(docs_root).as_posix()
        try:
            blocks = extract_document(doc_path)
        except Exception:
            # If one doc fails extraction, skip it instead of killing the whole index
            blocks = [ExtractedBlock(ref=None, text="")]

        for block in blocks:
            num_extracted_blocks += 1
            for ci, chunk in enumerate(chunk_text(block.text)):
                records.append(
                    {
                        "source_path": rel,
                        "page": block.ref,       # int page for PDF, sheet name for XLSX, or None
                        "chunk_index": ci,
                        "text": chunk,
                    }
                )
                texts.append(chunk)

        if not any(b.text.strip() for b in blocks):
            num_skipped_empty += 1

    if not texts:
        raise RuntimeError(
            "No extractable text found. If PDFs are scanned/image-only, OCR is required."
        )

    X = np.asarray(embedder.encode(texts, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, str(store_dir / "index.faiss"))
    (store_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (store_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "docs_root": str(docs_root),
                "store_dir": str(store_dir),
                "num_files_found": len(docs),
                "num_files_with_no_text": num_skipped_empty,
                "num_extracted_blocks": num_extracted_blocks,
                "num_chunks": len(records),
                "pdf_extractor": "pdfplumber" if _PDFPLUMBER_OK else ("pypdf" if _PYPDF_OK else "none"),
                "pandas_enabled": _PANDAS_OK,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✅ Indexed {len(records)} chunks from {len(docs)} files")
    print(f"   Store: {store_dir}")
    print(f"   records.json: {store_dir / 'records.json'}")
    print(f"   index.faiss:  {store_dir / 'index.faiss'}")


def main():
    # Defaults (match your earlier structure)
    base_dir = Path(__file__).resolve().parents[1]  # rag/ -> project root
    docs_root = base_dir / "resources" / "data"
    store_dir = base_dir / "resources" / ".rag_store"

    # Optional CLI overrides:
    # python rag/index.py --docs resources/data --store resources/.rag_store
    args = sys.argv[1:]
    if "--docs" in args:
        docs_root = Path(args[args.index("--docs") + 1])
    if "--store" in args:
        store_dir = Path(args[args.index("--store") + 1])
    if "--model" in args:
        model_name = args[args.index("--model") + 1]
    else:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    build_index(docs_root=docs_root, store_dir=store_dir, model_name=model_name)


if __name__ == "__main__":
    main()
