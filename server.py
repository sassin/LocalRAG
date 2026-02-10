# server.py
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag.tool import rag_search_2pass, rag_get_page
from llm.openai_client import openai_chat
from llm.gemini_client import gemini_chat
from prompts.research_prompt import build_chat_prompt, DEFAULT_CONFIG
from memory import InMemoryChatStore

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"

ACCESS_KEY = os.getenv("APP_ACCESS_KEY", "").strip()
DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai").strip().lower()
RETURN_EVIDENCE = os.getenv("RETURN_EVIDENCE", "0").strip() == "1"

_ALLOWED_PROVIDERS = {"openai", "gemini"}
if DEFAULT_PROVIDER not in _ALLOWED_PROVIDERS:
    DEFAULT_PROVIDER = "openai"

# Session memory store (Option B)
CHAT_STORE = InMemoryChatStore(ttl_seconds=6 * 60 * 60)

app = FastAPI()
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


def _require_key(request: Request):
    if not ACCESS_KEY:
        return
    supplied = request.headers.get("X-ACCESS-KEY", "")
    if supplied != ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_sources_from_evidence(evidence: str) -> list[str]:
    """
    Evidence headers look like:
      [paper.pdf p.4 c.12]
    Capture: "paper.pdf p.4"
    """
    sources = []
    for m in re.finditer(r"\[([^\]]+)\]", evidence or ""):
        hdr = m.group(1)  # e.g. "JAAD.pdf p.4 c.0"
        # keep file + page only
        parts = hdr.split()
        if not parts:
            continue
        file_part = parts[0]
        page_part = next((p for p in parts[1:] if p.startswith("p.")), "")
        s = (file_part + (" " + page_part if page_part else "")).strip()
        if s and s not in sources:
            sources.append(s)
    return sources[:8]


@app.get("/health")
def health():
    return {
        "ok": True,
        "default_provider": DEFAULT_PROVIDER,
        "auth_enabled": bool(ACCESS_KEY),
        "return_evidence": RETURN_EVIDENCE,
        "retrieval": "2pass+get_page",
        "memory": "in_memory",
    }


@app.get("/")
def root():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/api/chat")
async def chat(request: Request):
    _require_key(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_msg = (body.get("message") or "").strip()
    provider = (body.get("provider") or DEFAULT_PROVIDER).strip().lower()
    session_id = (body.get("session_id") or "").strip() or uuid.uuid4().hex

    if provider not in _ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="provider must be 'openai' or 'gemini'")

    if not user_msg:
        return JSONResponse({"answer": "Please enter a question.", "provider": provider, "session_id": session_id})

    state = CHAT_STORE.get(session_id)

    mode = (body.get("mode") or "rag").strip().lower()

    # Retrieval (always evidence-first)
    if mode == "get_page":
        source_path = (body.get("source_path") or "").strip()
        page = body.get("page")
        if not source_path or page is None:
            raise HTTPException(status_code=400, detail="mode=get_page requires source_path and page")

        evidence = rag_get_page(source_path=source_path, page=page)
        if evidence == "NO_HITS":
            return JSONResponse(
                {"answer": "I couldn’t find that page in the indexed documents.", "provider": provider, "session_id": session_id}
            )
    else:
        # Memory-aware query hinting (small, safe)
        ctx_hint = state.summary.strip()
        # keep it short to avoid poisoning retrieval
        query_for_retrieval = user_msg if not ctx_hint else f"{user_msg}\nContext hint: {ctx_hint}"
        evidence = rag_search_2pass(query_for_retrieval)
        if evidence == "NO_HITS":
            return JSONResponse(
                {"answer": "I couldn’t find relevant evidence in the indexed documents.", "provider": provider, "session_id": session_id}
            )

    # Update memory with sources (from evidence)
    used_sources = _extract_sources_from_evidence(evidence)
    state.add_sources(used_sources)

    # Build prompt with small session context
    context_block = state.build_context_block()
    prompt = build_chat_prompt(user_q=user_msg, evidence=evidence, cfg=DEFAULT_CONFIG, context=context_block)

    # Generation
    try:
        if provider == "gemini":
            answer = gemini_chat(prompt)
        else:
            answer = openai_chat(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # Save turn + update summary
    state.add_turn(user_msg, answer)
    state.update_summary_heuristic()

    payload = {"answer": answer, "provider": provider, "session_id": session_id}
    if RETURN_EVIDENCE:
        payload["evidence"] = evidence
        payload["mode"] = mode
        payload["context_used"] = context_block

    return JSONResponse(payload)
