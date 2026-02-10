# llm/gemini_client.py
import os
from google.genai import Client

_client = None

def _get_client() -> Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found. Set it in .env")
        _client = Client(api_key=api_key)
    return _client

def gemini_chat(prompt: str, model: str = "gemini-2.5-flash-lite") -> str:
    client = _get_client()
    resp = client.models.generate_content(model=model, contents=[prompt])
    try:
        return resp.text or ""
    except Exception:
        return str(resp)
