import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from fastapi.templating import Jinja2Templates
from fastapi import Request

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agents.pdf_research_agent import pdf_research_agent
import os
from fastapi import Header, HTTPException

app = FastAPI()

# Single session service for the server lifetime
session_service = InMemorySessionService()
runner = Runner(agent=pdf_research_agent, app_name="rag_web", session_service=session_service)

API_KEY = os.getenv("RAG_WEB_API_KEY", "")
templates = Jinja2Templates(directory="templates")

def require_api_key(x_api_key: str | None):
    if not API_KEY:
        # fail closed so you don't accidentally expose without auth
        raise HTTPException(status_code=500, detail="Server misconfigured: RAG_WEB_API_KEY missing")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class ChatRequest(BaseModel):
    user_id: str = "web_user"
    session_id: str = "web_session"
    message: str

@app.on_event("startup")
async def startup():
    # Create a default session so first request doesn't fail
    await session_service.create_session(app_name="rag_web", user_id="web_user", session_id="web_session")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    # Ensure session exists (create if new user/session)
    try:
        await session_service.get_session(app_name="rag_web", user_id=req.user_id, session_id=req.session_id)
    except Exception:
        await session_service.create_session(app_name="rag_web", user_id=req.user_id, session_id=req.session_id)

    msg = types.Content(parts=[types.Part(text=req.message)])

    final_text = None
    async for event in runner.run_async(user_id=req.user_id, session_id=req.session_id, new_message=msg):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "\n".join([p.text for p in event.content.parts if getattr(p, "text", None)])

    return {"reply": final_text or "(no response)"}
