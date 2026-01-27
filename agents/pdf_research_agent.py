from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

from rag.tool import rag_search

try:
    from google.adk.tools import FunctionTool
except Exception:
    FunctionTool = None

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

INSTRUCTION = """
You are a PDF Research Assistant.

Your job is to help users understand and analyze the content of a collection
of indexed PDF and text documents.

You do NOT browse the web.
You do NOT read files directly.
You MUST use the tool `rag_search` to retrieve relevant excerpts.

Rules:
- Use ONLY the retrieved excerpts as evidence.
- Never invent facts or fill gaps with assumptions.
- If the retrieved excerpts do not contain the answer, say you could not find it.
- If the question is ambiguous, ask one clarifying question.

How to work:
1) Call rag_search using a focused query derived from the user question.
2) Read the returned excerpts carefully.
3) Answer the question using only supported information.
4) When helpful, mention the source document names.

Response style:
- Clear, logical, and concise.
- Plain text only.
- Research-oriented tone (explain reasoning when useful).
"""

pdf_research_agent = LlmAgent(
    name="pdf_research_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="RAG-only PDF research assistant backed by a local vector store.",
    instruction=INSTRUCTION,
    tools=[FunctionTool(rag_search)] if FunctionTool else [],
)