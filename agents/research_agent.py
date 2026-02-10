from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

from rag.tool import rag_search_2pass, rag_get_page

from prompts.research_prompt import agent_instruction, DEFAULT_CONFIG

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

tools = []
if FunctionTool:
    tools = [
        FunctionTool(rag_search_2pass),
        FunctionTool(rag_get_page),
    ]

research_agent = LlmAgent(
    name="research_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="RAG-only research assistant backed by a local vector store.",
    instruction=agent_instruction(DEFAULT_CONFIG),
    tools=tools,
)
