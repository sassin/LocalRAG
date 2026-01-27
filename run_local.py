from dotenv import load_dotenv
load_dotenv()

from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner

from agents.rag_coordinator_agent import rag_coordinator_agent

session_service = InMemorySessionService()

app = App(
    name="rag_coordinator_app",
    root_agent=rag_coordinator_agent,
    plugins=[LoggingPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
)

Runner(app=app, session_service=session_service)
print("✅ RAG Project Coordinator is running")
print("Try asking questions via your ADK client or UI")

