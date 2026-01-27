import asyncio
from dotenv import load_dotenv
load_dotenv()

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agents.rag_coordinator_agent import rag_coordinator_agent


def main():
    session_service = InMemorySessionService()

    app_name = "rag_cli"
    user_id = "local_user"
    session_id = "local_session"

    # ✅ Create the session FIRST (required by your ADK version)
    asyncio.run(
        session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    )

    runner = Runner(
        agent=rag_coordinator_agent,
        app_name=app_name,
        session_service=session_service,
    )

    print("RAG Coordinator CLI. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        msg = types.Content(parts=[types.Part(text=q)])

        final_text = None
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "\n".join(
                    [p.text for p in event.content.parts if getattr(p, "text", None)]
                )

        print(f"\nAgent: {final_text or '(no response)'}\n")


if __name__ == "__main__":
    main()
