import asyncio
from dotenv import load_dotenv
load_dotenv()

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agents.pdf_research_agent import pdf_research_agent


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def main():
    session_service = InMemorySessionService()

    app_name = "pdf_cli"
    user_id = "local_user"
    session_id = "local_session"

    # Create session once (required by your ADK version)
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=pdf_research_agent,
        app_name=app_name,
        session_service=session_service,
    )

    print("PDF Research Agent CLI. Type 'exit' to quit.\n")

    while True:
        q = (await ainput("You: ")).strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        msg = types.Content(parts=[types.Part(text=q)])

        final_text = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=msg,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "\n".join(
                    [p.text for p in event.content.parts if getattr(p, "text", None)]
                )

        print(f"\nAgent: {final_text or '(no response)'}\n")


if __name__ == "__main__":
    asyncio.run(main())
