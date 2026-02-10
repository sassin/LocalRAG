import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agents.research_agent import research_agent


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def _graceful_shutdown():
    # Best-effort: cancel pending tasks so aiohttp cleans up before loop closes
    try:
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception:
        pass


async def main():
    session_service = InMemorySessionService()

    app_name = "pdf_cli"
    user_id = "local_user"
    session_id = "local_session"

    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(agent=research_agent, app_name=app_name, session_service=session_service)

    print("Research Agent CLI. Type 'exit' to quit.\n")

    try:
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
                        p.text for p in event.content.parts if getattr(p, "text", None)
                    )

            print(f"\nAgent: {final_text or '(no response)'}\n")

    finally:
        # IMPORTANT: attempt to close underlying network clients cleanly
        try:
            # Some ADK versions expose close/aclosedown indirectly; this is best-effort.
            await session_service.aclose()  # if implemented
        except Exception:
            pass

        await _graceful_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
