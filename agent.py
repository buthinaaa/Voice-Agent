from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import google

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant. Be concise and natural in your responses."
        )

async def entrypoint(ctx: agents.JobContext):
    # Create session with Live API model
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="models/gemini-2.0-flash-live-001",  # Live-compatible model
            voice="Puck",  # Valid voice
            temperature=0.8,
        ),
    )
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant()
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )