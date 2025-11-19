from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    room_io,
    ChatContext,
    ChatMessage,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation
from livekit.plugins.google.realtime import RealtimeModel 

from rag_system import SimpleRAG

load_dotenv(".env.local")

# Initialize RAG system once at startup
print("Initializing RAG system...")
rag = SimpleRAG(
    knowledge_base_path="knowledge_base.json",
    index_path="data",
    embedding_model="all-MiniLM-L6-v2",
    score_threshold=0.3
)
print("RAG system ready!")
print(f"Stats: {rag.get_index_stats()}")


class RagAssistant(Agent):
    """NexaMind Labs Assistant with RAG integration using Gemini Live API"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant for NexaMind Labs.

When I provide you with context from the knowledge base using the search_knowledge tool, use it to answer accurately.
If I tell you no relevant context was found, respond naturally and apologize that you do not have info about it.

IMPORTANT: When you need information about NexaMind Labs, ALWAYS call the search_knowledge tool first before answering.

Be conversational, friendly, and concise. Keep responses brief and natural for voice conversation."""
        )
    
    @function_tool
    async def search_knowledge(self, query: str, run_ctx: RunContext) -> str:
        """
        Search the NexaMind Labs knowledge base for information.
        
        Args:
            query: The question or topic to search for in the knowledge base.
        
        Returns:
            Relevant information from the knowledge base, or a message indicating no information was found.
        """
        print(f"🔍 RAG Tool called with query: '{query}'")
        
        # Search RAG - let score threshold filter relevance
        results = rag.search(query, top_k=3)
        
        if results:
            # Build context from top results
            context_parts = [
                "Here is relevant information from the NexaMind Labs knowledge base:",
                ""
            ]
            
            for i, result in enumerate(results, 1):
                context_parts.append(f"{i}. {result['answer']}")
            
            context = "\n".join(context_parts)
            
            print(f"✅ RAG: Found {len(results)} results (best score: {results[0]['score']:.3f})")
            return context
        else:
            # No relevant context found
            print("❌ RAG: No relevant results found")
            return "No relevant information found in the knowledge base for this query."


server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    assistant = RagAssistant()
    
    # Create Gemini Live API realtime model
    gemini_live = RealtimeModel(
        model="gemini-2.0-flash-live-001", 
        voice="Puck",
        instructions=assistant.instructions,
        temperature=0.8,
    )
    
    # Create session with ONLY the realtime model
    session = AgentSession(
        llm=gemini_live,
    )

    # Start session with noise cancellation
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() 
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP 
                    else noise_cancellation.BVC(),
            ),
        ),
    )

    # Initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and introduce yourself as the NexaMind Labs assistant. Offer to help with questions about the company. Remember to use the search_knowledge tool when you need information."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)