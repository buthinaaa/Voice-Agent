from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AgentServer, 
    AgentSession, 
    Agent, 
    room_io,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
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
    """NexaMind Labs Assistant with RAG integration"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant for NexaMind Labs.

When provided with context from the knowledge base, use it to answer accurately.
If no relevant context is provided, respond naturally and apologize for that you do not have info about it.

Be conversational, friendly, and concise. Keep responses brief and natural for voice conversation."""
        )
    
    async def on_user_turn_completed(
        self, 
        turn_ctx: ChatContext, 
        new_message: ChatMessage
    ) -> None:
        """
        Intercept user message and inject RAG context before LLM generates response.
        """
        user_text = new_message.text_content
        
        print(f"User query: '{user_text}'")
        
        # Always search RAG - let score threshold filter relevance
        results = rag.search(user_text, top_k=3)
        
        if results:
            # Build context from top results
            context_parts = [
                "Reference information from NexaMind Labs knowledge base:",
                ""
            ]
            
            for i, result in enumerate(results, 1):
                context_parts.append(f"{i}. {result['answer']}")
            
            context = "\n".join(context_parts)
            
            # Inject context as system-level instruction
            turn_ctx.add_message(
                role="user",
                content=f"[Context for this question]\n{context}\n\n[User Question]\n{user_text}"
            )
            
            print(f"RAG: Added {len(results)} results (best score: {results[0]['score']:.3f})")
        else:
            # No relevant context found - inform the model
            turn_ctx.add_message(
                role="user",
                content=f"[No relevant context found in knowledge base]\n\n[User Question]\n{user_text}"
            )
            print("RAG: No relevant results found")


server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    assistant = RagAssistant()
    
    # Create session with STT -> LLM -> TTS pipeline
    session = AgentSession(
        stt="deepgram/nova-3:en",
        llm="google/gemini-2.5-flash-lite",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
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
        instructions="Greet the user warmly and introduce yourself as the NexaMind Labs assistant. Offer to help with questions about the company."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)