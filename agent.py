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

# Initialize RAG system (only once at startup)
print("🚀 Initializing RAG system...")
rag = SimpleRAG(
    knowledge_base_path="knowledge_base.json",
    index_path="data",
    embedding_model="all-MiniLM-L6-v2"
)
print("✅ RAG system ready!")
print(f"📊 Stats: {rag.get_index_stats()}")


class RagAssistant(Agent):
    """NexaMind Labs Assistant with RAG using on_user_turn_completed"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant for NexaMind Labs.

You have access to detailed information about:
- Company information and products
- Services and capabilities  
- Pricing and plans
- Integrations and technical details
- Policies and procedures

When answering questions about NexaMind Labs, use the context provided to you.
Be conversational, friendly, and concise. Keep responses brief and natural for voice."""
        )
    
    async def on_user_turn_completed(
        self, 
        turn_ctx: ChatContext, 
        new_message: ChatMessage
    ) -> None:
        """
        Called after user finishes speaking.
        This is where we inject RAG context BEFORE the LLM generates a response.
        """
        # Get the user's message text
        user_text = new_message.text_content
        
        print(f"🔍 User said: '{user_text}'")
        
        # Check if this is a company-related question
        company_keywords = [
            'nexamind', 'company', 'product', 'service', 'pricing', 'price',
            'integration', 'trial', 'support', 'policy', 'do you', 'what is',
            'how do', 'can you', 'what are', 'tell me about'
        ]
        
        is_company_question = any(
            keyword in user_text.lower() 
            for keyword in company_keywords
        )
        
        if is_company_question:
            # Search the knowledge base
            print(f"🔍 Searching RAG for: '{user_text}'")
            results = rag.search(user_text, top_k=2)
            
            if results:
                # Build context from results
                context = "\n\nRelevant information from NexaMind Labs knowledge base:\n"
                for i, result in enumerate(results, 1):
                    context += f"\n{i}. {result['answer']}"
                
                # Inject context into the conversation
                turn_ctx.add_message(
                    role="assistant",
                    content=context
                )
                
                print(f"📚 Added RAG context (top score: {results[0]['score']:.3f})")
            else:
                print("⚠️ No relevant RAG results found")


server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    # Create assistant
    assistant = RagAssistant()
    
    # Create session
    session = AgentSession(
        stt="deepgram/nova-3:en",
        llm="google/gemini-2.5-flash-lite",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Start session
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