# LiveKit Voice Agent with RAG Integration
### 📹 Live Demo
[Watch the Live Demo Video here](https://drive.google.com/file/d/1ptKFeIQH-YMB-JNUzCbS82rMJiTb7QbP/view?usp=sharing)

A real-time voice AI agent that combines **Google Gemini Live API** with **Retrieval-Augmented Generation (RAG)** using function calling to provide accurate, knowledge-grounded responses about NexaMind Labs.

## Project Overview

This project solves a critical challenge: **integrating RAG with real-time streaming audio**. Traditional approaches require exposing intermediate text (STT → LLM → TTS), but Gemini Live API processes audio end-to-end. Our solution uses **function calling** - the AI model autonomously calls our RAG system when it needs information, achieving both real-time streaming AND accurate knowledge retrieval.

## Architecture

### System Flow

```
User Speech → LiveKit Room → Gemini Live API ⟷ RAG Function Tool (FAISS) → Audio Response
```

### How It Works

1. **User speaks** into browser microphone
2. **Audio streams** to LiveKit room in real-time
3. **Gemini Live API** processes speech directly (no separate STT)
4. **When company information is needed**, model calls `search_knowledge()` function
5. **RAG system** searches FAISS vector database with semantic similarity
6. **Top relevant results** returned to the model as context
7. **Model generates** audio response incorporating RAG information
8. **User hears** natural, accurate answer (no separate TTS)

### Key Components

**LiveKit Agents Framework**
- Real-time audio communication
- Room management and participant handling
- Noise cancellation

**Google Gemini Live API (gemini-2.0-flash-live-001)**
- Unified speech-to-speech processing
- Function calling capability
- Low-latency streaming audio

**RAG System (FAISS + sentence-transformers)**
- Vector database: FAISS IndexFlatIP (cosine similarity)
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Knowledge base: 31 Q&A pairs about NexaMind Labs
- Score threshold: 0.3 (filters irrelevant results)

**Function Calling Integration**
- `@function_tool` decorator exposes `search_knowledge()` to AI
- Model decides when to call function based on query context
- RAG results passed back as function return value
- Model uses context to generate grounded responses

### Why This Approach?

**Problem:** Gemini Live API is a black box - audio in, audio out. No access to intermediate text for RAG injection.

**Solution:** Function calling allows the model to request information when needed, maintaining real-time streaming while enabling knowledge retrieval.

**Advantages:**
- True real-time voice interaction
- Accurate, retrieval-grounded responses
- Natural conversation flow
- Model decides when RAG is needed

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Voice Framework | LiveKit Agents |
| AI Model | Google Gemini 2.0 Flash Live |
| Vector Database | FAISS (IndexFlatIP) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Frontend | React + LiveKit SDK |
| Backend | Python 3.10+ |

## Project Structure

```
.
├── agent.py                 # Main LiveKit agent with Gemini Live + RAG
├── rag_system.py           # FAISS vector search implementation
├── build_rag.py            # Script to build FAISS index
├── knowledge_base.json     # NexaMind Labs Q&A data (31 entries)
├── requirements.txt        # Python dependencies
├── .env.example           # Template for environment setup
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── frontend/              # React web interface
    ├── package.json
    └── ... (LiveKit React starter)
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Node.js 16+ (for frontend)
- Git

### API Keys Required

You need accounts and API keys for:

1. **LiveKit** - Get free account at [cloud.livekit.io](https://cloud.livekit.io)
   - Create project
   - Copy WebSocket URL, API Key, and API Secret

2. **Google Gemini** - Get API key at [ai.google.dev](https://ai.google.dev)
   - Sign in and create API key in Google AI Studio

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/buthinaaa/Voice-Agent.git
cd Voice-Agent
```

#### 2. Backend Setup

Create and activate virtual environment:

```bash
# Create virtual environment
python -m venv voice

# Activate it
# On macOS/Linux:
source voice/bin/activate
# On Windows:
voice\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

This installs LiveKit agents, Gemini integration, FAISS, sentence-transformers, and other dependencies.

#### 3. Configure Environment Variables

Create `.env.local` file in project root:

```bash
cp .env.example .env.local
```

Edit `.env.local` with your credentials:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key
```

#### 4. Build RAG Index

```bash
python build_rag.py
```

This will:
- Download sentence-transformers model (~90MB, first time only)
- Process 31 knowledge base entries
- Generate embeddings
- Build FAISS index
- Save to `data/` folder

Expected output shows index statistics and sample test queries.


#### 5. Frontend Setup

Open new terminal, navigate to frontend:

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_LIVEKIT_URL=wss://your-project.livekit.cloud
```

Must use same LiveKit URL as backend.

## Running the Application

### Start Backend Agent

In project root (with virtual environment activated):

```bash
# Development mode (with hot reload)
python agent.py dev

# OR Production mode
python agent.py start
```

Keep this terminal running. You should see:
```
Initializing RAG system...
Loading embedding model: all-MiniLM-L6-v2
Loading existing FAISS index...
Loaded FAISS index with 31 documents
RAG system ready!
...
INFO   livekit.agents   registered worker
```

### Start Frontend

In new terminal:

```bash
cd frontend
npm run dev
```

Opens at `http://localhost:3000`

### Use the Application

1. Open browser to `http://localhost:3000`
2. Enter room name (e.g., "test-room")
3. Click "Join" and allow microphone permissions
4. Agent will greet you
5. Start talking - ask questions about NexaMind Labs

## Testing

### Example Queries

**Should trigger RAG (company-specific):**
- "What is NexaMind Labs?"
- "Do you offer a free trial?"
- "What integrations are available?"
- "How much does it cost?"
- "Can I use my own AI model?"

**Won't trigger RAG (general knowledge):**
- "What's the weather today?"
- "Tell me a joke"
- "How are you?"

### Verify RAG is Working

When you ask company questions, check backend terminal for:

```
[RAG] Tool called with query: 'What is NexaMind Labs?'
[RAG] Found 1 results (best score: 0.923)
```

This confirms the AI is calling the RAG function and retrieving information.

## How RAG Integration Works

### The Challenge

Gemini Live API processes audio end-to-end without exposing intermediate text. Traditional RAG requires text to search and inject context.

### The Solution: Function Calling

**Step 1: Define Function Tool**
```python
@function_tool
async def search_knowledge(self, query: str, run_ctx: RunContext) -> str:
    """Search NexaMind Labs knowledge base"""
    results = rag.search(query, top_k=3)
    if results:
        return f"Relevant info: {results[0]['answer']}"
    return "No information found."
```

**Step 2: Model Recognizes Need**

When user asks "What integrations do you support?", Gemini Live:
1. Processes speech internally
2. Recognizes this needs company-specific info
3. Decides to call `search_knowledge("What integrations do you support?")`

**Step 3: RAG Executes**

```python
# Generate embedding for query
embedding = model.encode(query)

# Search FAISS index (cosine similarity)
scores, indices = index.search(embedding, top_k=3)

# Filter by threshold (0.3)
results = [r for r in results if r.score >= 0.3]

# Return best match
return results[0]['answer']
```

**Step 4: Model Uses Context**

Function returns: "Integrations include Slack, Microsoft Teams, Zendesk..."

Gemini Live incorporates this into audio response naturally.

**Step 5: User Hears Answer**

Audio response includes accurate information from knowledge base.

## RAG System Details

### Vector Database

- **Type:** FAISS IndexFlatIP (exact inner product search)
- **Normalization:** L2 normalized vectors for cosine similarity
- **Size:** 31 documents × 384 dimensions = ~120KB
- **Search Time:** 50-100ms per query

### Embedding Model

- **Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **Dimensions:** 384
- **Size:** ~90MB
- **Speed:** Fast inference on CPU
- **Quality:** Good balance of speed and semantic understanding

### Score Threshold

- **Current:** 0.3 (cosine similarity)
- **Range:** -1.0 to 1.0 (after normalization: 0.0 to 1.0)
- **Interpretation:**
  - 0.3-0.5: Moderate similarity
  - 0.5-0.7: Good match
  - 0.7+: Strong match

To adjust threshold, edit `agent.py`:
```python
rag = SimpleRAG(
    score_threshold=0.3  # Lower = more results, Higher = stricter
)
```

### Knowledge Base

31 Q&A pairs covering:
- Company information
- Products and services
- Pricing and plans
- Integrations
- Technical capabilities
- Policies and support

Located in `knowledge_base.json` - easily extensible.

## Configuration

### Adjust Gemini Voice

In `agent.py`:
```python
gemini_live = RealtimeModel(
    model="gemini-2.0-flash-live-001",
    voice="Puck",  # Options: Puck, Charon, Kore, Fenrir, Aoede
    temperature=0.8,  # 0.0 = deterministic, 1.0 = creative
)
```

### Change Embedding Model

In `build_rag.py`:
```python
rag = SimpleRAG(
    embedding_model="all-MiniLM-L6-v2"  # Or: "all-mpnet-base-v2"
)
```

Then rebuild index: `python build_rag.py`

## Troubleshooting

**Backend won't start - "ModuleNotFoundError"**
- Ensure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

**"Invalid API Key" error**
- Check `.env.local` exists and has correct keys
- Verify `GOOGLE_API_KEY` starts with `AIza`

**Frontend can't connect**
- Verify `LIVEKIT_URL` matches in both `.env.local` files
- Check LiveKit credentials at cloud.livekit.io

**Agent not responding to voice**
- Check browser microphone permissions
- Try different browser
- Verify agent terminal shows "participant joined"

**RAG not finding results**
- Lower score threshold to 0.2 in `agent.py`
- Rebuild index: `python build_rag.py`
- Check query relevance to knowledge base

**Slow first response**
- Normal: First query loads embedding model (~2-3 seconds)
- Subsequent queries are faster (~1 second)


## Limitations

1. **Function calling dependency:** Relies on model correctly identifying when to search
2. **Score threshold tuning:** May need adjustment for different query types
3. **Top-K limit:** Returns maximum 3 results per query
4. **English-optimized:** Embedding model works best with English


## License

Created for job assessment purposes.
