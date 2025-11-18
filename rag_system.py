"""
Simple RAG system for NexaMind Labs knowledge base
Uses FAISS for vector search with sentence-transformers embeddings
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer


# -------------------------------
#   GLOBAL MODEL LOADING (Fix #1)
# -------------------------------
_GLOBAL_EMBEDDING_MODEL = {}

def get_shared_embedding_model(model_name: str):
    """Load embedding model once and reuse across sessions."""
    if model_name not in _GLOBAL_EMBEDDING_MODEL:
        print(f"📚 Loading embedding model: {model_name}")
        _GLOBAL_EMBEDDING_MODEL[model_name] = SentenceTransformer(model_name)
    return _GLOBAL_EMBEDDING_MODEL[model_name]


class SimpleRAG:
    """Simple RAG system using FAISS and sentence-transformers"""
    
    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base.json",
        index_path: str = "data",
        embedding_model: str = "all-MiniLM-L6-v2",
        score_threshold: float = 0.40  # Fix #2 — ignore irrelevant results
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        self.score_threshold = score_threshold

        # Use shared embedding model (fast reuse)
        self.model = get_shared_embedding_model(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.documents = []
        
        if self._index_exists():
            print("✅ Loading existing FAISS index...")
            self._load_index()
        else:
            print("🔨 Building new FAISS index...")
            self._build_index()
    
    # --------------------
    #   INDEX MANAGEMENT
    # --------------------
    
    def _index_exists(self) -> bool:
        return (self.index_path / "faiss_index.bin").exists() and \
               (self.index_path / "metadata.pkl").exists()
    
    def _build_index(self):
        self.index_path.mkdir(exist_ok=True)

        # Load knowledge base
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        print(f"📖 Processing {len(kb_data)} knowledge base items...")
        
        texts = []
        self.documents = []
        
        for item in kb_data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            text = f"{question} {answer}".strip()
            
            texts.append(text)
            self.documents.append({
                'question': question,
                'answer': answer,
                'text': text
            })
        
        print("🧠 Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        faiss.normalize_L2(embeddings)

        print("🔨 Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, str(self.index_path / "faiss_index.bin"))
        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✅ FAISS index built and saved.")
        print(f"📊 Indexed {self.index.ntotal} documents")
    
    def _load_index(self):
        metadata_file = self.index_path / "metadata.pkl"

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        saved_dim = metadata['embedding_dim']
        
        if saved_dim != self.embedding_dim:
            print("⚠️ Dimension mismatch! Rebuilding index...")
            self._build_index()
            return
        
        self.index = faiss.read_index(str(self.index_path / "faiss_index.bin"))
        print(f"✅ Loaded FAISS index with {self.index.ntotal} documents")
    
    # --------------------
    #       SEARCH
    # --------------------
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None or not self.documents:
            return []
        
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.documents):
                continue
            
            # Fix #2 — apply score threshold
            if score < self.score_threshold:
                continue
            
            result = self.documents[idx].copy()
            result['score'] = float(score)
            results.append(result)
        
        return results
    
    # --------------------
    #   CONTEXT BUILDER
    # --------------------
    
    def get_context(self, query: str, top_k: int = 2) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""
        
        context = "Relevant information from NexaMind Labs knowledge base:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. Q: {result['question']}\n"
            context += f"   A: {result['answer']}\n\n"
        
        return context
    
    # --------------------
    #   STATISTICS
    # --------------------
    
    def get_index_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'model': self.embedding_model_name,
            'index_type': 'FAISS IndexFlatIP (Cosine Similarity)',
            'score_threshold': self.score_threshold,
        }
