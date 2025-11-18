"""
Simple RAG system for NexaMind Labs knowledge base
Uses FAISS for vector search with sentence-transformers embeddings
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    """Simple RAG system using FAISS and sentence-transformers"""
    
    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base.json",
        index_path: str = "data",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        print(f"📚 Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # FAISS index and metadata
        self.index = None
        self.documents = []
        
        if self._index_exists():
            print("✅ Loading existing FAISS index...")
            self._load_index()
        else:
            print("🔨 Building new FAISS index...")
            self._build_index()
    
    def _index_exists(self) -> bool:
        """Check if index files exist"""
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        return index_file.exists() and metadata_file.exists()
    
    def _build_index(self):
        """Build FAISS index from knowledge base"""
        # Create data directory
        self.index_path.mkdir(exist_ok=True)
        
        # Load knowledge base
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        print(f"📖 Processing {len(kb_data)} knowledge base items...")
        
        # Prepare documents and embeddings
        texts = []
        self.documents = []
        
        for item in kb_data:
            # Combine question and answer for better retrieval
            text = f"{item['question']} {item['answer']}"
            texts.append(text)
            self.documents.append({
                'question': item['question'],
                'answer': item['answer'],
                'text': text
            })
        
        # Generate embeddings in batch (faster)
        print("🧠 Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product = Cosine similarity for normalized vectors)
        print("🔨 Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        # Save index and metadata
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_file))
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✅ FAISS index built and saved to {self.index_path}")
        print(f"📊 Indexed {self.index.ntotal} documents")
    
    def _load_index(self):
        """Load existing FAISS index"""
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            self.documents = metadata['documents']
            saved_dim = metadata['embedding_dim']
        
        # Verify embedding dimensions match
        if saved_dim != self.embedding_dim:
            print(f"⚠️ Dimension mismatch! Rebuilding index...")
            self._build_index()
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        print(f"✅ Loaded FAISS index with {self.index.ntotal} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using FAISS
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with scores
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):  # -1 means not found
                result = self.documents[idx].copy()
                result['score'] = float(score)  # Cosine similarity score (0-1)
                results.append(result)
        
        return results
    
    def get_context(self, query: str, top_k: int = 2) -> str:
        """
        Get formatted context string for LLM
        
        Args:
            query: User's question
            top_k: Number of results to include
            
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return ""
        
        context = "Relevant information from NexaMind Labs knowledge base:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. Q: {result['question']}\n"
            context += f"   A: {result['answer']}\n\n"
        
        return context
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'model': self.embedding_model_name,
            'index_type': 'FAISS IndexFlatIP (Cosine Similarity)'
        }