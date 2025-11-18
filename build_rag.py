"""
Build FAISS index from knowledge_base.json
Run this once before starting the agent
"""

from rag_system import SimpleRAG

def main():
    print("=" * 60)
    print("Building FAISS RAG Index for NexaMind Labs")
    print("=" * 60)
    
    rag = SimpleRAG(
        knowledge_base_path="knowledge_base.json",
        index_path="data",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    print("\n" + "=" * 60)
    print("✅ RAG Index Built Successfully!")
    print("=" * 60)
    
    stats = rag.get_index_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   - Total Documents: {stats['total_documents']}")
    print(f"   - Embedding Dimension: {stats['embedding_dimension']}")
    print(f"   - Model: {stats['model']}")
    print(f"   - Index Type: {stats['index_type']}")
    
    # Test search
    print("\n🧪 Testing search functionality...")
    test_queries = [
        "What is NexaMind Labs?",
        "Do you offer a free trial?",
        "What integrations are available?"
    ]
    
    for query in test_queries:
        results = rag.search(query, top_k=1)
        if results:
            print(f"\n   Query: '{query}'")
            print(f"   Best Match: {results[0]['question']}")
            print(f"   Score: {results[0]['score']:.3f}")
    
    print("\n✅ All tests passed! You can now run agent.py")
    print("=" * 60)

if __name__ == "__main__":
    main()