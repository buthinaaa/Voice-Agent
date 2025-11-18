"""
Build FAISS index from knowledge_base.json
Run this once before starting the agent.
"""

from rag_system import SimpleRAG


def main():
    print("=" * 60)
    print("Building FAISS RAG Index for NexaMind Labs")
    print("=" * 60)

    # Create SimpleRAG instance — this automatically:
    # (1) loads index if exists
    # (2) otherwise builds a new one
    rag = SimpleRAG(
        knowledge_base_path="knowledge_base.json",
        index_path="data",
        embedding_model="all-MiniLM-L6-v2",
        score_threshold=0.40          # Same default as in SimpleRAG
    )

    print("\n" + "=" * 60)
    print("✅ RAG Index Ready!")
    print("=" * 60)

    # Show index stats
    stats = rag.get_index_stats()
    print("\n📊 Index Statistics:")
    print(f"   - Total Documents: {stats['total_documents']}")
    print(f"   - Embedding Dimension: {stats['embedding_dimension']}")
    print(f"   - Model: {stats['model']}")
    print(f"   - Index Type: {stats['index_type']}")
    print(f"   - Score Threshold: {stats['score_threshold']}")

    # Test search
    print("\n🧪 Testing search functionality with sample queries...")
    test_queries = [
        "What is NexaMind Labs?",
        "Do you offer a free trial?",
        "What integrations are available?"
    ]

    for query in test_queries:
        results = rag.search(query, top_k=1)
        if results:
            best = results[0]
            print(f"\n   Query: {query}")
            print(f"   Best Match: {best['question']}")
            print(f"   Score: {best['score']:.3f}")
        else:
            print(f"\n   Query: {query}")
            print("   ⚠ No relevant match (below score threshold).")

    print("\n✅ All tests completed! You can now run agent.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
