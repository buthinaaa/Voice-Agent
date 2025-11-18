import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# 1. Load the knowledge base (JSON file)
# -----------------------------------------
with open("faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert each FAQ entry into a single text chunk
documents = []
for entry in data:
    text = f"Question: {entry['question']}\nAnswer: {entry['answer']}"
    documents.append(text)

print(f"Loaded {len(documents)} FAQ entries.")

# -----------------------------------------
# 2. Load embedding model
# -----------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings (shape: N x 384)
embeddings = model.encode(documents, convert_to_numpy=True)

# Convert to float32 for FAISS
embeddings = embeddings.astype("float32")

print("Embeddings generated with shape:", embeddings.shape)

# -----------------------------------------
# 3. Build FAISS index
# -----------------------------------------
dimension = embeddings.shape[1]  # should be 384 for MiniLM
index = faiss.IndexFlatL2(dimension)

# Add vectors to FAISS
index.add(embeddings)

print("FAISS index created.")
print("Total vectors indexed:", index.ntotal)

# -----------------------------------------
# 4. Save FAISS index + metadata
# -----------------------------------------
faiss.write_index(index, "index.faiss")

# Save metadata (original documents)
with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print("Index saved as 'index.faiss' and metadata as 'documents.json'")
