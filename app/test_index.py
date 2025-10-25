
from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np

INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"

# Load models and data
print("Loading embedding model and index...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
print(f"Index loaded with {len(meta)} text chunks")

# Helper function
def retrieve(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [{"score": float(D[0][i]), **meta[I[0][i]]} for i in range(top_k)]
    return results

#### Test Queries 
test_queries = [
    "Who is Alice and where did she fall?",
    "Tell me about Aladdin's adventures.",
    "What did Gulliver see on his travels?",
    "Explain the Queen of Hearts.",
    "What is Neural Network",
]

for q in test_queries:
    print("\n" + "="*80)
    print(f"Query: {q}")
    results = retrieve(q)
    for r in results:
        print(f"\n Source: {r['source']}")
        print(f"Similarity: {r['score']:.3f}")
        print(f"Text Snippet: {r['text'][:300]}...")
