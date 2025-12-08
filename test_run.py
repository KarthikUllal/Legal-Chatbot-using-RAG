from backend.app.vector_store import VectorStore as v
from backend.app.provider_client import NVIDIAProvider as NP
from backend.app.config import settings

try:
    provider = NP(settings.NVIDIA_API_KEY)
except Exception as e:
    print("Provider error:", e)

# Get embedding
query_text = "Tell me about IPC Section 420?"
query_embedding = provider.get_embeddings([query_text])[0]

print("Embedding created successfully")

# Create VectorStore instance
store = v()

# Query vector database
result = store.query(query_embedding=query_embedding, n_results=5)

print(result.get("documents", "not found"))
print(" ")
print(result.get("metadatas", "not found"))

print(" ")
print(result.get("distance","not found"))
