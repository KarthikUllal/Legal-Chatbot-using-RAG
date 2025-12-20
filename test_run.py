from backend.app.provider_client import NVIDIAProvider
from backend.app.vector_store import VectorStore

n = NVIDIAProvider()
v = VectorStore()
query = "What is punishment for rape?"
q_embed = n.get_embeddings(query)[0]
res = v.query(q_embed, n_results=6)

print(res)
