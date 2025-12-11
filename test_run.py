from backend.app.provider_client import NVIDIAProvider
from backend.app.vector_store import VectorStore

n = NVIDIAProvider()
v = VectorStore()
query = "What is section 63 of BNS?"
q_embed = n.get_embeddings([query])[0]  # Note: get_embeddings expects a list
res = v.query(q_embed, n_results=6)

# FIX THIS LINE - "documents" not "document"
docs = res.get("documents", [[]])[0]  # ✅ CORRECTED
metas = res.get("metadatas", [[]])[0]

print(f"\n🔍 DEBUG RETRIEVAL FOR: '{query}'")
print("=" * 50)

if not docs:
    print("❌ NO DOCUMENTS RETRIEVED!")
else:
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"\n📄 Chunk {i+1}:")
        print(f"   Metadata: {meta}")
        if doc:
            print(f"   Content preview: {doc[:200]}...")
            doc_lower = doc.lower()
            print(f"   Contains 'motor': {'motor' in doc_lower}")
            print(f"   Contains 'vehicle': {'vehicle' in doc_lower}")
            print(f"   Contains 'rape': {'rape' in doc_lower}")
            print(f"   Contains 'bharatiya': {'bharatiya' in doc_lower}")
            print(f"   Contains 'nyaya': {'nyaya' in doc_lower}")
            print(f"   Contains 'sanhita': {'sanhita' in doc_lower}")
            print(f"   Contains 'section 63': {'section 63' in doc_lower}")
        else:
            print("   ❌ EMPTY DOCUMENT")
        print("-" * 30)