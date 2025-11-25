# backend/app/vector_store.py
from typing import List, Dict
import chromadb

# Handle imports for both direct execution and module import
try:
    from .config import settings
except ImportError:
    # Fallback for direct execution
    from config import settings

class VectorStore:
    def __init__(self, collection_name: str = "indian_laws"):
        # NEW ChromaDB client 
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        self.col_name = collection_name
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name, 
                metadata={"source": "legal_docs"}
            )

    def add_document(self, document_name: str, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """
        Automatically add entire document with auto-generated IDs
        """
        # Auto-generate IDs: "ipc_chunk_0", "ipc_chunk_1", ...
        ids = [f"{document_name}_chunk_{i}" for i in range(len(chunks))]
        
        # If no metadata provided, create basic metadata
        if metadatas is None:
            metadatas = [{"document": document_name, "chunk_id": i} 
                        for i in range(len(chunks))]
        
        self.collection.add(
            ids=ids, 
            documents=chunks, 
            embeddings=embeddings, 
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} chunks from {document_name}")

    def query(self, query_embedding: List[float], n_results: int = 4):
        return self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results, 
            include=["documents", "metadatas", "distances", "ids"]
        )

    def count(self):
        return self.collection.count()

    def list_collections(self):
        """List all collections"""
        return self.client.list_collections()

# Test code when run directly
if __name__ == "__main__":
    print("🧪 Testing VectorStore with NEW ChromaDB client")
    try:
        vs = VectorStore("test_collection")
        print(f"✅ Success! Collection count: {vs.count()}")
        print(f"✅ Collections: {[col.name for col in vs.list_collections()]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()