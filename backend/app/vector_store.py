# backend/app/vector_store.py
from typing import List, Dict
import chromadb

try:
    from .config import settings
except ImportError:
    from config import settings

class VectorStore:
    def __init__(self, collection_name: str = "indian_laws"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.col_name = collection_name
        
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name, 
                metadata={"source": "legal_docs"}
            )

    def add(self, ids: List[str], docs: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add documents to the collection"""
        self.collection.add(
            ids=ids, 
            documents=docs, 
            embeddings=embeddings, 
            metadatas=metadatas
        )

    def query(self, query_embedding: List[float], n_results: int = 4) -> Dict:
        """Query the vector store """
        return self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # changes on 6 dec 2026 time : 3.11 pm
        )

    def count(self):
        return self.collection.count()

    def list_collections(self):
        return self.client.list_collections()

