# backend/app/admin.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# ✅ LAZY INITIALIZATION - No circular imports
def get_engine():
    from .rag_engine import get_rag_engine

    return get_rag_engine()


@router.get("/stats")
async def get_admin_stats(engine=Depends(get_engine)):
    """Get detailed admin statistics"""
    try:
        logger.info("Admin stats endpoint called")
        stats = engine.get_stats()
        logger.info(f"Engine stats: {stats}")

        response_data = {
            "status": "success",
            "data": {
                "total_documents": stats.get("total_documents", 0),
                "vector_store_size": "N/A",
                "rag_engine_status": "active",
                "embedding_model": stats.get("embedding_model", "Unknown"),
                "llm_provider": stats.get("provider", "Unknown"),
            },
        }
        logger.info(f"Sending response: {response_data}")
        return response_data

    except Exception as e:
        logger.error(f"Admin stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/documents")
async def debug_documents(engine=Depends(get_engine)):
    """Debug endpoint to see what's actually in the vector store"""
    try:
        collection = engine.vstore.collection
        results = collection.get(include=["metadatas", "documents"])

        docs_info = {}
        for i, metadata in enumerate(results["metadatas"]):
            if metadata:
                doc_id = metadata.get("doc_id", "unknown")
                if doc_id not in docs_info:
                    docs_info[doc_id] = {
                        "name": metadata.get("act", doc_id),
                        "chunks": 0,
                        "sample_content": (
                            results["documents"][i][:100] + "..."
                            if i < len(results["documents"]) and results["documents"][i]
                            else "empty"
                        ),
                    }
                docs_info[doc_id]["chunks"] += 1

        return {
            "status": "success",
            "documents_found": docs_info,
            "total_chunks": len(results["metadatas"]),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/debug/retrieve/{query}")
async def debug_retrieval(query: str, engine=Depends(get_engine)):
    """Debug what documents are retrieved for a query"""
    try:
        collection = engine.vstore.collection
        query_embedding = engine.provider.get_embeddings([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=8,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_info = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                retrieved_info.append(
                    {
                        "rank": i + 1,
                        "distance": round(distance, 4),
                        "doc_id": meta.get("doc_id", "unknown") if meta else "unknown",
                        "act": meta.get("act", "unknown") if meta else "unknown",
                        "content_preview": (
                            doc[:200] + "..." if doc and len(doc) > 200 else doc
                        ),
                        "contains_420": "420" in doc.lower() if doc else False,
                        "contains_section": "section" in doc.lower() if doc else False,
                        "contains_ipc": "ipc" in doc.lower() if doc else False,
                    }
                )

        return {
            "status": "success",
            "query": query,
            "total_retrieved": len(retrieved_info),
            "retrieved_documents": retrieved_info,
        }

    except Exception as e:
        logger.error(f"Debug retrieval failed: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/documents")
async def list_documents(engine=Depends(get_engine)):
    """List all ingested documents"""
    try:
        # This would need enhancement to track documents properly
        return {
            "status": "success",
            "documents": [
                {
                    "id": "ipc",
                    "name": "Indian Penal Code",
                    "status": "ingested",
                    "chunks": 100,
                    "ingestion_date": "2025-01-01",
                }
            ],
        }
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, engine=Depends(get_engine)):
    """Delete a specific document"""
    try:
        return {
            "status": "success",
            "message": f"Document {doc_id} deletion endpoint - implement based on your vector store",
        }
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/debug/find-section/{section_number}")
async def find_section(section_number: str, engine=Depends(get_engine)):
    """Find specific section content"""
    collection = engine.vstore.collection
    results = collection.get(include=["metadatas", "documents"])
    
    section_matches = []
    search_terms = [
        f"Section {section_number}",
        f"section {section_number}",
        f"{section_number}.",
        f" {section_number} "
    ]
    
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        doc_lower = doc.lower()
        for term in search_terms:
            if term.lower() in doc_lower:
                section_matches.append({
                    "chunk_id": i,
                    "doc_id": meta.get("doc_id", "unknown"),
                    "term_found": term,
                    "preview": doc[:400] + "...",
                })
                break
    
    return {
        "section": section_number,
        "total_chunks_searched": len(results["documents"]),
        "matches_found": len(section_matches),
        "matches": section_matches
    }
