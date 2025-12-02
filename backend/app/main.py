# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import logging
from .config import settings
from .schemas import ChatRequest, ChatResponse, IngestPayload, SourceItem
from .rag_engine import RAGEngine, get_rag_engine
from .provider_client import get_best_provider
from .admin import router as admin_router
from .translation import translator
from .voice_processor import voice_processor
from .news_routes import news_router

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal RAG Chatbot",
    description="AI-powered legal assistant using RAG technology",
    version="1.0.0"
)

# CORS middleware - FIXED VERSION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=True,
    allow_methods=["*"],   # Allow ALL methods
    allow_headers=["*"],   # Allow ALL headers
)
app.include_router(admin_router)
#new router
app.include_router(news_router)

# Initialize RAG engine with best available provider
engine = get_rag_engine()

@app.get("/")
async def root():
    return {"message": "Legal RAG Chatbot API is running!"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        stats = engine.get_stats()
        return {
            "status": "healthy",
            "provider": type(engine.provider).__name__,
            "documents_count": stats.get("total_documents", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...), act_name: str = None):
    """
    Upload and ingest a PDF file into the knowledge base
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    dst = data_dir / file.filename
    try:
        with dst.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {dst}")
        
        # Ingest file into RAG engine
        success = engine.ingest_file(
            file_path=str(dst),
            doc_id=dst.stem, 
            act_name=act_name or dst.stem
        )
        
        if success:
            return {
                "status": "success", 
                "message": "File ingested successfully",
                "file": file.filename,
                "doc_id": dst.stem
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest file")
            
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        # Clean up uploaded file if ingestion fails
        try:
            if dst.exists():
                dst.unlink()
        except Exception as cleanup_error:
            logger.error(f"File cleanup failed: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/ingest-text")
async def ingest_text(payload: IngestPayload):
    """
    Ingest raw text content into the knowledge base
    """
    try:
        success = engine.ingest_text(
            doc_id=payload.doc_id,
            text=payload.text,
            metadata={"act": payload.act_name, "source_type": "direct_text"}
        )
        
        if success:
            return {
                "status": "success",
                "message": "Text ingested successfully",
                "doc_id": payload.doc_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest text")
            
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask a legal question and get an AI-powered answer
    """
    try:
        logger.info(f"Processing chat request: {req.question[:50]}...")
        
        # Use the complete RAG pipeline (retrieve + generate)
        response = engine.query(question=req.question, top_k=req.top_k)
        
        logger.info(f"Chat response generated with {len(response.sources)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = engine.get_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base"""
    try:
        success = engine.clear_collection()
        if success:
            return {"status": "success", "message": "Knowledge base cleared"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
    except Exception as e:
        logger.error(f"Clear operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Legal RAG Chatbot API starting up...")
    logger.info(f"Using provider: {type(engine.provider).__name__}")
    stats = engine.get_stats()
    logger.info(f"Initial documents count: {stats.get('total_documents', 0)}")

#langauage translation route
@app.post("/chat-translate", response_model=ChatResponse)
async def chat_with_translation(question: str, language: str = "en", top_k: int = 4):
    """
    Chat endpoint with language translation
    - question: User's question in any language
    - language: Language code for response (en, hi, kn, etc.)
    - top_k: Number of sources to use
    """
    try:
        logger.info(f"Translation chat: '{question[:50]}...' in {language}")
        
        # Use the simple translation approach
        response = engine.query_with_language(
            question=question, 
            language=language, 
            top_k=top_k
        )
        
        logger.info(f"Generated response in {language}")
        return response
        
    except Exception as e:
        logger.error(f"Translation chat failed: {e}")
        error_msg = f"Failed to process question: {str(e)}"
        if language != "en":
            error_msg = translator.translate_legal_response(error_msg, language)
        raise HTTPException(status_code=500, detail=error_msg)
    
#voice 
@app.post("/voice/process")
async def process_voice(audio_data: str, language: str = "en"):
    """
    Process voice audio and convert to text
    """
    try:
        text = voice_processor.process_audio(audio_data, language)
        return {
            "status": "success",
            "text": text,
            "language": language
        }
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages for voice recognition
    """
    return {
        "status": "success",
        "languages": voice_processor.supported_languages
    }

