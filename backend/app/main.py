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
from .ingestion import fetch_html_text
from datetime import datetime
from fastapi.responses import FileResponse, Response
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import io

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal RAG Chatbot",
    description="AI-powered legal assistant using RAG technology",
    version="1.0.0",
)

# CORS middleware - FIXED VERSION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow ALL methods
    allow_headers=["*"],  # Allow ALL headers
)
app.include_router(admin_router)
# new router
app.include_router(news_router)

# Initialize RAG engine with best available provider
engine = get_rag_engine()


@app.get("/")
async def root():
    return {"message": "Legal RAG Chatbot API is running!"}


# To check whether all the endpoints works or not
@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        stats = engine.get_stats()
        return {
            "status": "healthy",
            "provider": type(engine.provider).__name__,
            "documents_count": stats.get("total_documents", 0),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# function to ingest file or pdf into vectordb
@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...), act_name: str = None):
    """
    Upload and ingest a PDF file into the knowledge base
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
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
            file_path=str(dst), doc_id=dst.stem, act_name=act_name or dst.stem
        )

        if success:
            return {
                "status": "success",
                "message": "File ingested successfully",
                "file": file.filename,
                "doc_id": dst.stem,
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


# To ingest text into vectordb
@app.post("/ingest-text")
async def ingest_text(payload: IngestPayload):
    """
    Ingest raw text content into the knowledge base
    """
    try:
        success = engine.ingest_text(
            doc_id=payload.doc_id,
            text=payload.text,
            metadata={"act": payload.act_name, "source_type": "direct_text"},
        )

        if success:
            return {
                "status": "success",
                "message": "Text ingested successfully",
                "doc_id": payload.doc_id,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest text")

    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# To add Sources from legal website just like how we do it for pdf upload
@app.post("/ingest-url")
async def ingest_url(url: str, doc_id: str, act_name: str = None):
    """Ingest legal content from a URL"""
    try:
        text = fetch_html_text(url)

        # Use existing ingest_text function
        success = engine.ingest_text(
            doc_id=doc_id,
            text=text,
            metadata={
                "act": act_name or doc_id,
                "source_type": "webpage",
                "url": url,
                "ingestion_date": datetime.now().isoformat(),
            },
        )

        return {"status": "success" if success else "failed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chat Route : For actual conversation
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, session_id: str = "default"):
    """
    Ask a legal question and get an AI-powered answer
    """
    try:
        logger.info(f"Processing chat request: {req.question[:50]}...")

        # Use the complete RAG pipeline (retrieve + generate)
        response = engine.query(
            question=req.question, top_k=req.top_k, session_id=session_id
        )

        logger.info(f"Chat response generated with {len(response.sources)} sources")
        return response

    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}"
        )


# To get System Stats
@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = engine.get_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# function to clear all docs from vector db
@app.delete("/clear")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base"""
    try:
        success = engine.clear_collection()
        if success:
            return {"status": "success", "message": "Knowledge base cleared"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to clear knowledge base"
            )
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


# langauage translation route
@app.post("/chat-translate", response_model=ChatResponse)
async def chat_with_translation(
    question: str, language: str = "en", top_k: int = 4, session_id: str = "default"
):
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
            question=question, language=language, top_k=top_k, session_id=session_id
        )

        logger.info(f"Generated response in {language}")
        return response

    except Exception as e:
        logger.error(f"Translation chat failed: {e}")
        error_msg = f"Failed to process question: {str(e)}"
        if language != "en":
            error_msg = translator.translate_legal_response(error_msg, language)
        raise HTTPException(status_code=500, detail=error_msg)


# voice
@app.post("/voice/process")
async def process_voice(audio_data: str, language: str = "en"):
    """
    Process voice audio and convert to text
    """
    try:
        text = voice_processor.process_audio(audio_data, language)
        return {"status": "success", "text": text, "language": language}
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/transcript/download/{session_id}")
async def download_transcript_txt(session_id: str):
    transcript = engine.get_full_transcript(session_id)

    if not transcript:
        raise HTTPException(status_code=404, detail="No transcript found")

    content = format_lawyer_transcript(transcript)
    filename = f"legal_consultation_{session_id}.txt"

    return Response(
        content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

    

#helper function
def format_lawyer_transcript(transcript: list) -> str:
    lines = []

    lines.append("LEGAL AI CHATBOT – CONSULTATION TRANSCRIPT")
    lines.append("Jurisdiction : India")
    lines.append(f"Generated On : {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    lines.append("=" * 70)

    for idx, turn in enumerate(transcript, start=1):
        lines.append(f"\nCONSULTATION QUERY {idx}")
        lines.append("-" * 30)
        lines.append("Client Query:")
        lines.append(turn["user_query"])

        lines.append("\nLegal Opinion:")
        lines.append(turn["legal_response"])

        lines.append("\nDisclaimer:")
        lines.append(
            "This response is generated by an AI legal assistant for informational "
            "purposes only and does not constitute professional legal advice."
        )

        lines.append("=" * 70)

    return "\n".join(lines)


@app.get("/transcript/download/pdf/{session_id}")
async def download_transcript_pdf(session_id: str):
    transcript = engine.get_full_transcript(session_id)

    if not transcript:
        raise HTTPException(status_code=404, detail="No transcript found")

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1*inch,
        leftMargin=1*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )

    styles = getSampleStyleSheet()

    # Custom styles (lawyer-grade)
    styles.add(ParagraphStyle(
        name="TitleStyle",
        fontName="Times-Bold",
        fontSize=14,
        spaceAfter=14,
        alignment=1  # Center
    ))

    styles.add(ParagraphStyle(
        name="HeadingStyle",
        fontName="Times-Bold",
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="BodyStyle",
        fontName="Times-Roman",
        fontSize=11,
        spaceAfter=10,
        leading=14
    ))

    styles.add(ParagraphStyle(
        name="DisclaimerStyle",
        fontName="Times-Italic",
        fontSize=9,
        textColor="grey",
        spaceBefore=12
    ))

    elements = []

    # ===== Document Header =====
    elements.append(Paragraph(
        "LEGAL AI CHATBOT – CONSULTATION TRANSCRIPT",
        styles["TitleStyle"]
    ))

    elements.append(Paragraph(
        "<b>Jurisdiction:</b> India<br/>"
        f"<b>Generated On:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        styles["BodyStyle"]
    ))

    elements.append(Spacer(1, 12))

    # ===== Transcript Content =====
    for idx, turn in enumerate(transcript, start=1):
        elements.append(Paragraph(
            f"CONSULTATION QUERY {idx}",
            styles["HeadingStyle"]
        ))

        elements.append(Paragraph(
            "<b>Client Query:</b><br/>" + turn["user_query"],
            styles["BodyStyle"]
        ))

        elements.append(Paragraph(
            "<b>Legal Opinion:</b><br/>" + turn["legal_response"].replace("\n", "<br/>"),
            styles["BodyStyle"]
        ))

        elements.append(Paragraph(
            "Disclaimer: This response is generated by an AI legal assistant for "
            "informational purposes only and does not constitute professional legal advice.",
            styles["DisclaimerStyle"]
        ))

        elements.append(Spacer(1, 20))

    doc.build(elements)

    buffer.seek(0)

    filename = f"legal_consultation_{session_id}.pdf"

    return Response(
        buffer.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )