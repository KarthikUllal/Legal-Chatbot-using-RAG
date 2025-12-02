# backend/app/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict


# WHAT YOU SEND WHEN ADDING A DOCUMENT
class IngestPayload(BaseModel):
    doc_id: str  # Unique ID for this document (auto-generated)
    text: Optional[str] = None  # Direct text content (if no file)
    file_path: Optional[str] = None  # Path to PDF file (if uploading file)
    act_name: Optional[str] = (
        None  # Law name like "IPC" (optional - can be auto-detected)
    )


#  WHAT YOU SEND WHEN ASKING A QUESTION
class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # Chat session ID (for conversation history)
    question: str  # Your actual question
    top_k: int = 4  # How many reference sources to use


# INFORMATION ABOUT WHERE ANSWER CAME FROM
class SourceItem(BaseModel):
    id: str  # Which document was used
    source: Dict  # Extra info (page number, section, etc.)
    snippet: str  # The actual text used from document


# WHAT YOU GET BACK AS ANSWER
class ChatResponse(BaseModel):
    answer: str  # The actual answer to your question
    sources: List[SourceItem]  # Proof/references that support the answer


