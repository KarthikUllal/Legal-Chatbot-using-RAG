# backend/app/multilingual.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/multilingual", tags=["multilingual"])

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    target_language: str

# Simple translation mapping for demo - in production, use proper translation API
DEMO_TRANSLATIONS = {
    "hi": {  # Hindi
        "hello": "नमस्ते",
        "what is murder": "हत्या क्या है",
        "consumer rights": "उपभोक्ता अधिकार",
        "cyber crime": "साइबर अपराध"
    },
    "ta": {  # Tamil
        "hello": "வணக்கம்", 
        "what is murder": "கொலை என்றால் என்ன",
        "consumer rights": "நுகர்வோர் உரிமைகள்",
        "cyber crime": "சைபர் குற்றம்"
    },
    "te": {  # Telugu
        "hello": "హలో",
        "what is murder": "హత్య అంటే ఏమిటి",
        "consumer rights": "గ్రాహక హక్కులు", 
        "cyber crime": "సైబర్ క్రైమ్"
    }
}

@router.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text to target language (demo implementation)"""
    try:
        text_lower = request.text.lower()
        target_lang = request.target_language
        
        if target_lang in DEMO_TRANSLATIONS:
            translated = DEMO_TRANSLATIONS[target_lang].get(text_lower, f"[Translation: {request.text}]")
        else:
            translated = f"[Translation for {target_lang}: {request.text}]"
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated,
            target_language=target_lang
        )
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": [
            {"code": "en", "name": "English", "native_name": "English"},
            {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"},
            {"code": "ta", "name": "Tamil", "native_name": "தமிழ்"},
            {"code": "te", "name": "Telugu", "native_name": "తెలుగు"},
            {"code": "bn", "name": "Bengali", "native_name": "বাংলা"},
            {"code": "mr", "name": "Marathi", "native_name": "मराठी"}
        ]
    }