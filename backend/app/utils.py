# backend/app/utils.py
import re
import pymupdf

DISCLAIMER = (
    "This application provides general legal information only and is NOT a substitute for professional legal advice."
)\
#to clean extracted text from each pages
def clean_text(s: str) -> str:
    return " ".join(s.split())


def extract_clean_text_from_pdf(pdf_path: str) -> str:
    """
    Extract and clean text from PDF while handling PDF-specific issues.
    """
    import pdfplumber
    
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # PDF-aware text extraction
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    
    full_text = "\n".join(text_chunks)
    # Now you might need PDF-specific cleaning, not generic clean_text()
    return clean_pdf_text(full_text)

def clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDFs - handles PDF-specific issues.
    """
    # Fix common PDF extraction problems:
    # 1. Remove page numbers/headers
    # 2. Fix hyphenated words across line breaks
    # 3. Handle multi-column text ordering
    # 4. Remove PDF artifacts
    text = re.sub(r'-\n', '', text)  # Fix line-break hyphens
    text = re.sub(r'\n+', '\n', text)  # Reduce multiple newlines
    return text.strip()

