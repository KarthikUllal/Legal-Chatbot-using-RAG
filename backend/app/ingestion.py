# backend/app/ingestion.py
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
from typing import List
import concurrent.futures
from .config import settings
from .utils import clean_text


def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return clean_text("\n\n".join(pages))


def fetch_html_text(url: str) -> str:
    # STEP 1: Download the webpage
    resp = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": "LegalResearchBot/1.0 (+https://yourdomain.com)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        },
    )
    # Gets raw HTML like: <html><body><p>Hello World</p><script>...</script></body></html>

    # STEP 2: Check if download succeeded
    resp.raise_for_status()  # Crash if 404, 500 errors, etc.

    # STEP 3: Parse HTML into a structured tree
    soup = BeautifulSoup(resp.text, "html.parser")
    # Now we can navigate the HTML like a tree structure

    # STEP 4: Remove unwanted elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()  # Completely delete these tags
    # Removes: JavaScript code, CSS styles, hidden content

    # STEP 5: Extract only the visible text
    text = soup.get_text(separator="\n")
    # Converts: <p>Hello</p><p>World</p> → "Hello\nWorld"

    # STEP 6: Clean up extra spaces
    return clean_text(text)


def split_into_chunks(
    text: str, chunk_size: int = None, overlap: int = None
) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cs = chunk_size or settings.CHUNK_SIZE
    co = overlap or settings.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co)
    return splitter.split_text(text)


def process_multiple_documents_parallel(file_paths: List[Path]) -> List[str]:
    """
    Process multiple documents simultaneously using parallel processing.

    Args:
        file_paths (List[Path]): List of paths to PDF files

    Returns:
        List[str]: All chunks from all processed documents

    Example:
        chunks = process_multiple_documents_parallel([
            Path("doc1.pdf"),
            Path("doc2.pdf"),
            Path("doc3.pdf")
        ])
    """
    all_chunks = []

    def process_single_document(file_path):
        """Process one PDF file and return chunks"""
        text = load_pdf_text(file_path)
        chunks = split_into_chunks(text)
        return chunks

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all documents for processing
        future_to_file = {
            executor.submit(process_single_document, file_path): file_path
            for file_path in file_paths
        }

        # Wait for completion and collect results
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                print(f"✅ Completed: {file_path.name} - {len(chunks)} chunks")
            except Exception as e:
                print(f"❌ Failed: {file_path.name} - {e}")

    return all_chunks


def process_multiple_documents_sequential(file_paths: List[Path]) -> List[str]:
    """
    Process multiple documents one after another (more memory efficient).

    Args:
        file_paths (List[Path]): List of paths to PDF files

    Returns:
        List[str]: All chunks from all processed documents
    """
    all_chunks = []

    for file_path in file_paths:
        try:
            print(f"🔄 Processing: {file_path.name}")
            text = load_pdf_text(file_path)
            chunks = split_into_chunks(text)
            all_chunks.extend(chunks)
            print(f"✅ Completed: {file_path.name} - {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Failed: {file_path.name} - {e}")

    return all_chunks


# Helper function for admin usage
def get_processing_stats(chunks: List[str]) -> dict:
    """
    Get statistics about processed chunks.

    Args:
        chunks (List[str]): List of text chunks

    Returns:
        dict: Processing statistics
    """
    total_chunks = len(chunks)
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chunk_size = total_chars // total_chunks if total_chunks > 0 else 0

    return {
        "total_chunks": total_chunks,
        "total_characters": total_chars,
        "average_chunk_size": avg_chunk_size,
        "chunk_size_range": (
            f"{min(len(chunk) for chunk in chunks)}-{max(len(chunk) for chunk in chunks)}"
            if chunks
            else "0-0"
        ),
    }

#testing
if __name__ == "__main__":
    print("✅ Ingestion module loaded successfully!")
    print("Available functions:")
    print(" - load_pdf_text(path)")
    print(" - fetch_html_text(url)") 
    print(" - split_into_chunks(text)")