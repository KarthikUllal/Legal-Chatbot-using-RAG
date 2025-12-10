# Check how many pages your PDF has and what's on later pages
from pypdf import PdfReader
from pathlib import Path

bns_path = Path("./datas/bharathiya nyaya sanhita.pdf")
if bns_path.exists():
    reader = PdfReader(str(bns_path))
    
    print(f"📄 BNS PDF has {len(reader.pages)} pages")
    
    # Check different sections of the PDF
    page_samples = [
        (1, "First page"),
        (5, "Page 5"),
        (10, "Page 10"),
        (20, "Page 20"),
        (50, "Page 50"),
        (100, "Page 100")
    ]
    
    for page_num, desc in page_samples:
        if page_num < len(reader.pages):
            text = reader.pages[page_num].extract_text()
            print(f"\n{'='*60}")
            print(f"{desc} (Page {page_num}):")
            print(f"Text length: {len(text)} chars")
            print(f"Preview: {text[:300]}...")
            
            # Check content type
            if "section 304" in text.lower() and "whoever" in text.lower():
                print("✅ Contains ACTUAL LAW TEXT for Section 304!")
            elif "section" in text.lower() and len(text) < 500:
                print("⚠️ Likely TABLE OF CONTENTS")
            else:
                print("📝 Other content")