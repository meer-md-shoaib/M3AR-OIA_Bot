"""
Presidency University - Office of International Affairs (OIA)
PDF Ingestion Pipeline: Extracts structured plain text from PDF documents in data/pdfs/ into scraped_data/
Supports both pypdf and pdfplumber with full page-by-page text extraction.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR = BASE_DIR / "scraped_data"


def extract_pdf_text_pypdf(pdf_path: Path) -> str:
    """Extracts text from all pages using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        extracted_pages = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                extracted_pages.append(f"--- [Page {page_idx}] ---\n{text.strip()}")
        return "\n\n".join(extracted_pages)
    except Exception as e:
        print(f"pypdf extraction warning for {pdf_path.name}: {e}")
        return ""


def extract_pdf_text_pdfplumber(pdf_path: Path) -> str:
    """Extracts text from all pages using pdfplumber."""
    try:
        import pdfplumber
        extracted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_pages.append(f"--- [Page {page_idx}] ---\n{page_text.strip()}")
        return "\n\n".join(extracted_pages)
    except Exception as e:
        print(f"pdfplumber extraction warning for {pdf_path.name}: {e}")
        return ""


def extract_pdf_text(pdf_path: Path) -> str:
    """Extracts raw text across all pages in a given PDF document using available libraries."""
    print(f"Reading PDF: {pdf_path.name} ...")
    # Try pypdf first
    content = extract_pdf_text_pypdf(pdf_path)
    if not content:
        # Fall back to pdfplumber
        content = extract_pdf_text_pdfplumber(pdf_path)
    return content


def ingest_all_pdfs():
    """Scans data/pdfs/ and extracts text into scraped_data/*.txt"""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files detected in {PDF_DIR}.")
        return []

    processed_outputs = []
    for pdf_file in pdf_files:
        content = extract_pdf_text(pdf_file)
        if content:
            out_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"
            out_file.write_text(content, encoding="utf-8")
            print(f"Extracted: {pdf_file.name} -> {out_file.name} ({len(content)} characters)")
            processed_outputs.append(out_file)
        else:
            print(f"Notice: No text content could be extracted from {pdf_file.name}")

    print(f"\nPDF Ingestion complete. Total processed: {len(processed_outputs)} file(s).")
    return processed_outputs


if __name__ == "__main__":
    ingest_all_pdfs()
