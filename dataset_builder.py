"""
Presidency University - Office of International Affairs (OIA)
Fine-Tuning Dataset Builder: Converts knowledge base and ingested text files into high-quality instruction-tuning JSONL pairs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent
SCRAPED_DIR = BASE_DIR / "scraped_data"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "finetune_dataset.jsonl"

try:
    from knowledge import KNOWLEDGE_ENTRIES
except ImportError:
    KNOWLEDGE_ENTRIES = []

def build_qa_from_knowledge() -> List[Dict]:
    """Generates prompt-completion pairs from knowledge.py entries."""
    pairs = []
    system_intro = (
        "You are the official AI Counselor for the Office of International Affairs (OIA) at Presidency University, Bengaluru. "
        "Provide a precise, warm, and helpful answer grounded in official university guidelines."
    )

    for entry in KNOWLEDGE_ENTRIES:
        category = entry.get("category", "General")
        title = entry.get("title", "")
        content = entry.get("content", "")
        keywords = entry.get("keywords", [])

        # Direct topical question
        pairs.append({
            "prompt": f"{system_intro}\n\nStudent Inquiry: What are the official guidelines regarding {title.lower()}?",
            "completion": content
        })

        # Practical action questions based on category
        if "Visa" in category or "FRRO" in category:
            pairs.append({
                "prompt": f"{system_intro}\n\nStudent Inquiry: How does Presidency University assist international students with student visa and FRRO registration?",
                "completion": content
            })
        elif "Admissions" in category:
            pairs.append({
                "prompt": f"{system_intro}\n\nStudent Inquiry: Can you list the mandatory document checklist and eligibility criteria for foreign applicants?",
                "completion": content
            })
        elif "Scholarships" in category:
            pairs.append({
                "prompt": f"{system_intro}\n\nStudent Inquiry: Are there government or university scholarships like ICCR and Study In India for foreign students?",
                "completion": content
            })
        elif "Fees" in category:
            pairs.append({
                "prompt": f"{system_intro}\n\nStudent Inquiry: What are the accepted fee payment modes and wire transfer details for international students?",
                "completion": content
            })
        elif "Accommodation" in category:
            pairs.append({
                "prompt": f"{system_intro}\n\nStudent Inquiry: What accommodation, dining, and hostel facilities are provided for international students on campus?",
                "completion": content
            })

    return pairs

def build_qa_from_text(text: str, source_topic: str) -> List[Dict]:
    """Splits ingested plain text into informative chunks and builds prompt-completion pairs."""
    chunks = [c.strip() for c in re.split(r'\n{2,}', text) if len(c.strip()) > 70]
    pairs = []
    system_intro = (
        "You are the official AI Counselor for the Office of International Affairs (OIA) at Presidency University, Bengaluru. "
        "Answer the following question about " + source_topic + " accurately:"
    )

    for chunk in chunks:
        # Create student inquiry from the first line or topic
        lines = chunk.split("\n")
        first_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else chunk

        # Clean prompt topic
        inquiry_topic = re.sub(r'^[0-9\.\-\*\#\s]+', '', first_line)[:100]
        if not inquiry_topic or len(inquiry_topic) < 5:
            inquiry_topic = f"details regarding {source_topic}"

        pairs.append({
            "prompt": f"{system_intro}\n\nStudent Inquiry: Could you provide information on {inquiry_topic}?",
            "completion": chunk
        })

    return pairs

def build_dataset():
    """Compiles all data sources into data/finetune_dataset.jsonl"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    # 1. Add structured knowledge entries
    knowledge_pairs = build_qa_from_knowledge()
    all_pairs.extend(knowledge_pairs)
    print(f"Loaded {len(knowledge_pairs)} pairs from knowledge.py.")

    # 2. Add scraped and ingested text files
    if SCRAPED_DIR.exists():
        for txt_file in SCRAPED_DIR.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            source_topic = txt_file.stem.replace("_", " ").title()
            file_pairs = build_qa_from_text(text, source_topic)
            all_pairs.extend(file_pairs)
            print(f"Loaded {len(file_pairs)} pairs from {txt_file.name}.")

    # 3. Write out to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully compiled {len(all_pairs)} instruction-tuning pairs to {OUTPUT_FILE}")
    return OUTPUT_FILE

if __name__ == "__main__":
    build_dataset()
