"""
Presidency University - Office of International Affairs (OIA)
Image Matcher: Dynamically associates user student queries with verified campus, office, and collaboration photos.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "Photos"
MANIFEST_PATH = PHOTOS_DIR / "image_manifest.json"

def load_manifest() -> List[Dict[str, Any]]:
    """Loads image metadata manifest handling standard UTF-8 and BOM."""
    if not MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
        return data.get("images", [])
    except Exception as e:
        print(f"Warning: Failed to load image manifest ({e}).")
        return []

def extract_tokens(text: str) -> set:
    """Extracts lowercase alphabetic keywords."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {
        "the", "and", "for", "with", "that", "this", "what", "how", "can",
        "are", "have", "from", "you", "your", "about", "tell", "show", "give", "please"
    }
    return {w for w in words if w not in stopwords}

def find_relevant_images(user_query: str, max_images: int = 2) -> List[str]:
    """
    Scores each image in Photos/ against the user query based on topics, tags, title, and description.
    Returns list of absolute or relative file path strings.
    """
    manifest = load_manifest()
    if not manifest:
        return []

    query_tokens = extract_tokens(user_query)
    if not query_tokens:
        return []

    scored_images = []
    for item in manifest:
        score = 0
        filename = item.get("filename", "")
        img_path = PHOTOS_DIR / filename

        # Skip if image file does not physically exist
        if not img_path.exists():
            continue

        topics = [t.lower() for t in item.get("topics", [])]
        tags = [t.lower() for t in item.get("tags", [])]
        title = item.get("title", "").lower()
        desc = item.get("description", "").lower()

        for token in query_tokens:
            for topic in topics:
                if token in topic:
                    score += 4
            for tag in tags:
                if token in tag:
                    score += 3
            if token in title:
                score += 2
            if token in desc:
                score += 1

        if score > 0:
            scored_images.append((score, str(img_path)))

    scored_images.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in scored_images[:max_images]]

if __name__ == "__main__":
    test_queries = [
        "Where is the Office of International Affairs located and what are the office hours?",
        "Tell me about global partner universities and MoU collaborations",
        "What is the Taiwan Education Center at Presidency University?",
        "Where can international students wait or relax on campus?",
        "How do I consult an international advisor for student visa and FRRO?"
    ]

    print("=== Testing Image Matcher Standalone ===")
    for q in test_queries:
        matched = find_relevant_images(q, max_images=2)
        print(f"\nQuery: '{q}'")
        if matched:
            for p in matched:
                print(f"  -> Match: {Path(p).name} ({p})")
        else:
            print("  -> No matching photo")
