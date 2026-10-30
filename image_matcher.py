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
    """Extracts lowercase alphabetic keywords with basic singular normalization."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {
        "the", "and", "for", "with", "that", "this", "what", "how", "can",
        "are", "have", "from", "you", "your", "about", "tell", "show", "give", "please", "some"
    }
    tokens = set()
    for w in words:
        if w in stopwords:
            continue
        tokens.add(w)
        # Add singular form for simple plurals
        if w.endswith("s") and len(w) > 3:
            tokens.add(w[:-1])
        if w.endswith("ies") and len(w) > 4:
            tokens.add(w[:-3] + "y")
    return tokens

def find_relevant_images(user_query: str, max_images: int = 3) -> List[str]:
    """
    Scores each image in Photos/ against the user query based on topics, tags, title, and description.
    Supports broad photo queries like 'show campus photos' or 'all pics'.
    Returns list of absolute or relative file path strings.
    """
    manifest = load_manifest()
    if not manifest:
        return []

    q_lower = user_query.lower()
    query_tokens = extract_tokens(user_query)

    is_general_photo_req = any(w in q_lower for w in ["photo", "photos", "pic", "pics", "picture", "pictures", "image", "images", "gallery"])

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

        # Score matching tokens
        for token in query_tokens:
            for topic in topics:
                if token in topic:
                    score += 5
            for tag in tags:
                if token in tag:
                    score += 4
            if token in title:
                score += 3
            if token in desc:
                score += 1

        # Specific landmark / category bonuses
        if any(w in q_lower for w in ["confluence", "atrium", "convention", "global minds"]) and any(x in filename for x in ["6.", "9.", "12.", "13."]):
            score += 15
        if any(w in q_lower for w in ["flag", "flags", "country", "countries", "nation", "nations"]) and filename == "8.jpeg":
            score += 20
        if any(w in q_lower for w in ["boardroom", "delegation", "council", "meeting", "bilateral", "table"]) and any(x in filename for x in ["7.", "10.", "11."]):
            score += 15
        if any(w in q_lower for w in ["courtyard", "patio", "exterior", "building", "outside", "garden", "lawn"]) and any(x in filename for x in ["1.", "14.", "15.", "16."]):
            score += 12
        if any(w in q_lower for w in ["award", "ficci", "ranking", "rankings", "excellence", "trophy", "winner"]) and filename in ["campus_banner_2.jpeg", "presidency_campus_bg.jpeg"]:
            score += 20
        if any(w in q_lower for w in ["main gate", "portal", "b-school", "management", "ghrdc"]) and filename == "presidency_campus_bg.jpeg":
            score += 20

        # General photo boost if user asked for photos, gallery or all pics
        if is_general_photo_req:
            highlight_samples = [
                "1.jpeg", "6.jpeg", "8.jpeg", "campus_banner_2.jpeg",
                "3.jpeg", "15.jpeg", "11.jpeg", "presidency_campus_bg.jpeg"
            ]
            if filename in highlight_samples:
                score += (10 - highlight_samples.index(filename))

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
