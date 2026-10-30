"""
Presidency University - Office of International Affairs (OIA)
Image Embeddings & Multimodal Understanding Engine.

Capabilities:
1. [IMAGE EMBEDDINGS]: Generates dense 256-dimensional normalized visual feature embeddings for images.
2. [IMAGE-TO-TEXT DECODING]: Matches uploaded images against verified campus/office photos via cosine similarity
   and extracts OCR text / semantic descriptions.
3. [IMAGE-TO-RAG MATCHING]: Decodes uploaded images into text queries and retrieves official matching university results.
4. [TEXT-TO-IMAGE RETRIEVAL]: Dynamically retrieves verified campus photos when a student asks text queries.
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "Photos"
MANIFEST_PATH = PHOTOS_DIR / "image_manifest.json"

# Try importing pytesseract for optional OCR text decoding
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def load_image(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """Converts diverse image input formats into an RGB PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        return Image.open(p).convert("RGB")
    elif isinstance(image_input, bytes):
        import io
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def compute_image_embedding(image_input: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
    """
    [VISUAL IMAGE EMBEDDING]
    Computes a dense 256-dimensional normalized visual feature embedding.
    Combines:
    - 64-dim spatial grid color luminance (8x8 spatial grid)
    - 96-dim 3-channel color histogram (RGB distribution)
    - 96-dim spatial edge gradient magnitude distribution
    Normalized to unit L2 norm so dot product equals cosine similarity.
    """
    img = load_image(image_input)
    # Standardize scale
    resized = img.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # (64, 64, 3)

    # 1. 8x8 spatial grid average luminance (64 dims)
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    grid_features = gray.reshape(8, 8, 8, 8).mean(axis=(1, 3)).flatten()  # 64 dims

    # 2. RGB color histogram (32 bins per channel = 96 dims)
    r_hist, _ = np.histogram(arr[:, :, 0], bins=32, range=(0.0, 1.0), density=True)
    g_hist, _ = np.histogram(arr[:, :, 1], bins=32, range=(0.0, 1.0), density=True)
    b_hist, _ = np.histogram(arr[:, :, 2], bins=32, range=(0.0, 1.0), density=True)
    color_features = np.concatenate([r_hist, g_hist, b_hist])  # 96 dims

    # 3. Spatial horizontal & vertical edge gradients (96 dims)
    dx = np.diff(gray, axis=1)  # (64, 63)
    dy = np.diff(gray, axis=0)  # (63, 64)
    dx_hist, _ = np.histogram(dx, bins=48, range=(-0.5, 0.5), density=True)
    dy_hist, _ = np.histogram(dy, bins=48, range=(-0.5, 0.5), density=True)
    edge_features = np.concatenate([dx_hist, dy_hist])  # 96 dims

    # Combine into 256-dim embedding vector
    combined = np.concatenate([grid_features, color_features, edge_features])
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined = combined / norm

    return combined.astype(np.float32)


class ImageEmbeddingStore:
    """
    Vector store for campus photo visual embeddings.
    Indexes all photos in Photos/ alongside their manifest metadata.
    """

    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._index_catalog()

    def _index_catalog(self):
        """Indexes all verified photos from Photos/ and image_manifest.json."""
        manifest_items = {}
        if MANIFEST_PATH.exists():
            try:
                data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
                for item in data.get("images", []):
                    manifest_items[item.get("filename")] = item
            except Exception as e:
                print(f"Warning loading manifest in ImageEmbeddingStore: {e}")

        # Index all JPEG/PNG images in Photos/
        for img_p in list(PHOTOS_DIR.glob("*.jpeg")) + list(PHOTOS_DIR.glob("*.jpg")) + list(PHOTOS_DIR.glob("*.png")):
            try:
                emb = compute_image_embedding(img_p)
                self.embeddings[img_p.name] = emb

                meta = manifest_items.get(img_p.name, {
                    "filename": img_p.name,
                    "title": img_p.stem.replace("_", " ").title(),
                    "description": f"Verified Presidency University campus photograph: {img_p.stem.replace('_', ' ').title()}",
                    "tags": [img_p.stem.lower()],
                    "topics": ["campus", "presidency university"]
                })
                meta["path"] = str(img_p)
                self.metadata[img_p.name] = meta
            except Exception as err:
                print(f"Error computing visual embedding for {img_p.name}: {err}")

        print(f"Image Embedding Store initialized! Indexed {len(self.embeddings)} visual vectors.")

    def search_by_image(
        self,
        query_image: Union[str, Path, bytes, Image.Image],
        top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        [IMAGE-TO-IMAGE SEARCH]
        Matches an uploaded image against the catalog using cosine similarity of visual embeddings.
        """
        query_emb = compute_image_embedding(query_image)
        scored = []
        for filename, emb in self.embeddings.items():
            cosine_sim = float(np.dot(query_emb, emb))
            scored.append((cosine_sim, self.metadata[filename]))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, meta in scored[:top_k]:
            res = dict(meta)
            res["similarity_score"] = sim
            results.append(res)
        return results

    def search_by_text(self, text_query: str, max_images: int = 2) -> List[Dict[str, Any]]:
        """
        [TEXT-TO-IMAGE SEARCH]
        Finds matching campus photos based on query keywords matching topics, tags, and descriptions.
        """
        import re
        tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', text_query.lower()))
        stopwords = {"the", "and", "for", "with", "that", "this", "what", "how", "can", "are", "have", "from", "you", "your", "about", "tell", "show"}
        keywords = tokens - stopwords

        scored = []
        for filename, meta in self.metadata.items():
            score = 0
            topics = [t.lower() for t in meta.get("topics", [])]
            tags = [t.lower() for t in meta.get("tags", [])]
            title = meta.get("title", "").lower()
            desc = meta.get("description", "").lower()

            for kw in keywords:
                for topic in topics:
                    if kw in topic:
                        score += 4
                for tag in tags:
                    if kw in tag:
                        score += 3
                if kw in title:
                    score += 2
                if kw in desc:
                    score += 1

            if score > 0:
                res = dict(meta)
                res["text_score"] = score
                scored.append((score, res))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:max_images]]


_GLOBAL_IMAGE_STORE: Optional[ImageEmbeddingStore] = None

def get_image_store() -> ImageEmbeddingStore:
    """Returns singleton instance of the ImageEmbeddingStore."""
    global _GLOBAL_IMAGE_STORE
    if _GLOBAL_IMAGE_STORE is None:
        _GLOBAL_IMAGE_STORE = ImageEmbeddingStore()
    return _GLOBAL_IMAGE_STORE


# ==============================================================================
# IMAGE-TO-TEXT DECODING & MATCHING PIPELINE
# ==============================================================================

def decode_image_to_text(image_input: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
    """
    [DECODE IMAGE TO TEXT]
    Decodes an uploaded image into semantic text by:
    1. Matching visual embeddings against campus catalog photos (cosine similarity).
    2. Extracting OCR text if recognizable text appears in the image.
    3. Returning structured title, description, topics, and a decoded search prompt.
    """
    store = get_image_store()
    matches = store.search_by_image(image_input, top_k=1)
    best_match = matches[0] if matches else None

    # Optional OCR text extraction
    ocr_text = ""
    if HAS_PYTESSERACT:
        try:
            pil_img = load_image(image_input)
            ocr_text = pytesseract.image_to_string(pil_img).strip()
        except Exception:
            ocr_text = ""

    if best_match:
        title = best_match.get("title", "Campus Visual")
        desc = best_match.get("description", "")
        topics = ", ".join(best_match.get("topics", []))
        similarity = best_match.get("similarity_score", 0.0)

        # Formulate rich text query for ChromaDB RAG
        decoded_query = f"{title}. {desc} Topics: {topics}"
        if ocr_text:
            decoded_query += f" Extracted Text: {ocr_text}"

        return {
            "title": title,
            "description": desc,
            "matched_filename": best_match.get("filename"),
            "matched_path": best_match.get("path"),
            "similarity_score": similarity,
            "ocr_text": ocr_text,
            "decoded_query": decoded_query
        }
    else:
        return {
            "title": "Unidentified Image",
            "description": "General campus visual",
            "matched_filename": None,
            "similarity_score": 0.0,
            "ocr_text": ocr_text,
            "decoded_query": "Presidency University Office of International Affairs general information"
        }


def process_uploaded_image_rag(image_input: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
    """
    [COMPLETE MULTIMODAL PIPELINE]
    1. Upload Image
    2. Visual Embedding & Decode to Text
    3. ChromaDB Vector DB Retrieval
    4. Grounded Response Generation
    """
    from bot_engine import InternationalOfficeBot

    # Step 1 & 2: Decode image to text using visual embeddings
    decoded = decode_image_to_text(image_input)
    decoded_text = decoded["decoded_query"]

    # Step 3 & 4: Query RAG pipeline using decoded text
    bot = InternationalOfficeBot()
    response = bot.generate_response(user_message=decoded_text)

    # Attach matched image
    matched_img_path = decoded.get("matched_path")

    return {
        "decoded_image": decoded,
        "decoded_text": decoded_text,
        "matched_photo": matched_img_path,
        "counselor_response": response
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING IMAGE EMBEDDINGS & IMAGE-TO-TEXT DECODING PIPELINE")
    print("=" * 70)

    store = get_image_store()
    test_img = PHOTOS_DIR / "3.jpeg"  # Taiwan Education Center (TECC)

    print(f"\n1. Testing Image-to-Text on: {test_img.name}")
    decoded = decode_image_to_text(test_img)
    print(f"   Decoded Title:       {decoded['title']}")
    print(f"   Visual Match Score:  {decoded['similarity_score']:.4f}")
    print(f"   Decoded Query Text:  {decoded['decoded_query'][:120]}...\n")

    print("2. Testing Multimodal Image-to-RAG Matching Results:")
    result = process_uploaded_image_rag(test_img)
    print("--- Matching Results from ChromaDB RAG ---")
    print(result["counselor_response"][:300] + "...")
    print("=" * 70)
