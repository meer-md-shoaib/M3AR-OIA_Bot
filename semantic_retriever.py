"""
Presidency University - Office of International Affairs (OIA)
Semantic Retriever: Multi-source dense vector retriever powered by Sentence Transformers and FAISS.
Indexes structured knowledge entries alongside ingested PDF guides and web-scraped collaborations.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent
SCRAPED_DIR = BASE_DIR / "scraped_data"

# 1. Safely import structured entries from knowledge.py
try:
    from knowledge import KNOWLEDGE_ENTRIES
except ImportError:
    KNOWLEDGE_ENTRIES = [
        {
            "title": "Indian Student Visa (S-Visa) Guidelines",
            "content": "International students must enter India strictly on a valid 'Student Visa' (S-Visa) explicitly endorsing Presidency University."
        },
        {
            "title": "e-FRRO Compliance",
            "content": "All foreign national students must register with the FRRO within 14 days of landing in India via indianfrro.gov.in."
        },
        {
            "title": "Office of International Affairs Contact Details",
            "content": "Working Hours: Monday to Friday, 9:00 AM - 5:00 PM IST. Email: international@presidencyuniversity.in"
        }
    ]

# Try importing dense vector dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class SemanticRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the multi-source vector corpus and indexes entries."""
        print("Initializing Semantic Retriever...")
        self.corpus: List[str] = []
        self.model = None
        self.index = None
        self.corpus_embeddings = None

        # 1. Build Multi-Source Corpus from knowledge.py + scraped_data/*.txt
        self._build_corpus()

        # 2. Initialize SentenceTransformer if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print(f"Loading embedding model: {model_name} ...")
                self.model = SentenceTransformer(model_name)
                print(f"Encoding {len(self.corpus)} multi-source knowledge chunks...")
                embeddings = self.model.encode(self.corpus, convert_to_numpy=True, normalize_embeddings=True)
                self.corpus_embeddings = embeddings

                # Build FAISS index if library available
                if HAS_FAISS:
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors (cosine similarity)
                    self.index.add(embeddings)
                    print("FAISS IndexFlatIP index compiled successfully!")
                else:
                    print("FAISS not installed; utilizing high-speed NumPy cosine similarity.")
            except Exception as e:
                print(f"Notice: Dense embedding model deferred ({e}); using keyword ranking.")
                self.model = None
        else:
            print("sentence-transformers not installed; utilizing high-speed keyword retrieval.")

    def _build_corpus(self):
        """Compiles knowledge entries and scraped/ingested text documents."""
        # Add core knowledge entries
        for entry in KNOWLEDGE_ENTRIES:
            title = entry.get("title", "")
            content = entry.get("content", "")
            chunk = f"Document: {title}\nDetails: {content}"
            self.corpus.append(chunk)

        # Add ingested files from scraped_data/
        if SCRAPED_DIR.exists():
            for txt_file in SCRAPED_DIR.glob("*.txt"):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    topic = txt_file.stem.replace("_", " ").title()
                    # Split into natural paragraphs
                    sections = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 80]
                    for sec in sections:
                        self.corpus.append(f"Document: {topic} Resource\nDetails: {sec}")
                except Exception as err:
                    print(f"Notice reading {txt_file.name}: {err}")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Returns the top_k most relevant chunks for a given user query."""
        if not self.corpus:
            return []

        # Strategy 1: FAISS dense vector search
        if self.index is not None and self.model is not None:
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.corpus)))
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.corpus):
                    results.append(self.corpus[idx])
            return results

        # Strategy 2: NumPy Cosine Similarity
        if self.corpus_embeddings is not None and self.model is not None:
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [self.corpus[i] for i in top_indices if i < len(self.corpus)]

        # Strategy 3: High-speed keyword overlap fallback
        import re
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        stopwords = {"the", "and", "for", "with", "that", "this", "what", "how", "can", "are", "have", "from", "you", "your", "about", "tell"}
        tokens = words - stopwords

        scored = []
        for chunk in self.corpus:
            chunk_lower = chunk.lower()
            score = sum(3 if token in chunk_lower else 0 for token in tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return [c for _, c in scored[:top_k]]
        return self.corpus[:top_k]


if __name__ == "__main__":
    retriever = SemanticRetriever()

    sample_queries = [
        "What documents do I need to bring for international admission eligibility?",
        "How many days do I have to register with the FRRO after landing in India?",
        "What partner universities does Presidency have in Germany or Taiwan?",
        "Can I pay my tuition fees using an international wire transfer or SWIFT?"
    ]

    print("\n--- Running Multi-Source Vector Search Verification ---")
    for i, q in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: '{q}'")
        matched_chunks = retriever.retrieve(q, top_k=2)
        print("Retrieved Context:")
        for idx, chunk in enumerate(matched_chunks, 1):
            snippet = chunk.replace("\n", " ")[:160] + "..."
            print(f"  [{idx}] {snippet}")
