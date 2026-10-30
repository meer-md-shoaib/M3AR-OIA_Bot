"""
Presidency University - Office of International Affairs (OIA)
Semantic Retriever: Vector Database Retriever powered by ChromaDB.
Indexes structured knowledge entries alongside ingested PDF guides and web-scraped collaborations.
Provides dense vector retrieval with cosine similarity and automatic fallback.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent
SCRAPED_DIR = BASE_DIR / "scraped_data"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# 1. Safely import structured entries from knowledge.py
try:
    from knowledge import KNOWLEDGE_ENTRIES
except ImportError:
    KNOWLEDGE_ENTRIES = [
        {
            "category": "Visa & Immigration",
            "title": "Indian Student Visa (S-Visa) Guidelines",
            "content": "International students must enter India strictly on a valid 'Student Visa' (S-Visa) explicitly endorsing Presidency University."
        },
        {
            "category": "FRRO Registration",
            "title": "e-FRRO Compliance",
            "content": "All foreign national students must register with the FRRO within 14 days of landing in India via indianfrro.gov.in."
        },
        {
            "category": "Office Contacts",
            "title": "Office of International Affairs Contact Details",
            "content": "Working Hours: Monday to Friday, 9:00 AM - 5:00 PM IST. Email: international@presidencyuniversity.in | Helpline: +91 80 4925 5555."
        }
    ]

# Try importing ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Try importing dense vector dependencies as optional secondary fallbacks
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
    """
    ChromaDB Vector Database Retriever for the RAG pipeline.
    Handles indexing of multi-source university knowledge into ChromaDB
    and fast semantic nearest-neighbor retrieval.
    """

    def __init__(self, collection_name: str = "presidency_oia_knowledge"):
        print("Initializing Semantic Retriever with ChromaDB Vector Database...")
        self.collection_name = collection_name
        self.corpus: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        self.chroma_client = None
        self.collection = None

        # Fallback variables
        self.model = None
        self.index = None
        self.corpus_embeddings = None

        # 1. Compile multi-source knowledge items
        self._prepare_knowledge_corpus()

        # 2. Initialize ChromaDB Vector Database
        if HAS_CHROMADB:
            try:
                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                # Initialize PersistentClient for persistent vector storage
                self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

                # Populate / upsert documents into ChromaDB
                self._sync_chromadb()
                print(f"ChromaDB Vector DB ready! Collection '{self.collection_name}' has {self.collection.count()} vectors.")
            except Exception as e:
                print(f"Warning: ChromaDB initialization error ({e}); switching to in-memory fallback.")
                try:
                    self.chroma_client = chromadb.Client()
                    self.collection = self.chroma_client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self._sync_chromadb()
                    print(f"ChromaDB in-memory Vector DB ready ({self.collection.count()} vectors).")
                except Exception as err:
                    print(f"ChromaDB fallback failed: {err}")
                    self.collection = None
        else:
            print("ChromaDB is not installed. Run 'pip install chromadb' for vector DB features.")

        # 3. Setup secondary fallback (SentenceTransformers / FAISS / NumPy) if ChromaDB is unavailable
        if self.collection is None:
            self._setup_secondary_fallback()

    def _prepare_knowledge_corpus(self):
        """Compiles knowledge entries and scraped/ingested text documents."""
        # 1. Structured knowledge entries
        for idx, entry in enumerate(KNOWLEDGE_ENTRIES):
            title = entry.get("title", "")
            category = entry.get("category", "General")
            content = entry.get("content", "")
            doc_text = f"### {title}\nCategory: {category}\n{content}"

            doc_id = f"kb_{idx}"
            self.ids.append(doc_id)
            self.documents.append(doc_text)
            self.metadatas.append({
                "source": "knowledge_base",
                "title": title,
                "category": category,
                "id": doc_id
            })
            self.corpus.append(doc_text)

        # 2. Ingested plain-text files from scraped_data/
        if SCRAPED_DIR.exists():
            for txt_file in SCRAPED_DIR.glob("*.txt"):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    topic = txt_file.stem.replace("_", " ").title()
                    sections = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 80]
                    for sec_idx, sec in enumerate(sections):
                        doc_id = f"scraped_{txt_file.stem}_{sec_idx}"
                        doc_text = f"### {topic} Resource\n{sec}"
                        self.ids.append(doc_id)
                        self.documents.append(doc_text)
                        self.metadatas.append({
                            "source": "scraped_data",
                            "topic": topic,
                            "filename": txt_file.name,
                            "id": doc_id
                        })
                        self.corpus.append(doc_text)
                except Exception as err:
                    print(f"Notice reading {txt_file.name}: {err}")

    def _sync_chromadb(self):
        """Upserts all prepared documents and metadata into the ChromaDB collection."""
        if self.collection is None or not self.documents:
            return

        # ChromaDB upsert is idempotent: adds or updates records by ID
        # Batch in chunks of 100 for maximum performance
        batch_size = 100
        for i in range(0, len(self.documents), batch_size):
            batch_ids = self.ids[i:i + batch_size]
            batch_docs = self.documents[i:i + batch_size]
            batch_meta = self.metadatas[i:i + batch_size]
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )

    def _setup_secondary_fallback(self):
        """Initializes SentenceTransformer / FAISS as secondary fallback if ChromaDB is missing."""
        if HAS_SENTENCE_TRANSFORMERS and self.corpus:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = self.model.encode(self.corpus, convert_to_numpy=True, normalize_embeddings=True)
                self.corpus_embeddings = embeddings
                if HAS_FAISS:
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(embeddings)
            except Exception as e:
                print(f"Secondary fallback notice: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        [RAG Step 1: RETRIEVAL]
        Queries ChromaDB Vector DB using semantic similarity and returns top-k context passages.
        """
        if not query or not query.strip():
            return self.corpus[:top_k] if self.corpus else []

        # 1. Primary Strategy: ChromaDB Vector Database Retrieval
        if self.collection is not None:
            try:
                total_docs = self.collection.count()
                if total_docs > 0:
                    k = min(top_k, total_docs)
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=k
                    )
                    documents_list = results.get("documents", [[]])
                    if documents_list and len(documents_list[0]) > 0:
                        return documents_list[0]
            except Exception as err:
                print(f"ChromaDB retrieval warning ({err}); trying fallback.")

        # 2. Secondary Strategy: FAISS dense vector search
        if self.index is not None and self.model is not None:
            try:
                query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                distances, indices = self.index.search(query_embedding, min(top_k, len(self.corpus)))
                results = []
                for idx in indices[0]:
                    if idx != -1 and idx < len(self.corpus):
                        results.append(self.corpus[idx])
                if results:
                    return results
            except Exception:
                pass

        # 3. Tertiary Strategy: NumPy Cosine Similarity
        if self.corpus_embeddings is not None and self.model is not None:
            try:
                query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [self.corpus[i] for i in top_indices if i < len(self.corpus)]
            except Exception:
                pass

        # 4. Final Fallback: Keyword Overlap
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

    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top matches with text, metadata tags, and distance metrics from ChromaDB.
        """
        if self.collection is not None:
            try:
                k = min(top_k, max(1, self.collection.count()))
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )
                output = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]

                for i in range(len(docs)):
                    output.append({
                        "document": docs[i],
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": dists[i] if i < len(dists) else None
                    })
                return output
            except Exception as e:
                print(f"ChromaDB detailed query error: {e}")

        # Fallback to simple list format
        return [{"document": doc, "metadata": {}, "distance": None} for doc in self.retrieve(query, top_k=top_k)]


if __name__ == "__main__":
    retriever = SemanticRetriever()

    sample_queries = [
        "What documents do I need to bring for international admission eligibility?",
        "How many days do I have to register with the FRRO after landing in India?",
        "What partner universities does Presidency have in Germany or Taiwan?",
        "Can I pay my tuition fees using an international wire transfer or SWIFT?"
    ]

    print("\n--- Running ChromaDB Vector Database Search Verification ---")
    for i, q in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: '{q}'")
        matched_chunks = retriever.retrieve(q, top_k=2)
        print("Retrieved Context from ChromaDB:")
        for idx, chunk in enumerate(matched_chunks, 1):
            snippet = chunk.replace("\n", " ")[:160] + "..."
            print(f"  [{idx}] {snippet}")
