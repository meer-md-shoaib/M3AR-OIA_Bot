import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Safely import the actual data structure from knowledge.py
try:
    from knowledge import KNOWLEDGE_ENTRIES
except ImportError:
    # Fallback mock array matching your team's exact structure if run completely isolated
    KNOWLEDGE_ENTRIES = [
        {
            "title": "Indian Student Visa (S-Visa) Guidelines",
            "content": "International students must enter India strictly on a valid 'Student Visa' (S-Visa). The Student Visa must explicitly mention 'Presidency University'."
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

class SemanticRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initializes the retriever, transforms dictionaries into strings, and indexes them via FAISS."""
        print("Initializing Semantic Retriever...")
        self.model = SentenceTransformer(model_name)
        
        # 2. Transform the structured dictionary data into searchable text chunks
        self.corpus = []
        for entry in KNOWLEDGE_ENTRIES:
            # Combine Title and Content for a rich text context
            combined_text = f"Document: {entry['title']}\nDetails: {entry['content']}"
            self.corpus.append(combined_text)
        
        # 3. Generate text embeddings
        print(f"Encoding {len(self.corpus)} structured knowledge entries...")
        corpus_embeddings = self.model.encode(self.corpus, convert_to_numpy=True)
        
        # 4. Get vector layout metrics and compile the FAISS index
        dimension = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(corpus_embeddings)
        print("FAISS index built successfully from KNOWLEDGE_ENTRIES!")

    def retrieve(self, query: str, top_k: int = 3):
        """Encodes the user query and returns the top_k closest structured matches."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.corpus):
                results.append(self.corpus[idx])
                
        return results

# ==========================================
# PRODUCTION STANDALONE TEST BLOCK
# ==========================================
if __name__ == "__main__":
    # Fire up the retriever
    retriever = SemanticRetriever()
    
    # 3 targeted student intent queries mapped directly to your database keys
    sample_queries = [
        "What documents do I need to bring for international admission eligibility?",
        "How many days do I have to register with the FRRO after landing?",
        "Can I pay my tuition fees using an international wire transfer or SWIFT?"
    ]
    
    print("\n--- Running Live Vector Search Verification ---")
    for i, query in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        matched_chunks = retriever.retrieve(query, top_k=2)
        
        print("Retrieved Context:")
        for idx, chunk in enumerate(matched_chunks, 1):
            print(f"  [{idx}] {chunk}\n")
