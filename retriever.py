"""
Presidency University - Office of International Affairs (OIA)
RAG Pipeline: Retrieval, Augmentation, and Generation with ChromaDB Vector DB.

This module coordinates:
1. [RETRIEVAL]: Semantic nearest-neighbor vector search powered by ChromaDB.
2. [AUGMENTATION]: Context injection with official university guidelines, mandatory compliance, and conversation history.
3. [GENERATION]: Grounded response generation based strictly on augmented context.
"""

import sys
import re
from typing import List, Dict, Optional, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from knowledge import KNOWLEDGE_ENTRIES

# Import ChromaDB-backed Semantic Retriever
try:
    from semantic_retriever import SemanticRetriever
    _GLOBAL_CHROMA_RETRIEVER: Optional[SemanticRetriever] = None

    def get_chroma_retriever() -> SemanticRetriever:
        global _GLOBAL_CHROMA_RETRIEVER
        if _GLOBAL_CHROMA_RETRIEVER is None:
            _GLOBAL_CHROMA_RETRIEVER = SemanticRetriever()
        return _GLOBAL_CHROMA_RETRIEVER

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    get_chroma_retriever = None


def extract_keywords(text: str) -> set:
    """Extracts informative query terms, filtering common stopwords."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {"the", "and", "for", "with", "that", "this", "what", "how", "can", "are", "have", "from", "you", "your", "about", "tell"}
    return {w for w in words if w not in stopwords}


# ==============================================================================
# PHASE 1: RETRIEVAL (Vector DB via ChromaDB)
# ==============================================================================

def retrieve_relevant_context(user_query: str, max_entries: int = 3) -> str:
    """
    [RAG PHASE 1: RETRIEVAL]
    Retrieves the most semantically relevant knowledge chunks using ChromaDB Vector Database.
    Falls back gracefully to keyword overlap if the vector database is unreachable.
    """
    if not user_query or not user_query.strip():
        # Default fallback to admissions overview
        return f"### {KNOWLEDGE_ENTRIES[0]['title']}:\n{KNOWLEDGE_ENTRIES[0]['content']}"

    # 1. Primary Vector DB Retrieval via ChromaDB
    if HAS_CHROMA:
        try:
            retriever = get_chroma_retriever()
            retrieved_docs = retriever.retrieve(user_query, top_k=max_entries)
            if retrieved_docs:
                return "\n\n".join(retrieved_docs)
        except Exception as e:
            print(f"ChromaDB retrieval notice: {e}; using keyword ranking.")

    # 2. Secondary Fallback: Keyword-overlap ranking across knowledge base
    query_tokens = extract_keywords(user_query)
    scored_entries = []
    for entry in KNOWLEDGE_ENTRIES:
        score = 0
        entry_keywords = set(entry.get("keywords", []))
        entry_content = (entry.get("title", "") + " " + entry.get("content", "")).lower()

        for token in query_tokens:
            if token in entry_keywords:
                score += 3
            elif token in entry_content:
                score += 1

        if score > 0:
            scored_entries.append((score, entry))

    scored_entries.sort(key=lambda x: x[0], reverse=True)

    if not scored_entries:
        top_entries = [KNOWLEDGE_ENTRIES[0], KNOWLEDGE_ENTRIES[-1]]
    else:
        top_entries = [entry for _, entry in scored_entries[:max_entries]]

    context_parts = []
    for e in top_entries:
        context_parts.append(f"### {e['title']}:\n{e['content']}")

    return "\n\n".join(context_parts)


# ==============================================================================
# PHASE 2: AUGMENTATION (Contextual Prompt Engineering)
# ==============================================================================

def augment_prompt(
    user_query: str,
    retrieved_context: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    campus_images: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    [RAG PHASE 2: AUGMENTATION]
    Combines the user query with ChromaDB-retrieved facts, university compliance rules,
    and conversational history to produce an augmented prompt for the generator.
    """
    # 1. Retrieve context if not explicitly provided
    if retrieved_context is None:
        retrieved_context = retrieve_relevant_context(user_query)

    # 2. Official System Directive
    system_instruction = (
        "You are the official AI Counselor for the Office of International Affairs (OIA) at Presidency University, Bengaluru.\n"
        "Assist prospective and current foreign students warmly, accurately, and authoritatively.\n"
        "Guidelines:\n"
        "1. Base your answer strictly on the official Presidency University knowledge context provided below.\n"
        "2. When answering visa-related questions, always highlight the mandatory Indian Student Visa (S-Visa) and the strict 14-day e-FRRO registration rule upon arrival.\n"
        "3. Provide official contact details when helpful: international@presidencyuniversity.in | +91 80 4925 5555.\n\n"
        f"=== [CHROMADB RETRIEVED KNOWLEDGE CONTEXT] ===\n{retrieved_context}\n=============================================="
    )

    # 3. Format augmented chat messages
    messages = [{"role": "system", "content": system_instruction}]

    if history:
        for turn_user, turn_bot in history[-3:]:
            if turn_user:
                messages.append({"role": "user", "content": str(turn_user)})
            if turn_bot:
                messages.append({"role": "assistant", "content": str(turn_bot)})

    messages.append({"role": "user", "content": str(user_query)})

    return {
        "augmented_messages": messages,
        "system_instruction": system_instruction,
        "retrieved_context": retrieved_context,
        "user_query": user_query,
        "campus_images": campus_images or []
    }


# ==============================================================================
# PHASE 3: GENERATION (Grounded Response Synthesis)
# ==============================================================================

def generate_grounded_answer(user_query: str, max_new_tokens: int = 250) -> str:
    """
    [RAG PHASE 3: GENERATION]
    Runs full RAG pipeline: ChromaDB Retrieval -> Prompt Augmentation -> Generation.
    """
    try:
        from bot_engine import InternationalOfficeBot
        bot = InternationalOfficeBot()
        return bot.generate_response(user_message=user_query, max_new_tokens=max_new_tokens)
    except Exception as e:
        # Grounded fallback generation from retrieved context
        context = retrieve_relevant_context(user_query)
        return (
            f"Presidency University International Office Counselor:\n\n"
            f"Based on our official records:\n{context}\n\n"
            f"For further assistance, please contact the Office of International Affairs at "
            f"international@presidencyuniversity.in or call +91 80 4925 5555."
        )


if __name__ == "__main__":
    test_query = "What is the mandatory 14-day FRRO registration rule for international students?"
    print("=" * 70)
    print("RAG PIPELINE TEST: ChromaDB Vector DB -> Augmentation -> Generation")
    print("=" * 70)
    print(f"User Query: {test_query}\n")

    # 1. RETRIEVE
    print("[1. RETRIEVING from ChromaDB Vector DB]")
    context = retrieve_relevant_context(test_query)
    print(context[:300] + "...\n")

    # 2. AUGMENT
    print("[2. AUGMENTING Prompt]")
    augmented = augment_prompt(test_query, retrieved_context=context)
    print(f"Augmented System Prompt Length: {len(augmented['system_instruction'])} chars")
    print(f"Total Chat Turns: {len(augmented['augmented_messages'])}\n")

    # 3. GENERATE
    print("[3. GENERATING Response]")
    answer = generate_grounded_answer(test_query)
    print(f"Generated Response:\n{answer}\n")
    print("=" * 70)
