"""
Context Retriever for Presidency University International Office.
Matches user query against university guidelines for high-precision responses.
"""

import re
from typing import List, Dict
from knowledge import KNOWLEDGE_ENTRIES

def extract_keywords(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Exclude common stop words
    stopwords = {"the", "and", "for", "with", "that", "this", "what", "how", "can", "are", "have", "from", "you", "your", "about", "tell"}
    return {w for w in words if w not in stopwords}

def retrieve_relevant_context(user_query: str, max_entries: int = 2) -> str:
    """
    Ranks knowledge entries based on keyword overlap and semantic relevance.
    Returns formatted context string to ground the transformer model.
    """
    query_tokens = extract_keywords(user_query)
    if not query_tokens:
        # Default to general overview
        return f"University Guidelines: {KNOWLEDGE_ENTRIES[0]['title']}\n{KNOWLEDGE_ENTRIES[0]['content']}"

    scored_entries = []
    for entry in KNOWLEDGE_ENTRIES:
        score = 0
        entry_keywords = set(entry.get("keywords", []))
        entry_content = (entry.get("title", "") + " " + entry.get("content", "")).lower()

        # Check keyword matches
        for token in query_tokens:
            if token in entry_keywords:
                score += 3
            elif token in entry_content:
                score += 1

        if score > 0:
            scored_entries.append((score, entry))

    # Sort descending by score
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    if not scored_entries:
        # Fallback to admission and contact info
        top_entries = [KNOWLEDGE_ENTRIES[0], KNOWLEDGE_ENTRIES[-1]]
    else:
        top_entries = [entry for _, entry in scored_entries[:max_entries]]

    context_parts = []
    for e in top_entries:
        context_parts.append(f"### {e['title']}:\n{e['content']}")

    return "\n\n".join(context_parts)

