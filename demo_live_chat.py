"""
Live demonstration script for Presidency University International Office Chatbot.
Simulates student inquiries, tests inference speed, validates RAG answers, and surfaces relevant campus images.
"""

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bot_engine import InternationalOfficeBot
from image_matcher import find_relevant_images

def run_live_demonstration():
    print("=" * 75)
    print("  PRESIDENCY UNIVERSITY - OFFICE OF INTERNATIONAL AFFAIRS (OIA)")
    print("  LIVE CHATBOT DEMONSTRATION & INFERENCE BENCHMARK")
    print("  Semantic RAG + Multimodal Visual Matching")
    print("=" * 75)

    bot = InternationalOfficeBot()

    test_queries = [
        "What are the visa requirements for foreign students, and what is the 14-day FRRO rule?",
        "Tell me about global partner universities and MoU collaborations in Europe or USA",
        "Where can international students wait or relax on campus and who do I contact for help?"
    ]

    history = []

    for idx, query in enumerate(test_queries, 1):
        print(f"\n[Test Case {idx}]")
        print(f"Student: \"{query}\"")

        start_time = time.time()
        response = bot.generate_response(user_message=query, history=history)
        elapsed = time.time() - start_time

        print(f"\nCounselor Response (Generated in {elapsed:.2f}s):\n")
        print(response)

        matched_imgs = find_relevant_images(query, max_images=2)
        if matched_imgs:
            names = ", ".join(Path(p).name for p in matched_imgs)
            print(f"\n📸 [Surfaced Campus Photos]: {names}")

        print("-" * 75)
        history.append((query, response))

if __name__ == "__main__":
    run_live_demonstration()
