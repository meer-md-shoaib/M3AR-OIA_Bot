"""
Live demonstration script for Presidency University International Office Chatbot.
Simulates student inquiries, tests inference speed, and validates answers.
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bot_engine import InternationalOfficeBot

def run_live_demonstration():
    print("=" * 75)
    print("  PRESIDENCY UNIVERSITY - OFFICE OF INTERNATIONAL AFFAIRS (OIA)")
    print("  LIVE CHATBOT DEMONSTRATION & INFERENCE BENCHMARK")
    print("=" * 75)

    bot = InternationalOfficeBot()
    
    test_queries = [
        "What are the visa requirements for foreign students, and what is the 14-day FRRO rule?",
        "Are there scholarships like ICCR or Study in India available for international applicants?",
        "Can international students stay in university hostels and how are fees paid from abroad?"
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
        print("-" * 75)

        # Append to conversation history to test multi-turn retention
        history.append((query, response))

if __name__ == "__main__":
    run_live_demonstration()
