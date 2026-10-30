"""
Command Line Interface for Presidency University International Office Chatbot.
Fast, interactive terminal session powered by transformers pipeline and FAISS Semantic RAG.
"""

import sys
from pathlib import Path
from bot_engine import InternationalOfficeBot
from image_matcher import find_relevant_images

def main():
    print("=" * 70)
    print("  PRESIDENCY UNIVERSITY - OFFICE OF INTERNATIONAL AFFAIRS (OIA)")
    print("  Fast Transformer Pipeline AI Chatbot (Qwen2.5-0.5B-Instruct)")
    print("  Integrated: FAISS Semantic RAG + Campus Image Matcher")
    print("=" * 70)
    print("Type your questions below. Type 'exit' or 'quit' to end the chat.")
    print("=" * 70)

    bot = InternationalOfficeBot()
    history = []

    print("\nChatbot ready! How can I help you today?\n")

    while True:
        try:
            user_input = input("Student: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nThank you for reaching out to Presidency University International Office. Have a great day!")
                break

            response = bot.generate_response(user_input, history=history)
            print(f"\nCounselor: {response}\n")

            matched_imgs = find_relevant_images(user_input, max_images=2)
            if matched_imgs:
                img_names = ", ".join(Path(p).name for p in matched_imgs)
                print(f"📸 [Relevant Visuals Attached]: {img_names}")

            print("-" * 70)
            history.append((user_input, response))
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

if __name__ == "__main__":
    main()
