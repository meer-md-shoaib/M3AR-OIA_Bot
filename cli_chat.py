"""
Command Line Interface for Presidency University International Office Chatbot.
Fast, interactive terminal session powered by transformers pipeline.
"""

import sys
from bot_engine import InternationalOfficeBot

def main():
    print("=" * 68)
    print("  PRESIDENCY UNIVERSITY - OFFICE OF INTERNATIONAL AFFAIRS (OIA)")
    print("  Fast Transformer Pipeline AI Chatbot (Qwen2.5-0.5B-Instruct)")
    print("=" * 68)
    print("Type your questions below. Type 'exit' or 'quit' to end the chat.")
    print("=" * 68)

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
            print("-" * 68)
            
            history.append((user_input, response))
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

if __name__ == "__main__":
    main()

