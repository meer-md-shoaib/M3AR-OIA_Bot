"""
Presidency University International Office Chatbot Engine.
Powered by Hugging Face Transformers text-generation pipeline.
"""

import os
import torch
from transformers import pipeline
from retriever import retrieve_relevant_context

# Optimize PyTorch CPU threading for fast interactive inference
if not torch.cuda.is_available():
    torch.set_num_threads(min(4, os.cpu_count() or 4))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

class InternationalOfficeBot:
    def __init__(self, model_name: str = MODEL_ID):
        print(f"Loading transformer text-generation pipeline with model '{model_name}'...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the text-generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch.float32,
            device=self.device
        )
        self.tokenizer = self.generator.tokenizer
        print(f"Model loaded successfully on {self.device_str.upper()}!")

    def parse_history(self, history) -> list:
        """Standardizes history from Gradio or CLI into list of (user_text, bot_text) tuples."""
        if not history:
            return []
        
        parsed = []
        # Case 1: Gradio messages format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
            u_temp = None
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    u_temp = content
                elif role == "assistant" and u_temp is not None:
                    parsed.append((u_temp, content))
                    u_temp = None
            return parsed

        # Case 2: List of tuples/lists: [[user_msg, bot_msg], ...]
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parsed.append((str(item[0] or ""), str(item[1] or "")))
        return parsed

    def generate_response(
        self, 
        user_message: str, 
        history=None, 
        temperature: float = 0.3, 
        max_new_tokens: int = 250
    ) -> str:
        """
        Generates a fast, contextual answer to an international student's inquiry.
        """
        parsed_history = self.parse_history(history)

        # Retrieve relevant Presidency University information
        knowledge_context = retrieve_relevant_context(user_message)

        # Construct system prompt
        system_prompt = (
            "You are the official AI Counselor for the Office of International Affairs at Presidency University.\n"
            "Your role is to assist international applicants and foreign students warmly, accurately, and clearly.\n"
            "Guidelines:\n"
            "1. Answer concisely using the official university facts provided below.\n"
            "2. When discussing visas, always mention the Indian Student Visa (S-Visa) and the mandatory 14-day FRRO registration upon arrival.\n"
            "3. If needed, provide the official International Office contact: international@presidencyuniversity.in.\n\n"
            f"[OFFICIAL PRESIDENCY UNIVERSITY INFORMATION]:\n{knowledge_context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        # Include recent history (last 3 turns for speed and concise context)
        for h_user, h_bot in parsed_history[-3:]:
            if h_user:
                messages.append({"role": "user", "content": h_user})
            if h_bot:
                messages.append({"role": "assistant", "content": h_bot})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response using transformer pipeline
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0.05 else False,
            temperature=max(0.1, temperature),
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = outputs[0]["generated_text"]
        
        # Extract assistant response after prompt
        response = full_text[len(prompt):].strip()
        return response
