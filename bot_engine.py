"""
Presidency University International Office Chatbot Engine.
Powered by Hugging Face Transformers, Multi-Source FAISS Semantic RAG,
Inline Base64 Campus Photo Retrieval, and Real-Time Token Streaming.
"""

import os
import base64
from pathlib import Path
from threading import Thread
from typing import List, Tuple, Generator, Optional
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

from semantic_retriever import SemanticRetriever
from image_matcher import find_relevant_images, load_manifest

# Optimize CPU threads for low-latency interactive inference
if not torch.cuda.is_available():
    torch.set_num_threads(min(4, os.cpu_count() or 4))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def extract_text_content(content) -> str:
    """Safely extracts plain string from any Gradio message content structure."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item.get("content", ""))))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content or "")

def format_image_base64_markdown(img_path_str: str) -> str:
    """Converts local image path into inline Base64 Markdown for 100% reliable browser rendering."""
    p = Path(img_path_str)
    if not p.exists():
        return ""
    
    manifest = load_manifest()
    title = p.name
    for item in manifest:
        if item.get("filename") == p.name:
            title = item.get("title", p.name)
            break

    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"\n\n**📸 {title}**\n\n![{title}](data:image/jpeg;base64,{b64})\n"
    except Exception as e:
        print(f"Warning encoding image {p.name}: {e}")
        return ""

class InternationalOfficeBot:
    def __init__(self, model_name: str = MODEL_ID, use_quantization: bool = False):
        print(f"Loading International Office AI Counselor with model '{model_name}'...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Initialize Multi-Source Semantic RAG Retriever
        self.retriever = SemanticRetriever()

        # 2. Check 4-bit Quantization availability (BitsAndBytes)
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                print("4-bit NF4 Quantization enabled.")
            except ImportError:
                print("bitsandbytes not installed; continuing with standard precision.")

        # 3. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.generator = None
        else:
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=self.device
            )
            self.model = self.generator.model
            self.tokenizer = self.generator.tokenizer

        print(f"International Office Bot ready on {self.device_str.upper()}!")

    def parse_history(self, history) -> List[Tuple[str, str]]:
        """Standardizes conversation history from Gradio or CLI into [(user, assistant)] text tuples."""
        if not history:
            return []

        parsed = []
        # Case 1: Gradio messages format: [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
        if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
            u_temp = None
            for msg in history:
                role = msg.get("role")
                raw_content = msg.get("content", "")
                content_str = extract_text_content(raw_content)
                if role == "user":
                    u_temp = content_str
                elif role == "assistant" and u_temp is not None:
                    parsed.append((u_temp, content_str))
                    u_temp = None
            return parsed

        # Case 2: List of tuples/lists: [[user, bot], ...]
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg = extract_text_content(item[0])
                bot_msg = extract_text_content(item[1])
                parsed.append((user_msg, bot_msg))
        return parsed

    def build_prompt_messages(self, user_message: str, history=None) -> Tuple[List[dict], str, List[str]]:
        """Retrieves semantic context and relevant images, then builds clean text messages."""
        # 1. Semantic RAG context
        clean_user_message = extract_text_content(user_message)
        retrieved_chunks = self.retriever.retrieve(clean_user_message, top_k=3)
        knowledge_context = "\n\n".join(retrieved_chunks)

        # 2. Associated campus photos
        matched_images = find_relevant_images(clean_user_message, max_images=2)

        # 3. Official System Prompt
        system_prompt = (
            "You are the official AI Counselor for the Office of International Affairs (OIA) at Presidency University, Bengaluru.\n"
            "Your role is to assist prospective international applicants and foreign students warmly, accurately, and clearly.\n"
            "Guidelines:\n"
            "1. Answer concisely using the official university facts provided in the knowledge context below.\n"
            "2. When discussing visas, always highlight the mandatory Indian Student Visa (S-Visa) and the strict 14-day e-FRRO registration rule upon landing.\n"
            "3. If needed, provide the official contact: international@presidencyuniversity.in | Helpline: +91 80 4925 5555.\n\n"
            f"[OFFICIAL PRESIDENCY UNIVERSITY INFORMATION]:\n{knowledge_context}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Append recent conversation history (last 3 turns)
        parsed_history = self.parse_history(history)
        for h_user, h_bot in parsed_history[-3:]:
            if h_user:
                messages.append({"role": "user", "content": str(h_user)})
            if h_bot:
                messages.append({"role": "assistant", "content": str(h_bot)})

        # Append current user prompt as pure string
        messages.append({"role": "user", "content": str(clean_user_message)})

        return messages, knowledge_context, matched_images

    def generate_response(
        self,
        user_message: str,
        history=None,
        temperature: float = 0.3,
        max_new_tokens: int = 250
    ) -> str:
        """Synchronous response generation for CLI and automated evaluation."""
        messages, _, matched_images = self.build_prompt_messages(user_message, history)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if self.generator is not None:
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
            response = full_text[len(prompt):].strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0.05 else False,
                    temperature=max(0.1, temperature),
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        if matched_images:
            image_blocks = [format_image_base64_markdown(img_p) for img_p in matched_images]
            response += "".join(image_blocks)

        return response

    def generate_response_streaming(
        self,
        user_message: str,
        history=None,
        temperature: float = 0.3,
        max_new_tokens: int = 250
    ) -> Generator[str, None, None]:
        """
        Token-by-token streaming generator for real-time interactive UI updates.
        Renders verified base64 images seamlessly inline.
        """
        messages, _, matched_images = self.build_prompt_messages(user_message, history)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0.05 else False,
            temperature=max(0.1, temperature),
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated_text = ""
        for new_text in streamer:
            accumulated_text += new_text
            yield accumulated_text

        thread.join()

        # Append inline base64 images if matching photos were found
        if matched_images:
            image_blocks = [format_image_base64_markdown(img_p) for img_p in matched_images]
            if image_blocks:
                accumulated_text += "".join(image_blocks)
                yield accumulated_text
