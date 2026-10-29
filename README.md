# 🎓 Presidency University - Office of International Affairs (OIA) Chatbot

A high-speed, lightweight conversational AI chatbot built with **Hugging Face Transformers Pipeline** and **Qwen2.5-0.5B-Instruct**. Tailored specifically for the **Office of International Affairs at Presidency University** to assist foreign and international students with admissions, visa regulations, e-FRRO compliance, scholarships, fees, and campus life.

---

## ⚡ Key Highlights

- **Ultra-Fast Generation on CPU**: Powered by `Qwen/Qwen2.5-0.5B-Instruct` (~500M parameters), delivering conversational turns in 1–2 seconds without requiring a dedicated GPU.
- **Hugging Face `pipeline("text-generation")`**: Clean, modular implementation following Hugging Face best practices.
- **University Domain Grounding**: Equipped with a curated knowledge base covering:
  - **Admissions & Eligibility**: AIU equivalence, required certificates, degree eligibility.
  - **Indian Student Visa (S-Visa)**: Firm acceptance letters, consulate process, S-Visa requirements.
  - **FRRO / e-FRRO Compliance**: Mandatory 14-day registration, Form C, and university support.
  - **Scholarships & Aid**: ICCR A2A portal, Study in India (SII), and University International Merit Scholarships.
  - **Fee Payments**: International wire transfers, SWIFT/IBAN, NRE/NRO accounts.
  - **Hostels & Amenities**: International student residences, meal plans, medical centers, and emergency contacts.
- **Two User Interfaces**:
  1. **Interactive Web UI**: Modern, styled web interface powered by Gradio (`python app.py`).
  2. **Terminal / CLI Chat**: Direct, instant console session (`python cli_chat.py`).

---

## 📁 Project Structure

```
presidency_international_bot/
│── app.py             # Gradio Web Interface with Presidency University branding
│── cli_chat.py        # Interactive Terminal / Command Line Chatbot
│── bot_engine.py      # Core transformer pipeline & prompt template logic
│── retriever.py       # Context retrieval engine matching student queries
│── knowledge.py       # Curated Presidency University International Office knowledge base
│── requirements.txt   # Dependencies (transformers, torch, gradio)
└── README.md          # Documentation and usage guide
```

---

## 🚀 Quickstart

### 1. Run the Interactive Web UI
In PowerShell or Terminal, navigate to the directory and run:
```bash
python app.py
```
Then open your browser to **http://127.0.0.1:7860**.

### 2. Run the Terminal / CLI Chat
For instant command-line conversations without opening a browser:
```bash
python cli_chat.py
```

---

## 💬 Sample Inquiries Handled

- *"What documents do I need to submit for international admission?"*
- *"How do I apply for an Indian Student Visa (S-Visa)?"*
- *"What is the 14-day rule for FRRO registration after arriving in India?"*
- *"Are ICCR and Study in India scholarships accepted at Presidency University?"*
- *"How do I wire my tuition fees internationally?"*
- *"What are the hostel accommodations and Form C procedures for foreign students?"*

