# 🎓 Presidency University - Office of International Affairs (OIA) AI Counselor

A high-speed, lightweight conversational AI chatbot built with **Hugging Face Transformers Pipeline**, **Qwen2.5-0.5B-Instruct**, **FAISS Semantic RAG**, and **Multimodal Campus Photo Integration**. Specifically engineered for the **Office of International Affairs (OIA) at Presidency University, Bengaluru**, to assist prospective and enrolled international students with admissions, visa guidelines, e-FRRO compliance, global partner collaborations, fee payments, and campus life.

---

## ⚡ Key System Features

- **⚡ Real-Time Token Streaming**: High-speed, responsive token generation via `TextIteratorStreamer` with conversational turns streaming directly to students.
- **🔍 ChromaDB Vector Database & Semantic RAG**: Persistent vector database storage (`data/chroma_db`) indexing multi-source knowledge (structured entries, scraped web pages, handbooks) using `chromadb` with cosine distance and multi-tier fallback.
- **🔄 Complete 3-Stage RAG Pipeline**:
  - **1. Retrieval**: ChromaDB semantic nearest-neighbor search extracting high-relevance university facts.
  - **2. Augmentation**: Contextual prompt engineering injecting official policies, mandatory 14-day e-FRRO compliance, and history.
  - **3. Generation**: Real-time token streaming and grounded response synthesis.
- **📸 Multimodal Campus Photo Support**: Automatically tags and retrieves relevant verified campus photos, international lounge areas, TECC plaques, and global partner collaboration walls matching student queries.
- **🌐 Web Scraper & Ingestion Pipeline**: Scrapes live international collaborations directly from the university portal (`scraper.py`) and extracts text from PDFs (`pdf_ingester.py`).
- **🎯 LoRA Fine-Tuning Pipeline**: Standalone dataset generator (`dataset_builder.py`) and ready-to-run Google Colab notebook (`finetune_colab.ipynb`) using PEFT, SFTTrainer, and 4-bit NF4 quantization on a free T4 GPU.
- **📊 10-Scenario Automated Evaluation Suite**: Golden benchmark accuracy test (`evaluate.py`) measuring keyword recall, visa compliance facts, and grounded responses.

---

## 📁 Project Structure

```
M3AR-OIA_Bot/
├── Photos/
│   ├── 1.jpeg ... 5.jpeg        # Verified campus & OIA office photos
│   └── image_manifest.json      # Structured tags, topics & visual descriptions
├── data/
│   ├── pdfs/                    # Source PDF repository for student handbooks
│   └── finetune_dataset.jsonl   # Instruction-tuning pairs generated for LoRA
├── scraped_data/
│   ├── collaborations.txt       # Scraped from live university collaborations portal
│   └── *.txt                    # Extracted plain text from ingested PDFs
├── app.py                       # Modern Gradio Web UI with streaming & photo display
├── bot_engine.py                # Core engine: ChromaDB RAG, prompt augmentation & token streaming
├── semantic_retriever.py        # ChromaDB persistent vector database retriever
├── retriever.py                 # RAG pipeline module (Retrieval, Augmentation, Generation)
├── image_matcher.py             # Topic & tag relevance scorer for campus photos
├── scraper.py                   # Web scraper for presidencyuniversity.in international pages
├── pdf_ingester.py              # PDF text extraction pipeline using pdfplumber
├── dataset_builder.py           # Instruction-tuning JSONL dataset generator
├── finetune_colab.ipynb         # Complete Colab notebook for LoRA fine-tuning
├── evaluate.py                  # 10-scenario golden benchmark accuracy evaluation
├── cli_chat.py                  # Interactive command-line console session
├── demo_live_chat.py            # Live demonstration benchmark script
├── knowledge.py                 # Core knowledge base for admissions, visa & FRRO
├── requirements.txt             # Comprehensive project dependencies
└── README.md                    # Documentation & operational guide
```

---

## 🚀 Quickstart Guide

### 1. Run the Interactive Web UI
```bash
python app.py
```
Open your web browser at **http://127.0.0.1:7860**.

### 2. Run the Terminal / CLI Chat
```bash
python cli_chat.py
```

### 3. Run Web Scraping & PDF Ingestion
```bash
python scraper.py
python pdf_ingester.py
```

### 4. Build Fine-Tuning Dataset
```bash
python dataset_builder.py
```

### 5. Run the Automated Evaluation Benchmark
```bash
python evaluate.py
```

---

## 👥 The Team: M³AR
* **M**eer
* **M**anoj
* **M**aitri
* **A**mulya
* **R**ashmi
