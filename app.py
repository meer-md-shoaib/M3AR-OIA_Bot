"""
Presidency University - Office of International Affairs (OIA)
Interactive AI Counselor & Multimodal Visual Application.

Features:
- Real-time Grounded RAG with ChromaDB Vector Database
- Multimodal Image Upload: Visual Embeddings -> Decodes Image to Text -> Queries RAG Knowledge Base
- Text-to-Image Matching: Renders verified campus photos based on student text inquiries
- Official Handbooks & Ingested Document Delivery (PDFs)
- Dual Engine: Modern Gradio UI with Flask fallback if Windows OS blocks pandas C-extensions
"""

import sys
import base64
from pathlib import Path
from typing import Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bot_engine import InternationalOfficeBot
from image_embeddings import process_uploaded_image_rag, decode_image_to_text

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "Photos"
PDF_DIR = BASE_DIR / "data" / "pdfs"
BG_IMAGE_PATH = PHOTOS_DIR / "presidency_campus_bg.jpeg"

print("Initializing Presidency University International Office Counselor...")
bot = InternationalOfficeBot()

RAW_EXAMPLES = [
    "Please give me the official international student handbook and rules document.",
    "Where is the Office of International Affairs located and what does the building look like?",
    "Show me the Taiwan Education Center (TECC) plaque and explain collaborations.",
    "What is the e-FRRO registration procedure and the mandatory 14-day rule?",
    "What are the admission requirements and document checklist for foreign students?",
    "What are the hostel accommodation, Form C, and dining facilities for foreign students?",
    "Tell me about global partner universities and MoU collaborations in Europe or USA"
]

# Check Gradio availability
HAS_GRADIO = False
try:
    import gradio as gr
    HAS_GRADIO = True
except Exception as e:
    print(f"Notice: Gradio environment check deferred ({e}).")
    HAS_GRADIO = False


def run_gradio_app():
    """Runs the full Gradio Web Application with streaming, visual recognition, and handbooks."""
    def chat_stream_fn(message, history):
        if not message or not str(message).strip():
            yield ""
            return
        for partial_response in bot.generate_response_streaming(
            user_message=str(message),
            history=history,
            temperature=0.3,
            max_new_tokens=250
        ):
            yield partial_response

    def handle_image_upload(uploaded_image):
        if uploaded_image is None:
            return "Please upload an image.", "", None
        try:
            result = process_uploaded_image_rag(uploaded_image)
            decoded = result["decoded_image"]
            analysis_md = (
                f"### 🔍 Visual Recognition Analysis\n"
                f"- **Recognized Location / Plaque**: **{decoded.get('title', 'Unknown')}**\n"
                f"- **Visual Embedding Similarity**: `{decoded.get('similarity_score', 0.0):.4f}`\n"
                f"- **Decoded Semantic Context**: {decoded.get('description', '')}\n"
            )
            if decoded.get("ocr_text"):
                analysis_md += f"- **Extracted Inscription**: _{decoded.get('ocr_text')}_\n"
            return analysis_md, result["counselor_response"], decoded.get("matched_path")
        except Exception as e:
            return f"Error analyzing image: {e}", "", None

    with gr.Blocks(title="Presidency University - International Office AI") as demo:
        gr.HTML("""
        <div style="text-align:center;background:linear-gradient(135deg,#0b2545,#134074);color:white;padding:22px;border-radius:14px;margin-bottom:15px;">
            <h1 style="margin:0;font-size:26px;">🎓 Presidency University</h1>
            <p style="margin:5px 0 0 0;font-size:15px;color:#e0f2fe;">Office of International Affairs (OIA) • Multimodal AI Counselor</p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("💬 Student Counselor Chat"):
                gr.ChatInterface(
                    fn=chat_stream_fn,
                    examples=RAW_EXAMPLES,
                    chatbot=gr.Chatbot(height=480),
                    textbox=gr.Textbox(placeholder="Ask anything about admissions, visas, FRRO 14-day rules, or request the handbook..."),
                )

            with gr.Tab("📸 Visual Recognition & Image-to-RAG"):
                gr.Markdown("Upload an image of a campus building, TECC plaque, or lounge. Visual embeddings will decode it to text and query ChromaDB RAG!")
                with gr.Row():
                    img_in = gr.Image(type="filepath", label="Upload Campus Photo")
                    rec_btn = gr.Button("🔍 Recognize & Retrieve Information", variant="primary")
                analysis_box = gr.Markdown()
                matched_photo_box = gr.Image(label="Matched Verified Photo", interactive=False)
                counselor_box = gr.Markdown(label="University Results")
                rec_btn.click(handle_image_upload, inputs=[img_in], outputs=[analysis_box, counselor_box, matched_photo_box])

            with gr.Tab("📄 Official Documents & Handbooks"):
                gr.Markdown("""
                ### 📚 Ingested University Publications & Documents
                - **📘 International Student Handbook**: `data/pdfs/international_student_handbook.pdf`
                - **🌐 OIA Annual Bulletin & Guide (64 Pages)**: `data/pdfs/OIA_info.pdf`
                - **📋 OIA Interview & Department FAQs**: `data/pdfs/OIA_Questions.pdf`
                """)

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


def run_flask_app():
    """High-performance interactive web application powered by Flask (runs 100% offline, zero C-extension blocks)."""
    from flask import Flask, request, jsonify, render_template_string, send_from_directory

    app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>International Office Assistant | Presidency University</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --text-navy: #0f284e;
                --text-sub: #13365e;
                --text-muted: #53687e;
                --text-time: #8e9fae;
                --accent-blue: #1d70f5;
                --accent-hover: #1560dc;
                --user-bubble-bg: #cbe5ff;
                --user-bubble-text: #0b2545;
                --bot-bubble-bg: #f0f4f8;
                --bot-bubble-text: #1a2c3f;
                --bg-canvas: #edf3f9;
                --card-bg: #ffffff;
                --border-subtle: #dbe7f3;
                --border-pill: #d6e3f0;
            }
            * { box-sizing: border-box; }
            body {
                font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: var(--bg-canvas);
                background-image: radial-gradient(#d5e4f5 1px, transparent 1px);
                background-size: 24px 24px;
                color: var(--text-navy);
                margin: 0;
                padding: 24px 16px;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
            }
            .app-window {
                width: 100%;
                max-width: 1060px;
                background: var(--card-bg);
                border-radius: 24px;
                box-shadow: 0 16px 50px rgba(15, 34, 60, 0.09), 0 2px 10px rgba(15, 34, 60, 0.04);
                border: 1px solid rgba(214, 227, 240, 0.8);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                position: relative;
            }

            /* HERO BANNER SECTION */
            .hero-banner {
                background: linear-gradient(135deg, #e3effd 0%, #edf5ff 55%, #f8fbff 100%);
                padding: 34px 40px 42px 40px;
                position: relative;
                overflow: hidden;
                border-bottom: 1px solid #dce8f5;
            }
            .hero-bg-graphics {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }
            .hero-content {
                position: relative;
                z-index: 2;
                max-width: 580px;
            }
            .uni-tag {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(29, 112, 245, 0.09);
                border: 1px solid rgba(29, 112, 245, 0.22);
                color: #1d70f5;
                font-size: 12.5px;
                font-weight: 600;
                padding: 4px 12px;
                border-radius: 20px;
                margin-bottom: 12px;
                letter-spacing: 0.2px;
            }
            .hero-title {
                font-size: 34px;
                font-weight: 700;
                color: var(--text-navy);
                margin: 0 0 6px 0;
                letter-spacing: -0.5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .hero-subtitle {
                font-size: 21px;
                font-weight: 700;
                color: var(--text-sub);
                margin: 0 0 10px 0;
                letter-spacing: -0.2px;
            }
            .hero-desc {
                font-size: 14px;
                line-height: 1.55;
                color: var(--text-muted);
                margin: 0;
                max-width: 520px;
            }

            /* Cursive Callout Badge (Top Right) */
            .cursive-badge {
                position: absolute;
                top: 24px;
                right: 32px;
                z-index: 3;
                text-align: center;
                transform: rotate(5deg);
                pointer-events: none;
            }
            .cursive-text {
                font-family: 'Caveat', cursive;
                font-size: 24px;
                font-weight: 700;
                color: #16365c;
                line-height: 1.15;
                letter-spacing: 0.5px;
            }
            .cursive-swoosh {
                width: 110px;
                height: 18px;
                margin-top: -2px;
                filter: drop-shadow(0 2px 4px rgba(245, 158, 11, 0.3));
            }

            /* Landmark Skyline Illustration (Bottom Right of Hero) */
            .skyline-svg {
                position: absolute;
                bottom: 0;
                right: 30px;
                height: 130px;
                width: auto;
                z-index: 2;
                opacity: 0.95;
                pointer-events: none;
            }

            /* FLOATING "YOU CAN ASK ME ABOUT" BAR */
            .ask-card-wrapper {
                padding: 0 32px;
                margin-top: -22px;
                position: relative;
                z-index: 10;
            }
            .ask-card {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid var(--border-subtle);
                box-shadow: 0 8px 30px rgba(15, 40, 78, 0.08);
                padding: 12px 18px;
                display: flex;
                align-items: center;
                gap: 12px;
                overflow-x: auto;
                scrollbar-width: none;
            }
            .ask-card::-webkit-scrollbar { display: none; }
            .ask-label {
                font-size: 13.5px;
                font-weight: 700;
                color: var(--text-navy);
                white-space: nowrap;
                margin-right: 4px;
            }
            .ask-pills-container {
                display: flex;
                align-items: center;
                gap: 10px;
                flex-wrap: nowrap;
            }
            .ask-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: #f4f8fd;
                border: 1px solid var(--border-pill);
                color: var(--text-navy);
                font-size: 13px;
                font-weight: 600;
                padding: 7px 14px;
                border-radius: 12px;
                cursor: pointer;
                white-space: nowrap;
                transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
                font-family: inherit;
            }
            .ask-pill:hover {
                background: #e9f2fd;
                border-color: #bad3f5;
                color: var(--accent-blue);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(29, 112, 245, 0.12);
            }
            .ask-pill .pill-icon-box {
                width: 24px;
                height: 24px;
                border-radius: 7px;
                background: #e1effe;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 13px;
                flex-shrink: 0;
            }

            /* CHAT FEED */
            .chat-feed {
                background: #fbfcfe;
                padding: 26px 36px 18px 36px;
                height: 480px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 18px;
                scroll-behavior: smooth;
            }
            .chat-msg-row {
                display: flex;
                gap: 12px;
                max-width: 82%;
                animation: popIn 0.25s ease-out;
            }
            @keyframes popIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .chat-msg-row.bot {
                align-self: flex-start;
            }
            .chat-msg-row.user {
                align-self: flex-end;
                flex-direction: row-reverse;
            }
            .avatar {
                width: 38px;
                height: 38px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }
            .avatar.bot {
                background: #0d2847;
                color: #ffffff;
                box-shadow: 0 2px 8px rgba(13, 40, 71, 0.22);
            }
            .avatar.user {
                background: #cce0fc;
                color: #1d70f5;
            }
            .msg-col {
                display: flex;
                flex-direction: column;
            }
            .msg-bubble {
                padding: 14px 18px;
                border-radius: 18px;
                font-size: 14.5px;
                line-height: 1.55;
            }
            .chat-msg-row.bot .msg-bubble {
                background: var(--bot-bubble-bg);
                color: var(--bot-bubble-text);
                border-top-left-radius: 4px;
                border: 1px solid rgba(220, 230, 240, 0.7);
            }
            .chat-msg-row.user .msg-bubble {
                background: var(--user-bubble-bg);
                color: var(--user-bubble-text);
                border-top-right-radius: 4px;
            }
            .msg-time {
                font-size: 11px;
                color: var(--text-time);
                margin-top: 5px;
                padding: 0 4px;
            }
            .chat-msg-row.bot .msg-time {
                text-align: left;
            }
            .chat-msg-row.user .msg-time {
                text-align: right;
            }

            .chat-p {
                margin: 6px 0;
                line-height: 1.55;
            }
            .chat-p:first-child { margin-top: 0; }
            .chat-p:last-child { margin-bottom: 0; }
            .chat-list {
                margin: 6px 0 8px 0;
                padding-left: 20px;
            }
            .chat-list li {
                margin-bottom: 5px;
                line-height: 1.5;
            }
            .chat-list li:last-child {
                margin-bottom: 0;
            }
            .chat-link {
                color: #1d70f5;
                text-decoration: underline;
                font-weight: 500;
            }

            /* Verified Campus Photo Card inside Chat */
            .chat-photo-card {
                margin: 12px 0 4px 0;
                background: #ffffff;
                border: 1px solid var(--border-subtle);
                border-radius: 14px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(15, 40, 78, 0.08);
                max-width: 440px;
            }
            .chat-photo-header {
                padding: 8px 14px;
                background: #f4f8fd;
                border-bottom: 1px solid var(--border-subtle);
                font-size: 13px;
                font-weight: 600;
                color: var(--text-sub);
            }
            .chat-photo-img-wrapper {
                background: #f8fafc;
                display: flex;
                align-items: center;
                justify-content: center;
                max-height: 260px;
                overflow: hidden;
                position: relative;
            }
            .chat-photo-img {
                width: 100%;
                height: auto;
                max-height: 260px;
                object-fit: cover;
                display: block;
                cursor: pointer;
                transition: transform 0.25s ease;
            }
            .chat-photo-img:hover {
                transform: scale(1.02);
            }
            .chat-photo-desc {
                padding: 10px 14px;
                font-size: 12.5px;
                color: var(--text-muted);
                line-height: 1.45;
                background: #ffffff;
                border-top: 1px solid #f1f5f9;
            }

            /* BOTTOM FLOATING PILL INPUT BAR */
            .input-section {
                padding: 10px 32px 18px 32px;
                background: #ffffff;
            }
            .pill-input-bar {
                background: #ffffff;
                border-radius: 36px;
                border: 1.5px solid var(--border-pill);
                box-shadow: 0 4px 18px rgba(15, 34, 60, 0.05);
                padding: 6px 8px 6px 16px;
                display: flex;
                align-items: center;
                gap: 12px;
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            .pill-input-bar:focus-within {
                border-color: #93c5fd;
                box-shadow: 0 6px 22px rgba(29, 112, 245, 0.12);
            }
            .btn-attach {
                background: transparent;
                border: none;
                cursor: pointer;
                padding: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: background 0.2s;
                color: #5d7285;
            }
            .btn-attach:hover {
                background: #eef5fc;
                color: var(--accent-blue);
            }
            .pill-input-bar input[type="text"] {
                flex: 1;
                border: none;
                outline: none;
                background: transparent;
                font-family: inherit;
                font-size: 14.5px;
                color: var(--text-navy);
            }
            .pill-input-bar input[type="text"]::placeholder {
                color: #8b9baa;
            }
            .btn-send {
                width: 42px;
                height: 42px;
                border-radius: 50%;
                background: var(--accent-blue);
                color: #ffffff;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.2s, background 0.2s, box-shadow 0.2s;
                box-shadow: 0 3px 10px rgba(29, 112, 245, 0.35);
                flex-shrink: 0;
            }
            .btn-send:hover {
                background: var(--accent-hover);
                transform: scale(1.05);
                box-shadow: 0 4px 14px rgba(29, 112, 245, 0.45);
            }
            .btn-send:active {
                transform: scale(0.96);
            }

            /* COLLAPSIBLE RESOURCE DRAWER (Handbooks, Photos, Vectors) */
            .resource-drawer-toggle {
                text-align: center;
                padding: 10px 32px 14px 32px;
                background: #ffffff;
                border-top: 1px solid #f1f5f9;
            }
            .btn-toggle-resources {
                background: #f4f8fd;
                border: 1px solid var(--border-pill);
                color: var(--text-sub);
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-family: inherit;
            }
            .btn-toggle-resources:hover {
                background: #eaf2fd;
                color: var(--accent-blue);
                border-color: #bad3f5;
            }
            .resource-shelf {
                display: none;
                padding: 16px 32px 24px 32px;
                background: #fbfcfe;
                border-top: 1px solid #eef3f8;
            }
            .resource-shelf.open {
                display: block;
            }
            .doc-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 12px;
                margin-top: 10px;
            }
            .doc-card {
                background: #ffffff;
                border: 1px solid var(--border-subtle);
                border-radius: 12px;
                padding: 12px 14px;
                display: flex;
                align-items: center;
                gap: 12px;
                text-decoration: none;
                color: inherit;
                transition: all 0.2s ease;
            }
            .doc-card:hover {
                background: #f4f8fd;
                border-color: #bad3f5;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(15, 34, 60, 0.06);
            }
            .doc-icon { font-size: 24px; }
            .doc-meta strong { display: block; font-size: 13.5px; color: var(--text-navy); }
            .doc-meta span { font-size: 11.5px; color: var(--text-muted); }
            .doc-btn {
                margin-left: auto;
                background: #eef5fc;
                color: var(--accent-blue);
                font-size: 11.5px;
                font-weight: 600;
                padding: 5px 9px;
                border-radius: 8px;
                white-space: nowrap;
            }

            .photo-gallery-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(145px, 1fr));
                gap: 10px;
                margin-top: 12px;
            }
            .photo-thumb-card {
                background: #ffffff;
                border: 1px solid var(--border-subtle);
                border-radius: 10px;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
                display: flex;
                flex-direction: column;
                text-decoration: none;
            }
            .photo-thumb-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 16px rgba(15, 34, 60, 0.12);
                border-color: #bad3f5;
            }
            .photo-thumb-img {
                width: 100%;
                height: 90px;
                object-fit: cover;
                display: block;
                background: #eef2f6;
            }
            .photo-thumb-title {
                padding: 6px 8px;
                font-size: 11px;
                font-weight: 600;
                color: var(--text-navy);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                background: #f8fbff;
                border-top: 1px solid #f1f5f9;
            }

            /* Lightbox Modal */
            .lightbox {
                display: none;
                position: fixed;
                z-index: 9999;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(6px);
                align-items: center;
                justify-content: center;
                flex-direction: column;
                cursor: zoom-out;
            }
            .lightbox.active { display: flex; }
            .lightbox img {
                max-width: 90vw;
                max-height: 80vh;
                border-radius: 12px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            }
            .lightbox-caption {
                margin-top: 14px;
                color: #ffffff;
                font-size: 14px;
                text-align: center;
            }

            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .hero-banner { padding: 24px 20px 32px 20px; }
                .hero-title { font-size: 28px; }
                .hero-subtitle { font-size: 18px; }
                .cursive-badge { display: none; }
                .skyline-svg { display: none; }
                .ask-card-wrapper { padding: 0 16px; margin-top: -16px; }
                .chat-feed { padding: 18px 16px; height: 420px; }
                .input-section { padding: 8px 16px 16px 16px; }
            }
        </style>
    </head>
    <body>
        <div class="app-window">
            <!-- HERO BANNER SECTION -->
            <div class="hero-banner">
                <!-- Background Graphics: World Map & Flight Trail -->
                <svg class="hero-bg-graphics" viewBox="0 0 1000 240" preserveAspectRatio="none">
                    <!-- Subtle World Map Silhouette -->
                    <g opacity="0.28" fill="#a8cffb">
                        <!-- North America -->
                        <path d="M 120 40 Q 150 30 180 45 Q 210 65 190 95 Q 160 120 140 100 Q 110 80 120 40 Z" />
                        <!-- South America -->
                        <path d="M 190 120 Q 220 130 210 170 Q 195 200 180 175 Q 170 140 190 120 Z" />
                        <!-- Europe -->
                        <path d="M 430 45 Q 470 35 490 55 Q 480 80 450 75 Q 430 65 430 45 Z" />
                        <!-- Africa -->
                        <path d="M 440 90 Q 485 85 495 120 Q 480 170 455 180 Q 430 150 440 90 Z" />
                        <!-- Asia -->
                        <path d="M 510 40 Q 640 25 720 70 Q 700 120 620 110 Q 550 115 510 75 Z" />
                        <!-- Australia -->
                        <path d="M 720 150 Q 770 145 780 175 Q 750 195 720 180 Z" />
                    </g>
                    <!-- Curving Flight Trajectory -->
                    <path d="M 820 190 C 720 120 610 80 490 110 S 330 160 210 80" fill="none" stroke="#60a5fa" stroke-width="2" stroke-dasharray="6,6" opacity="0.65" />
                    <!-- Airplane Silhouette -->
                    <g transform="translate(365, 125) rotate(-22)" fill="#1d70f5">
                        <polygon points="0,0 22,-8 16,-2 28,0 16,2 22,8 0,0" />
                    </g>
                </svg>

                <!-- Top-Right Cursive Calligraphy Badge -->
                <div class="cursive-badge">
                    <div class="cursive-text">Explore<br>Learn<br>Experience<br>the World</div>
                    <svg class="cursive-swoosh" viewBox="0 0 120 20" fill="none">
                        <path d="M 6 15 C 40 4 80 6 114 14" stroke="#f59e0b" stroke-width="3" stroke-linecap="round" />
                    </svg>
                </div>

                <!-- World Landmarks Skyline Silhouette (Right Side) -->
                <svg class="skyline-svg" viewBox="0 0 320 140" fill="none">
                    <!-- Eiffel Tower -->
                    <path d="M 80 140 L 92 80 L 96 40 L 98 10 L 100 40 L 104 80 L 116 140 L 106 140 L 102 95 Q 98 85 94 95 L 90 140 Z" fill="#475569" opacity="0.85" />
                    <!-- Big Ben Clock Tower -->
                    <rect x="135" y="45" width="22" height="95" fill="#f59e0b" opacity="0.75" />
                    <polygon points="135,45 146,20 157,45" fill="#d97706" opacity="0.85" />
                    <circle cx="146" cy="55" r="5" fill="#ffffff" />
                    <!-- Taj Mahal Dome & Minarets -->
                    <path d="M 210 140 L 210 80 Q 230 45 250 80 L 250 140 Z" fill="#93c5fd" opacity="0.85" />
                    <path d="M 230 45 L 230 35" stroke="#1d70f5" stroke-width="2" />
                    <rect x="195" y="70" width="6" height="70" fill="#64748b" opacity="0.75" />
                    <rect x="259" y="70" width="6" height="70" fill="#64748b" opacity="0.75" />
                    <!-- Colosseum Arches -->
                    <path d="M 275 140 L 275 95 Q 315 95 315 140 Z" fill="#cbd5e1" opacity="0.9" />
                    <!-- Trees & Green Foliage Foundation -->
                    <ellipse cx="60" cy="138" rx="28" ry="12" fill="#10b981" opacity="0.8" />
                    <ellipse cx="180" cy="138" rx="30" ry="12" fill="#059669" opacity="0.8" />
                    <ellipse cx="280" cy="138" rx="32" ry="12" fill="#10b981" opacity="0.8" />
                </svg>

                <!-- Hero Text -->
                <div class="hero-content">
                    <div class="uni-tag">🎓 Presidency University • Bengaluru</div>
                    <h1 class="hero-title">Hello! 👋</h1>
                    <h2 class="hero-subtitle">I’m your International Office Assistant</h2>
                    <p class="hero-desc">Get quick information about study abroad programs, admissions, visas, scholarships, partner universities and more.</p>
                </div>
            </div>

            <!-- FLOATING "YOU CAN ASK ME ABOUT" BAR -->
            <div class="ask-card-wrapper">
                <div class="ask-card">
                    <span class="ask-label">You can ask me about:</span>
                    <div class="ask-pills-container">
                        <button type="button" class="ask-pill" onclick="askPreset('What are the study abroad programs available?')">
                            <span class="pill-icon-box">🎓</span>
                            <span>Study Abroad Programs</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('What are the admission requirements and document checklist for international students?')">
                            <span class="pill-icon-box">📄</span>
                            <span>Admissions & Application</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('What is the mandatory 14-day FRRO registration timeframe and student visa rule?')">
                            <span class="pill-icon-box">🛂</span>
                            <span>Visas & Travel</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('What scholarships are available for international students including ICCR and Study in India?')">
                            <span class="pill-icon-box">🛡️</span>
                            <span>Scholarships</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('Tell me about global partner universities and collaborations.')">
                            <span class="pill-icon-box">👥</span>
                            <span>Partner Universities</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('Show me photos of the Confluence Convention Center')">
                            <span class="pill-icon-box">🏛️</span>
                            <span>Confluence Center</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('What awards and national rankings has Presidency University won?')">
                            <span class="pill-icon-box">🏆</span>
                            <span>Awards & Rankings</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('Show me partner country flags and international delegation suites')">
                            <span class="pill-icon-box">🚩</span>
                            <span>Partner Flags & Delegations</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('Show me verified campus photos and office facilities')">
                            <span class="pill-icon-box">📸</span>
                            <span>Campus Photo Tour</span>
                        </button>
                        <button type="button" class="ask-pill" onclick="askPreset('What are the upcoming events, intake deadlines, and orientation schedule?')">
                            <span class="pill-icon-box">📅</span>
                            <span>Events & Deadlines</span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- CHAT FEED -->
            <div class="chat-feed" id="chat">
                <!-- Initial Bot Message -->
                <div class="chat-msg-row bot">
                    <div class="avatar bot">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="2" y1="12" x2="22" y2="12"></line>
                            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                        </svg>
                    </div>
                    <div class="msg-col">
                        <div class="msg-bubble">
                            <p class="chat-p">Hello! 👋</p>
                            <p class="chat-p">I’m the International Office Assistant. I can help you with information about study abroad programs, admissions, visas, scholarships, partner universities and more.</p>
                            <p class="chat-p">How can I assist you today?</p>
                        </div>
                        <div class="msg-time" id="initialTime">10:00 AM</div>
                    </div>
                </div>
            </div>

            <!-- BOTTOM PILL INPUT BAR -->
            <div class="input-section">
                <div class="pill-input-bar">
                    <!-- Hidden File Input for Paperclip -->
                    <input type="file" id="fileUpload" accept="image/*" style="display:none" onchange="handleFilePicked(event)">
                    <!-- Paperclip Attachment Button -->
                    <button type="button" class="btn-attach" onclick="document.getElementById('fileUpload').click()" title="Upload campus photo for visual recognition & RAG">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                        </svg>
                    </button>
                    <!-- Text Message Input -->
                    <input type="text" id="query" placeholder="Type your message here..." autocomplete="off">
                    <!-- Send Button -->
                    <button type="button" class="btn-send" onclick="sendQuery()" title="Send message">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
                        </svg>
                    </button>
                </div>
            </div>

            <!-- COLLAPSIBLE RESOURCE DRAWER (Handbooks, Photos, Vectors) -->
            <div class="resource-drawer-toggle">
                <button type="button" class="btn-toggle-resources" onclick="toggleResourceShelf()">
                    <span>📑 Official Handbooks & Multimodal Resources (18 Campus Photos)</span>
                    <span id="shelfArrow">▼</span>
                </button>
            </div>
            <div class="resource-shelf" id="resourceShelf">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <span style="font-size:13px; font-weight:700; color:var(--text-navy);">📚 Ingested University Document Library</span>
                    <span style="font-size:12px; color:#10b981; font-weight:600; display:flex; align-items:center; gap:5px;">
                        <span>●</span> ChromaDB 81 Vectors & 256-D Visual Embeddings Online
                    </span>
                </div>
                <div class="doc-grid">
                    <a href="/download/international_student_handbook.pdf" target="_blank" class="doc-card">
                        <div class="doc-icon">📘</div>
                        <div class="doc-meta">
                            <strong>Student Handbook</strong>
                            <span>Arrival rules, 14-day FRRO & hostels</span>
                        </div>
                        <div class="doc-btn">PDF ⬇️</div>
                    </a>
                    <a href="/download/OIA_info.pdf" target="_blank" class="doc-card">
                        <div class="doc-icon">🌐</div>
                        <div class="doc-meta">
                            <strong>OIA Guide (64 Pages)</strong>
                            <span>165+ University MoUs & TECC Center</span>
                        </div>
                        <div class="doc-btn">PDF ⬇️</div>
                    </a>
                    <a href="/download/OIA_Questions.pdf" target="_blank" class="doc-card">
                        <div class="doc-icon">📋</div>
                        <div class="doc-meta">
                            <strong>OIA Inquiries & FAQs</strong>
                            <span>Admissions & assessment guidance</span>
                        </div>
                        <div class="doc-btn">PDF ⬇️</div>
                    </a>
                </div>

                <!-- VERIFIED CAMPUS PHOTO GALLERY (ALL 18 PHOTOS) -->
                <div style="margin-top:24px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <span style="font-size:13px; font-weight:700; color:var(--text-navy);">📸 Verified Campus Visual Gallery (18 Photos)</span>
                    <span style="font-size:11.5px; color:var(--text-muted);">Click any photo to enlarge in lightbox</span>
                </div>
                <div class="photo-gallery-grid">
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/1.jpeg', 'Office of International Affairs - Main Entrance')">
                        <img src="/photos/1.jpeg" class="photo-thumb-img" alt="OIA Main Entrance" loading="lazy">
                        <div class="photo-thumb-title">🏛️ OIA Main Entrance</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/2.jpeg', 'International Student Visitor Lounge & Reception')">
                        <img src="/photos/2.jpeg" class="photo-thumb-img" alt="Visitor Lounge" loading="lazy">
                        <div class="photo-thumb-title">🛋️ Visitor Lounge</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/3.jpeg', 'Taiwan Education Center (TECC) Plaque & Collaboration')">
                        <img src="/photos/3.jpeg" class="photo-thumb-img" alt="Taiwan Education Center" loading="lazy">
                        <div class="photo-thumb-title">🇹🇼 TECC Plaque</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/4.jpeg', 'Global Partner Universities & Collaborations Wall')">
                        <img src="/photos/4.jpeg" class="photo-thumb-img" alt="Global Partners Wall" loading="lazy">
                        <div class="photo-thumb-title">🌐 165+ Partners Wall</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/5.jpeg', 'OIA Counseling & International Advisory Cabins')">
                        <img src="/photos/5.jpeg" class="photo-thumb-img" alt="Advising Cabins" loading="lazy">
                        <div class="photo-thumb-title">💬 Advising Cabins</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/6.jpeg', 'Confluence – Global Convention Center & International Atrium')">
                        <img src="/photos/6.jpeg" class="photo-thumb-img" alt="Confluence Atrium" loading="lazy">
                        <div class="photo-thumb-title">✨ Confluence Atrium</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/7.jpeg', 'International Delegation Conference & Boardroom')">
                        <img src="/photos/7.jpeg" class="photo-thumb-img" alt="Delegation Boardroom" loading="lazy">
                        <div class="photo-thumb-title">🤝 Delegation Suite</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/8.jpeg', 'International Partner Nations & National Flags Display')">
                        <img src="/photos/8.jpeg" class="photo-thumb-img" alt="Partner Flags" loading="lazy">
                        <div class="photo-thumb-title">🚩 Partner Flags</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/9.jpeg', 'Confluence Hall Entrance & Floral Archway')">
                        <img src="/photos/9.jpeg" class="photo-thumb-img" alt="Confluence Floral Arch" loading="lazy">
                        <div class="photo-thumb-title">🌸 Floral Archway</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/10.jpeg', 'International Council Chamber & Delegation Suite')">
                        <img src="/photos/10.jpeg" class="photo-thumb-img" alt="Council Chamber" loading="lazy">
                        <div class="photo-thumb-title">🏛️ Council Chamber</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/11.jpeg', 'International Bilateral Meeting & Executive Boardroom')">
                        <img src="/photos/11.jpeg" class="photo-thumb-img" alt="Executive Meeting" loading="lazy">
                        <div class="photo-thumb-title">💼 Bilateral Suite</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/12.jpeg', 'Confluence International Convention Center Entrance')">
                        <img src="/photos/12.jpeg" class="photo-thumb-img" alt="Convention Center Portal" loading="lazy">
                        <div class="photo-thumb-title">🚪 Confluence Portal</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/13.jpeg', 'Confluence Promenade & Botanical Indoor Garden')">
                        <img src="/photos/13.jpeg" class="photo-thumb-img" alt="Indoor Gardens" loading="lazy">
                        <div class="photo-thumb-title">🌿 Botanical Garden</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/14.jpeg', 'OIA Courtyard & Shaded Patio')">
                        <img src="/photos/14.jpeg" class="photo-thumb-img" alt="OIA Courtyard" loading="lazy">
                        <div class="photo-thumb-title">☀️ Courtyard Patio</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/15.jpeg', 'Presidency University OIA Campus Gardens & Frontal View')">
                        <img src="/photos/15.jpeg" class="photo-thumb-img" alt="Campus Gardens" loading="lazy">
                        <div class="photo-thumb-title">🌳 Campus Gardens</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/16.jpeg', 'OIA Courtyard & Diplomatic Glass Pavilion')">
                        <img src="/photos/16.jpeg" class="photo-thumb-img" alt="Glass Pavilion" loading="lazy">
                        <div class="photo-thumb-title">🏢 Glass Pavilion</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/campus_banner_2.jpeg', 'FICCI Higher Education Excellence Award – University of the Year')">
                        <img src="/photos/campus_banner_2.jpeg" class="photo-thumb-img" alt="FICCI University of the Year" loading="lazy">
                        <div class="photo-thumb-title">🏆 FICCI Award</div>
                    </div>
                    <div class="photo-thumb-card" onclick="openLightbox('/photos/presidency_campus_bg.jpeg', 'Presidency University Main Campus Portal & B-School Rankings')">
                        <img src="/photos/presidency_campus_bg.jpeg" class="photo-thumb-img" alt="Main Campus Portal" loading="lazy">
                        <div class="photo-thumb-title">🎓 Main Portal</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- LIGHTBOX MODAL FOR CAMPUS PHOTOS -->
        <div class="lightbox" id="lightbox" onclick="closeLightbox()">
            <img id="lightboxImg" src="" alt="Campus Photo">
            <div class="lightbox-caption" id="lightboxCaption"></div>
        </div>

        <script>
            function getCurrentTime() {
                const now = new Date();
                let hours = now.getHours();
                const minutes = now.getMinutes().toString().padStart(2, '0');
                const ampm = hours >= 12 ? 'PM' : 'AM';
                hours = hours % 12;
                hours = hours ? hours : 12;
                return hours + ':' + minutes + ' ' + ampm;
            }

            // Set current time for initial message
            document.getElementById('initialTime').innerText = getCurrentTime();

            // Pure JavaScript Markdown & HTML Formatter
            function formatChatContent(rawText) {
                if (!rawText) return "";
                let text = rawText.trim();

                // Format Markdown bold **text** to <strong>text</strong>
                text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');

                // Format Markdown italics *text* or _text_
                text = text.replace(/(?<!\\*)\\*(?!\\*)(.*?)(?<!\\*)\\*(?!\\*)/g, '<em>$1</em>');
                text = text.replace(/(?<!_)_(?!_)(.*?)(?<!_)_(?!_)/g, '<em>$1</em>');

                // Format Markdown links [text](url)
                text = text.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" class="chat-link">$1</a>');

                // Split into lines and group list items cleanly into single <ul> blocks
                const rawLines = text.split('\\n');
                let inList = false;
                let htmlParts = [];

                for (let i = 0; i < rawLines.length; i++) {
                    let line = rawLines[i].trim();

                    // Check if line is a bullet item
                    if (line.startsWith('• ') || line.startsWith('- ') || line.startsWith('* ')) {
                        if (!inList) {
                            htmlParts.push('<ul class="chat-list">');
                            inList = true;
                        }
                        let content = line.replace(/^[•\\-\\*]\\s*/, '');
                        htmlParts.push('<li>' + content + '</li>');
                    } else if (line === '') {
                        let nextIsBullet = false;
                        for (let j = i + 1; j < rawLines.length; j++) {
                            let nextTrim = rawLines[j].trim();
                            if (nextTrim !== '') {
                                if (nextTrim.startsWith('• ') || nextTrim.startsWith('- ') || nextTrim.startsWith('* ')) {
                                    nextIsBullet = true;
                                }
                                break;
                            }
                        }
                        if (!nextIsBullet && inList) {
                            htmlParts.push('</ul>');
                            inList = false;
                        }
                    } else {
                        if (inList) {
                            htmlParts.push('</ul>');
                            inList = false;
                        }
                        if (line.startsWith('<div') || line.startsWith('</div') || line.startsWith('<img') || line.startsWith('<hr')) {
                            htmlParts.push(line);
                        } else {
                            htmlParts.push('<p class="chat-p">' + line + '</p>');
                        }
                    }
                }
                if (inList) htmlParts.push('</ul>');

                return htmlParts.join('');
            }

            function appendMsg(role, text) {
                const box = document.getElementById("chat");
                const row = document.createElement("div");
                row.className = "chat-msg-row " + role;

                const avatar = document.createElement("div");
                avatar.className = "avatar " + role;
                if (role === "bot") {
                    avatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>';
                } else {
                    avatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>';
                }

                const msgCol = document.createElement("div");
                msgCol.className = "msg-col";

                const bubble = document.createElement("div");
                bubble.className = "msg-bubble";
                bubble.innerHTML = formatChatContent(text);

                const timeDiv = document.createElement("div");
                timeDiv.className = "msg-time";
                timeDiv.innerText = getCurrentTime();

                msgCol.appendChild(bubble);
                msgCol.appendChild(timeDiv);

                row.appendChild(avatar);
                row.appendChild(msgCol);
                box.appendChild(row);
                box.scrollTop = box.scrollHeight;

                // Attach click handlers to any rendered campus photos for lightbox
                bubble.querySelectorAll('.chat-photo-img').forEach(img => {
                    img.addEventListener('click', () => {
                        openLightbox(img.src, img.alt);
                    });
                });
            }

            async function sendQuery() {
                const qInput = document.getElementById("query");
                const q = qInput.value.trim();
                if (!q) return;

                appendMsg("user", q);
                qInput.value = "";

                // Show typing indicator
                const box = document.getElementById("chat");
                const typingRow = document.createElement("div");
                typingRow.className = "chat-msg-row bot typing-indicator";
                typingRow.id = "typingRow";
                typingRow.innerHTML = `
                    <div class="avatar bot">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
                    </div>
                    <div class="msg-col">
                        <div class="msg-bubble" style="color:var(--text-muted);"><em>Consulting official university regulations...</em></div>
                    </div>
                `;
                box.appendChild(typingRow);
                box.scrollTop = box.scrollHeight;

                try {
                    const res = await fetch("/api/chat", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({message: q})
                    });
                    const data = await res.json();
                    document.getElementById("typingRow")?.remove();
                    appendMsg("bot", data.response);
                } catch (err) {
                    document.getElementById("typingRow")?.remove();
                    appendMsg("bot", "I apologize, but I encountered an error connecting to the server. Please try again or email us at international@presidencyuniversity.in.");
                }
            }

            function askPreset(presetText) {
                document.getElementById("query").value = presetText;
                sendQuery();
            }

            document.getElementById("query").addEventListener("keypress", (e) => {
                if (e.key === "Enter") sendQuery();
            });

            // Handle Multimodal Image Upload via Paperclip button
            async function handleFilePicked(event) {
                const file = event.target.files[0];
                if (!file) return;

                appendMsg("user", "📸 Uploaded photo: " + file.name);

                // Show analyzing state
                const box = document.getElementById("chat");
                const analyzingRow = document.createElement("div");
                analyzingRow.className = "chat-msg-row bot";
                analyzingRow.id = "analyzingRow";
                analyzingRow.innerHTML = `
                    <div class="avatar bot">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
                    </div>
                    <div class="msg-col">
                        <div class="msg-bubble" style="color:var(--accent-blue);">
                            <em>✨ Computing 256-D visual embeddings & querying ChromaDB RAG...</em>
                        </div>
                    </div>
                `;
                box.appendChild(analyzingRow);
                box.scrollTop = box.scrollHeight;

                const formData = new FormData();
                formData.append("image", file);

                try {
                    const res = await fetch("/api/upload_image", {
                        method: "POST",
                        body: formData
                    });
                    const data = await res.json();
                    document.getElementById("analyzingRow")?.remove();
                    appendMsg("bot", "🔍 **Recognized Campus Location:** " + data.title + " *(Similarity: `" + data.similarity + "`)*\\n\\n" + data.response);
                } catch (err) {
                    document.getElementById("analyzingRow")?.remove();
                    appendMsg("bot", "Error analyzing image visual embeddings. Please ensure the file is a valid image.");
                }
                // Reset file input
                event.target.value = "";
            }

            // Lightbox Modal functions
            function openLightbox(src, caption) {
                const lb = document.getElementById("lightbox");
                document.getElementById("lightboxImg").src = src;
                document.getElementById("lightboxCaption").innerText = caption || "";
                lb.classList.add("active");
            }
            function closeLightbox() {
                document.getElementById("lightbox").classList.remove("active");
            }

            // Toggle Resource Shelf
            function toggleResourceShelf() {
                const shelf = document.getElementById("resourceShelf");
                const arrow = document.getElementById("shelfArrow");
                if (shelf.classList.contains("open")) {
                    shelf.classList.remove("open");
                    arrow.innerText = "▼";
                } else {
                    shelf.classList.add("open");
                    arrow.innerText = "▲";
                }
            }
        </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/chat", methods=["POST"])
    def chat_api():
        data = request.get_json() or {}
        msg = data.get("message", "")
        resp = bot.generate_response(user_message=msg)
        return jsonify({"response": resp})

    @app.route("/api/upload_image", methods=["POST"])
    def upload_image_api():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        img_bytes = file.read()
        res = process_uploaded_image_rag(img_bytes)
        decoded = res["decoded_image"]
        return jsonify({
            "title": decoded.get("title", "Campus Visual"),
            "similarity": f"{decoded.get('similarity_score', 0.0):.4f}",
            "response": res["counselor_response"]
        })

    @app.route("/photos/<filename>")
    def serve_photo(filename):
        return send_from_directory(PHOTOS_DIR, filename)

    @app.route("/download/<filename>")
    def download_pdf(filename):
        return send_from_directory(PDF_DIR, filename, as_attachment=True)

    print("\nStarting Presidency University OIA Web Application on http://127.0.0.1:7860 ...")
    app.run(host="127.0.0.1", port=7860, debug=False)


if __name__ == "__main__":
    if HAS_GRADIO:
        try:
            run_gradio_app()
        except Exception as err:
            print(f"Gradio launch exception ({err}); launching high-performance web interface.")
            run_flask_app()
    else:
        run_flask_app()
