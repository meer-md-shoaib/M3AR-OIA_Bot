"""
Presidency University - Office of International Affairs (OIA)
Interactive AI Counselor Web Application
Features:
- Presidency University Bangalore Campus Visual Background
- Real-time Token Streaming
- FAISS Multi-Source Semantic RAG
- Seamless Inline Campus Photo Rendering
- Clean Example Queries (No parameter table or sliders)
"""

import base64
from pathlib import Path
import gradio as gr
from bot_engine import InternationalOfficeBot

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "Photos"
BG_IMAGE_PATH = PHOTOS_DIR / "presidency_campus_bg.jpeg"

print("Initializing Presidency University International Office Chatbot...")
bot = InternationalOfficeBot()

def chat_stream_fn(message, history):
    """Streaming chat handler with real-time token yield and multimodal image attachments."""
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

# Clean example questions for foreign & prospective students
RAW_EXAMPLES = [
    "What are the admission requirements and document checklist for foreign students?",
    "How do I apply for an Indian Student Visa (S-Visa)?",
    "What is the e-FRRO registration procedure and the mandatory 14-day rule?",
    "Are there scholarships available for international students (like ICCR or SII)?",
    "What are the hostel accommodation and dining facilities for international students?",
    "Tell me about global partner universities and MoU collaborations in Europe or USA",
    "Where is the Office of International Affairs located and what does the building look like?"
]

# Generate Base64 background image CSS for 100% reliable local/remote rendering
if BG_IMAGE_PATH.exists():
    try:
        bg_b64 = base64.b64encode(BG_IMAGE_PATH.read_bytes()).decode("utf-8")
        bg_css = f"""
        body, .gradio-container {{
            background: linear-gradient(135deg, rgba(11, 37, 69, 0.85) 0%, rgba(19, 64, 116, 0.88) 100%),
                        url('data:image/jpeg;base64,{bg_b64}') no-repeat center center fixed !important;
            background-size: cover !important;
            background-position: center !important;
            min-height: 100vh !important;
        }}
        """
    except Exception as e:
        print(f"Notice reading background image: {e}")
        bg_css = ""
else:
    bg_css = ""

custom_css = bg_css + """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}
.header-box {
    text-align: center;
    background: linear-gradient(135deg, rgba(11, 37, 69, 0.95) 0%, rgba(19, 64, 116, 0.92) 60%, rgba(0, 119, 182, 0.95) 100%) !important;
    backdrop-filter: blur(12px) !important;
    color: white;
    padding: 24px 20px;
    border-radius: 16px;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.25);
}
.header-box h1 {
    color: #ffffff !important;
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header-box p {
    color: #e0f2fe !important;
    margin-top: 6px;
    font-size: 15px;
    font-weight: 400;
}
.badge-container {
    margin-top: 10px;
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
.badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16) !important;
    backdrop-filter: blur(8px) !important;
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255, 255, 255, 0.3);
}
/* Glassmorphism styling for chatbot card */
.chatbot, .gradio-chatbot {
    background: rgba(255, 255, 255, 0.96) !important;
    backdrop-filter: blur(14px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.28) !important;
}
/* Clean styling for example queries pills */
.examples-holder, .examples, .gr-samples {
    background: transparent !important;
    margin-top: 12px !important;
}
.examples button, .example-btn, .table-wrap {
    background: rgba(255, 255, 255, 0.94) !important;
    color: #0b2545 !important;
    border: 1px solid rgba(11, 37, 69, 0.18) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
}
.examples button:hover {
    background: #ffffff !important;
    border-color: #0077b6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0, 119, 182, 0.25) !important;
}
"""

with gr.Blocks(title="Presidency University - International Office AI") as demo:
    gr.HTML("""
    <div class="header-box">
        <h1>🎓 Presidency University</h1>
        <p>Office of International Affairs (OIA) • Virtual AI Counselor</p>
        <div class="badge-container">
            <span class="badge">⚡ REAL-TIME STREAMING</span>
            <span class="badge">🔍 FAISS SEMANTIC RAG</span>
            <span class="badge">📸 CAMPUS PHOTO INTEGRATION</span>
            <span class="badge">🏛️ BENGALURU CAMPUS</span>
        </div>
    </div>
    """)

    chat_interface = gr.ChatInterface(
        fn=chat_stream_fn,
        examples=RAW_EXAMPLES,
        cache_examples=False,
        chatbot=gr.Chatbot(
            placeholder="<strong>Welcome to Presidency University Office of International Affairs!</strong><br>I am your official AI Counselor. Ask me anything about international admissions, Indian Student Visa (S-Visa), e-FRRO 14-day compliance, ICCR / SII scholarships, campus hostels, or our 165+ global partner universities across 50+ countries.",
            height=490
        ),
        textbox=gr.Textbox(
            placeholder="Type your inquiry here (e.g. 'Where is the Office of International Affairs located and what does it look like?')...",
            container=False,
            scale=8,
            submit_btn="Ask Counselor"
        ),
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css,
        allowed_paths=[str(PHOTOS_DIR)]
    )
