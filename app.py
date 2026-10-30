"""
Presidency University - Office of International Affairs (OIA)
Interactive AI Counselor Web Application
Features:
- Live Presidency University Bengaluru Campus Background (Aerial Fountain View)
- Real-time Token Streaming
- ChromaDB Multi-Source Semantic RAG (All PDFs, Live Website & Reach Us Directory)
- High-Contrast Visible Input Box
- Public Gradio Live Link Sharing
- Inline Verified Campus Photo Integration
"""

import base64
from pathlib import Path
import gradio as gr
from bot_engine import InternationalOfficeBot

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "Photos"
BG_IMAGE_PATH = PHOTOS_DIR / "presidency_campus_bg.jpeg"

print("Initializing Presidency University International Office Chatbot with ChromaDB...")
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

# Clean example questions reflecting all PDFs, contacts, and live collaborations
RAW_EXAMPLES = [
    "What are the university board lines and admission contact numbers?",
    "Tell me about the International Affairs department interview report findings",
    "What global academic collaboration MoU was signed with Miami University (Ohio, USA)?",
    "What are the student exchange programs and Chancellor's Bon Voyage Scholarship on the live website?",
    "What is the e-FRRO registration procedure and the mandatory 14-day rule for foreign students?",
    "Show me what the Presidency University Bengaluru campus and central fountain look like",
    "What are the hostel accommodation and dining facilities for international students?",
    "Are there scholarships available for international students (like ICCR or Study In India)?"
]

# Generate Base64 background image CSS for 100% reliable local/remote rendering
if BG_IMAGE_PATH.exists():
    try:
        bg_b64 = base64.b64encode(BG_IMAGE_PATH.read_bytes()).decode("utf-8")
        bg_css = f"""
        body, .gradio-container {{
            background: linear-gradient(135deg, rgba(11, 37, 69, 0.70) 0%, rgba(19, 64, 116, 0.72) 100%),
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
    max-width: 1080px !important;
    margin: auto !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}

/* Header Banner */
.header-box {
    text-align: center;
    background: linear-gradient(135deg, rgba(11, 37, 69, 0.94) 0%, rgba(19, 64, 116, 0.92) 60%, rgba(0, 119, 182, 0.94) 100%) !important;
    backdrop-filter: blur(14px) !important;
    color: white;
    padding: 22px 20px 18px 20px;
    border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.28);
}
.header-box h1 {
    color: #ffffff !important;
    margin: 0;
    font-size: 27px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header-box p {
    color: #e0f2fe !important;
    margin-top: 5px;
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
    background: rgba(255, 255, 255, 0.18) !important;
    backdrop-filter: blur(8px) !important;
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255, 255, 255, 0.35);
}

/* Official Reach Us bar inside header */
.reach-us-bar {
    margin-top: 12px;
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 8px 14px;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 16px;
    font-size: 12px;
    color: #f0fdf4;
}
.reach-us-bar strong {
    color: #ffffff;
}

/* Glassmorphism styling for chatbot conversation box */
.chatbot, .gradio-chatbot {
    background: rgba(255, 255, 255, 0.97) !important;
    backdrop-filter: blur(14px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.3) !important;
}

/* HIGH-VISIBILITY TEXTBOX STYLING (User Request) */
.gradio-container textarea,
.gradio-container input[type="text"],
textarea[data-testid="textbox"],
.gr-textbox textarea,
.input-container textarea {
    background: #ffffff !important;
    color: #0b2545 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border: 2.5px solid #0077b6 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    box-shadow: 0 4px 20px rgba(0, 119, 182, 0.25), 0 2px 6px rgba(0,0,0,0.12) !important;
    outline: none !important;
    transition: all 0.2s ease !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: #0096c7 !important;
    box-shadow: 0 6px 26px rgba(0, 150, 199, 0.45) !important;
    background: #ffffff !important;
}
.gradio-container textarea::placeholder {
    color: #334155 !important;
    font-weight: 500 !important;
    opacity: 0.85 !important;
}

/* Prominent Send Button */
button.primary, button.submit-btn, button[aria-label="Ask Counselor"] {
    background: linear-gradient(135deg, #0077b6 0%, #023e8a 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    box-shadow: 0 4px 14px rgba(0, 119, 182, 0.4) !important;
    padding: 10px 22px !important;
    font-size: 15px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
button.primary:hover, button[aria-label="Ask Counselor"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0, 119, 182, 0.55) !important;
}

/* Clean styling for example queries pills */
.examples-holder, .examples, .gr-samples {
    background: transparent !important;
    margin-top: 12px !important;
}
.examples button, .example-btn, .table-wrap {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #0b2545 !important;
    border: 1px solid rgba(11, 37, 69, 0.22) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 9px 15px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
}
.examples button:hover {
    background: #ffffff !important;
    border-color: #0077b6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0, 119, 182, 0.28) !important;
}
"""

with gr.Blocks(title="Presidency University - International Office AI") as demo:
    gr.HTML("""
    <div class="header-box">
        <h1>🎓 Presidency University</h1>
        <p>Office of International Affairs (OIA) • Virtual AI Counselor</p>
        <div class="badge-container">
            <span class="badge">⚡ REAL-TIME STREAMING</span>
            <span class="badge">🧠 CHROMADB SEMANTIC VECTOR STORE</span>
            <span class="badge">🌐 LIVE COLLABORATIONS SYNC</span>
            <span class="badge">📸 VERIFIED CAMPUS PHOTOS</span>
        </div>
        <div class="reach-us-bar">
            <span>📞 Board Line: <strong>+91 080-23093500</strong>, <strong>+91 080-44787350</strong></span>
            <span>🎓 Admissions: <strong>+91 9022092222</strong>, <strong>+91 9024060000</strong></span>
            <span>✉️ <strong>admission@presidencyuniversity.in</strong></span>
            <span>🌐 <strong>presidencyuniversity.in</strong></span>
        </div>
    </div>
    """)

    chat_interface = gr.ChatInterface(
        fn=chat_stream_fn,
        examples=RAW_EXAMPLES,
        cache_examples=False,
        chatbot=gr.Chatbot(
            placeholder="<strong>Welcome to Presidency University Office of International Affairs!</strong><br>I am your official AI Counselor powered by ChromaDB. Ask me anything about admission phone lines, Miami University global academic MoU, Department interview reports, Indian Student Visa (S-Visa), 14-day e-FRRO registration, ICCR & Chancellor's Bon Voyage scholarships, hostels, or our 165+ global partner universities across 33+ countries.",
            height=490
        ),
        textbox=gr.Textbox(
            placeholder="Type your inquiry here (e.g. 'What are the admission contact numbers?' or 'Tell me about the Miami University MoU')...",
            container=False,
            scale=8,
            submit_btn="Ask Counselor"
        ),
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css,
        allowed_paths=[str(PHOTOS_DIR)]
    )
