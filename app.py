"""
Presidency University - Office of International Affairs (OIA)
Interactive AI Chatbot Web Application
"""

import gradio as gr
from bot_engine import InternationalOfficeBot

print("Initializing Presidency University International Office Chatbot...")
bot = InternationalOfficeBot()

def chat_fn(message, history, temperature, max_tokens):
    if not message.strip():
        return ""
    return bot.generate_response(
        user_message=message,
        history=history,
        temperature=float(temperature),
        max_new_tokens=int(max_tokens)
    )

RAW_EXAMPLES = [
    "What are the admission requirements and documents for foreign students?",
    "How do I apply for an Indian Student Visa (S-Visa)?",
    "What is the FRRO registration procedure and the 14-day rule?",
    "Are there scholarships available for international students (like ICCR or SII)?",
    "What are the hostel and dining facilities for international students?",
    "How can I pay my tuition fees from abroad and who do I contact?"
]

# Formatted for Gradio ChatInterface with additional inputs [message, temperature, max_tokens]
EXAMPLE_ROWS = [[q, 0.3, 250] for q in RAW_EXAMPLES]

custom_css = """
.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header-box {
    text-align: center;
    background: linear-gradient(135deg, #0b2545 0%, #134074 100%);
    color: white;
    padding: 24px 20px;
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
}
.header-box h1 {
    color: #eef4f8;
    margin: 0;
    font-size: 26px;
    font-weight: 700;
}
.header-box p {
    color: #8da9c4;
    margin-top: 6px;
    font-size: 14px;
}
.badge {
    display: inline-block;
    background: #0077b6;
    color: #ffffff;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    margin-top: 8px;
    letter-spacing: 0.5px;
}
"""

with gr.Blocks(title="Presidency University - International Office AI") as demo:
    gr.HTML("""
    <div class="header-box">
        <h1>🎓 Presidency University</h1>
        <p>Office of International Affairs (OIA) • Virtual AI Counselor</p>
        <span class="badge">FAST INFERENCE • TRANSFORMER PIPELINE • QWEN2.5-0.5B</span>
    </div>
    """)

    with gr.Accordion("⚙️ Generation Settings (Optional)", open=False):
        with gr.Row():
            temp_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Creativity / Temperature (Lower = More deterministic)",
            )
            tokens_slider = gr.Slider(
                minimum=64,
                maximum=512,
                value=250,
                step=32,
                label="Max New Tokens (Controls response length & speed)",
            )

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[temp_slider, tokens_slider],
        examples=EXAMPLE_ROWS,
        cache_examples=False,
        chatbot=gr.Chatbot(
            placeholder="<strong>Welcome to Presidency University International Office!</strong><br>Ask any question about international admissions, Indian student visas, e-FRRO registration, ICCR scholarships, or hostel life.",
            height=480
        ),
        textbox=gr.Textbox(
            placeholder="Type your inquiry here (e.g. 'What documents do I need for a student visa?')...",
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
        css=custom_css
    )
