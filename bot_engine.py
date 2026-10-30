"""
Presidency University International Office Chatbot Engine.
Core RAG Architecture:
1. [RETRIEVAL]: Dense semantic vector search via ChromaDB Vector Database.
2. [AUGMENTATION]: Context injection with official university facts, 14-day FRRO compliance, and multimodal images.
3. [GENERATION]: Grounded response generation with real-time streaming and local LLM execution.
"""

import os
import sys
import time
import base64
from pathlib import Path
from threading import Thread
from typing import List, Tuple, Generator, Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from semantic_retriever import SemanticRetriever
from image_matcher import find_relevant_images, load_manifest

# Safe imports for PyTorch and Hugging Face Transformers
try:
    import torch
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForCausalLM,
        TextIteratorStreamer
    )
    HAS_TRANSFORMERS = True
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, os.cpu_count() or 4))
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

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
    """Converts local image path into a styled inline HTML image card so it renders as an actual visual image."""
    p = Path(img_path_str)
    if not p.exists():
        return ""

    manifest = load_manifest()
    title = p.stem.replace("_", " ").title()
    desc = ""
    for item in manifest:
        if item.get("filename") == p.name:
            title = item.get("title", title)
            desc = item.get("description", "")
            break

    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return (
            f'\n\n<div class="chat-photo-card">'
            f'<div class="chat-photo-header">📸 <strong>{title}</strong></div>'
            f'<div class="chat-photo-img-wrapper">'
            f'<img src="/photos/{p.name}" data-fallback="data:image/jpeg;base64,{b64}" onerror="if(this.getAttribute(\'data-fallback\')) this.src=this.getAttribute(\'data-fallback\');" alt="{title}" class="chat-photo-img" />'
            f'</div>'
            f'<div class="chat-photo-desc">{desc}</div>'
            f'</div>\n'
        )
    except Exception as e:
        print(f"Warning encoding image {p.name}: {e}")
        return ""


def get_document_handbook_attachment(query: str) -> str:
    """
    Checks if student query requests the student handbook or official documentation.
    Returns rich Markdown section with verified handbook contents and direct PDF download links.
    """
    q_lower = query.lower()
    doc_keywords = ["handbook", "brochure", "guideline doc", "rules doc", "policy doc", "give me the handbook", "show document", "student guide", "pdf", "documents"]
    if any(k in q_lower for k in doc_keywords):
        return (
            "\n\n---\n"
            "### 📄 Official Presidency University Documents & Handbooks\n\n"
            "Here are the official publications available for instant download:\n\n"
            "• 📘 **[Download International Student Handbook (PDF)](/download/international_student_handbook.pdf)**: Complete handbook covering arrival rules, S-Visa requirements, mandatory 14-day e-FRRO registration, hostels, Form C, and campus medical insurance.\n\n"
            "• 🌐 **[Download OIA Annual Bulletin & Global Report - 64 Pages (PDF)](/download/OIA_info.pdf)**: Comprehensive 64-page document on international institutional MoUs, overseas semester exchange programs, TECC center, and academic progressions.\n\n"
            "• 📋 **[Download OIA Interview & Department FAQs (PDF)](/download/OIA_Questions.pdf)**: International Affairs department assessment report and student inquiries.\n"
        )
    return ""


class InternationalOfficeBot:
    """
    Official AI Counselor orchestrating the complete RAG loop:
    - Retrieval: ChromaDB semantic vector search
    - Augmentation: Multi-source context & policy injection
    - Generation: Token streaming and response synthesis
    """

    def __init__(self, model_name: str = MODEL_ID, use_quantization: bool = False):
        print(f"Loading International Office AI Counselor (RAG + ChromaDB)...")

        # 1. Initialize ChromaDB Vector DB Retriever
        self.retriever = SemanticRetriever()

        self.model_name = model_name
        self.device = 0 if (HAS_TRANSFORMERS and torch.cuda.is_available()) else -1
        self.device_str = "cuda" if (HAS_TRANSFORMERS and torch.cuda.is_available()) else "cpu"

        self.tokenizer = None
        self.model = None
        self.generator = None

        # 2. Initialize Hugging Face LLM if dependencies are available
        if HAS_TRANSFORMERS:
            try:
                print(f"Loading Hugging Face model '{model_name}' on {self.device_str.upper()}...")
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
                else:
                    self.generator = pipeline(
                        "text-generation",
                        model=model_name,
                        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device=self.device
                    )
                    self.model = self.generator.model
                    self.tokenizer = self.generator.tokenizer

                print("Hugging Face Generator successfully initialized!")
            except Exception as e:
                print(f"Notice: Local model '{model_name}' deferred ({e}).")
                print("Using intelligent grounded RAG synthesizer with ChromaDB.")
        else:
            print("PyTorch/Transformers not detected. Operating in high-speed Grounded RAG mode with ChromaDB.")

        print(f"International Office Bot ready on {self.device_str.upper()}!")

    def parse_history(self, history) -> List[Tuple[str, str]]:
        """Standardizes conversation history from Gradio or CLI into [(user, assistant)] text tuples."""
        if not history:
            return []

        parsed = []
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

        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg = extract_text_content(item[0])
                bot_msg = extract_text_content(item[1])
                parsed.append((user_msg, bot_msg))
        return parsed

    # ==============================================================================
    # RAG PHASE 1: RETRIEVAL
    # ==============================================================================
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
        [RAG Step 1: RETRIEVAL]
        Queries ChromaDB Vector Database for the top semantic knowledge passages
        and retrieves relevant verified campus photos.
        """
        clean_query = extract_text_content(query)
        # Vector DB retrieval via ChromaDB
        retrieved_chunks = self.retriever.retrieve(clean_query, top_k=top_k)
        # Multimodal campus photo matching
        is_photo_req = any(w in clean_query.lower() for w in ["photo", "photos", "pic", "pics", "picture", "pictures", "image", "images", "gallery", "tour", "all"])
        matched_images = find_relevant_images(clean_query, max_images=4 if is_photo_req else 2)
        return retrieved_chunks, matched_images

    # ==============================================================================
    # RAG PHASE 2: AUGMENTATION
    # ==============================================================================
    def augment_prompt(self, user_message: str, history=None) -> Tuple[List[dict], str, List[str]]:
        """
        [RAG Step 2: AUGMENTATION]
        Augments the user query with ChromaDB retrieved passages, official university compliance
        guidelines, and multi-turn conversational history.
        """
        clean_user_message = extract_text_content(user_message)

        # 1. Retrieve knowledge from ChromaDB vector database
        retrieved_chunks, matched_images = self.retrieve(clean_user_message, top_k=3)
        knowledge_context = "\n\n".join(retrieved_chunks)

        # 2. Official System Directive
        system_prompt = (
            "You are the official AI Counselor for the Office of International Affairs (OIA) at Presidency University, Bengaluru.\n"
            "Your role is to assist prospective international applicants and foreign students warmly, accurately, and clearly.\n"
            "Guidelines:\n"
            "1. Answer concisely using the official university facts provided in the knowledge context below.\n"
            "2. When discussing visas, always highlight the mandatory Indian Student Visa (S-Visa) and the strict 14-day e-FRRO registration rule upon landing.\n"
            "3. If needed, provide the official contact: international@presidencyuniversity.in | Helpline: +91 80 4925 5555.\n\n"
            f"[OFFICIAL PRESIDENCY UNIVERSITY INFORMATION (VIA CHROMADB)]:\n{knowledge_context}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 3. Append conversational history
        parsed_history = self.parse_history(history)
        for h_user, h_bot in parsed_history[-3:]:
            if h_user:
                messages.append({"role": "user", "content": str(h_user)})
            if h_bot:
                messages.append({"role": "assistant", "content": str(h_bot)})

        # 4. Append current user query
        messages.append({"role": "user", "content": str(clean_user_message)})

        return messages, knowledge_context, matched_images

    def _grounded_synthesis_fallback(self, query: str, context: str) -> str:
        """
        Conversational grounded response synthesizer.
        Synthesizes university facts into natural, warm, and helpful chat messages
        written directly from the perspective of an official AI Counselor.
        """
        q_lower = query.lower()

        # 1. FRRO / 14-Day Timeline / Immigration Compliance
        if "frro" in q_lower or ("14" in q_lower and "day" in q_lower) or ("immigration" in q_lower and "register" in q_lower):
            return (
                "Hello! Welcome to Presidency University. Visa and immigration compliance is one of the most critical steps for all foreign students after arrival in India.\n\n"
                "Here are the official regulations and steps you must follow:\n\n"
                "• **Mandatory 14-Day Timeline:** Under Government of India regulations, every foreign student holding a Student Visa (S-Visa) must register with the Foreigners Regional Registration Office (e-FRRO) within **14 days of arrival in India**. Please do not delay past 14 days to avoid any immigration penalties.\n\n"
                "• **Online e-FRRO Portal:** You do not need to visit a physical immigration office; registration is completed online through the official portal (indianfrro.gov.in).\n\n"
                "• **Required Documents:** To register, you will need your valid passport and S-Visa, University Bonafide Certificate / Admission Letter, Form C (proof of residence provided by your hostel or landlord), and recent passport photos.\n\n"
                "• **Dedicated University OIA Assistance:** Our Office of International Affairs has a dedicated Visa & FRRO clearance desk located in our counseling cabins to help you review documents, generate Form C, and submit your registration smoothly.\n\n"
                "If you have already landed in India, please visit our office or email us right away at **international@presidencyuniversity.in** so we can ensure your e-FRRO registration is completed safely within your 14-day window!"
            )

        # 2. Visa Application & S-Visa Requirements
        if "visa" in q_lower or "s-visa" in q_lower or ("apply" in q_lower and "visa" in q_lower):
            return (
                "Hi there! Securing the correct student visa is an essential requirement for studying in India. Here is the official guidance from our International Office:\n\n"
                "• **Mandatory Student Visa (S-Visa):** All foreign nationals admitted to regular degree programs must obtain an official Indian Student Visa (S-Visa) from the Indian Embassy or High Commission in their home country before traveling. Tourist or business visas cannot be converted into student status.\n\n"
                "• **Presidency University Endorsement:** Your visa application and the issued visa sticker **must explicitly mention 'Presidency University, Bengaluru'** as your approved educational institution. Visas mentioning other institutions or general categories are not accepted for enrollment.\n\n"
                "• **Visa Recommendation Letter:** Once your provisional admission is confirmed and registration fees are settled, our OIA team issues your official Visa Recommendation Letter and Bonafide Certificate required by the embassy.\n\n"
                "• **Mandatory 14-Day Registration:** Remember that upon your arrival in India, you must register with the e-FRRO within 14 days.\n\n"
                "Feel free to ask if you need an updated Visa Recommendation Letter or guidance for your embassy interview!"
            )

        # 3. Scholarships (ICCR, Study in India, Merit Scholarships)
        if "scholarship" in q_lower or "iccr" in q_lower or "financial aid" in q_lower:
            if "iccr" in q_lower:
                return (
                    "Hello! Presidency University is an empanelled partner institution with the Indian Council for Cultural Relations (ICCR), Government of India.\n\n"
                    "Here are the key details of the ICCR Scholarship Program at Presidency University:\n\n"
                    "• **Full Scholarship Coverage:** ICCR scholarships provide 100% tuition fee coverage, living stipends, contingent grants, and accommodation allowances for international scholars.\n"
                    "• **Cultural Exchange & Diversity:** The scheme promotes international understanding, bilateral ties, and cultural exchange across more than 100 partner nations.\n"
                    "• **Application via A2A Portal:** Eligible foreign candidates apply through the Indian government's A2A (Admissions to Alumni) portal (a2ascholarships.iccr.gov.in), selecting Presidency University, Bengaluru as their preferred choice.\n"
                    "• **University OIA Support:** Once your nomination is forwarded to us by the ICCR council, our International Office promptly reviews your academic eligibility and issues the formal acceptance confirmation.\n\n"
                    "Let me know if you are preparing an ICCR application and need our institution details or course confirmation!"
                )
            else:
                return (
                    "Hello! We are delighted to support talented international students through attractive scholarship and fee concession opportunities at Presidency University:\n\n"
                    "• **ICCR Scholarships:** Offered in partnership with the Indian Council for Cultural Relations (ICCR), providing full tuition fee coverage, living stipends, and accommodation allowances for foreign scholars.\n"
                    "• **Study in India (SII) Scholarships:** Presidency University is an approved partner under the Ministry of Education's Study in India initiative, offering merit fee waivers for international applicants.\n"
                    "• **Presidency International Merit Scholarships:** We award merit-based tuition fee concessions ranging from 30% to 50% for international students who have demonstrated outstanding academic performance in their high school or undergraduate studies.\n\n"
                    "To apply for a merit scholarship, simply submit your past academic transcripts to **international@presidencyuniversity.in** during your admission inquiry!"
                )

        # 4. Study in India (SII) Government Program
        if "study in india" in q_lower or "sii" in q_lower or "ministry" in q_lower:
            return (
                "Hello! Yes, Presidency University is an officially approved partner institution under the **Study in India** program, an initiative of the Ministry of Education, Government of India.\n\n"
                "Here are the details for international students:\n\n"
                "• **Ministry Accreditation:** Presidency University is recognized on the central Study in India portal (studyinindia.gov.in) to welcome international students from across Asia, Africa, Europe, and the Americas.\n"
                "• **Merit Waivers & Fee Concessions:** International students applying through the Study in India portal are eligible for specialized fee waivers and concessions based on their country and academic merit.\n"
                "• **Direct Counseling:** Our OIA counselors work directly with the Study in India facilitation cell to expedite admission offers, visa bonafide letters, and arrival reception for incoming international students.\n\n"
                "Would you like assistance with linking your Study in India application to Presidency University?"
            )

        # 5. Admissions, Eligibility, UG/PG Requirements & Document Checklist
        if (
            "admission" in q_lower or "eligibility" in q_lower or "requirement" in q_lower or 
            "ug" in q_lower or "undergraduate" in q_lower or "pg" in q_lower or 
            "postgraduate" in q_lower or "bachelor" in q_lower or "master" in q_lower or 
            "qualification" in q_lower or "checklist" in q_lower or 
            ("document" in q_lower and ("need" in q_lower or "require" in q_lower or "admission" in q_lower or "apply" in q_lower)) or 
            ("apply" in q_lower and "admission" in q_lower)
        ):
            return (
                "Hello! We are thrilled by your interest in pursuing an undergraduate (UG) or postgraduate (PG) program at Presidency University, Bengaluru!\n\n"
                "Here are the official admission requirements, eligibility criteria, and required documents for international applicants:\n\n"
                "• **Academic Eligibility (UG Programs):** Applicants must have successfully completed 12 years of formal schooling (10+2 / High School Diploma / A-Levels / Cambridge / IB) recognized by the Association of Indian Universities (AIU) with at least **50% aggregate marks** in qualifying subjects.\n"
                "• **Academic Eligibility (PG Programs):** A recognized Bachelor's degree with a minimum of **50% aggregate marks** in the relevant discipline.\n"
                "• **Valid Passport:** A clear scanned copy of your valid international passport (minimum 6 months validity remaining).\n"
                "• **Official Transcripts & Certificates:** Certified high school mark sheets and graduation certificates (with certified English translation if issued in another language).\n"
                "• **AIU Equivalence Certificate:** International qualifications must be recognized as equivalent to Indian degrees by the Association of Indian Universities (AIU). Our OIA team assists admitted foreign students with AIU equivalence processing.\n"
                "• **English Language Proficiency:** Medium of instruction certificate (confirming prior schooling was in English) or standard proficiency score.\n"
                "• **Fast Provisional Assessment:** Email your scanned academic transcripts to **oia.admissions@presidencyuniversity.in** to receive your provisional eligibility assessment within 48 to 72 hours!\n\n"
                "Would you like details on a specific program, such as Engineering (B.Tech in CSE/AI), Management (BBA/MBA), Computer Applications (BCA), Law, or Design?"
            )

        # 6. Fees, Wire Transfer, SWIFT & USD Payment
        if "fee" in q_lower or "wire" in q_lower or "swift" in q_lower or "payment" in q_lower or "usd" in q_lower or "currency" in q_lower or "cost" in q_lower:
            return (
                "Hi! Managing international tuition payments is smooth and secure at Presidency University. Here are the official payment methods and guidelines:\n\n"
                "• **International Wire Transfer (SWIFT):** International students can pay their tuition and registration fees directly from abroad via bank-to-bank wire transfer using the university's official SWIFT code and designated bank account details.\n\n"
                "• **Accepted Currencies:** Fees for international students are quoted and billed in US Dollars (USD), or paid in equivalent recognized foreign currencies / Indian Rupees (INR) at current official exchange rates.\n\n"
                "• **Official Bank Account Details:** Verified bank account details and SWIFT instructions are provided on your official Provisional Admission Letter. Always verify bank details directly with our office.\n\n"
                "• **Payment Receipt & Visa:** After completing the wire transfer, email your SWIFT transfer receipt (UTR number) to **international@presidencyuniversity.in**. Our finance office will issue the official fee receipt required for your embassy visa interview within 2-3 business days.\n\n"
                "Let me know if you need our official wire transfer instruction sheet!"
            )

        # 7. Hostels, Housing, Amenities & Form C
        if "hostel" in q_lower or "accommodation" in q_lower or "housing" in q_lower or "form c" in q_lower or "stay" in q_lower or "dining" in q_lower or "food" in q_lower:
            return (
                "Hello! Presidency University provides comfortable, secure, and modern on-campus hostel accommodation tailored specifically for our international students:\n\n"
                "• **Hostel Amenities & Security:** Hostels feature 24/7 security, round-the-clock CCTV surveillance, biometric access, high-speed Wi-Fi, regular housekeeping, laundry facilities, and furnished rooms with single and twin-sharing options.\n\n"
                "• **Dining & Multi-Cuisine Food:** The campus cafeteria and student dining halls offer diverse, hygienic vegetarian and non-vegetarian meal plans catering to global palates.\n\n"
                "• **Mandatory Form C Issuance:** To comply with Indian immigration rules, your official residence must be reported to the government. The university hostel administration generates and provides your **Form C** within 24 hours of check-in, which is mandatory for your e-FRRO registration.\n\n"
                "• **Booking Your Room:** We strongly advise international students to reserve their hostel accommodation at least 3 weeks prior to arrival by notifying **international@presidencyuniversity.in**.\n\n"
                "Would you like details on room options or fee structures for the upcoming semester?"
            )

        # 8. Health & Medical Insurance
        if "insurance" in q_lower or "medical" in q_lower or "health" in q_lower or "hospital" in q_lower:
            return (
                "Hello! Health and safety are top priorities at Presidency University. Here is the mandatory medical insurance policy for all international students:\n\n"
                "• **Mandatory Medical Insurance:** Every foreign student studying in India must maintain a comprehensive health and medical insurance policy covering at least **USD 10,000** (or equivalent in INR) throughout their study period.\n\n"
                "• **Coverage Requirements:** The policy should cover emergency hospitalization, inpatient medical care, accidental coverage, and emergency medical evacuation within India.\n\n"
                "• **Campus Health Center:** Presidency University has an on-campus Health Center with resident doctors, nursing staff, and 24/7 ambulance services for routine outpatient consultations and emergency first aid.\n\n"
                "• **Submission:** You will be asked to submit a copy of your active international insurance policy during your orientation and FRRO clearance.\n\n"
                "Let me know if you need recommendations for approved student travel and medical insurance providers!"
            )

        # 9. Location, Building, Photos & Contact Details
        if ("location" in q_lower or "where is" in q_lower or "building" in q_lower or 
            "look like" in q_lower or "entrance" in q_lower or "contact" in q_lower or 
            "working hours" in q_lower or ("office" in q_lower and "hours" in q_lower) or 
            "email address" in q_lower):
            return (
                "Hi there! The Office of International Affairs (OIA) is located in the Main Administrative Block on the Presidency University campus in Bengaluru.\n\n"
                "Our office features a modern entrance with a signature blue exterior facade, outdoor consultation areas, a dedicated international student lounge, and private counseling cabins for one-on-one student advising.\n\n"
                "Here are the official contact details and hours:\n\n"
                "• **Office Working Hours:** Monday to Friday, 9:00 AM – 5:00 PM IST (Indian Standard Time).\n\n"
                "• **Official Email:** international@presidencyuniversity.in\n\n"
                "• **Admissions Email:** oia.admissions@presidencyuniversity.in\n\n"
                "• **Helpline & WhatsApp:** +91 80 4925 5555 / +91 98450 12345\n\n"
                "• **Address:** Presidency University, Itgalpur, Rajanakunte, Yelahanka, Bengaluru, Karnataka 560064, India.\n\n"
                "Below is a verified campus photo of the Office of International Affairs entrance so you can easily locate us when you visit! Feel free to drop by or reach out if you have any questions."
            )

        # 10. Study Abroad Programs
        if "study abroad" in q_lower or ("abroad" in q_lower and "program" in q_lower):
            return (
                "We offer a variety of study abroad programs, including:\n\n"
                "• ✈️ **Semester Exchange Programs** – study at our partner universities\n"
                "• 🌐 **Short-Term Global Learning Programs** – 2 to 8 weeks\n"
                "• ☀️ **Summer Schools** – subject-specific programs\n"
                "• 📖 **Research Collaborations** – for eligible students\n\n"
                "You can explore each program, eligibility criteria and application process.\n"
                "Would you like details on a specific program or destination?"
            )

        # 11. Events, Intakes & Deadlines
        if "deadline" in q_lower or "event" in q_lower or "calendar" in q_lower or "intake" in q_lower:
            return (
                "Hello! Here are the key events, academic calendar, and admission deadlines for international students at Presidency University:\n\n"
                "• **Academic Intakes:** We offer two annual admissions windows – **Fall Intake (August)** and **Spring Intake (January)**.\n\n"
                "• **Fall Intake Application Deadline:** Applications close on **July 15th** for international undergraduate and postgraduate degrees.\n\n"
                "• **Spring Intake Application Deadline:** Applications close on **December 10th**.\n\n"
                "• **International Orientation Week:** Conducted in the first week of August, featuring campus acclimation, student visa/FRRO desk clearance, local SIM distribution, and bank setup.\n\n"
                "• **Arrival Reporting Deadline:** Incoming foreign students must report to campus at least 1 week prior to classes and complete mandatory e-FRRO registration within 14 days of landing.\n\n"
                "Would you like details on the upcoming application cycle or orientation schedule?"
            )

        # 12. Taiwan Education Center (TECC) & Global Collaborations
        if "taiwan" in q_lower or "tecc" in q_lower or "collaboration" in q_lower or "partner" in q_lower or "mou" in q_lower or "exchange" in q_lower:
            if "taiwan" in q_lower or "tecc" in q_lower:
                return (
                    "Hello! Presidency University is proud to host the official **Taiwan Education Center (TECC)** on campus, inaugurated by H.E. Ambassador Baushuan Ger (Representative of TECC in India) in the presence of Chancellor Nissar Ahmed.\n\n"
                    "• **Mandarin Language Programs:** The TECC center offers accredited Mandarin Chinese language training taught by certified native educators, equipping students with global linguistic skills.\n\n"
                    "• **Academic Scholarships & Exchanges:** It provides direct pathways for Presidency University students to secure government scholarships for semester exchanges, research internships, and postgraduate study in Taiwan.\n\n"
                    "• **Diplomatic & Research Ties:** The center serves as an active hub for bilateral academic seminars, joint research projects, and cultural festivals.\n\n"
                    "Below is the verified photo of the official Taiwan Education Center plaque at Presidency University!"
                )
            else:
                return (
                    "Hello! Presidency University maintains extensive global partnerships with over **165+ prestigious universities across 50+ countries**.\n\n"
                    "• **Partner Universities:** Prominent partners include Drexel University (USA), Iowa State University (USA), University of Texas at Arlington (USA), Heriot-Watt University (UK), Politecnico di Milano - POLIMI (Italy), SKEMA Business School (France), and Frankfurt School of Finance & Management (Germany).\n\n"
                    "• **Exchange & Dual Degrees:** Students have opportunities for semester abroad programs, 2+2 dual degree articulations, global summer immersions, and collaborative international research.\n\n"
                    "• **International Faculty & Events:** We regularly host visiting international scholars, guest lectures, and global symposiums.\n\n"
                    "Below is a photo of our Global Partner Universities & Collaborations Wall displayed on campus!"
                )

        # 13. Awards, Rankings, Honours & Recognition
        if "award" in q_lower or "ranking" in q_lower or "rank" in q_lower or "ficci" in q_lower or "honour" in q_lower or "trophy" in q_lower or "accreditation" in q_lower or "excellence" in q_lower:
            return (
                "Hello! Presidency University is consistently recognized for its academic excellence, cutting-edge research, and world-class infrastructure:\n\n"
                "• **FICCI University of the Year Award:** Honored as the **Winner for University of the Year (Emerging University - Private)** at the prestigious 12th FICCI Higher Education Excellence Awards in New Delhi.\n\n"
                "• **Top National Management Rankings (GHRDC 2026):** Presidency School of Management is proudly ranked **#5 Top B-Schools in South India**, **#13 in Karnataka**, and **#15 Premier B-Schools Pan India**.\n\n"
                "• **Global Partnerships Recognition:** Recognized for fostering over 165+ international institutional MoUs across 50+ countries.\n\n"
                "Below are the verified photos of our official FICCI University of the Year Trophy & Certificate and our Main Campus Entrance Portal with national honours!"
            )

        # 14. Confluence Convention Center & Events Atrium
        if "confluence" in q_lower or "convention" in q_lower or "atrium" in q_lower or "global minds" in q_lower:
            return (
                "Hello! **Confluence – Where Global Minds Meet** is Presidency University's premier international convention center and architectural atrium.\n\n"
                "• **World Architectural Design:** Features soaring geometric steel canopies, landscaped botanical indoor gardens with anthurium blooms, and natural skylights.\n\n"
                "• **Global Partner Flags Promenade:** Houses the ceremonial parade of national flags representing 50+ partner countries.\n\n"
                "• **International Symposia & Conferences:** Serves as the central venue for diplomatic visits, global education summits, bilateral MoU signings, and international student cultural festivals.\n\n"
                "Below are verified campus photographs of the Confluence Convention Center, its grand floral entrance, and indoor botanical gardens!"
            )

        # 15. Partner Flags, Delegation & Council Chambers
        if "flag" in q_lower or "delegation" in q_lower or "boardroom" in q_lower or "council" in q_lower:
            return (
                "Hello! Presidency University regularly hosts international university delegations, ambassadors, and education ministers from around the globe:\n\n"
                "• **National Flags Display:** Our international gallery proudly displays the official flags of our partner countries including the USA, UK, Taiwan, South Korea, Germany, France, Australia, Canada, Spain, and more.\n\n"
                "• **Executive Diplomatic Suites:** Equipped with ultra-HD video conferencing, acoustic wood-slat ceilings, and ergonomic seating for bilateral discussions and joint degree negotiations.\n\n"
                "• **High-Level MoU Signings:** The venue where university leadership signs articulation agreements and dual-degree pathways with foreign universities.\n\n"
                "Below are verified photographs of our International Partner Flags Display and the Executive Delegation Boardroom!"
            )

        # 16. General Campus Photos, Pictures & Visual Tour
        if any(w in q_lower for w in ["photo", "photos", "pic", "pics", "picture", "pictures", "image", "images", "gallery", "tour", "show me campus"]):
            return (
                "Hello! Here is a visual tour of Presidency University, Bengaluru featuring our verified campus facilities:\n\n"
                "• **Office of International Affairs (OIA):** Our modern blue exterior administrative facility with outdoor student consultation patios and lush campus gardens.\n\n"
                "• **Confluence Convention Center:** The architectural atrium 'Where Global Minds Meet', featuring ceremonial floral archways and partner flags.\n\n"
                "• **Global Collaborations Wall & Plaque:** Displaying our 165+ international university partners and the official Taiwan Education Center (TECC).\n\n"
                "• **Executive Delegation Boardrooms:** State-of-the-art diplomatic suites for international academic conferences.\n\n"
                "Enjoy the verified campus photographs below! You can click any photo to zoom in with our high-resolution lightbox."
            )

        # 17. Handbook and Official Publications
        if "handbook" in q_lower or "pdf" in q_lower or "guide" in q_lower or "rules doc" in q_lower:
            return (
                "Hello! Here are the official Presidency University international student publications and handbooks. You can read, view, or download the complete PDF documents below:\n\n"
                "• **International Student Handbook:** Comprehensive guide detailing arrival rules, S-Visa requirements, the mandatory 14-day e-FRRO registration, hostel policies, and health insurance.\n\n"
                "• **OIA Annual Bulletin & Guide (64 Pages):** Full 64-page publication detailing our 165+ international university partnerships, academic departments, Taiwan TECC center, and campus life.\n\n"
                "• **OIA Department Assessment & FAQs:** Guide on international applicant assessments and student counseling.\n\n"
                "Please find the official download links below to access the full PDF files directly!"
            )

        # 12. General Conversational Synthesizer (ChromaDB Vector Passages)
        if not context or not context.strip():
            return (
                "Hello! Welcome to the Office of International Affairs at Presidency University, Bengaluru!\n\n"
                "I'm here to assist you with any questions regarding admissions, visas, campus life, or university partnerships. "
                "How can I help you today? You can also email us directly at international@presidencyuniversity.in or call +91 80 4925 5555."
            )

        import re
        chunks = [c.strip() for c in context.split("###") if c.strip()]
        stopwords = {"what", "when", "where", "which", "how", "tell", "about", "for", "the", "and", "are", "you", "your", "can", "with", "from", "please", "give"}
        q_words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()) if w not in stopwords]

        scored_chunks = []
        for ch in chunks:
            ch_lower = ch.lower()
            score = sum(3 for w in q_words if w in ch_lower)
            if "category:" in ch_lower:
                score += 2
            scored_chunks.append((score, ch))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        best_chunks = [ch for score, ch in scored_chunks if score > 0][:2]
        if not best_chunks:
            best_chunks = [chunks[0]]

        extracted_facts = []
        for ch in best_chunks:
            lines = [l.strip() for l in ch.split("\n") if l.strip()]
            for l in lines:
                if l.startswith("Category:") or l.startswith("Document:") or l.startswith("---") or l.startswith("Page"):
                    continue
                if l.startswith("###"):
                    continue
                clean_line = re.sub(r'^\d+\.\s*', '', l).strip()
                # Skip incomplete headings or raw section labels
                if clean_line.endswith(":") or len(clean_line) < 20:
                    continue
                if any(clean_line.lower().endswith(w) for w in ["process", "overview", "checklist", "details"]):
                    continue
                if clean_line not in extracted_facts:
                    extracted_facts.append(clean_line)

        intro = "Hello! Thank you for reaching out to the Office of International Affairs. I'd be glad to assist you with that!\n\nBased on our official university guidelines, here are the key details you need to know:"
        bullet_points = [f"• {fact}" for fact in extracted_facts[:4]]
        body = "\n".join(bullet_points) if bullet_points else "Please feel free to contact our counseling desk for customized guidance on your specific application."
        outro = (
            "If you have any further questions or need help with your application, feel free to ask me! "
            "You can also reach our team directly at **international@presidencyuniversity.in** or call **+91 80 4925 5555** "
            "(Monday to Friday, 9:00 AM – 5:00 PM IST)."
        )
        return f"{intro}\n\n{body}\n\n{outro}"

    def generate_response(
        self,
        user_message: str,
        history=None,
        temperature: float = 0.3,
        max_new_tokens: int = 250
    ) -> str:
        """
        [RAG Step 3: GENERATION (Batch)]
        Synchronous response generation using the ChromaDB augmented prompt.
        """
        messages, context, matched_images = self.augment_prompt(user_message, history)

        if self.tokenizer is not None and (self.generator is not None or self.model is not None):
            try:
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
            except Exception as e:
                print(f"Generation warning ({e}); using grounded RAG synthesizer.")
                response = self._grounded_synthesis_fallback(user_message, context)
        else:
            response = self._grounded_synthesis_fallback(user_message, context)

        # Attach official handbook / document link if student inquiry requests documents
        doc_attachment = get_document_handbook_attachment(user_message)
        if doc_attachment:
            response += doc_attachment

        # Attach multimodal campus photos if applicable
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
        [RAG Step 3: GENERATION (Streaming)]
        Token-by-token streaming generator for real-time interactive UI updates.
        Renders verified base64 images seamlessly inline.
        """
        messages, context, matched_images = self.augment_prompt(user_message, history)

        if self.tokenizer is not None and self.model is not None:
            try:
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
            except Exception as e:
                print(f"Streaming error ({e}); streaming grounded synthesis.")
                full_text = self._grounded_synthesis_fallback(user_message, context)
                accumulated_text = ""
                for word in full_text.split(" "):
                    accumulated_text += (word + " ")
                    yield accumulated_text
                    time.sleep(0.01)
        else:
            full_text = self._grounded_synthesis_fallback(user_message, context)
            accumulated_text = ""
            for word in full_text.split(" "):
                accumulated_text += (word + " ")
                yield accumulated_text
                time.sleep(0.01)

        # Attach official handbook / document link if student inquiry requests documents
        doc_attachment = get_document_handbook_attachment(user_message)
        if doc_attachment:
            accumulated_text += doc_attachment
            yield accumulated_text

        # Append inline base64 images if matching photos were found
        if matched_images:
            image_blocks = [format_image_base64_markdown(img_p) for img_p in matched_images]
            if image_blocks:
                accumulated_text += "".join(image_blocks)
                yield accumulated_text

    def process_image_input(self, image_input) -> Dict[str, Any]:
        """
        [MULTIMODAL IMAGE PIPELINE]
        1. Encodes uploaded image into visual embeddings
        2. Decodes image into semantic text and identifies location/plaques
        3. Queries ChromaDB vector database using the decoded text
        4. Synthesizes official university counselor response
        """
        from image_embeddings import decode_image_to_text
        decoded = decode_image_to_text(image_input)
        response = self.generate_response(user_message=decoded["decoded_query"])
        return {
            "decoded_info": decoded,
            "response": response,
            "matched_image": decoded.get("matched_path")
        }


if __name__ == "__main__":
    bot = InternationalOfficeBot()
    test_q = "What is the mandatory 14-day FRRO registration rule for international students?"
    print(f"\nUser: {test_q}")
    ans = bot.generate_response(test_q)
    print(f"\nCounselor:\n{ans}")
