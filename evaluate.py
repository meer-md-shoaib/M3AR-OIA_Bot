"""
Presidency University - Office of International Affairs (OIA)
Evaluation Suite: Automated benchmark testing against 10 critical international student inquiry scenarios.
Scores bot grounded accuracy, keyword recall, and policy compliance.
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bot_engine import InternationalOfficeBot

GOLDEN_BENCHMARK = [
    {
        "id": 1,
        "question": "What is the mandatory FRRO registration timeframe and rule for international students after arrival?",
        "expected_keywords": ["14", "frro", "register", "india"],
        "category": "Visa & Immigration Compliance"
    },
    {
        "id": 2,
        "question": "What government and university scholarships are available for foreign students?",
        "expected_keywords": ["iccr", "scholarship", "merit"],
        "category": "Scholarships & Financial Aid"
    },
    {
        "id": 3,
        "question": "How do I apply for an Indian Student Visa and what must it mention?",
        "expected_keywords": ["visa", "student", "presidency"],
        "category": "Visa & Immigration"
    },
    {
        "id": 4,
        "question": "What documents and qualifications are needed for international undergraduate or postgraduate admission?",
        "expected_keywords": ["passport", "transcripts", "aiu"],
        "category": "Admissions & Eligibility"
    },
    {
        "id": 5,
        "question": "How can an international student pay tuition fees from abroad and in what currency?",
        "expected_keywords": ["wire", "swift", "usd"],
        "category": "Fees & Payments"
    },
    {
        "id": 6,
        "question": "What accommodation and hostel facilities are provided for international students on campus?",
        "expected_keywords": ["hostel", "security", "form c"],
        "category": "Campus Life & Housing"
    },
    {
        "id": 7,
        "question": "Tell me about the ICCR scholarship program at Presidency University.",
        "expected_keywords": ["iccr", "cultural", "tuition"],
        "category": "Scholarships"
    },
    {
        "id": 8,
        "question": "Is Presidency University an approved partner under the Study in India program?",
        "expected_keywords": ["study in india", "ministry"],
        "category": "Government Programs"
    },
    {
        "id": 9,
        "question": "What health and medical insurance policy is required for international students in India?",
        "expected_keywords": ["insurance", "10,000", "medical"],
        "category": "Health & Insurance"
    },
    {
        "id": 10,
        "question": "What are the working hours, email address, and location of the Office of International Affairs?",
        "expected_keywords": ["monday", "friday", "international@presidencyuniversity.in"],
        "category": "Office Contacts"
    }
]

def run_evaluation():
    print("=" * 70)
    print("PRESIDENCY UNIVERSITY OIA BOT - ACCURACY EVALUATION SUITE")
    print("=" * 70)
    print(f"Total Benchmark Tests: {len(GOLDEN_BENCHMARK)}")
    print("Loading AI Counselor and Knowledge Indices...")

    start_init = time.time()
    bot = InternationalOfficeBot()
    print(f"Bot initialized in {time.time() - start_init:.2f}s\n")

    passed_tests = 0
    start_eval = time.time()

    for item in GOLDEN_BENCHMARK:
        q_id = item["id"]
        question = item["question"]
        expected_kws = item["expected_keywords"]
        cat = item["category"]

        print(f"[{q_id:02d}/10] Testing: {cat}")
        print(f"       Inquiry: \"{question}\"")

        t0 = time.time()
        response = bot.generate_response(user_message=question, temperature=0.2, max_new_tokens=220)
        gen_time = time.time() - t0

        resp_lower = response.lower()
        matched_kws = [kw for kw in expected_kws if kw.lower() in resp_lower]
        missing_kws = [kw for kw in expected_kws if kw.lower() not in resp_lower]

        is_pass = len(matched_kws) >= max(2, int(len(expected_kws) * 0.7))

        if is_pass:
            passed_tests += 1
            print(f"       Status: PASS ({len(matched_kws)}/{len(expected_kws)} keywords hit in {gen_time:.2f}s)")
        else:
            print(f"       Status: FAIL (Missing keywords: {missing_kws})")

        print(f"       Sample Response: {response.replace('\n', ' ')[:140]}...\n")

    total_time = time.time() - start_eval
    score_pct = (passed_tests / len(GOLDEN_BENCHMARK)) * 100

    print("=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed_tests} / {len(GOLDEN_BENCHMARK)} benchmark tests")
    print(f"Overall Accuracy: {score_pct:.1f}%")
    print(f"Total Evaluation Time: {total_time:.2f}s (Avg {total_time/len(GOLDEN_BENCHMARK):.2f}s per query)")

    if score_pct >= 80:
        print("Result: BENCHMARK PASSED (Production Ready)")
    else:
        print("Result: BENCHMARK BELOW THRESHOLD")
    print("=" * 70)

    return passed_tests, len(GOLDEN_BENCHMARK)

if __name__ == "__main__":
    passed, total = run_evaluation()
    sys.exit(0 if passed >= 8 else 1)
