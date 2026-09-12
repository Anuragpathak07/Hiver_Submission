import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Windows UTF-8 console setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

class ResponseGenerator:
    def __init__(self):
        self.client = None
        if GROQ_AVAILABLE and GROQ_API_KEY and not GROQ_API_KEY.startswith("your_"):
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
            except Exception:
                self.client = None

    def generate_response(self, customer_message: str, intent: str, retrieved_cases: list) -> dict:
        """
        Generates a support response grounded strictly in retrieved historical cases.
        """
        evidence_snippets = []
        for i, case in enumerate(retrieved_cases, 1):
            evidence_snippets.append(
                f"Evidence #{i} (Case ID: {case.get('case_id', 'N/A')}, Similarity: {case.get('similarity_score', 0):.2f}):\n"
                f"Customer Query: {case.get('customer_message', '')}\n"
                f"Historical Response: {case.get('final_support_response', '')}\n"
            )

        evidence_text = "\n".join(evidence_snippets) if evidence_snippets else "No historical evidence found."

        system_prompt = (
            "You are an expert AI customer support agent for AppleSupport on Twitter.\n"
            "Your task is to draft a concise, empathetic, and professional support reply to the customer.\n"
            "CRITICAL RULE: Base your reply strictly on the provided historical support evidence. "
            "Do NOT invent unsupported troubleshooting steps, URLs, or promises not grounded in the evidence.\n"
            "If the customer requires diagnostic steps, ask concise questions or provide standard AppleSupport links from evidence."
        )

        user_prompt = (
            f"Customer Message: \"{customer_message}\"\n"
            f"Detected Intent: {intent}\n\n"
            f"Historical Evidence:\n{evidence_text}\n\n"
            f"Draft a response to the customer."
        )

        # Use Groq API if configured and available
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024
                )
                draft = response.choices[0].message.content.strip()
                # Clean any reasoning tags if emitted by model
                if "<think>" in draft and "</think>" in draft:
                    draft = draft.split("</think>")[-1].strip()
                return {
                    "generated_response": draft,
                    "model": MODEL_NAME,
                    "grounded_evidence_case_ids": [c.get("case_id") for c in retrieved_cases]
                }
            except Exception as e:
                # Fallback to grounded rule template if Groq call fails
                pass

        # Grounded Fallback Response Generator
        if retrieved_cases and retrieved_cases[0].get("final_support_response"):
            fallback_response = (
                f"Thanks for reaching out! {retrieved_cases[0]['final_support_response']} "
                f"Feel free to send us a DM if you need further help."
            )
        else:
            fallback_response = (
                "Thanks for reaching out to AppleSupport! We'd love to help get this resolved. "
                "Please send us a DM with your device model and current iOS version to get started."
            )

        return {
            "generated_response": fallback_response,
            "model": "grounded_template_fallback",
            "grounded_evidence_case_ids": [c.get("case_id") for c in retrieved_cases]
        }

if __name__ == "__main__":
    generator = ResponseGenerator()
    test_retrieved = [{
        "case_id": "apple_101",
        "customer_message": "Battery life is terrible on iOS 11",
        "final_support_response": "We're here to help. Send us a DM with your iOS version in Settings > General > About.",
        "similarity_score": 0.89
    }]
    res = generator.generate_response("My battery is dying fast after update", "battery_life_issue", test_retrieved)
    print("Generated Response:")
    print(res["generated_response"])
