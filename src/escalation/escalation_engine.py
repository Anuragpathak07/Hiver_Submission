import sys
from pathlib import Path

# Windows UTF-8 console setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HIGH_RISK_INTENTS = {"account_identity_issue", "other_unclear"}
SENSITIVE_KEYWORDS = {"billing", "charged", "invoice", "refund", "hacked", "stolen", "security", "lawsuit", "police", "legal"}

class EscalationEngine:
    def __init__(self, min_classifier_confidence: float = 0.50, min_retrieval_similarity: float = 0.40):
        self.min_classifier_confidence = min_classifier_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def evaluate(self, customer_message: str, predicted_intent: str, classifier_confidence: float, retrieved_cases: list) -> dict:
        """
        Determines whether to AUTO_HANDLE or ESCALATE customer query.
        """
        msg_lower = customer_message.lower()

        # Rule 1: High-Risk Intent
        if predicted_intent in HIGH_RISK_INTENTS:
            return {
                "decision": "ESCALATE",
                "escalate": True,
                "escalation_reason": f"High-risk or ambiguous intent detected: '{predicted_intent}'. Requires human specialist.",
                "rule_triggered": "HIGH_RISK_INTENT"
            }

        # Rule 2: Low Classifier Confidence
        if classifier_confidence < self.min_classifier_confidence:
            return {
                "decision": "ESCALATE",
                "escalate": True,
                "escalation_reason": f"Low intent classification confidence ({classifier_confidence:.2f} < threshold {self.min_classifier_confidence}).",
                "rule_triggered": "LOW_CLASSIFIER_CONFIDENCE"
            }

        # Rule 3: Poor Retrieval Quality / Lack of Evidence
        top_sim = retrieved_cases[0]["similarity_score"] if retrieved_cases else 0.0
        if top_sim < self.min_retrieval_similarity:
            return {
                "decision": "ESCALATE",
                "escalate": True,
                "escalation_reason": f"Insufficient retrieval evidence (top similarity {top_sim:.2f} < threshold {self.min_retrieval_similarity}).",
                "rule_triggered": "INSUFFICIENT_RETRIEVAL_EVIDENCE"
            }

        # Rule 4: Sensitive Keyword Trigger
        for kw in SENSITIVE_KEYWORDS:
            if kw in msg_lower:
                return {
                    "decision": "ESCALATE",
                    "escalate": True,
                    "escalation_reason": f"Sensitive financial or security term detected: '{kw}'. Escalating for human safety.",
                    "rule_triggered": "SENSITIVE_KEYWORD"
                }

        # Auto Handle
        return {
            "decision": "AUTO_HANDLE",
            "escalate": False,
            "escalation_reason": f"Sufficient confidence ({classifier_confidence:.2f}), strong retrieval grounding ({top_sim:.2f}), and low risk intent.",
            "rule_triggered": "AUTO_HANDLE_RULES_PASSED"
        }

if __name__ == "__main__":
    esc = EscalationEngine()
    print("Testing Escalation Engine...")
    test_res = esc.evaluate(
        customer_message="I was charged $20 unexpectedly",
        predicted_intent="other_unclear",
        classifier_confidence=0.85,
        retrieved_cases=[{"similarity_score": 0.65}]
    )
    print(test_res)
