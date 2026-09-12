import os
import sys
import json
import argparse
from pathlib import Path
import joblib
import numpy as np

# Windows UTF-8 console setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add src to path
sys.path.append(str(PROJECT_ROOT / "src"))

from sentence_transformers import SentenceTransformer
from retrieval.retrieval_engine import RetrievalEngine
from generation.generator import ResponseGenerator
from escalation.escalation_engine import EscalationEngine

MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "intent_classifier.joblib"

class SupportAgentPipeline:
    def __init__(self, sample_retrieval_size: int = 30000):
        print("Initializing Hiver Support Agent Pipeline...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(CACHE_DIR))
        self.intent_classifier = joblib.load(MODEL_PATH)
        self.retrieval_engine = RetrievalEngine(sample_size=sample_retrieval_size)
        self.response_generator = ResponseGenerator()
        self.escalation_engine = EscalationEngine()
        print("Pipeline initialization complete!")

    def process_query(self, customer_message: str) -> dict:
        clean_msg = customer_message.strip()

        # 1. Intent Classification
        query_emb = self.embedder.encode([clean_msg], normalize_embeddings=True)
        probs = self.intent_classifier.predict_proba(query_emb)[0]
        max_idx = np.argmax(probs)
        predicted_intent = str(self.intent_classifier.classes_[max_idx])
        confidence = float(probs[max_idx])

        # 2. Historical Case Retrieval
        retrieved = self.retrieval_engine.retrieve(clean_msg, top_k=3, method="semantic")

        # 3. Escalation Decision
        escalation_result = self.escalation_engine.evaluate(
            customer_message=clean_msg,
            predicted_intent=predicted_intent,
            classifier_confidence=confidence,
            retrieved_cases=retrieved
        )

        # 4. Response Generation
        generation_result = self.response_generator.generate_response(
            customer_message=clean_msg,
            intent=predicted_intent,
            retrieved_cases=retrieved
        )

        return {
            "customer_message": clean_msg,
            "intent_classification": {
                "predicted_intent": predicted_intent,
                "confidence": confidence
            },
            "retrieval": {
                "retrieved_cases_count": len(retrieved),
                "top_cases": retrieved
            },
            "escalation": escalation_result,
            "response_generation": generation_result
        }

def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent Pipeline CLI")
    parser.add_argument("--query", type=str, default="My iPhone battery drains within 2 hours after updating to iOS 11.", help="Customer support query")
    args = parser.parse_args()

    pipeline = SupportAgentPipeline(sample_retrieval_size=10000)
    result = pipeline.process_query(args.query)

    print("\n" + "=" * 80)
    print("HIVER SUPPORT AGENT PIPELINE RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
