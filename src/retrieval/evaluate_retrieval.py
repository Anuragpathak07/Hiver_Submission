import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(PROJECT_ROOT / "src"))

from retrieval_engine import RetrievalEngine
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_annotated.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

def evaluate_retrieval():
    print("=" * 80)
    print("HISTORICAL RESPONSE RETRIEVAL EVALUATION (LEXICAL VS SEMANTIC)")
    print("=" * 80)

    golden_df = pd.read_csv(GOLDEN_FILE)
    engine = RetrievalEngine(sample_size=30000)

    top_k = 3
    results_summary = {}

    for method in ["lexical", "semantic"]:
        print(f"\nEvaluating {method.upper()} retrieval on {len(golden_df)} Golden Set queries...")
        intent_match_counts = []
        similarity_scores = []

        for _, row in golden_df.iterrows():
            query = row["clean_message"]
            true_intent = row["human_intent"]

            retrieved = engine.retrieve(query, top_k=top_k, method=method)
            if not retrieved:
                continue

            # Check if any top-K retrieved item shares the customer's true intent
            retrieved_intents = [item["intent"] for item in retrieved]
            intent_match = any(intent == true_intent for intent in retrieved_intents)
            intent_match_counts.append(1 if intent_match else 0)

            # Average similarity score of top-1
            similarity_scores.append(retrieved[0]["similarity_score"])

        intent_match_rate = float(np.mean(intent_match_counts))
        avg_top1_sim = float(np.mean(similarity_scores))

        results_summary[method] = {
            "intent_match_rate_at_k": intent_match_rate,
            "avg_top1_similarity": avg_top1_sim,
            "top_k": top_k
        }

        print(f"   {method.capitalize()} Intent Match Rate @ {top_k}: {intent_match_rate:.4f} ({intent_match_rate*100:.2f}%)")
        print(f"   {method.capitalize()} Avg Top-1 Similarity: {avg_top1_sim:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "retrieval_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nSaved retrieval evaluation metrics to: {out_path}")

if __name__ == "__main__":
    evaluate_retrieval()
