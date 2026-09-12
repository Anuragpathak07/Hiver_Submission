import os
import sys
import json
import pandas as pd
from pathlib import Path

# Windows UTF-8 setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_annotated.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

sys.path.append(str(PROJECT_ROOT / "src"))
from pipeline import SupportAgentPipeline

def run_evaluation_harness():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE EVALUATION HARNESS")
    print("=" * 80)

    golden_df = pd.read_csv(GOLDEN_FILE)
    print(f"Loaded {len(golden_df)} Golden Set evaluation records.")

    pipeline = SupportAgentPipeline(sample_retrieval_size=20000)

    harness_results = []
    
    for idx, row in golden_df.iterrows():
        case_id = str(row["case_id"])
        query = str(row["clean_message"])
        human_intent = str(row["human_intent"])

        res = pipeline.process_query(query)

        pred_intent = res["intent_classification"]["predicted_intent"]
        confidence = res["intent_classification"]["confidence"]
        escalation = res["escalation"]
        retrieved = res["retrieval"]["top_cases"]

        intent_correct = (pred_intent == human_intent)
        top1_sim = retrieved[0]["similarity_score"] if retrieved else 0.0
        retrieval_intent_match = any(c["intent"] == human_intent for c in retrieved)

        harness_results.append({
            "case_id": case_id,
            "query": query,
            "human_intent": human_intent,
            "predicted_intent": pred_intent,
            "confidence": confidence,
            "intent_correct": intent_correct,
            "retrieval_top1_similarity": top1_sim,
            "retrieval_intent_match": retrieval_intent_match,
            "escalate": escalation["escalate"],
            "escalation_reason": escalation["escalation_reason"],
            "rule_triggered": escalation["rule_triggered"]
        })

    results_df = pd.DataFrame(harness_results)

    # Metrics computation
    accuracy = float(results_df["intent_correct"].mean())
    retrieval_match_rate = float(results_df["retrieval_intent_match"].mean())
    avg_top1_sim = float(results_df["retrieval_top1_similarity"].mean())
    escalation_rate = float(results_df["escalate"].mean())
    auto_handle_rate = float(1.0 - escalation_rate)

    summary = {
        "dataset_size": len(golden_df),
        "intent_accuracy": accuracy,
        "retrieval_intent_match_rate_at_3": retrieval_match_rate,
        "retrieval_avg_top1_similarity": avg_top1_sim,
        "escalation_distribution": {
            "escalated_count": int(results_df["escalate"].sum()),
            "auto_handled_count": int((~results_df["escalate"]).sum()),
            "escalation_rate": escalation_rate,
            "auto_handle_rate": auto_handle_rate
        },
        "escalation_rules_breakdown": results_df["rule_triggered"].value_counts().to_dict()
    }

    print("\n" + "=" * 80)
    print("EVALUATION HARNESS SUMMARY RESULTS")
    print("=" * 80)
    print(f"Intent Classification Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Retrieval Intent Match Rate @ 3:    {retrieval_match_rate:.4f} ({retrieval_match_rate*100:.2f}%)")
    print(f"Retrieval Top-1 Avg Similarity:      {avg_top1_sim:.4f}")
    print(f"Escalation Rate (Human Specialist): {escalation_rate:.4f} ({escalation_rate*100:.2f}%)")
    print(f"Auto-Handle Rate (AI Agent):        {auto_handle_rate:.4f} ({auto_handle_rate*100:.2f}%)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "full_evaluation_harness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved comprehensive evaluation report to: {out_file}")

if __name__ == "__main__":
    run_evaluation_harness()
