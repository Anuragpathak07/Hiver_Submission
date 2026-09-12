import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

# Windows UTF-8 console output setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

# Paths
ALL_CASES_FILE = PROJECT_ROOT / "data" / "processed" / "applesupport_cases_with_intents.csv"
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_annotated.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    print("=" * 80)
    print("INTENT CLASSIFIER EVALUATION: BASELINES VS IMPROVED MODEL")
    print("=" * 80)

    # 1. Load Datasets
    print("\n1. Loading datasets...")
    all_cases = pd.read_csv(ALL_CASES_FILE)
    golden_df = pd.read_csv(GOLDEN_FILE)

    print(f"Total dataset cases: {len(all_cases):,}")
    print(f"Golden Set evaluation cases: {len(golden_df):,}")

    # Exclude Golden Set case IDs from training set to prevent data leakage
    golden_case_ids = set(golden_df["case_id"].astype(str))
    train_df = all_cases[~all_cases["case_id"].astype(str).isin(golden_case_ids)].copy()
    
    print(f"Training dataset size (excluding Golden Set): {len(train_df):,}")

    # Data preparation
    X_train_text = train_df["clean_message"].fillna("").astype(str).tolist()
    y_train = train_df["intent"].astype(str).tolist()

    X_test_text = golden_df["clean_message"].fillna("").astype(str).tolist()
    y_test = golden_df["human_intent"].astype(str).tolist()

    classes = sorted(list(set(y_test)))

    results = {}

    # ---------------------------------------------------------
    # Baseline 1: Majority Class Baseline
    # ---------------------------------------------------------
    print("\n2. Training Baseline 1: Majority Class Baseline...")
    majority_class = pd.Series(y_train).mode()[0]
    y_pred_majority = [majority_class] * len(y_test)

    acc_maj = accuracy_score(y_test, y_pred_majority)
    macro_f1_maj = f1_score(y_test, y_pred_majority, average="macro", zero_division=0)
    weighted_f1_maj = f1_score(y_test, y_pred_majority, average="weighted", zero_division=0)

    results["majority_baseline"] = {
        "accuracy": float(acc_maj),
        "macro_f1": float(macro_f1_maj),
        "weighted_f1": float(weighted_f1_maj),
        "majority_class": majority_class
    }
    print(f"   Accuracy: {acc_maj:.4f} | Macro-F1: {macro_f1_maj:.4f}")

    # ---------------------------------------------------------
    # Baseline 2: TF-IDF + Logistic Regression Baseline
    # ---------------------------------------------------------
    print("\n3. Training Baseline 2: TF-IDF + Logistic Regression Baseline...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    clf_tfidf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_tfidf.fit(X_train_tfidf, y_train)

    y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)

    acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
    macro_f1_tfidf = f1_score(y_test, y_pred_tfidf, average="macro", zero_division=0)
    weighted_f1_tfidf = f1_score(y_test, y_pred_tfidf, average="weighted", zero_division=0)

    results["tfidf_logistic_regression"] = {
        "accuracy": float(acc_tfidf),
        "macro_f1": float(macro_f1_tfidf),
        "weighted_f1": float(weighted_f1_tfidf),
        "per_class_report": classification_report(y_test, y_pred_tfidf, output_dict=True, zero_division=0)
    }
    print(f"   Accuracy: {acc_tfidf:.4f} | Macro-F1: {macro_f1_tfidf:.4f}")

    # ---------------------------------------------------------
    # Improved Model: SentenceTransformer Embeddings + Logistic Regression
    # ---------------------------------------------------------
    print("\n4. Training Improved Model: SentenceTransformer + Logistic Regression...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(CACHE_DIR))

    # Sample a representative slice of non-golden data for fast, high-quality training if full set is large
    # Using 30,000 cases for training fast and dense embedding classifier
    if len(train_df) > 30000:
        train_sub = train_df.sample(n=30000, random_state=42).reset_index(drop=True)
    else:
        train_sub = train_df.reset_index(drop=True)

    X_train_sub_text = train_sub["clean_message"].fillna("").astype(str).tolist()
    y_train_sub = train_sub["intent"].astype(str).tolist()

    print(f"   Encoding {len(train_sub):,} training samples...")
    X_train_emb = embedder.encode(X_train_sub_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"   Encoding {len(golden_df):,} Golden Set test samples...")
    X_test_emb = embedder.encode(X_test_text, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    clf_improved = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf_improved.fit(X_train_emb, y_train_sub)

    y_pred_improved = clf_improved.predict(X_test_emb)

    acc_imp = accuracy_score(y_test, y_pred_improved)
    macro_f1_imp = f1_score(y_test, y_pred_improved, average="macro", zero_division=0)
    weighted_f1_imp = f1_score(y_test, y_pred_improved, average="weighted", zero_division=0)

    results["improved_sentence_transformer_classifier"] = {
        "accuracy": float(acc_imp),
        "macro_f1": float(macro_f1_imp),
        "weighted_f1": float(weighted_f1_imp),
        "per_class_report": classification_report(y_test, y_pred_improved, output_dict=True, zero_division=0)
    }
    print(f"   Accuracy: {acc_imp:.4f} | Macro-F1: {macro_f1_imp:.4f}")

    # ---------------------------------------------------------
    # Comparison Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL COMPARISON ON GOLDEN SET")
    print("=" * 80)
    print(f"{'Model':<45} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 83)
    print(f"{'1. Majority Class Baseline':<45} | {acc_maj:<10.4f} | {macro_f1_maj:<10.4f} | {weighted_f1_maj:<10.4f}")
    print(f"{'2. TF-IDF + Logistic Regression':<45} | {acc_tfidf:<10.4f} | {macro_f1_tfidf:<10.4f} | {weighted_f1_tfidf:<10.4f}")
    print(f"{'3. SentenceTransformer + LogReg (Improved)':<45} | {acc_imp:<10.4f} | {macro_f1_imp:<10.4f} | {weighted_f1_imp:<10.4f}")

    # Save artifacts & results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "intent_evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save best classifier model
    joblib.dump(clf_improved, MODELS_DIR / "intent_classifier.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    print(f"\nSaved evaluation metrics to: {results_path}")
    print(f"Saved trained classifier model to: {MODELS_DIR / 'intent_classifier.joblib'}")

if __name__ == "__main__":
    main()
