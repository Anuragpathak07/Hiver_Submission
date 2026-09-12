import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Windows UTF-8 console setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

CASES_FILE = PROJECT_ROOT / "data" / "processed" / "applesupport_cases_with_intents.csv"
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_annotated.csv"
RETRIEVAL_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "retrieval_cache"

class RetrievalEngine:
    def __init__(self, sample_size: int = 40000, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sample_size = sample_size
        self.model_name = embedding_model
        self.corpus_df = None
        self.semantic_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embedder = None
        
        self._load_corpus_and_build_indices()

    def _load_corpus_and_build_indices(self):
        RETRIEVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        corpus_csv_path = RETRIEVAL_CACHE_DIR / "retrieval_corpus.csv"
        embeddings_path = RETRIEVAL_CACHE_DIR / "retrieval_embeddings.npy"

        all_cases = pd.read_csv(CASES_FILE)
        golden_df = pd.read_csv(GOLDEN_FILE)
        golden_ids = set(golden_df["case_id"].astype(str))

        # Filter strictly for cases with support responses, excluding Golden Set
        valid_cases = all_cases[
            (all_cases["evidence_status"] == "SUPPORT_RESPONSE_PRESENT") &
            (all_cases["final_support_response"].notna()) &
            (~all_cases["case_id"].astype(str).isin(golden_ids))
        ].copy()

        valid_cases = valid_cases.reset_index(drop=True)

        if len(valid_cases) > self.sample_size:
            self.corpus_df = valid_cases.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        else:
            self.corpus_df = valid_cases

        # Build Lexical TF-IDF index
        print("Building Lexical TF-IDF retrieval index...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2)
        corpus_texts = self.corpus_df["clean_message"].fillna("").astype(str).tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)

        # Build Semantic Embedding index
        print("Building Semantic Embedding retrieval index...")
        self.embedder = SentenceTransformer(self.model_name, cache_folder=str(CACHE_DIR))

        if embeddings_path.exists() and len(np.load(embeddings_path)) == len(self.corpus_df):
            print("Loading precomputed retrieval embeddings from cache...")
            self.semantic_embeddings = np.load(embeddings_path)
        else:
            print(f"Computing semantic embeddings for {len(self.corpus_df):,} corpus items...")
            self.semantic_embeddings = self.embedder.encode(
                corpus_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
            )
            np.save(embeddings_path, self.semantic_embeddings)
            self.corpus_df.to_csv(corpus_csv_path, index=False)

    def retrieve(self, query_text: str, top_k: int = 3, method: str = "semantic"):
        clean_query = str(query_text).strip()
        if not clean_query:
            return []

        if method == "lexical":
            query_vec = self.tfidf_vectorizer.transform([clean_query])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        elif method == "semantic":
            query_emb = self.embedder.encode([clean_query], normalize_embeddings=True)
            sim_scores = cosine_similarity(query_emb, self.semantic_embeddings).flatten()
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.corpus_df.iloc[idx]
            results.append({
                "case_id": str(row["case_id"]),
                "customer_message": str(row["customer_message"]),
                "final_support_response": str(row["final_support_response"]),
                "intent": str(row["intent"]),
                "similarity_score": float(sim_scores[idx])
            })

        return results

if __name__ == "__main__":
    print("Testing Retrieval Engine...")
    engine = RetrievalEngine(sample_size=10000)
    query = "My battery is draining really fast on iOS 11"
    print(f"\nQuery: {query}\n")
    print("Semantic Results:")
    for res in engine.retrieve(query, top_k=2, method="semantic"):
        print(f"- [Score: {res['similarity_score']:.4f}] ({res['intent']}) {res['final_support_response']}")
