import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Project root:
# hiver-support-agent/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fix Windows HuggingFace cache symlink permission errors by setting local cache directory
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applesupport_cases_clean.csv"
)
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "intent_clusters.csv"

SAMPLE_SIZE = 10000
N_CLUSTERS = 12
RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading cases...")

df = pd.read_csv(INPUT_FILE)

df = df[df["customer_message"].notna()].copy()
df["customer_message"] = df["customer_message"].astype(str).str.strip()

# Remove extremely short messages
df = df[df["customer_message"].str.len() >= 10]

print(f"Total usable cases: {len(df):,}")


# ============================================================
# 2. SAMPLE DATA
# ============================================================

if len(df) > SAMPLE_SIZE:
    sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)
else:
    sample = df.reset_index(drop=True)

print(f"Clustering {len(sample):,} cases")


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(CACHE_DIR)
)

print("Creating embeddings...")

embeddings = model.encode(
    sample["clean_message"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings)


# ============================================================
# 4. CLUSTER
# ============================================================

print("\nClustering...")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

sample["cluster_id"] = kmeans.fit_predict(embeddings)


# ============================================================
# 5. FIND REPRESENTATIVE EXAMPLES
# ============================================================

print("\nFinding representative examples...")

# Distance from each example to its cluster center
distances = kmeans.transform(embeddings)
MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_kmeans.joblib"
)

joblib.dump(kmeans, MODEL_FILE)

print(f"Saved KMeans model to: {MODEL_FILE}")

representatives = []

for cluster_id in range(N_CLUSTERS):

    cluster_indices = np.where(
        sample["cluster_id"].values == cluster_id
    )[0]

    # Closest examples to cluster centroid
    cluster_distances = distances[
        cluster_indices,
        cluster_id
    ]

    closest_indices = cluster_indices[
        np.argsort(cluster_distances)[:10]
    ]

    for idx in closest_indices:
        representatives.append({
            "cluster_id": cluster_id,
            "customer_message": sample.iloc[idx]["clean_message"]
        })


representative_df = pd.DataFrame(representatives)


# ============================================================
# 6. SHOW CLUSTERS
# ============================================================

print("\n" + "=" * 80)
print("CLUSTER SUMMARY")
print("=" * 80)

for cluster_id in range(N_CLUSTERS):

    cluster_data = representative_df[
        representative_df["cluster_id"] == cluster_id
    ]

    count = (sample["cluster_id"] == cluster_id).sum()

    print(f"\nCLUSTER {cluster_id}  ({count:,} examples)")
    print("-" * 60)

    for message in cluster_data["customer_message"].head(10):
        print(f"- {message}")


# ============================================================
# 7. SAVE RESULTS
# ============================================================

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

sample[
    [
        "case_id",
        "customer_message",
        "cluster_id"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)