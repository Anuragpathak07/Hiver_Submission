import os
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Windows UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Same HuggingFace cache as cluster_intents.py
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applesupport_cases_clean.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_kmeans.joblib"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cluster_to_intent.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applesupport_cases_with_intents.csv"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BATCH_SIZE = 64


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading cases...")

df = pd.read_csv(INPUT_FILE)

df = df[
    df["customer_message"].notna()
].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

df = df[
    df["customer_message"].str.len() >= 10
].copy()

df = df.reset_index(drop=True)

print(f"Cases to classify: {len(df):,}")


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL,
    cache_folder=str(CACHE_DIR)
)

print("Loading KMeans model...")

kmeans = joblib.load(MODEL_FILE)


# ============================================================
# 3. LOAD CLUSTER → INTENT MAPPING
# ============================================================

print("Loading taxonomy mapping...")

with open(
    MAPPING_FILE,
    "r",
    encoding="utf-8"
) as f:
    cluster_to_intent = json.load(f)

print(
    f"Loaded {len(cluster_to_intent)} "
    "cluster mappings"
)


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

embeddings = model.encode(
    df["clean_message"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings)


# ============================================================
# 5. PREDICT CLUSTERS
# ============================================================

print("\nPredicting clusters...")

cluster_ids = kmeans.predict(
    embeddings
)

df["cluster_id"] = cluster_ids


# ============================================================
# 6. MAP CLUSTER → INTENT
# ============================================================

print("Mapping clusters to intents...")

df["intent"] = [
    cluster_to_intent[str(cluster_id)]
    for cluster_id in cluster_ids
]


# ============================================================
# 7. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved classified cases to:"
)
print(OUTPUT_FILE)


# ============================================================
# 8. DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("INTENT DISTRIBUTION")
print("=" * 70)

distribution = (
    df["intent"]
    .value_counts()
    .rename_axis("intent")
    .reset_index(name="count")
)

distribution["percentage"] = (
    distribution["count"]
    / len(df)
    * 100
)

print(
    distribution.to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)