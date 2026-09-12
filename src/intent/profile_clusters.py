import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_clusters.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cluster_profiles.csv"
)

TOP_KEYWORDS = 10
TOP_EXAMPLES = 8


# ============================================================
# LOAD
# ============================================================

print("Loading clustered data...")

df = pd.read_csv(INPUT_FILE)

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
)

print(f"Cases: {len(df):,}")
print(f"Clusters: {df['cluster_id'].nunique()}")


# ============================================================
# TF-IDF
# ============================================================

print("\nCalculating TF-IDF keywords...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    max_features=10000
)

X = vectorizer.fit_transform(df["customer_message"])

feature_names = np.array(vectorizer.get_feature_names_out())


# ============================================================
# PROFILE EACH CLUSTER
# ============================================================

profiles = []

for cluster_id in sorted(df["cluster_id"].unique()):

    cluster_indices = np.where(
        df["cluster_id"].values == cluster_id
    )[0]

    cluster_matrix = X[cluster_indices]

    # Average TF-IDF score within cluster
    mean_scores = np.asarray(
        cluster_matrix.mean(axis=0)
    ).flatten()

    top_indices = mean_scores.argsort()[
        ::-1
    ][:TOP_KEYWORDS]

    keywords = feature_names[top_indices]

    cluster_df = df[
        df["cluster_id"] == cluster_id
    ]

    # Pick examples with highest TF-IDF density
    example_scores = np.asarray(
        cluster_matrix.sum(axis=1)
    ).flatten()

    best_examples = cluster_indices[
        np.argsort(example_scores)[::-1][:TOP_EXAMPLES]
    ]

    examples = [
        df.iloc[idx]["customer_message"]
        for idx in best_examples
    ]

    profiles.append({
        "cluster_id": cluster_id,
        "cluster_size": len(cluster_df),
        "keywords": " | ".join(keywords),
        "examples": " || ".join(examples)
    })


# ============================================================
# SAVE
# ============================================================

profiles_df = pd.DataFrame(profiles)

profiles_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCluster profiles:")

for _, row in profiles_df.iterrows():

    print("\n" + "=" * 80)

    print(
        f"CLUSTER {int(row['cluster_id'])}"
        f" | SIZE: {int(row['cluster_size'])}"
    )

    print(
        f"KEYWORDS: {row['keywords']}"
    )

    print("EXAMPLES:")

    for example in row["examples"].split(" || "):
        print(f"- {example}")

print("\nSaved:")
print(OUTPUT_FILE)