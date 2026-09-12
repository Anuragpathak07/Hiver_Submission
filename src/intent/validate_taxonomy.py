import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TAXONOMY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_taxonomy.json"
)


def main():
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    intents = taxonomy["intents"]

    # 1. Intent count
    assert 8 <= len(intents) <= 12, (
        f"Expected 8-12 intents, got {len(intents)}"
    )

    # 2. Required fields
    required = {
        "intent",
        "description",
        "example_problem",
        "cluster_ids",
        "mapping_reason",
    }

    for item in intents:
        missing = required - set(item.keys())

        assert not missing, (
            f"{item.get('intent', '<unknown>')} "
            f"missing fields: {missing}"
        )

    # 3. Unique intent names
    names = [item["intent"] for item in intents]

    assert len(names) == len(set(names)), (
        "Duplicate intent names found"
    )

    # 4. Cluster coverage
    cluster_ids = []

    for item in intents:
        cluster_ids.extend(item["cluster_ids"])

    counts = Counter(cluster_ids)

    expected = set(range(12))
    actual = set(cluster_ids)

    missing_clusters = expected - actual
    unknown_clusters = actual - expected
    duplicate_clusters = {
        cluster_id
        for cluster_id, count in counts.items()
        if count > 1
    }

    assert not missing_clusters, (
        f"Missing clusters: {missing_clusters}"
    )

    assert not unknown_clusters, (
        f"Unknown clusters: {unknown_clusters}"
    )

    assert not duplicate_clusters, (
        f"Clusters mapped more than once: {duplicate_clusters}"
    )

    print("================================")
    print("TAXONOMY VALIDATION PASSED")
    print("================================")
    print(f"Intents: {len(intents)}")
    print(f"Clusters: {sorted(actual)}")

    print("\nMapping:")
    for item in intents:
        print(
            f"{item['intent']:<40} "
            f"{item['cluster_ids']}"
        )


if __name__ == "__main__":
    main()