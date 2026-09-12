import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TAXONOMY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_taxonomy.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cluster_to_intent.json"
)


def main():

    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    mapping = {}

    for item in taxonomy["intents"]:
        intent = item["intent"]

        for cluster_id in item["cluster_ids"]:

            cluster_id = str(cluster_id)

            if cluster_id in mapping:
                raise ValueError(
                    f"Cluster {cluster_id} mapped twice"
                )

            mapping[cluster_id] = intent

    expected = {str(i) for i in range(12)}

    if set(mapping.keys()) != expected:
        raise ValueError(
            f"Invalid cluster coverage.\n"
            f"Expected: {expected}\n"
            f"Actual: {set(mapping.keys())}"
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            mapping,
            f,
            indent=2
        )

    print(f"Saved: {OUTPUT_FILE}")

    print("\nCluster → Intent")

    for cluster_id in sorted(
        mapping,
        key=lambda x: int(x)
    ):
        print(
            f"Cluster {cluster_id:2} → "
            f"{mapping[cluster_id]}"
        )


if __name__ == "__main__":
    main()