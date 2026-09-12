import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applesupport_cases_with_intents.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_candidates.csv"
)

TOTAL_SAMPLES = 200
RANDOM_STATE = 42


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Total cases: {len(df):,}")

    # Number of examples per intent
    intents = sorted(df["intent"].unique())

    samples_per_intent = TOTAL_SAMPLES // len(intents)

    remainder = TOTAL_SAMPLES % len(intents)

    sampled_parts = []

    for i, intent in enumerate(intents):

        intent_df = df[
            df["intent"] == intent
        ]

        n = samples_per_intent

        if i < remainder:
            n += 1

        n = min(n, len(intent_df))

        sample = intent_df.sample(
            n=n,
            random_state=RANDOM_STATE + i
        )

        sampled_parts.append(sample)

    golden = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    # Shuffle final dataset
    golden = golden.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Add human annotation column
    golden["human_intent"] = ""

    # Add optional notes column
    golden["annotation_notes"] = ""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    golden.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved {len(golden)} "
        f"golden candidates to:"
    )

    print(OUTPUT_FILE)

    print("\nCandidate distribution:")

    print(
        golden["intent"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()