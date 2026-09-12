from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_annotated.csv"
)


# ============================================================
# Intent taxonomy
# ============================================================

INTENTS = [
    "account_identity_issue",
    "battery_life_issue",
    "ios_update_issues",
    "iphone_hardware_issue",
    "keyboard_autocorrect_issue",
    "macbook_issues",
    "music_audio_issue",
    "other_unclear",
]


def print_intents():
    print("\nChoose the correct intent:\n")

    for i, intent in enumerate(INTENTS, start=1):
        print(f"  {i}. {intent}")

    print("\nCommands:")
    print("  c = accept suggested intent")
    print("  s = skip")
    print("  q = save and quit")


def annotate_case(row):
    print("\n" + "=" * 80)

    print(f"Case ID: {row['case_id']}")

    print("\nCustomer message:")
    print("-" * 80)
    print(row["customer_message"])

    print("\nSuggested intent:")
    print("-" * 80)
    print(row["intent"])

    print_intents()

    while True:
        choice = input("\nYour choice: ").strip().lower()

        # Accept suggested label
        if choice == "c":
            selected_intent = row["intent"]

            print(f"\nAccepted suggested intent: {selected_intent}")

            notes = input(
                "Annotation note "
                "(press Enter if none): "
            ).strip()

            return selected_intent, notes

        # Skip
        if choice == "s":
            return "SKIP", ""

        # Save and quit
        if choice == "q":
            return "QUIT", ""

        # Select another intent
        if choice.isdigit():
            number = int(choice)

            if 1 <= number <= len(INTENTS):
                selected_intent = INTENTS[number - 1]

                print(f"\nSelected: {selected_intent}")

                notes = input(
                    "Annotation note "
                    "(press Enter if none): "
                ).strip()

                return selected_intent, notes

        print(
            "Invalid choice. Enter a number, c, s, or q."
        )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Create annotation columns if they don't exist
    # --------------------------------------------------------

    if "human_intent" not in df.columns:
        df["human_intent"] = pd.Series("", index=df.index, dtype="string")
    else:
        df["human_intent"] = df["human_intent"].fillna("").astype("string")

    if "annotation_notes" not in df.columns:
        df["annotation_notes"] = pd.Series("", index=df.index, dtype="string")
    else:
        df["annotation_notes"] = df["annotation_notes"].fillna("").astype("string")

    if "label_source" not in df.columns:
        df["label_source"] = pd.Series("", index=df.index, dtype="string")
    else:
        df["label_source"] = df["label_source"].fillna("").astype("string")

    # --------------------------------------------------------
    # Resume from previous progress
    # --------------------------------------------------------

    completed = (
        df["human_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    )

    remaining = df[~completed]

    total = len(df)
    already_done = completed.sum()

    print("=" * 80)
    print("APPLE SUPPORT GOLDEN SET ANNOTATION")
    print("=" * 80)

    print(f"\nTotal cases:      {total}")
    print(f"Already labelled: {already_done}")
    print(f"Remaining:        {len(remaining)}")

    if len(remaining) == 0:
        print("\nAll cases have already been annotated.")
        print(f"Saved file: {OUTPUT_FILE}")
        return

    # --------------------------------------------------------
    # Annotate cases
    # --------------------------------------------------------

    for index in remaining.index:

        row = df.loc[index]

        selected_intent, notes = annotate_case(row)

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if selected_intent == "QUIT":
            df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
            )

            print("\nProgress saved.")
            print(f"File: {OUTPUT_FILE}")
            return

        # ----------------------------------------------------
        # Skip
        # ----------------------------------------------------

        if selected_intent == "SKIP":
            print("Skipped.")
            continue

        # ----------------------------------------------------
        # Save annotation
        # ----------------------------------------------------

        df.loc[index, "human_intent"] = selected_intent
        df.loc[index, "annotation_notes"] = notes
        df.loc[index, "label_source"] = "human_verified"

        # Save after every annotation
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        completed_count = (
            df["human_intent"]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ).sum()

        print(
            f"\nSaved. Progress: "
            f"{completed_count}/{total}"
        )

    print("\n" + "=" * 80)
    print("ANNOTATION COMPLETE")
    print("=" * 80)

    print(f"\nGolden set saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()