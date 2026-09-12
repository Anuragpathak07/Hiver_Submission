from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "golden" / "golden_annotated.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nGOLDEN SET VALIDATION")
print("=" * 50)


# --------------------------------------------------
# Expected taxonomy
# --------------------------------------------------

VALID_INTENTS = [
    "ios_update_issues",
    "other_unclear",
    "macbook_issues",
    "keyboard_autocorrect_issue",
    "music_audio_issue",
    "iphone_hardware_issue",
    "account_identity_issue",
    "battery_life_issue",
]


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

total = len(df)

missing_human_labels = df["human_intent"].isna().sum()

invalid_human_labels = (
    ~df["human_intent"].isin(VALID_INTENTS)
).sum()

human_verified = (
    df["label_source"].astype(str).str.lower() == "human_verified"
).sum()


print(f"Total examples:       {total}")
print(f"Human labels:         {total - missing_human_labels}")
print(f"Missing labels:       {missing_human_labels}")
print(f"Invalid labels:       {invalid_human_labels}")
print(f"Human verified:       {human_verified}")


# --------------------------------------------------
# Silver vs Human agreement
# --------------------------------------------------

agreement = (
    df["intent"] == df["human_intent"]
).mean()

print(
    f"\nSilver-label agreement: "
    f"{agreement:.2%}"
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

print("\nCONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(
    df["human_intent"],
    df["intent"],
    labels=VALID_INTENTS,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"TRUE: {x}" for x in VALID_INTENTS],
    columns=[f"PRED: {x}" for x in VALID_INTENTS],
)

print(cm_df)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        df["human_intent"],
        df["intent"],
        labels=VALID_INTENTS,
        zero_division=0,
        digits=3,
    )
)


# --------------------------------------------------
# Disagreements
# --------------------------------------------------

disagreements = df[
    df["intent"] != df["human_intent"]
][
    [
        "case_id",
        "customer_message",
        "intent",
        "human_intent",
        "annotation_notes",
    ]
]

print("\nDISAGREEMENTS")
print("=" * 50)

print(f"Number of disagreements: {len(disagreements)}")

if len(disagreements) > 0:
    print(disagreements.to_string(index=False))


# --------------------------------------------------
# Final checks
# --------------------------------------------------

print("\nFINAL CHECKS")
print("=" * 50)

checks_passed = True

if total < 150 or total > 250:
    print("FAIL: Golden set must contain 150–250 examples.")
    checks_passed = False

if missing_human_labels > 0:
    print("FAIL: Some examples are missing human_intent.")
    checks_passed = False

if invalid_human_labels > 0:
    print("FAIL: Invalid human intent labels found.")
    checks_passed = False

if human_verified != total:
    print("FAIL: Not all examples are marked human_verified.")
    checks_passed = False

if checks_passed:
    print("ALL GOLDEN SET CHECKS PASSED")
else:
    print("GOLDEN SET VALIDATION FAILED")