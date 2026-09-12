import re
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applesupport_cases.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applesupport_cases_clean.csv"
)


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtags but preserve the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize repeated punctuation
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    # Normalize repeated characters
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    return text.strip()


print("Loading cases...")

df = pd.read_csv(INPUT_FILE)

df = df[df["customer_message"].notna()].copy()

df["clean_message"] = df["customer_message"].apply(clean_text)

# Remove messages that become too short
df = df[df["clean_message"].str.len() >= 10]

print(f"Original cases: {len(df):,}")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved cleaned cases:")
print(OUTPUT_FILE)