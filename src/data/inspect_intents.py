import pandas as pd

INPUT_FILE = "processed/applesupport_cases.csv"

df = pd.read_csv(INPUT_FILE)

print("Total cases:", len(df))

# Sample customer messages
sample = df[
    ["case_id", "customer_message"]
].sample(
    n=300,
    random_state=42
)

sample.to_csv(
    "processed/applesupport_intent_sample.csv",
    index=False
)

print(
    "\nSaved 300 customer messages to:"
    " processed/applesupport_intent_sample.csv"
)

print("\nFirst 30 examples:\n")

for i, row in sample.head(30).iterrows():

    print(
        f"\n[{row['case_id']}] "
        f"{row['customer_message']}"
    )