import pandas as pd

df = pd.read_csv("raw/twcs.csv")

print("Total rows:", len(df))
print("Unique tweet IDs:", df["tweet_id"].nunique())
print("Duplicate tweet IDs:", df["tweet_id"].duplicated().sum())

duplicates = df[
    df["tweet_id"].duplicated(keep=False)
].sort_values("tweet_id")

print("\nFirst duplicate tweet IDs:")
print(
    duplicates[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "in_response_to_tweet_id",
            "response_tweet_id"
        ]
    ].head(20).to_string(index=False)
)