import pandas as pd

df = pd.read_csv("raw/twcs.csv")

APPLE = "AppleSupport"

# AppleSupport's tweets
apple = df[
    (df["author_id"].astype(str) == APPLE)
].copy()

# Keep only AppleSupport tweets that are replies
apple_replies = apple[
    apple["in_response_to_tweet_id"].notna()
].copy()

print("AppleSupport tweets:", len(apple))
print("AppleSupport replies:", len(apple_replies))


# ------------------------------------------------------------
# Inspect a few conversations by following parent links
# ------------------------------------------------------------

tweets = df.set_index("tweet_id")


def trace(tweet_id, max_turns=10):

    chain = []
    current = int(tweet_id)

    for _ in range(max_turns):

        if current not in tweets.index:
            break

        row = tweets.loc[current]

        chain.append({
            "tweet_id": current,
            "author": row["author_id"],
            "speaker": (
                "CUSTOMER"
                if row["inbound"]
                else "SUPPORT"
            ),
            "text": row["text"],
            "parent": row["in_response_to_tweet_id"]
        })

        parent = row["in_response_to_tweet_id"]

        if pd.isna(parent):
            break

        current = int(parent)

    return chain[::-1]


# ------------------------------------------------------------
# Show 10 AppleSupport conversations
# ------------------------------------------------------------

for tweet_id in apple_replies["tweet_id"].head(10):

    print("\n" + "=" * 80)
    print("STARTING FROM APPLESUPPORT TWEET:", tweet_id)
    print("=" * 80)

    chain = trace(tweet_id)

    for i, turn in enumerate(chain, 1):

        print(
            f"{i}. "
            f"{turn['speaker']} "
            f"[{turn['tweet_id']}]\n"
            f"   {turn['text']}\n"
            f"   parent={turn['parent']}"
        )