import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("raw/twcs.csv")
OUTPUT_DIR = Path("processed")

APPLE_SUPPORT_AUTHOR = "AppleSupport"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading TWCS dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total tweets: {len(df):,}")


# ============================================================
# 2. VERIFY APPLESUPPORT
# ============================================================

support = df[df["inbound"] == False]

apple = support[
    support["author_id"].astype(str)
    == APPLE_SUPPORT_AUTHOR
].copy()

if apple.empty:
    raise ValueError(
        "AppleSupport was not found."
    )

print(
    f"AppleSupport tweets: "
    f"{len(apple):,}"
)


# ============================================================
# 3. CREATE TWEET LOOKUP
# ============================================================

tweets = df.set_index("tweet_id")


# ============================================================
# 4. FIND ROOT OF EVERY TWEET
# ============================================================

root_cache = {}


def find_root(tweet_id):

    tweet_id = int(tweet_id)

    # Already calculated
    if tweet_id in root_cache:
        return root_cache[tweet_id]

    current = tweet_id
    visited = []

    while True:

        if current in root_cache:

            root = root_cache[current]
            break

        if current not in tweets.index:

            root = current
            break

        visited.append(current)

        row = tweets.loc[current]

        parent = row["in_response_to_tweet_id"]

        # No parent = root tweet
        if pd.isna(parent):

            root = current
            break

        try:
            parent = int(parent)
        except (ValueError, TypeError):

            root = current
            break

        current = parent

    # Cache every tweet in this chain
    for tweet in visited:
        root_cache[tweet] = root

    return root


# ============================================================
# 5. FIND ROOTS FOR APPLESUPPORT TWEETS
# ============================================================

print("\nFinding conversation roots...")

apple["root_tweet_id"] = (
    apple["tweet_id"]
    .apply(find_root)
)


print(
    "Unique conversation roots containing "
    "AppleSupport:"
)

print(
    apple["root_tweet_id"].nunique()
)


# ============================================================
# 6. GET ALL TWEETS BELONGING TO THOSE ROOTS
# ============================================================

apple_roots = set(
    apple["root_tweet_id"]
)


print(
    "\nCollecting conversation tweets..."
)


# Calculate root for every tweet.
#
# This is the expensive part, but root_cache makes it
# considerably cheaper after the first traversal.

df["root_tweet_id"] = (
    df["tweet_id"]
    .apply(find_root)
)


# Keep only conversations that contain AppleSupport
conversation_df = df[
    df["root_tweet_id"].isin(apple_roots)
].copy()


print(
    f"Tweets in AppleSupport conversations: "
    f"{len(conversation_df):,}"
)


# ============================================================
# 7. CREATE CONVERSATION ID
# ============================================================

conversation_df["conversation_id"] = (
    "apple_"
    + conversation_df["root_tweet_id"]
    .astype(str)
)


# ============================================================
# 8. IDENTIFY SPEAKER
# ============================================================

conversation_df["speaker"] = (
    conversation_df["inbound"]
    .map({
        True: "CUSTOMER",
        False: "SUPPORT"
    })
)


# ============================================================
# 9. SORT CHRONOLOGICALLY
# ============================================================

conversation_df["created_at"] = pd.to_datetime(
    conversation_df["created_at"]
)

conversation_df = conversation_df.sort_values(
    [
        "conversation_id",
        "created_at",
        "tweet_id"
    ]
)


# ============================================================
# 10. CREATE TURN NUMBER
# ============================================================

conversation_df["turn_number"] = (
    conversation_df
    .groupby("conversation_id")
    .cumcount()
    + 1
)


# ============================================================
# 11. SELECT REQUIRED COLUMNS
# ============================================================

conversation_df = conversation_df[
    [
        "conversation_id",
        "turn_number",
        "tweet_id",
        "speaker",
        "text",
        "created_at"
    ]
]


# ============================================================
# 12. SAVE CONVERSATION CORPUS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

conversation_file = (
    OUTPUT_DIR /
    "applesupport_conversations.csv"
)

conversation_df.to_csv(
    conversation_file,
    index=False
)


print(
    f"\nSaved conversation corpus:"
    f"\n{conversation_file}"
)

print(
    f"Unique conversations: "
    f"{conversation_df['conversation_id'].nunique():,}"
)


# ============================================================
# 13. CREATE CASE-LEVEL DATASET
# ============================================================

cases = []

for conversation_id, group in (
    conversation_df
    .groupby("conversation_id")
):

    group = group.sort_values(
        "turn_number"
    )

    customers = group[
        group["speaker"] == "CUSTOMER"
    ]

    support_messages = group[
        group["speaker"] == "SUPPORT"
    ]

    if customers.empty:
        continue

    first_customer = customers.iloc[0]

    support_texts = (
        support_messages["text"]
        .tolist()
    )

    cases.append({

        "case_id":
            conversation_id,

        "customer_message":
            first_customer["text"],

        "support_conversation":
            "\n".join(
                f"{row['speaker']}: "
                f"{row['text']}"
                for _, row in group.iterrows()
            ),

        "final_support_response":
            (
                support_texts[-1]
                if support_texts
                else ""
            ),

        "num_turns":
            len(group),

        "num_customer_turns":
            len(customers),

        "num_support_turns":
            len(support_messages),

        # This is NOT a resolution label.
        "evidence_status":
            (
                "NO_SUPPORT_RESPONSE"
                if support_messages.empty
                else "SUPPORT_RESPONSE_PRESENT"
            )
    })


cases_df = pd.DataFrame(cases)


# ============================================================
# 14. SAVE CASE DATASET
# ============================================================

case_file = (
    OUTPUT_DIR /
    "applesupport_cases.csv"
)

cases_df.to_csv(
    case_file,
    index=False
)

print(
    f"Saved case dataset:"
    f"\n{case_file}"
)

print(
    f"Cases: {len(cases_df):,}"
)


# ============================================================
# 15. PRINT SAMPLE CONVERSATIONS
# ============================================================

print("\n")
print("=" * 80)
print("SAMPLE CONVERSATIONS")
print("=" * 80)


sample_conversations = (
    conversation_df[
        "conversation_id"
    ]
    .drop_duplicates()
    .head(5)
)


for conversation_id in sample_conversations:

    print("\n" + "-" * 80)

    print(conversation_id)

    conversation = conversation_df[
        conversation_df["conversation_id"]
        == conversation_id
    ]

    for _, row in conversation.iterrows():

        print(
            f"{row['turn_number']}. "
            f"{row['speaker']}: "
            f"{row['text']}"
        )


print("\nDone.")