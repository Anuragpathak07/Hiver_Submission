import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cluster_profiles.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_taxonomy.json"
)

load_dotenv(PROJECT_ROOT / ".env")

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    clusters = df.to_dict(orient="records")

    clusters_text = json.dumps(
        clusters,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are designing a customer-support intent taxonomy
for AppleSupport Twitter conversations.

We have exactly 12 discovered clusters.

IMPORTANT:
Your response MUST contain between 8 and 12 intents.

MOST IMPORTANT REQUIREMENT:
EVERY cluster ID from 0 through 11 MUST appear
EXACTLY ONCE across all cluster_ids fields.

You are consolidating noisy embedding clusters into
business-level customer-support intents.

Rules:

- An intent represents the underlying CUSTOMER PROBLEM.
- Do NOT create an intent based only on wording.
- Do NOT create an intent for a language.
- Do NOT create an intent for a single word or phrase.
- Merge clusters when they represent the same underlying problem.
- Different clusters can map to the same intent.
- Use "other_unclear" for genuinely ambiguous cases.
- Do not invent problems unsupported by the data.
- Prefer broad but useful support categories.
- Use short snake_case intent names.

Known examples of consolidation that may be appropriate:

- Multiple iOS update/performance clusters may belong
  to one broader iOS update or system stability intent.
- Multiple keyboard/"I" glitch clusters may belong
  to one keyboard/autocorrect/system input intent.
- Multiple Apple ID/iCloud/App Store issues may be
  consolidated if the underlying problem is similar.

However, DO NOT blindly follow these examples.
Use the actual cluster evidence.

Before producing the final JSON, internally verify:

1. There are 8-12 intents.
2. Cluster IDs are exactly:
   0,1,2,3,4,5,6,7,8,9,10,11
3. Every cluster appears exactly once.
4. No cluster is missing.
5. No cluster appears twice.

CLUSTERS:

{clusters_text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in customer-support "
                    "intent taxonomy design."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "support_taxonomy",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intents": {
                            "type": "array",
                            "minItems": 8,
                            "maxItems": 12,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "intent": {
                                        "type": "string"
                                    },
                                    "description": {
                                        "type": "string"
                                    },
                                    "example_problem": {
                                        "type": "string"
                                    },
                                    "cluster_ids": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer"
                                        }
                                    },
                                    "mapping_reason": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "intent",
                                    "description",
                                    "example_problem",
                                    "cluster_ids",
                                    "mapping_reason"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["intents"],
                    "additionalProperties": False
                }
            }
        },
        temperature=0.1,
        max_tokens=6000,
    )

    content = response.choices[0].message.content

    taxonomy = json.loads(content)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            taxonomy,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved taxonomy to: {OUTPUT_FILE}"
    )

    print("\nCandidate intents:")

    for item in taxonomy["intents"]:
        print(
            f"- {item['intent']} "
            f"(clusters={item['cluster_ids']})"
        )


if __name__ == "__main__":
    main()