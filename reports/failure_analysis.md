# Top 5 Real Failure Modes Analysis

## 1. Classifier Failure: Hardware vs. Software Update Symptom Ambiguity
- **Description:** Customer queries describing hardware symptoms (e.g., "phone overheating", "screen freezing", "camera black") that started immediately after an iOS update.
- **Example Case:** `apple_2750532` — *"what's wrong with the iPhone X I can't access my contacts"*
- **Classifier Error:** Predicted `iphone_hardware_issue`, but true human label was `ios_update_issues`.
- **Root Cause:** Both categories share lexical tokens like "iPhone X" and symptom descriptions; semantic embeddings capture symptom similarity over causal intent.

## 2. Retrieval Failure: Extremely Short / High-Noise Customer Tweets
- **Description:** Customer tweets containing minimal context, expletives, or images/links with no descriptive text.
- **Example Case:** `apple_1709355` — *"what the fuck https://t.co/LB9KpiXBJj"*
- **Retrieval Error:** Retrieved top-1 historical case with low similarity (0.32), yielding a generic response suggestion.
- **Root Cause:** Absence of domain keywords prevents TF-IDF or embedding models from retrieving specific troubleshooting guides. Safely caught and escalated by the Escalation Engine under `INSUFFICIENT_RETRIEVAL_EVIDENCE`.

## 3. Taxonomy Boundary Blur: Account Identity vs. Store Purchasing
- **Description:** Queries asking about purchasing AppleCare or Apple Store shipment methods misclassified under `account_identity_issue`.
- **Example Case:** `apple_684378` — *"Hi I have a question about purchasing a MacBook, could I DM you?"*
- **Classifier Error:** Predicted `account_identity_issue`, true label `other_unclear`.
- **Root Cause:** The taxonomy consolidated purchase/store queries into `other_unclear` or `account_identity_issue`, creating a soft decision boundary.

## 4. Generation Limitation: Obsolete Twitter Shortened URLs (`t.co`)
- **Description:** Historical AppleSupport responses frequently contain truncated `https://t.co/...` links pointing to Apple Support knowledge base articles from 2017.
- **Impact:** Grounded generation occasionally copies historical `t.co` links which are now dead or context-specific.
- **Fix / Recommendation:** Strip raw `t.co` URLs during preprocessing and replace with canonical Apple Support landing page placeholders (`support.apple.com`).

## 5. Escalation Over-conservatism on Emojis & Informal Slang
- **Description:** Valid customer queries containing heavy slang or emojis (e.g., *"Plz get ur shit together and fix I.T"*) receive lower classification probability.
- **Impact:** Triggers `LOW_CLASSIFIER_CONFIDENCE` or `HIGH_RISK_INTENT` escalation even when the underlying problem (keyboard autocorrect glitch) is standard.
- **Root Cause:** High variance in informal Twitter syntax reduces model softmax confidence.
