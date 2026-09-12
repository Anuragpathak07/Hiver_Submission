# Hiver AI Customer Support Agent for AppleSupport — Final Report

**Author:** Anurag Pathak  
**Submission For:** Hiver SDE Intern Take-Home Assignment  
**Target Brand:** AppleSupport (Customer Support on Twitter Dataset)  
**Repository:** https://github.com/Anuragpathak07/Hiver_Submission.git  

---

## Executive Summary & Headline Results

| Evaluation Metric | System Result | Benchmark Baseline | Production Significance |
| :--- | :--- | :--- | :--- |
| **Intent Classification Accuracy** | **85.00%** | **15.50%** (Majority Class)<br>**79.50%** (TF-IDF + LogReg) | **+5.50% improvement** over strong TF-IDF baseline across 8 intents. |
| **Intent Classification Macro-F1** | **0.8512** | **0.0335** (Majority Class)<br>**0.7954** (TF-IDF + LogReg) | High balanced performance with zero class collapse. |
| **Historical Retrieval Match @ 3** | **87.50%** | **69.50%** (Lexical TF-IDF) | **+18.00% improvement** using semantic vector embeddings. |
| **Retrieval Top-1 Avg Similarity** | **0.7142** | **0.3711** (Lexical TF-IDF) | Captures technical symptom paraphrasing accurately. |
| **LLM-as-Judge Quality Score** | **4.25 / 5.0** | **80.00%** Rubric Pass Rate | Evaluated across 6 quality criteria (correctness, grounding, tone). |
| **Human-Judge Agreement** | **90.00%** | 30-case validation subset | Strong alignment between automated LLM judge and human evaluation. |
| **Auto-Handle Rate (AI Agent)** | **73.00%** | 146 / 200 Golden Set cases | Safely resolves routine diagnostic & update queries. |
| **Escalation Rate (Human Specialist)**| **27.00%** | 54 / 200 Golden Set cases | Safely escalates high-risk identity, billing, or low-confidence queries. |

---

## 1. Problem Framing & Brand Selection

### 1.1 Brand Selection Analysis
The primary dataset (`thoughtvector/customer-support-on-twitter`) contains ~3 million tweets across dozens of brands. We evaluated candidate high-volume brands (`AmazonHelp`, `AppleSupport`, `Uber_Support`, `Delta`) to select the optimal target:

- **Why AppleSupport?**
  1. **Technical Diagnostic Depth:** AppleSupport conversations feature multi-turn diagnostic troubleshooting (e.g. asking for iOS versions, device models, storage usage, SMC resets).
  2. **High Volume & Recurrence:** Over 106,000+ tweets centered around major software releases (iOS 11, macOS High Sierra), creating recurring support patterns.
  3. **High Resolution Ground Truth:** Support agents consistently provide actionable links (`support.apple.com`), step-by-step workarounds, or structured DM escalation.

---

### 1.2 What "Good" Means for AppleSupport
For AppleSupport on Twitter, an AI support agent must meet three operational criteria:
1. **Grounded Precision:** Responses must be strictly backed by verified AppleSupport resolutions without inventing troubleshooting steps or hallucinating non-existent URLs.
2. **Conservative Escalation:** High-risk queries (account recovery, billing disputes) or low-confidence predictions must be escalated to human agents with a clear diagnostic reason.
3. **Reproducible Proof:** Offline evaluation must demonstrate measurable superiority over trivial and simple baselines on a clean, isolated benchmark.

**What We Chose Not to Build:**
- **Automated Live Twitter Posting:** The agent generates drafts for agent-assist review rather than executing unmonitored live tweets.
- **Multilingual Support:** Optimization is restricted to English queries (non-English queries are caught by `other_unclear` / escalation).
- **Multi-Modal Vision Processing:** Image links (`t.co` screenshots) are handled via text mentions without running vision models.

---

## 2. Dataset Journey & Intent Discovery

### 2.1 Conversation Tree Reconstruction
Raw tweets are flat and unorganized. To turn tweets into structured support cases:
1. We parsed `in_response_to_tweet_id` recursively to trace every tweet back to its original root conversation.
2. Grouped root conversations into **Cases** containing: Initial Customer Query, Complete Multi-Turn Conversation, and Final Support Agent Response.
3. Extracted **80,672 complete AppleSupport cases** stored at `data/processed/applesupport_cases.csv`.

---

### 2.2 Intent Discovery & Taxonomy Consolidation
Rather than imposing an arbitrary top-down taxonomy, we used a data-driven bottom-up approach:
1. **Semantic Embeddings:** Generated 384-dimensional dense vector representations of customer queries using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Unsupervised Clustering:** Ran KMeans clustering ($k=12$) to group semantically similar queries.
3. **LLM Consolidation:** Prompted Groq (`openai/gpt-oss-20b`) with strict JSON Schema constraints to merge redundant clusters into an **8-intent taxonomy**:
   - `ios_update_issues`: System slowdowns, freeze glitches, and installation errors post-update.
   - `battery_life_issue`: Rapid battery drain, overheating, charging degradation.
   - `macbook_issues`: macOS High Sierra crashes, Mac boot loops, display blackouts.
   - `keyboard_autocorrect_issue`: iOS 11 autocorrect bug rendering "I" as a question mark box.
   - `music_audio_issue`: Apple Music playback errors, iTunes library syncing failures.
   - `iphone_hardware_issue`: Physical speaker buzzing, screen glass damage, camera hardware failure.
   - `account_identity_issue`: Apple ID recovery, passcode lockouts, two-factor authentication.
   - `other_unclear`: Ambiguous queries, general brand praise/ranting, or non-technical chatter.

---

## 3. Hand-Annotation & Golden Set Methodology

- **Sampling:** Selected 200 cases using stratified sampling (~25 cases per intent) with fixed seed `random_state=42`.
- **Data Isolation:** All 200 Golden Set case IDs were **strictly excluded** from classifier training datasets and retrieval vector indices to prevent data leakage.
- **Human Verification:** Evaluated each candidate case via an interactive CLI tool (`src/evaluation/annotate_golden.py`).
- **Silver vs. Human Agreement:** **92.50%** (185 out of 200 cases matched). 15 disagreements were reviewed and human-verified labels frozen at `data/golden/golden_annotated.csv`.

---

## 4. Results vs. Two Baselines

We evaluated three intent classification models on the frozen 200-example Golden Set:

1. **Baseline 1 (Trivial): Majority Class Baseline**
   - Always predicts `ios_update_issues`.
   - *Accuracy:* `15.50%` | *Macro-F1:* `0.0335`
2. **Baseline 2 (Simple): TF-IDF + Logistic Regression**
   - Fits `TfidfVectorizer(max_features=10000)` + `LogisticRegression(C=1.0)`.
   - *Accuracy:* `79.50%` | *Macro-F1:* `0.7954`
3. **Our Model (Improved): SentenceTransformer + Calibrated LogReg**
   - Dense embeddings (`all-MiniLM-L6-v2`) + `LogisticRegression(C=2.0)`.
   - *Accuracy:* **`85.00%`** | *Macro-F1:* **`0.8512`**

---

## 5. Top 5 Real Failure Modes Analysis

1. **Hardware vs. Software Symptom Ambiguity:** Update-induced hardware symptoms (e.g. screen freeze after iOS 11) misclassified as hardware. (`apple_2750532`)
2. **Short / Noisy Customer Queries:** High-expletive or image-only tweets (e.g. *"what the fuck"*) yielding low retrieval similarity (safely escalated). (`apple_1709355`)
3. **Taxonomy Boundary Blur:** Store purchasing queries overlapping between `account_identity_issue` and `other_unclear`. (`apple_684378`)
4. **Retrieval Response Divergence:** High query similarity match where the historical support response was an incomplete DM request. (`apple_14838`)
5. **Emoji & Informal Slang Over-conservatism:** High informal syntax variance lowering classifier confidence score. (`apple_164246`)

---

## 6. "What is Misleading About My Headline Number?" (Mandatory Section)

1. **Margin of Error on Small Golden Set ($N=200$):** 95% confidence interval spans **[80.0%, 90.0%]** ($\pm 5.0\%$).
2. **Stratified vs. Natural Class Distribution:** Stratified sampling overrepresents rare classes compared to real-world volume where update issues dominate >55% of queries.
3. **Silver Pre-Annotation Confirmation Bias:** Evaluators accepted silver labels in 92.5% of cases, introducing mild confirmation bias.
4. **Temporal Shift (2017 Dataset):** Models trained on 2017 iOS 11 issues will degrade on modern iOS 17/18 queries without continuous retraining.
5. **Offline Retrieval Metric vs. Resolution:** Intent match @ 3 measures class overlap, not whether the historical response resolved the customer's specific problem.

---

## 7. Next-Week Improvement Plan
1. **Blind Double-Annotation:** Annotate 1,000 cases blindly without pre-labeled silver intents.
2. **URL Canonicalization:** Map raw 2017 `t.co` shortlinks to canonical live `support.apple.com` article endpoints.
3. **Contrastive Fine-Tuning:** Fine-tune `all-MiniLM-L6-v2` using contrastive learning on AppleSupport query-response pairs.
4. **LLM Response Filtering:** Exclude historical responses that are mere DM requests before populating vector index.

---

## 8. Complete Decision Log (12 Key Decisions)
1. **Target Brand Selection (`AppleSupport`):** High volume (106K+ tweets), multi-turn diagnostic threads.
2. **Case Unit of Analysis:** Reconstructed flat tweets into 80,672 complete support cases.
3. **Embedding-Based Discovery:** Used `all-MiniLM-L6-v2` + KMeans ($k=12$).
4. **Groq LLM Consolidation:** Merged redundant clusters into 8 frozen intents via JSON Schema.
5. **Strict Data Isolation:** Excluded all 200 Golden Set case IDs from training corpora.
6. **SentenceTransformer Classifier:** Outperformed TF-IDF (85.00% vs 79.50%).
7. **Dual Retrieval Indexing:** Built primary semantic vector search and secondary TF-IDF search.
8. **Grounded Generation:** Prompted Groq (`openai/gpt-oss-20b`) with historical grounding.
9. **Multi-Tiered Escalation Engine:** Combined intent risk, confidence (<0.50), similarity (<0.40), and keywords.
10. **LLM-as-Judge Evaluator:** 6-criterion quality rubric with ~90% human agreement.
11. **Text Normalization:** Stripped mentions (`@user`) while preserving hashtags (`#iOS11`).
12. **Local Caching & Reproducibility:** Enabled <5 minute offline execution from root directory.
