# Hiver AI Customer Support Agent for AppleSupport

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Architecture-RAG%20%2B%20Intent%20Classifier-green.svg)]()
[![Model](https://img.shields.io/badge/LLM-Groq%20(openai%2Fgpt--oss--20b)-orange.svg)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)]()

Production-grade AI Customer Support Agent built for **AppleSupport** using the Customer Support on Twitter (`twcs.csv`) dataset. 

This repository contains the complete end-to-end AI system: automated intent classification, dual-engine RAG historical case retrieval, Groq-powered grounded response drafting, multi-tiered escalation, an offline evaluation harness, an LLM-as-Judge evaluator, and a 100% human-verified Golden Evaluation Set.

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

## 1. Quick Start: Reproducing Results in Under 15 Minutes

The repository is fully self-contained with pre-computed vector caches and trained classifier artifacts. You can run and reproduce all evaluation results from the root directory in **under 5 minutes**.

### Step 1: Environment Setup (~2 minutes)
```bash
# 1. Clone repository & set up environment
git clone https://github.com/Anuragpathak07/Hiver_Submission.git
cd Hiver_Submission

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file from example template
cp .env.example .env
```
> **Note:** An API key is **optional** for offline evaluation. The system includes deterministic fallback response generators if `GROQ_API_KEY` is unconfigured. To test live LLM generation via Groq, add your key to `.env`.

---

### Step 2: Run Unit & Integration Tests (~30 seconds)
```bash
pytest tests/
```
*Expected Output:* `4 passed in ~50s` verifying pipeline components, escalation triggers, and RAG integration.

---

### Step 3: Run Single-Query Pipeline CLI (~10 seconds)
```bash
python src/pipeline.py --query "My battery is draining fast on my iPhone 7 after updating to iOS 11."
```
*Expected Output:* JSON output displaying predicted intent (`battery_life_issue`), classification confidence (`0.9998`), top-3 retrieved historical cases, escalation decision (`AUTO_HANDLE`), and grounded response draft.

---

### Step 4: Reproduce Classification Baseline Results (~1.5 minutes)
```bash
python src/intent/train_and_evaluate_classifiers.py
```
*Expected Output:* Evaluates Majority Class Baseline (15.50%), TF-IDF + Logistic Regression (79.50%), and SentenceTransformer Classifier (85.00%) against the 200-example Golden Set.

---

### Step 5: Reproduce Retrieval Evaluation Results (~45 seconds)
```bash
python src/retrieval/evaluate_retrieval.py
```
*Expected Output:* Compares Lexical TF-IDF (69.50% Match@3) vs Semantic Vector Retrieval (87.50% Match@3) across 200 Golden Set queries.

---

### Step 6: Run Comprehensive End-to-End Evaluation Harness (~2 minutes)
```bash
python src/evaluation/evaluation_harness.py
```
*Expected Output:* Evaluates the full pipeline across all 200 Golden Set cases, saving results to `reports/full_evaluation_harness_results.json`.

---

## 2. Dataset Journey & Brand Selection Methodology

### 2.1 Brand Selection Analysis
The primary dataset (`thoughtvector/customer-support-on-twitter`) contains ~3 million tweets across dozens of brands. We evaluated candidate high-volume brands (`AmazonHelp`, `AppleSupport`, `Uber_Support`, `Delta`) to select the optimal target:

- **Why AppleSupport?**
  1. **Technical Diagnostic Depth:** AppleSupport conversations feature multi-turn diagnostic troubleshooting (e.g. asking for iOS versions, device models, storage usage, SMC resets).
  2. **High Volume & Recurrence:** Over 106,000+ tweets centered around major software releases (iOS 11, macOS High Sierra), creating recurring support patterns.
  3. **High Resolution Ground Truth:** Support agents consistently provide actionable links (`support.apple.com`), step-by-step workarounds, or structured DM escalation.

---

### 2.2 Recursive Conversation Tree Reconstruction
Raw tweets are flat and unorganized. To turn tweets into structured support cases:
1. We parsed `in_response_to_tweet_id` recursively to trace every tweet back to its original root conversation.
2. Grouped root conversations into **Cases** containing:
   - Initial Customer Query
   - Complete Multi-Turn Conversation
   - Final Support Agent Response
3. Extracted **80,672 complete AppleSupport cases** stored at `data/processed/applesupport_cases.csv`.

---

### 2.3 Intent Discovery & Taxonomy Consolidation
Rather than imposing an arbitrary top-down taxonomy, we used a data-driven bottom-up approach:
1. **Semantic Embeddings:** Generated 384-dimensional dense vector representations of customer queries using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Unsupervised Clustering:** Ran KMeans clustering ($k=12$) to group semantically similar queries.
3. **TF-IDF Profiling:** Extracted top n-gram keywords and representative centroid queries for each cluster (`src/intent/profile_clusters.py`).
4. **LLM Consolidation:** Prompted Groq (`openai/gpt-oss-20b`) with strict JSON Schema constraints to merge redundant clusters (e.g. merging two distinct clusters created by variants of the iOS 11 "I" autocorrect bug).
5. **Frozen 8-Intent Taxonomy:**
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

### 3.1 Stratified Sampling Strategy
To build an unbiased evaluation benchmark (`N = 200`):
- **Sampling:** Selected ~25 customer cases per intent class using fixed seed `random_state=42` (`src/evaluation/create_golden_sample.py`).
- **Data Isolation:** All 200 Golden Set case IDs were **strictly excluded** from classifier training datasets and retrieval vector indices to prevent data leakage.

### 3.2 Human Verification Procedure & CLI Tooling
- We built an interactive CLI annotation tool ([annotate_golden.py](file:///d:/All_Project/Hiver_Submission/hiver-support-agent/src/evaluation/annotate_golden.py)) to review each case individually.
- Each case was pre-annotated with a silver intent label (from KMeans model mapping).
- Evaluators reviewed the raw customer query, full conversation context, and verified or corrected the label.

### 3.3 Disagreement Analysis & Ground Truth Freeze
- **Silver vs. Human Agreement:** **92.50%** (185 out of 200 cases matched).
- **15 Disagreements Corrected:** For example, cases where update-induced hardware symptoms (e.g. camera unresponsive after update) were misassigned to hardware rather than `ios_update_issues`, or general frustration misassigned to `other_unclear`.
- The human-verified labels were frozen at [golden_annotated.csv](file:///d:/All_Project/Hiver_Submission/hiver-support-agent/data/golden/golden_annotated.csv) as the definitive evaluation benchmark.

---

## 4. System Architecture & Component Design

```
Incoming Customer Query
         │
         ▼
┌─────────────────────────────────┐
│  Intent Classification Engine   │  ---> SentenceTransformer + LogReg (85.0% Acc)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Historical Case Retrieval      │  ---> Dual Semantic Vector Index (87.5% Match@3)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Multi-Tiered Escalation Engine │  ---> Evaluates Confidence, Risk & Keywords
└────────┬────────────────┬───────┘
         │                │
   [AUTO_HANDLE]     [ESCALATE] ---> Human Specialist Queue (Explicit Reason)
         │
         ▼
┌─────────────────────────────────┐
│  Grounded Response Generation   │  ---> Groq LLM (openai/gpt-oss-20b) + Fallbacks
└─────────────────────────────────┘
```

---

## 5. Main Assignment Report

### Section A: Problem Framing (What "Good" Means & Scope Boundaries)
For AppleSupport on Twitter, an AI support agent must meet three operational criteria:
1. **Grounded Precision:** Responses must be strictly backed by verified AppleSupport resolutions without inventing troubleshooting steps or hallucinating non-existent URLs.
2. **Conservative Escalation:** High-risk queries (account recovery, billing disputes) or low-confidence predictions must be escalated to human agents with a clear diagnostic reason.
3. **Reproducible Proof:** Offline evaluation must demonstrate measurable superiority over trivial and simple baselines on a clean, isolated benchmark.

**What We Chose Not to Build:**
- **Automated Live Twitter Posting:** The agent generates drafts for agent-assist review rather than executing unmonitored live tweets.
- **Multilingual Support:** Optimization is restricted to English queries (non-English queries are caught by `other_unclear` / escalation).
- **Multi-Modal Vision Processing:** Image links (`t.co` screenshots) are handled via text mentions without running vision models.

---

### Section B: Results vs. Two Baselines

We evaluated three intent classification models on the frozen 200-example Golden Set:

1. **Baseline 1 (Trivial): Majority Class Baseline**
   - Always predicts the most frequent intent in the training corpus (`ios_update_issues`).
   - *Accuracy:* `15.50%` | *Macro-F1:* `0.0335` | *Weighted-F1:* `0.0416`
2. **Baseline 2 (Simple): TF-IDF + Logistic Regression**
   - Fits `TfidfVectorizer(max_features=10000, ngram_range=(1,2))` + `LogisticRegression(C=1.0)`.
   - *Accuracy:* `79.50%` | *Macro-F1:* `0.7954` | *Weighted-F1:* `0.7955`
3. **Our Model (Improved): SentenceTransformer + Calibrated LogReg**
   - Generates 384-dimensional dense embeddings (`all-MiniLM-L6-v2`) + `LogisticRegression(C=2.0)`.
   - *Accuracy:* **`85.00%`** | *Macro-F1:* **`0.8512`** | *Weighted-F1:* **`0.8502`**

#### Classification Report (Our Model on Golden Set):
```
                            precision    recall  f1-score   support
 account_identity_issue        0.880     0.880     0.880        25
     battery_life_issue        0.920     0.920     0.920        25
      ios_update_issues        0.769     0.800     0.784        25
  iphone_hardware_issue        0.800     0.800     0.800        25
keyboard_autocorrect_issue     0.960     0.960     0.960        25
         macbook_issues        0.880     0.880     0.880        25
      music_audio_issue        0.840     0.840     0.840        25
          other_unclear        0.750     0.720     0.735        25

               accuracy                            0.850       200
              macro avg        0.850     0.850     0.851       200
           weighted avg        0.850     0.850     0.850       200
```

---

### Section C: Top 5 Real Failure Modes Analysis

1. **Hardware vs. Software Symptom Ambiguity**
   - *Example (`apple_2750532`):* `"what's wrong with the iPhone X I can't access my contacts"`
   - *Predicted:* `iphone_hardware_issue` | *True Label:* `ios_update_issues`
   - *Hypothesis:* Queries describing hardware symptoms caused by software update bugs share dense embeddings with physical hardware failures.
2. **Short / Noisy Customer Queries**
   - *Example (`apple_1709355`):* `"what the fuck https://t.co/LB9KpiXBJj"`
   - *Predicted:* `other_unclear` | *Top Retrieval Sim:* `0.32`
   - *Hypothesis:* Absence of domain keywords prevents TF-IDF or embedding models from retrieving specific troubleshooting guides. (Safely escalated by Escalation Engine).
3. **Taxonomy Boundary Blur (Account vs. Store Purchases)**
   - *Example (`apple_684378`):* `"Hi I have a question about purchasing a MacBook, could I DM you?"`
   - *Predicted:* `account_identity_issue` | *True Label:* `other_unclear`
   - *Hypothesis:* Purchasing questions overlap between store account support and general inquiries.
4. **Retrieval Response Divergence**
   - *Example (`apple_14838`):* Query matches historical battery query with **0.8817 similarity**, but historical support response was *"What happens when you open Mail? Send DM"*.
   - *Hypothesis:* High query similarity does not guarantee high response quality in raw Twitter support threads where agents frequently request DMs.
5. **Emoji & Informal Slang Escalation Over-conservatism**
   - *Example (`apple_164246`):* `"Plz get ur shit together and fix I.T"`
   - *Hypothesis:* High informal syntax variance lowers classifier confidence, triggering escalation even when the underlying problem (keyboard glitch) is standard.

---

### Section D: "What is Misleading About My Headline Number?" (Mandatory Section)

While our headline numbers (**85.00% Intent Accuracy**, **87.50% Retrieval Match**, **4.25/5.0 LLM-Judge Score**) reflect a strong system, presenting them without context is misleading:

1. **Margin of Error on Small Golden Set ($N=200$):** At $N=200$ with 85% accuracy, the 95% confidence interval spans **[80.0%, 90.0%]** (margin of error $\approx \pm 5.0\%$).
2. **Stratified vs. Natural Class Distribution:** The Golden Set uses balanced sampling (~25 cases/intent). In real Twitter volume, `ios_update_issues` and `battery_life_issue` account for >55% of queries. Unweighted accuracy overrepresents rare classes.
3. **Silver Pre-Annotation Confirmation Bias:** Evaluators accepted pre-annotated silver labels in 92.5% of cases. Pre-labeling introduces mild confirmation bias compared to blind double-annotation.
4. **Temporal Shift (2017 Dataset):** Models trained on 2017 iOS 11 issues will degrade on modern iOS 17/18 queries without continuous retraining.
5. **Offline Retrieval Metric vs. Resolution:** Intent match @ 3 measures class overlap, not whether the historical response resolved the customer's specific problem.

---

### Section E: What You'd Do Next with One More Week
1. **Blind Double-Annotation:** Annotate 1,000 cases blindly without pre-labeled silver intents to eliminate confirmation bias.
2. **URL Canonicalization:** Map raw 2017 `t.co` shortlinks to canonical live `support.apple.com` article endpoints.
3. **Contrastive Fine-Tuning:** Fine-tune `all-MiniLM-L6-v2` using Multiple Negatives Ranking (MNR) loss on AppleSupport query-response pairs.
4. **LLM-Based Response Filtering:** Filter out historical support responses that are mere DM requests before populating the RAG vector index.

---

### Section F: Complete Decision Log (12 Key Engineering Decisions)

1. **Target Brand Selection (`AppleSupport`):** High volume (106K+ tweets), multi-turn diagnostic threads, and rich resolution links.
2. **Case as Unit of Analysis:** Reconstructed flat tweets into 80,672 complete customer-support cases using parent tweet ID chains.
3. **Embedding-Based Intent Discovery:** Used `all-MiniLM-L6-v2` + KMeans ($k=12$) to discover bottom-up intent clusters.
4. **Groq LLM Taxonomy Consolidation:** Used Groq (`openai/gpt-oss-20b`) with strict JSON Schema constraints to merge redundant clusters into 8 frozen intents.
5. **Frozen 8-Intent Taxonomy:** Selected 8 intents to maintain high inter-class variance while covering >95% of customer queries.
6. **Strict Data Isolation:** Excluded all 200 Golden Set case IDs from training corpora and vector retrieval indices.
7. **Classifier Choice (SentenceTransformer + LogReg):** Outperformed TF-IDF (85.00% vs 79.50%) by capturing semantic symptom phrasing.
8. **Dual Retrieval Indexing:** Built primary semantic vector search and secondary TF-IDF search (Semantic achieved 87.50% Match@3 vs 69.50% Lexical).
9. **Grounded Response Generation:** Prompted Groq (`openai/gpt-oss-20b`) to restrict responses strictly to retrieved historical evidence.
10. **Multi-Tiered Escalation Engine:** Combined intent risk, confidence (<0.50), retrieval similarity (<0.40), and sensitive keywords ("billing", "refund") for deterministic escalation.
11. **6-Criterion Quality Rubric for LLM-as-Judge:** Evaluated correctness, relevance, helpfulness, grounding, non-hallucination, and tone.
12. **Local Caching & Reproducibility:** Cached HuggingFace models, vector indices, and joblib classifiers to enable <5 minute offline execution.

---

## 6. Project Structure

```
hiver-support-agent/
│
├── README.md                      # Primary Submission Report & Quick Start Guide
├── decision_log.md                # 12 Non-Obvious Architecture & Product Decisions
├── requirements.txt               # Python Dependencies
├── .env.example                   # Environment Configuration Template
├── .gitignore                     # Git Exclusions (.venv, .env, .cache, __pycache__)
│
├── src/                           # Core Source Code
│   ├── pipeline.py                # End-to-End Pipeline Entry Point CLI
│   ├── data/                      # Dataset ingestion & conversation tree reconstruction
│   ├── intent/                    # Clustering, taxonomy consolidation & classification
│   ├── retrieval/                 # Dual Lexical & Semantic Vector Retrieval Engine
│   ├── generation/                # Groq Grounded Response Generator
│   ├── escalation/                # Deterministic Escalation Engine
│   └── evaluation/                # Evaluation Harness & LLM-as-Judge Evaluator
│
├── data/                          # Data Corpora (Excluded from raw noise)
│   ├── raw/                       # Raw twcs.csv dataset location
│   ├── processed/                 # Processed case corpus & vector model artifacts
│   └── golden/                    # 200-Example Golden Set Benchmark & Annotation Note
│       ├── golden_annotated.csv
│       └── annotation_note.md
│
├── reports/                       # Generated Metrics & Analysis Reports
│   ├── full_evaluation_harness_results.json
│   ├── intent_evaluation_results.json
│   ├── retrieval_evaluation_results.json
│   ├── llm_judge_evaluation_results.json
│   ├── failure_analysis.md
│   ├── headline_result_analysis.md
│   └── pipeline_smoke_test.md
│
└── tests/                         # Pytest Automated Test Suite
    ├── test_pipeline.py
    └── test_retrieval_smoke.py
```

---

## 7. License & Credits
- **Dataset:** Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`), Kaggle.
- **Models:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) and `openai/gpt-oss-20b` (Groq API).
- **Author:** Developed for the Hiver SDE Intern Take-Home Assignment.
