# Headline Result Analysis: "What is Misleading About My Headline Number?"

## Headline Metric
> **"The Hiver Support Agent achieved 85.00% Intent Classification Accuracy, 87.50% Retrieval Intent Match @ 3, a 4.25/5.0 LLM-Judge Quality Score, and 73.0% Safe Auto-Handling on AppleSupport Twitter queries."**

While these numbers demonstrate a strong functional pipeline, presenting them without context is misleading for production deployment due to the following critical methodological and evaluation limitations:

---

## 1. Small Sample Size & Margin of Error (N = 200)
- **Limitation:** The evaluation benchmark relies on a 200-example Golden Set.
- **Statistical Reality:** At N = 200 with an 85.0% success rate, the 95% confidence interval spans **[80.0%, 90.0%]** (margin of error ~±5.0%). A larger production sample (N ≥ 1,000) is required for tighter confidence bounds.

## 2. Stratified vs. Natural Class Distribution
- **Limitation:** The Golden Set was constructed using balanced stratified sampling (~25 cases per intent across 8 intents).
- **Production Reality:** In the real-world dataset, `ios_update_issues` and `battery_life_issue` account for over **55% of all incoming volume**, while `macbook_issues` account for <5%. Performance on stratified evaluation does not reflect true volume-weighted operational accuracy.

## 3. Silver-Label Pre-Annotation Guidance Bias
- **Limitation:** The Golden Set candidates were pre-annotated with silver labels before human verification.
- **Bias:** Human annotators accepted the silver label in 92.5% of cases. Pre-labeling introduces confirmation bias compared to blind double-annotation from raw text.

## 4. Twitter Support Domain Artifacts & Temporal Drift (2017 Dataset)
- **Limitation:** The corpus reflects AppleSupport Twitter conversations from late 2017 (dominated by iOS 11 updates and the "I" autocorrect bug).
- **Drift:** Modern Apple Support queries (iOS 17/18, Apple Intelligence, USB-C accessories) feature different symptom vocabulary. 85% accuracy on 2017 data will degrade under temporal distribution shift without continuous retraining.

## 5. Offline Retrieval Metric vs. True Problem Resolution
- **Limitation:** The 87.50% Retrieval Match Rate measures whether retrieved cases share the same intent class as the query.
- **Reality:** Intent match does not guarantee that the historical troubleshooting steps solved the customer's specific edge-case problem.

## 6. LLM-as-Judge Alignment & Synthetic Quality Scoring
- **Limitation:** Response quality scored 4.25/5.0 using automated LLM judging (`openai/gpt-oss-20b` via Groq).
- **Reality:** LLM judges measure logical consistency and prompt adherence, but cannot evaluate true customer satisfaction, emotional sentiment, or physical resolution.
