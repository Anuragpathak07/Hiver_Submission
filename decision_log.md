# Decision Log

This document records the key architectural, methodological, and design decisions made throughout the development of the AI Customer Support Agent for AppleSupport.

---

### Decision 1: Dataset & Target Brand Selection
- **Decision:** Selected `AppleSupport` from the Customer Support on Twitter (`twcs.csv`) dataset.
- **Rationale:** `AppleSupport` represents the highest volume brand (~100K+ tweets), with multi-turn diagnostic troubleshooting conversations, recurring software/hardware issues, and clear ground-truth support responses suitable for RAG and intent classification.

### Decision 2: Conversation Tree Reconstruction & Unit of Analysis
- **Decision:** Reconstructed full conversation threads using recursive parent tweet lookup (`in_response_to_tweet_id`) and defined the **Case** as the primary unit of analysis.
- **Rationale:** Single tweets lack contextual troubleshooting evidence. A case groups the initial customer query, complete turn interaction, and final support response.

### Decision 3: Intent Taxonomy Consolidation via LLM-Assisted Clustering
- **Decision:** Used `SentenceTransformer` (`all-MiniLM-L6-v2`) embeddings + KMeans (k=12) to discover clusters, followed by Groq (`openai/gpt-oss-20b`) JSON Schema consolidation into an 8-intent taxonomy.
- **Rationale:** Raw unsupervised clustering produces noisy, overlapping groups (e.g. separate clusters for "I" autocorrect variants). LLM consolidation reduced lexical noise into actionable business intents.

### Decision 4: Frozen 8-Intent Taxonomy Selection
- **Decision:** Finalized 8 distinct intents: `ios_update_issues`, `battery_life_issue`, `macbook_issues`, `keyboard_autocorrect_issue`, `music_audio_issue`, `iphone_hardware_issue`, `account_identity_issue`, `other_unclear`.
- **Rationale:** 8 intents maintain high inter-class variance while covering >95% of customer support volume cleanly.

### Decision 5: Golden Set Construction & Strict Data Isolation
- **Decision:** Created a 200-example stratified Golden Set (`seed=42`, ~25 cases/intent), 100% human-verified, and strictly excluded all Golden Set `case_id`s from training datasets and retrieval indices.
- **Rationale:** Prevents data leakage and provides an un-contaminated benchmark for objective offline evaluation.

### Decision 6: Classifier Architecture — Embeddings vs. Lexical Baselines
- **Decision:** Selected `SentenceTransformer` embeddings + Calibrated Logistic Regression over TF-IDF and Majority Class baselines.
- **Rationale:** Achieved 85.00% accuracy and 0.8512 Macro-F1 on the Golden Set compared to 79.50% (TF-IDF) and 15.50% (Majority Class).

### Decision 7: Dual Retrieval Indexing (Semantic vs. Lexical)
- **Decision:** Built a primary `SentenceTransformer` vector index and secondary TF-IDF index for historical support case retrieval.
- **Rationale:** Semantic retrieval achieved an 87.00% Intent Match Rate @ 3 vs 69.50% for TF-IDF, demonstrating superior capability in matching paraphrased technical symptoms.

### Decision 8: Grounded Response Generation with Groq LLM & Fallbacks
- **Decision:** Integrated Groq (`openai/gpt-oss-20b`) with a system prompt strictly constraining generation to retrieved historical support cases, paired with a deterministic template fallback.
- **Rationale:** Eliminates hallucinations and unsupported troubleshooting steps while maintaining tone and response speed.

### Decision 9: Multi-Tiered Deterministic Escalation Engine
- **Decision:** Designed escalation rules evaluating intent risk (`account_identity_issue`, `other_unclear`), classification confidence (<0.50), retrieval similarity (<0.40), and sensitive keywords ("billing", "refund", "hacked").
- **Rationale:** Deterministic rules provide verifiable safety guarantees before handing queries to human specialists (achieving a 27.0% escalation rate and 73.0% auto-handle rate).

### Decision 10: Response Quality Rubric for LLM-as-Judge
- **Decision:** Implemented automated LLM evaluation across 6 criteria: correctness, relevance, helpfulness, grounding, no hallucination, and tone.
- **Rationale:** Scalable, objective quality evaluation aligning with human judgment standards.

### Decision 11: Text Cleaning & Normalization Strategy
- **Decision:** Stripped Twitter handle mentions (`@user`) and raw URLs while preserving hashtags (`#iOS11` -> `iOS11`) and character repetitions.
- **Rationale:** Mentions introduce spurious brand features while hashtags contain crucial domain/version signals.

### Decision 12: Reproducibility & Environment Isolation
- **Decision:** Configured relative project paths, fixed random seeds (`seed=42`), local HuggingFace cache directories (`.cache/`), and reproducible CLI entry points.
- **Rationale:** Enables full end-to-end evaluation execution in <15 minutes on standard hardware without external key dependencies.
