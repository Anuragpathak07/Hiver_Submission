# Golden Set Annotation & Methodology Note

## Overview
The Golden Set (`data/golden/golden_annotated.csv`) serves as the evaluation ground truth for evaluating intent classification models on AppleSupport Twitter customer queries.

## Sampling Methodology
- **Total Size:** 200 examples
- **Sampling Strategy:** Stratified sampling across the 8 finalized taxonomy intents (~25 cases per intent class).
- **Random Seed:** `42` (`random_state=42`)
- **Source Corpus:** Derived from `data/processed/applesupport_cases_with_intents.csv` (80,672 total usable customer support cases).

## Annotation & Verification Procedure
1. **Silver-Label Pre-Annotation:** Each case was initially tagged with a silver intent label assigned via `all-MiniLM-L6-v2` semantic embeddings, KMeans clustering, and LLM-assisted cluster consolidation.
2. **Human Verification:** All 200 candidate examples were individually reviewed and annotated by a human evaluator using `src/evaluation/annotate_golden.py`.
3. **Disagreement Handling:**
   - **Silver vs. Human Agreement:** 92.50% (185 out of 200 cases agreed).
   - **Disagreements (15 cases):** Reviewed and corrected to reflect the true underlying customer intent. For example, cases where update-induced hardware symptoms were misassigned to hardware rather than `ios_update_issues` or general frustration misassigned to `other_unclear`.
4. **Data Isolation:** The entire 200-example Golden Set is strictly frozen for evaluation. All training of baseline and candidate intent models is restricted to non-Golden Set data.
