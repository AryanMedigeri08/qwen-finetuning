# Data audit - Bitext Customer Support dataset (auto-generated 2026-11-18)

Source: `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (CDLA-Sharing-1.0, Bitext Innovations). Hybrid synthetic data: no real customer data.

## Size and structure
- Rows: **26,872**, columns: flags, instruction, category, intent, response. Null cells: 0.
- Intents: **27**, categories: **11**.

## Duplicates
- Exact duplicate (instruction, response, intent) pairs: **0**
- Rows whose instruction text already appears earlier: **2237** (of which 989 instruction strings have >1 distinct response)
- Extra near-duplicates after lower-casing/stripping punctuation: **562**

## Class balance
- Rows per intent: min **950**, median **998**, max **1000** (max/min = 1.05) -> essentially balanced.

## Responses
- Words per response: min 9, median 90, p95 206, max 402.
- Responses with fewer than 20 words: 61.
- Responses containing a `{{placeholder}}`: **48.4%**; distinct placeholder names: 389.
- Responses containing a URL / e-mail / phone-like string: 57.
- Instructions flagged with noise (typos, tag Z): 19.7%.

## Risks and how the project handles them
1. **Synthetic, templated language** - many responses share boilerplate openings, so ROUGE-L/BERTScore can look good without real understanding. Mitigation: report them next to intent F1, TSR and human evaluation.
2. **Near-paraphrase leakage** - a random split would put paraphrases of one query in train and test. Mitigation: held-out intents + stratified split + cross-split overlap check in the manifest step.
3. **Placeholders instead of facts** - the model learns where facts go but not the facts. Mitigation: keep slots in targets (T12), fill them with tools, measure leakage.
4. **Generic policies** - Bitext responses contain no KharidiKart-specific numbers, so hallucination of policy numbers cannot be judged from this data alone. Mitigation: our own knowledge base with concrete numbers.
5. **Held-out intents remove whole categories** - a held-out intent that is the only member of its category removes that category from training.
6. **Small number of distinct intents (27) and clean label noise** - intent F1 on this data is likely to be high for a fine-tuned model; do not over-interpret it.
