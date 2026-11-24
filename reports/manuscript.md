# Fine tuning and agentic grounding for ecommerce support

Team 10 · MIT Academy of Engineering, Alandi, Pune

## Abstract

We adapted Qwen2.5-1.5B-Instruct to customer support using a low-rank adapter, then connected it to an order database and a versioned policy index for the fictional KharidiKart store. On the original Bitext test split, the adapted model increased macro-F1 from 0.615 to 0.999 over a five-example baseline and increased BERTScore-F1 from 0.887 to 0.923. Placeholder leakage increased from 21.1% to 46.9%. On 100 matched Tesla T4 scenarios, the verified workflow achieved 89% factual task success and 88% cited task success. Removing the Verifier reduced the latter to 71%. Both versions safely handled all 40 scripted safety prompts. The smallest tested training budget used 1,000 rows and achieved 97.8% of the full-data factual task success. Independent ratings remain open.

## I. Research questions

The study compares prompting and fine-tuning on language and routing, measures whether agentic grounding improves scenario success, tests the Verifier's effect, estimates the smallest effective data budget, and measures generalisation and safe handling. The language benchmark and the scenario benchmark have different gold evidence. The original language references contain placeholders. The scenario references are concrete policy clauses, transaction states, and tool arguments.

## II. Data and adaptation

The Bitext Customer Support LLM Chatbot Training Dataset provides English instructions, support responses, labelled intents, and generation flags. The split uses seed 42 and excludes check_invoice, review, and switch_account from training. The train split contains 19,100 pairs. The held-out file contains 2,997 pairs and the standard test contains 2,388. The SFT export contains 23,875 records because it adds routing targets to a subset of the response examples. About one fifth of the SFT records teach the same model to emit a JSON intent and category.

The adapter uses rank 16, scaling 32, dropout 0.05, seven projection types, and assistant-token-only loss. The original run uses 4-bit NF4 and one epoch. The saved model card records 18.46 million trainable parameters and identifies the original manifest. The training notebook's loss plot, model card, original prompts, and hashed manifest are included with the repository.

## III. Agent and backend design

The Router reads the customer request and identifies a Bitext intent, category and confidence. It extracts order and refund references deterministically. A low confidence result asks for clarification. Recent session context preserves a pending lookup slot, an owned read reference, or the arguments of a pending mutation.

The Knowledge agent retrieves ranked sections from twenty policy documents using BGE normalized embeddings and FAISS inner-product search. The policy packet supplies a set of allowable answers with human-readable section citations. The Action agent uses typed tool arguments and scoped ownership queries. Before executing a cancellation, refund or address update it requests an explicit confirmation tied to that action and customer. Cancellation and refund creation share one database transaction.

The Verifier checks whether the draft exactly matches an approved, evidence-derived answer and whether it contains a placeholder. A rejected policy draft can be revised once by choosing a verified passage. The workflow reserves its final tool call for escalation. The hard limits are six graph steps and three tool calls per turn. The reference store uses seed 10 and contains 200 fictional customers and 500 orders.

## IV. Evaluation setup

C0 uses the base model with a zero-shot prompt. C1 adds five fixed examples. C2 applies the QLoRA adapter without tools. C3 applies the workflow to the base model, C4 combines the workflow with the adapter, and C5 removes the Verifier from C4. All comparisons use greedy generation. The language test measures intent macro-F1, ROUGE-L, BERTScore-F1, perplexity and placeholder leakage. The provided notebook records a Tesla T4 systems benchmark for base and adapted weights.

The scripted scenario benchmark contains 25 policy questions, 25 reads, 20 confirmed mutations, 10 missing-slot cases, 10 denied requests, and 10 escalations. Each case starts with a copy of the same seed database. Task success checks the required facts and final database state. A configuration without tools can receive credit on an action case if it explicitly declines and leaves the database unchanged. Cited task success also checks the expected citation where applicable and excludes flagged answers. Tool accuracy compares the handler name and arguments. An automated unsupported-claim flag identifies suspicious responses for independent review.

The 40-prompt suite probes instruction overrides, cross-customer requests, out-of-scope content, premature confirmation and emotional escalation. C5 still runs ownership checks and confirmation logic, so the ablation measures the draft-verification component while preserving database transaction rules.

## V. Measured results

| Configuration | Test intent macro-F1 | ROUGE-L | BERTScore-F1 | Perplexity | Placeholder leakage |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 base zero-shot | 0.501 | 0.214 | 0.874 | 4.630 | 9.4% |
| C1 base five-shot | 0.615 | 0.267 | 0.887 | 4.048 | 21.1% |
| C2 adapted | 0.999 | 0.446 | 0.923 | 1.626 | 46.9% |

These measurements come from the original notebook's test table. The adapter's held-out intent macro-F1 fell to 0.348 and BERTScore-F1 fell to 0.898. The difference suggests weak zero-shot routing for labels omitted from SFT, despite substantial improvement on labels the model saw during training. The higher placeholder leakage is consistent with learning the training response format.

The strict Router reached macro-F1 of 0.9992 on 2,388 test inputs and 0.3604 on 2,997 held-out inputs in the T4 run. After retry, 16.3% of the held-out responses still failed to produce matching intent and category labels. This difference measures routing generalization to the three withheld intents.

The BGE section index reached 95.0% Recall@3 and 0.869 MRR@3 over 60 authored questions. The 200-token windows with 50-token overlap reached 100% Recall@3 and 0.878 MRR@3 when any indexed window overlapping the target section counted as a hit. The E5 follow-up held Router decisions fixed across seven settings on 25 information scenarios. Each setting achieved 17/25 cited task successes on both Tesla T4 NF4 and the exploratory Q4_K_M CPU runtime. The section index retrieved the expected passage on 22/25 cases, tying four window settings on T4. The tie-break favored the existing section index with k=3.

| Configuration | T4 task success | Cited task success | Unsupported-claim flags | Placeholder leakage |
| --- | ---: | ---: | ---: | ---: |
| C0 Base zero-shot | 11% | 10% | 17% | 11% |
| C1 Base five-shot | 9% | 5% | 13% | 1% |
| C2 Adapter | 1% | 0% | 3% | 68% |
| C3 Base and agents | 40% | 40% | 0% | 0% |
| C4 Adapter and verified agents | 89% | 88% | 0% | 0% |
| C5 Adapter and agents without the Verifier | 75% | 71% | 7% | 0% |

Each row used the same 100 scenarios on a Tesla T4 with NF4. Adding the workflow to the adapter increased factual task success by 88 percentage points from C2 to C4. With the workflow held fixed, the adapter increased factual task success by 49 points from C3 to C4. Removing the Verifier reduced cited task success by 17 points and produced seven unsupported-claim flags. C4 and C5 both safely handled all 40 red-team cases. The exploratory Q4_K_M CPU run also measured 88% cited task success for C4 and 71% for C5.

The T4 error taxonomy assigned nine routing failures, one retrieval miss, and two answer or citation failures to C4. C5 recorded nine routing failures, one retrieval miss, and 19 answer or citation failures. In the exploratory CPU review, one answer quoted the non-metro delivery window for a metro question. The Verifier could confirm the quoted window against its supporting passage while missing the distinction in the question.

## VI. Training-data budget

We stratified the four subsamples by intent with seed 42 and kept each selected row's response record and any associated Router training record. We evaluated each adapter on the same 2,388 language test inputs and 100 C4 scenarios on Tesla T4 with NF4. The full-data row uses the original 19,100-row adapter.

| Training rows | Epochs | Intent macro-F1 | BERTScore-F1 | C4 factual success | C4 cited success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 0.9794 | 0.9132 | 87% | 86% |
| 2,000 | 3 | 0.9874 | 0.9183 | 90% | 89% |
| 5,000 | 2 | 0.9950 | 0.9211 | 91% | 90% |
| 10,000 | 1 | 0.9943 | 0.9203 | 92% | 91% |
| 19,100 | 1 | 0.9992 | 0.9226 | 89% | 88% |

For RQ4, 95% of the 89% full-data factual task success is 84.55%. The 1,000-row adapter was the smallest tested budget that cleared this threshold. It used 5.2% of the training rows and achieved 97.8% of the full-data task success. The 10,000-row run led the scenario table by three cases over full data. Each budget was trained with one seed, and the epoch schedules varied, so this difference has no estimate of run-to-run variation.

## VII. Human evaluation

The 100 blinded T4 responses are ready for independent review. Two raters will score correctness, helpfulness, and tone, with weighted Cohen's kappa reported for each dimension.

## VIII. Limitations

The store uses a mock database, a small authored policy base, and fixed evaluation fixtures. The interface simulates identity through a demo account selector. Router token confidence remains uncalibrated. Exact evidence matching can reject valid paraphrases, and a retrieved passage can be factually correct while answering the wrong question. The authored policy suite and synthetic support prompts do not establish performance on production traffic.

## References

1. Bitext Innovations. Bitext Customer Support LLM Chatbot Training Dataset. CDLA-Sharing-1.0. [Dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
2. Qwen Team. Qwen2.5-1.5B-Instruct. Apache-2.0. [Model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
3. BAAI. bge-small-en-v1.5. [Embedding model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
4. Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. QLoRA: Efficient Finetuning of Quantized LLMs. 2023. [Preprint](https://arxiv.org/abs/2305.14314)
