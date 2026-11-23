# Fine tuning and agentic grounding for ecommerce support

Team 10 · MIT Academy of Engineering, Alandi, Pune

## Abstract

We adapted Qwen2.5-1.5B-Instruct to customer support using a low-rank adapter, then connected it to an order database and a versioned policy index for the fictional KharidiKart store. On the original Bitext test split, the adapted model increased macro-F1 from 0.615 to 0.999 over a five-example baseline and increased BERTScore-F1 from 0.887 to 0.923. Placeholder leakage also increased from 21.1% to 46.9%, motivating a workflow that fills facts from SQLite or quotes supporting policy sections. The final policy index retrieved the labelled section in the top three for 57 of 60 questions. Matched model-backed scenario results and independent ratings are required before drawing conclusions about the workflow's effect on task success and hallucination.

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

The scripted scenario benchmark contains 25 policy questions, 25 reads, 20 confirmed mutations, 10 missing-slot cases, 10 denied requests, and 10 escalations. Each case starts with a copy of the same seed database. A success satisfies the scenario's required facts and citations and checks the final database state. Tool accuracy compares both the handler name and arguments. An automated unsupported-claim flag identifies suspicious responses for independent review.

The 40-prompt suite probes instruction overrides, cross-customer requests, out-of-scope content, premature confirmation and emotional escalation. C5 still runs ownership checks and confirmation logic, so the ablation measures the draft-verification component while preserving database transaction rules.

## V. Measured results

| Configuration | Test intent macro-F1 | ROUGE-L | BERTScore-F1 | Perplexity | Placeholder leakage |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 base zero-shot | 0.501 | 0.214 | 0.874 | 4.630 | 9.4% |
| C1 base five-shot | 0.615 | 0.267 | 0.887 | 4.048 | 21.1% |
| C2 adapted | 0.999 | 0.446 | 0.923 | 1.626 | 46.9% |

These measurements come from the original notebook's test table. The adapter's held-out intent macro-F1 fell to 0.348 and BERTScore-F1 fell to 0.898. The difference suggests weak zero-shot routing for labels omitted from SFT, despite substantial improvement on labels the model saw during training. The higher placeholder leakage is consistent with learning the training response format.

The BGE section index reached 95.0% Recall@3 and 0.869 MRR@3 over 60 authored questions. The 200-token windows with 50-token overlap reached 100% Recall@3 and 0.878 MRR@3 when any indexed window overlapping the target section counted as a hit. The section index is kept for direct citation selection because its units correspond to answerable policy passages. The window comparison measures section coverage. It still needs a model-backed information scenario comparison to establish which strategy gives the best end-to-end answers.

## VI. Remaining analyses

Use one CUDA runtime to execute the six configurations on the same scenario suite. Compare C2 and C4 for grounding, C3 and C4 for the incremental effect of fine-tuning, and C4 and C5 for the Verifier. Review unsupported-claim flags and false blocks against the recorded tool output and policy packet. Use the four additional training budgets to find the smallest budget reaching 95% of full-data task success. Collect two independent reviews of 100 responses and report weighted Cohen's kappa for each rating question.

## VII. Limitations

The store uses a mock database, a small authored policy base, and fixed evaluation fixtures. The interface simulates identity through a demo account selector. Router token confidence remains uncalibrated. Exact evidence matching can reject valid paraphrases, and a retrieved passage can be factually correct while answering the wrong question. The authored policy suite and synthetic support prompts do not establish performance on production traffic.

## References

1. Bitext Innovations. Bitext Customer Support LLM Chatbot Training Dataset. CDLA-Sharing-1.0. [Dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
2. Qwen Team. Qwen2.5-1.5B-Instruct. Apache-2.0. [Model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
3. BAAI. bge-small-en-v1.5. [Embedding model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
4. Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. QLoRA: Efficient Finetuning of Quantized LLMs. 2023. [Preprint](https://arxiv.org/abs/2305.14314)
