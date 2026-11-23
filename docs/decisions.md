# Design decisions

## D1 - Placeholder strategy (T12)
Keep {{slots}} in training targets. The model learns where facts go; agents fill them from verified tool output;
the Verifier rejects any final answer that still contains a slot. Placeholder-leakage rate (C2 vs C4) quantifies the gap.
Sign-off: S1 [ ]  S3 [ ]

## D2 - Router prompt lists all 27 intents
The routing system prompt lists all 27 intent names (including the 3 held-out ones) so that held-out generalisation
(RQ5) measures whether the fine-tuned model can route to a label it never saw as a training target.

## D3 - Decoding
Greedy decoding (do_sample=False, repetition_penalty=1.0 set explicitly; Qwen's default 1.05 is NOT used) for every reported number.
