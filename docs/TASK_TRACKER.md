# SahayakLLM task tracker

The task descriptions, owners and weeks follow Section 27 of the brief. Recovered results come from the original notebook exports. Review items require the people named in the brief.

| ID | Task | Owner | Week | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| T01 | Create GitHub repo with folders data/, kb/, db/, src/agents/, src/tools/, notebooks/, eval/, app/, reports/; add .gitignore, licence note and branch protection on main | S3 | 1 | Team review | GitHub milestones pushed. Member commits and branch-protection setup pending. |
| T02 | Set up Colab/Kaggle GPU notebook; write and pin requirements.txt; load the model in 4-bit and generate one reply | S1 | 1 | Partial | Requirements pinned. A fresh CUDA/NF4 smoke run is pending. |
| T03 | Create experiment log (MLflow or results CSV) with columns run_id, config, data_manifest_hash, seed, all metrics, date | S4 | 1 | Review pending | CSV header in eval/results_template.csv. |
| T04 | Record team configuration: team number (Faker seed), three held-out intents, global seed 42 | S4 | 1 | Recovered | config/team.yaml preserves Team 10 settings and seed 42 from the training export. |
| T05 | Load Bitext with the datasets library; report rows, columns, dtypes and 10 sample rows | S1 | 1 | Recovered | notebooks/01_data_and_baselines.ipynb and reports/data_audit.md. |
| T06 | Plot intent and category distributions; confirm the intent list | S1 | 1 | Recovered | figures/T06_intent_category_distribution.png. |
| T07 | Tokenise instructions and responses with the Qwen tokenizer; plot length histograms; choose max_seq_length covering ≥ 98% of pairs | S1 | 1 | Recovered | figures/T07_token_lengths.png and data audit. |
| T08 | Analyse the flags column against the dataset card: meaning and count of each tag | S2 | 1 | Recovered | figures/T08_flags.png. |
| T09 | Extract all {{placeholder}} strings from responses by regex; frequency table; map each to DB field, KB document or static config | S3 | 1 | Recovered | Placeholder map from the original analysis export. |
| T10 | Write a one-page data audit: duplicates, near-duplicates, odd responses, class balance, risks | S4 | 1 | Review pending | reports/data_audit.md preserves the prior audit. |
| T11 | Clean text (whitespace, quotes); drop exact duplicate pairs; log rows removed | S1 | 1 | Recovered | src/data/clean.py and original cleaning log. |
| T12 | Decide and document placeholder strategy (keep {{slots}} in targets so the model learns where facts go; agents fill them) | S1 | 2 | Review pending | docs/decisions.md records the placeholder strategy. |
| T13 | Remove the three held-out intents into held_out_test.jsonl | S1 | 2 | Verified | Held-out split recovered and checked against the original manifest. |
| T14 | Stratified 80/10/10 split on remaining intents; write manifest with per-intent counts and SHA-256 hashes | S1 | 2 | Verified | All ordinary split file hashes against data/splits/manifest.json. |
| T15 | Create fixed 300-row stratified eval-mini from validation | S4 | 2 | Verified | Restored the hashed 300-row eval-mini. |
| T16 | Convert records to chat-message format via apply_chat_template; build routing records (instruction → intent JSON), ~20% of train | S1 | 2 | Verified | Restored both SFT files using data/splits/sft_manifest.json. |
| T17 | Implement evaluation harness: intent accuracy/macro-F1, ROUGE-L, BERTScore, perplexity, placeholder leakage, latency | S4 | 2 | Baselines audited | Replaying archived base predictions matched 20 reported values. Aggregating saved per-example BERTScores matched four more. |
| T18 | Run B0 zero-shot on eval-mini and test; save predictions | S1 | 2 | Recovered | B0 test and eval-mini results in results/results_log.csv. |
| T19 | Run B1 few-shot with five fixed exemplars | S1 | 2 | Recovered | B1 rows and fixed exemplars. |
| T20 | Compare three system-prompt variants on eval-mini; freeze the best | S2 | 2 | Recovered | Frozen prompts and results/prompt_comparison.csv. |
| T21 | Configure QLoRA (Section 11); record trainable parameter count and % of total | S1 | 3 | Recovered | docs/MODEL_CARD.md and the provided adapter configuration. |
| T22 | Train on full train split (multi-task); log train/val loss every 50 steps; save adapter checkpoints | S1 | 3 | Recovered | Training notebook and Figure 2. |
| T23 | Evaluate fine-tuned model (C2) on test and held-out-intent sets; compare with B0/B1 | S1 | 3 | Recovered | results/training_notebook_e1.csv includes C2 test and held-out runs. |
| T24 | Measure inference cost: tokens/s, peak GPU memory, p50/p95 latency for base vs fine-tuned | S4 | 3 | Recovered | results/training_notebook_systems.csv records the Tesla T4 probes. |
| T25 | Data-budget runs: 1k, 2k, 5k, 10k, full training rows | S1 | 4 | Pending compute | Train the four smaller budgets and run each adapter inside C4. |
| T26 | Upload final adapter (private HF repo or release zip) with model card: data, hyperparameters, limitations | S1 | 4 | Verified | Supplied weights match the public Hub release. Fresh CPU NF4 inference succeeded. |
| T27 | Write 20–25 KharidiKart policy documents (Section 9.1) with concrete numbers | S2 | 2 | Verified | 20 Markdown policies, each over 250 words, under policies/. |
| T28 | Chunk (300 tokens, 50 overlap), embed with bge-small-en-v1.5, build FAISS index with doc/section metadata | S2 | 3 | Verified | One-command section index with BGE and FAISS. E5 compares token windows separately. |
| T29 | Create 60 question → gold-section pairs; compute Recall@3 and MRR | S2 | 3 | Verified | 60 labelled questions and measured recall@3 and MRR. |
| T30 | Tune chunk size (200/300/500) × k (3/5); select best (Experiment E5) | S2 | 3 | Runner ready | Six retrieval configurations measured. The C4 runner compares end-to-end task success and exact citations on 25 information scenarios. T4 measurements pending. |
| T31 | Design SQLite schema (Section 9.2); draw ER diagram | S3 | 2 | Implemented | schema.sql and ER diagram in docs/ARCHITECTURE.md. |
| T32 | Write Faker seed script (team seed): 200 customers, 500 orders in all statuses, refunds, invoices | S3 | 2 | Verified | 200 customers, 500 orders, invoices and refunds with seed 10. |
| T33 | Implement 8 tool functions (Section 15) with JSON schemas, validation and structured errors | S3 | 3 | Verified | Eight validated tool schemas and handlers in sahayak/tools.py. |
| T34 | Write pytest unit tests: ≥ 3 per tool incl. invalid id, wrong customer, not-allowed state | S3 | 3 | Verified | Tool tests cover valid and rejected arguments, ownership and order status. |
| T35 | Define typed AgentState and trace-log format | S3 | 4 | Review pending | Typed state and event traces in sahayak/graph.py. |
| T36 | Router Agent: routing-mode prompt, JSON parsing with retry, regex slot extraction, log-prob confidence | S1 | 4 | Partial | JSON routing and token scores implemented. Full model-backed intent evaluation pending. |
| T37 | Knowledge Agent: retrieve top-k, draft cited answer | S2 | 4 | Verified | Knowledge retrieves top-k and prepares cited drafts. |
| T38 | Action Agent: intent→tool map, missing-slot questions, confirmation for irreversible tools, placeholder filling | S3 | 4 | Verified | Tracking, cancellation, refund and address confirmation flows tested. |
| T39 | Verifier Agent: placeholder, order-fact, policy-number and guardrail checks; PASS/REVISE/ESCALATE | S4 | 5 | Verified | Tests cover five unsupported drafts and the one-retry path. |
| T40 | Build LangGraph StateGraph with conditional edges and step budget; export graph diagram | S3 | 5 | Implemented | LangGraph and architecture diagram. Turn and tool caps covered by fixture checks. |
| T41 | Multi-turn memory: last 6 turns plus pending slot/confirmation state | S3 | 5 | Verified | Multi-turn slot collection, confirmation and dropped confirmation on account change. |
| T42 | Write 100-scenario suite (Section 16.2) against the team database | All | 5 | Verified | 100 scenarios with the prescribed category counts. |
| T43 | Scenario runner: runs any config, computes TSR, tool accuracy, hallucination, leakage, steps, latency | S4 | 5 | Implemented | Scenario and safety runner with isolated fixtures and resumable output. |
| T44 | Run C0–C5 on test set and scenarios; fill main table (Section 17); plot Figure 1 | S4 | 6 | CPU evidence | Each configuration ran 100 CPU scenarios. The comparison CSV and figure are available. Matched T4 run pending. |
| T45 | Human evaluation: 100 C2 and C4 responses, two raters, 1–5 rubric; Cohen's kappa | All | 6 | Pending raters | A blinded CPU pilot sheet with 100 responses was generated. Two independent scores are required. |
| T46 | Error analysis with R1–R7 taxonomy on all failed C4/C5 scenarios | S4 | 6 | CPU evidence | Case-level labels and counts were exported for C4 and C5. The manuscript reviews examples. |
| T47 | Red-team suite (40 prompts, Section 21); safe-handling rate for C4 and C5 | S4 | 7 | CPU evidence | Both C4 and C5 passed all 40 prompts on the Q4_K_M CPU run. Matched T4 run pending. |
| T48 | Streamlit Chat and Database pages with confirmation buttons and citations | S4 | 7 | Verified | Chat, confirmation and database tested through Streamlit. |
| T49 | Streamlit Trace viewer and Experiments pages | S4 | 7 | Verified | Trace viewer and Experiments pages verified. Results are read from files. |
| T50 | Optional extension (Section 23), if chosen | All | 7 | Optional | Outside the required submission. |
| T51 | Reproducibility README: one command per stage (split, train, index, seed, run, app) | S3 | 8 | Clean checkout checked | A fresh GitHub clone at `666ce10` restored all seven split files, seeded SQLite, rebuilt and scored the BGE/FAISS index, passed 43 tests, and reproduced 97/100 cited task successes and 40/40 safety cases in fixture mode. Used the installed dependencies and cached BGE weights. |
| T52 | Research-paper-style report (IEEE format, 6–8 pages): abstract, method, experiments, RQ answers, limitations | All | 8 | Partial | Six-page manuscript draft includes the training evidence, CPU comparison and error analysis. Final IEEE paper depends on matched runs and human ratings. |
| T53 | Five-minute demo video, slides, individual contribution logs; viva preparation | All | 8 | Partial | Captioned demo, editable slides and walkthrough are available. Individual contribution logs and viva are pending. |
