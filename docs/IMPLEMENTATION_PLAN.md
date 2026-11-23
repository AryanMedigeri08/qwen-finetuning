# SahayakLLM implementation plan

The Qwen2.5-1.5B-Instruct QLoRA adapter is the starting point. Use the original training prompts, few-shot examples, and split manifest. Keep the held-out intents assigned to Team 10: `check_invoice`, `review`, and `switch_account`. Use only fictional customer records.

| Milestone | Brief tasks | Evidence required |
| --- | --- | --- |
| 1. Recover training provenance | T01–T26 | Archive the frozen prompts, hashes, code, available results, and model configuration. Smoke-test the supplied adapter with the original base model. |
| 2. Build trusted data sources | T27–T34 | Write 20–25 Markdown policies, seed 200 customers and 500 orders, publish eight validated tool schemas and tests, and measure BGE/FAISS recall@3 and MRR on 60 labelled questions. |
| 3. Build the agent workflow | T35–T41 | Connect Router, Knowledge, Action, and Verifier in LangGraph. Verify ownership, per-action confirmation, turn budgets, and retry behaviour. Export the graph and trace evidence. |
| 4. Run controlled experiments | T42–T47 | Execute 100 scenarios and 40 safety prompts. Run C0–C5 with the same fixtures and decoding. Collect E1–E6, record failures, and recruit two raters for the 100-response review. |
| 5. Deliver the demo and report | T48–T53 | Run the five-page Streamlit demo. Generate figures from recorded results, assemble the IEEE report and presentation, and record the five-minute walkthrough. T50 is optional. |

Publish GitHub commits as each milestone produces usable evidence. Record model provenance, runtime and result scope with every run. Report unavailable measurements as pending.

The [task tracker](TASK_TRACKER.md) records the evidence and the remaining checks for each of the 53 brief tasks. The training evidence, retriever, backend, agents and dashboard are in the repository. The scenario fixtures and safety prompts have both deterministic integration and model-backed CPU results. Use Notebook 03 for the matched CUDA experiment and to prepare the two-rater evaluation.

The Python package under sahayak/ implements the database, tools, retrieval, orchestration and evaluation roles from the brief. The Markdown policies are under policies/. The production index uses document sections as citation units. The retrieval benchmark measures the requested token-window configurations alongside that section index.
