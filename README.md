# SahayakLLM

SahayakLLM is a four-agent support assistant for the fictional KharidiKart store. Its Router uses the supplied Qwen2.5-1.5B-Instruct QLoRA adapter. The Knowledge agent searches Markdown policies with BGE and FAISS. The Action agent uses eight validated SQLite tools. The Verifier checks drafts against their evidence and permits one revision.

## Run the application

Use Python 3.10 or later. Run the commands from the repository root.

    python -m pip install -r requirements.txt
    python -m sahayak.db
    python -m sahayak.retrieval
    python -m streamlit run app.py

The first model-backed use loads the Qwen base and the adapter. The app accepts the Hugging Face adapter ID or a local directory. Run the following command to restore the supplied archive.

    python -m sahayak.provenance --adapter path/to/C2_adapter.zip

The application uses .runtime/C2_adapter automatically when present. SAHAYAK_ADAPTER and SAHAYAK_BASE override the model paths. Set SAHAYAK_NF4=1 on a CUDA runtime with the GPU dependencies. For a quick demo without model downloads, set SAHAYAK_DEMO=1 before launching Streamlit. This mode uses rules and lexical retrieval and has its own validation results.

For model-backed use on a machine with limited memory, see the [local CPU setup](docs/LOCAL_INFERENCE.md).

Select C0001 for the scripted demo. Ask about the UPI refund timeline, track KK-10001, then ask to cancel it. Review the confirmation, approve it, and open Database to see the refund. The account selector is for the fictional demo. Use a fresh database file through SAHAYAK_DB for another cancellation walkthrough.

## Reproduce the training and evaluation

| Stage | Command or entry point |
| --- | --- |
| Audit, split, and reproduce B0/B1 | Run the cells in [Notebook 01](notebooks/01_data_and_baselines.ipynb). The original Team 10 prompt and split manifests are under prompts/ and data/splits/. |
| Restore the original split export | Run `python -m sahayak.provenance --splits path/to/SahayakLLM.zip`. This verifies file hashes against the checked-in manifests. The raw Bitext splits remain local. |
| Reproduce C2 or train the data budgets | Run [Notebook 02](notebooks/02_finetune_and_budget.ipynb) on a Kaggle T4 with the Notebook 01 outputs attached. Enable RUN_DATA_BUDGET for the additional budgets. |
| Seed a mock database | Run `python -m sahayak.db --seed 10`. The date and identifiers are fixed for repeatable fixtures. |
| Build and score the index | Run `python -m sahayak.retrieval --evaluate results/retrieval.json --sweep results/e5_retrieval_sweep.csv`. |
| Compare retrieval settings end to end | Run `python -m sahayak.retrieval_eval --nf4 --output results/runs/t4/e5`. This compares the section index with six token-window settings on the same 25 information scenarios. |
| Check the pipeline | Run `python -m pytest -q` after installing requirements-eval.txt. |
| Run the fixture-only integration suite | Run `python -m sahayak.benchmark --configs demo --output .runtime/demo-check`. |
| Run the matched Qwen benchmark | Run [Notebook 03](notebooks/03_evaluate_agents.ipynb) on a CUDA runner, or run `python -m sahayak.benchmark --nf4 --configs C0 C1 C3 C2 C4 C5 --resume --output results/runs/t4` after installing requirements-gpu.txt. Use the same GPU and precision throughout. |
| Check the Router on the frozen test split | Run `python -m sahayak.language_eval --router-only --nf4 --split test --output results/runs/t4/router-test`. Repeat with held_out_test to check generalization. |
| Build the experiment table and figures | Run `python -m sahayak.research --runs results/runs/t4`. |
| Prepare independent human ratings | Run `python -m sahayak.research --runs results/runs/t4 --human-sheet .runtime/ratings.csv`. Share the prompts, evidence, and rating rubric with two raters. After both have completed the sheet, run `python -m sahayak.research --raters rater1.csv rater2.csv`. |

The benchmark uses 100 scripted scenarios and a 40-prompt red-team suite. Each scenario starts with a fresh copy of the seed database. Task success checks the expected facts and database outcome. Cited task success also requires the expected citation where applicable and excludes flagged answers. The unsupported-claim flag uses a pattern check against approved drafts for human review. The Verifier accepts a policy answer when it exactly matches a retrieved approved answer and citation. Its retry uses the selected policy excerpt directly. A write tool rechecks ownership and eligibility inside its transaction.

## Repository map

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md) and [task tracker](docs/TASK_TRACKER.md)
- [Architecture, schema and demo](docs/ARCHITECTURE.md)
- [Model card](docs/MODEL_CARD.md) and [data audit](reports/data_audit.md)
- [Training experiment table](results/training_notebook_e1.csv) and [retrieval benchmark](results/retrieval.json)
- [All six CPU configurations](results/cpu_scenarios.csv), [per-scenario scores](results/cpu_scenario_cases.csv) and [safety cases](results/cpu_safety_cases.csv)
- [Research manuscript draft](reports/SahayakLLM.pdf), [editable source](reports/manuscript.md) and [demo walkthrough](docs/DEMO.md)

## Attribution

The [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) is by Bitext Innovations and licensed under CDLA-Sharing-1.0. The [Qwen base model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) is Apache-2.0. The [published adapter](https://huggingface.co/aryan22medigeri/llm-qwen2.5-1.5b-qlora) has its own model card.
