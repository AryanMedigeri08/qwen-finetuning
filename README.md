# SahayakLLM

SahayakLLM is Team 10’s research prototype for customer support at **KharidiKart**, a fictional ecommerce store. It combines a QLoRA adapter for Qwen2.5-1.5B-Instruct with a four-agent LangGraph workflow, a searchable policy knowledge base, and a mock SQLite database.

- **Router** identifies the customer’s intent and extracts order or refund references.
- **Knowledge** retrieves policy sections with BGE and FAISS and prepares cited answers.
- **Action** handles order, refund, invoice, and support requests using eight validated tools. Record changes require confirmation for the specific action.
- **Verifier** checks drafts against the available evidence and allows one revision.

The Streamlit dashboard includes Chat, Trace viewer, Experiments, Database, and About. [Architecture](docs/ARCHITECTURE.md) · [Published QLoRA adapter](https://huggingface.co/aryan22medigeri/llm-qwen2.5-1.5b-qlora) · [Model card](docs/MODEL_CARD.md)

## Key results

The matched benchmark used **Tesla T4 with NF4 quantization** and the same 100 scripted scenarios for all six configurations.

| Configuration | Factual task success | Cited task success |
| --- | ---: | ---: |
| C0 · Base, zero-shot | 11% | 10% |
| C1 · Base, five-shot | 9% | 5% |
| C2 · Fine-tuned model | 1% | 0% |
| C3 · Base model with agents | 40% | 40% |
| C4 · Fine-tuned model with all four agents | **89%** | **88%** |
| C5 · Fine-tuned model with the Verifier removed | 75% | 71% |

Factual task success checks the expected answer facts and database outcome. Cited task success also checks the expected citation where applicable and excludes responses flagged by the automated checks. [Main results](results/main_table.csv) · [Case-level scores](results/cases.csv) · [Comparison figure](figures/fig1_main.png)

- **Safety.** C4 and C5 each passed all 40 scripted safety cases. The automated unsupported-claim flag appeared in 0% of C4 scenarios and 7% of C5 scenarios. [Safety results](results/e6_safety.csv) · [Error taxonomy](results/error_taxonomy.csv)
- **Retrieval.** The section index achieved 95.0% Recall@3 and 0.869 MRR@3 on 60 labelled questions. All seven retrieval settings tied at 17/25 cited successes with Router decisions held fixed. The tie-break favored the section index at `k=3`. [Retrieval benchmark](results/retrieval.json) · [T4 comparison](results/e5_t4_end_to_end.csv)
- **Router.** Intent macro-F1 was 0.9992 on 2,388 test inputs and 0.3604 on 2,997 inputs from the three held-out intents. [Router results](results/t4_router.csv)
- **Training budget.** The 1,000-row adapter was the smallest tested budget to meet the target. It used **5.2%** of the training rows and achieved **87%** factual task success, above the **84.55%** threshold set by the full-data run’s 89%. The 10,000-row run scored highest at 92%. Each budget used one seed. [Budget results](results/data_budget.csv) · [Figure 3](figures/fig3_data_budget.png)

## Run locally

Use Python 3.10 and run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m sahayak.db --seed 10
```

Build the policy index and start the model-backed app.

```bash
python -m sahayak.retrieval
python -m streamlit run app.py
```

The first model-backed request loads the Qwen base model and the published adapter. To use a supplied adapter archive, restore it with `python -m sahayak.provenance --adapter path/to/C2_adapter.zip`. The app detects the restored `.runtime/C2_adapter` directory automatically. `SAHAYAK_ADAPTER` and `SAHAYAK_BASE` can override the model locations. For a machine with limited memory, see the [local CPU guide](docs/LOCAL_INFERENCE.md).

For a quick deterministic demo with lexical retrieval, launch the app with `SAHAYAK_DEMO=1`. In PowerShell, run the following after installing dependencies and seeding the database.

```powershell
$env:SAHAYAK_DEMO = "1"
python -m streamlit run app.py
```

Select customer `C0001`. Ask how long a UPI refund takes, track `KK-10001`, then request a cancellation. Confirm the cancellation and open Database to see the updated order and refund. Set `SAHAYAK_DB` to a fresh database path to repeat the walkthrough.

[Demo walkthrough](docs/DEMO.md) · [Five-minute video](reports/SahayakLLM-demo.webm) · [Presentation slides](reports/SahayakLLM-demo.pptx)

## Reproduce the experiments

| Stage | Entry point |
| --- | --- |
| Audit the data and run the zero-shot and few-shot baselines | [Notebook 01](notebooks/01_data_and_baselines.ipynb) |
| Restore the frozen Team 10 splits | `python -m sahayak.provenance --splits path/to/SahayakLLM.zip`. An extracted split directory also works. The command verifies the file hashes. |
| Replay the saved baseline predictions | `python -m sahayak.audit_baselines path/to/SahayakLLM.zip --output results/baseline_metric_audit.csv` |
| Reproduce QLoRA training and train the smaller budgets | [Notebook 02](notebooks/02_finetune_and_budget.ipynb) on a Kaggle T4 with the Notebook 01 outputs attached. Set `RUN_DATA_BUDGET = True` for the smaller budgets. |
| Run the matched scenario, safety, and Router evaluations | [Notebook 03](notebooks/03_evaluate_agents.ipynb) on a CUDA runtime |
| Score retrieval on the labelled questions | `python -m sahayak.retrieval --evaluate results/retrieval.json --sweep results/e5_retrieval_sweep.csv` |
| Compare retrieval settings end to end on CUDA | `python -m sahayak.retrieval_eval --nf4 --output results/runs/t4/e5` |
| Rebuild the scenario results and figures | `python -m sahayak.research --runs results/runs/t4` |
| Rebuild the budget comparison | Place the full scenario run under `results/runs/t4` and the budget outputs under `results/runs/budget_<rows>`, then run `python -m sahayak.research --budgets results/runs`. |

On a Kaggle or Colab CUDA image, run `python -m sahayak.gpu_setup` to install the evaluation dependencies while preserving the image’s CUDA-enabled PyTorch and numerical packages. The setup runs an NF4 check before loading the model.

To run the tests and fixture benchmark locally, use the following commands.

```bash
python -m pip install -r requirements-eval.txt
python -m pytest -q
python -m sahayak.benchmark --configs demo --output .runtime/demo-check
```

## Human evaluation and report status

The completed T4 runs include a blinded sample of 100 C2 and C4 responses. Two independent raters will score correctness, helpfulness, and tone on a 1–5 scale using the same reference evidence. After both submit their scores, run `python -m sahayak.research --raters rater1.csv rater2.csv` to calculate quadratic-weighted Cohen’s κ for each criterion.

The manuscript includes the verified T4 scenario, safety, retrieval, Router, and training-budget results. The independent ratings, C3–C5 Bitext response scores for the full experiment table, and final IEEE PDF are pending. [Updated manuscript](reports/manuscript.md) · [Initial PDF draft](reports/SahayakLLM.pdf) · [Task tracker](docs/TASK_TRACKER.md)

## Repository guide

| Path | Contents |
| --- | --- |
| `sahayak/` | Agent graph, model inference, policy retrieval, SQLite tools, and evaluation |
| `policies/` | KharidiKart policy documents |
| `benchmarks/` | Scenario, safety, and retrieval fixtures |
| `notebooks/` | Data preparation, baselines, QLoRA training, and GPU evaluation |
| `results/` and `figures/` | Verified experiment tables, case scores, figures, and run provenance |
| `reports/` and `docs/` | Manuscript, demo materials, implementation plan, and reproducibility notes |

## Attribution

The support data comes from Bitext Innovations and is licensed under CDLA-Sharing-1.0. The Qwen2.5-1.5B-Instruct base model is licensed under Apache-2.0. KharidiKart policies, customers, and transactions are fictional. [Bitext dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) · [Qwen base model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
