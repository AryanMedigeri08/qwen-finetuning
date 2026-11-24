# Response ratings

[Rater 1](rater1.csv) and [Rater 2](rater2.csv) score the same 100 blinded responses. Match rows by `response_id`. Each row has a separate 1–5 score for correctness, helpfulness, and tone.

Run the agreement calculation from the repository root.

```bash
python -m pip install -r requirements-eval.txt
python -m sahayak.research --raters rating/rater1.csv rating/rater2.csv
```

The command reports quadratic-weighted Cohen’s kappa for each of the three criteria.
