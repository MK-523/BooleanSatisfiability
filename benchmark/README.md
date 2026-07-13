# SAT Policy Audit and Benchmark

This directory contains a reproducible audit and evaluation of the preserved
policy-gradient experiment in [`legacy/original`](../legacy/original).

The result is deliberately negative: the checked-in policy does not condition on the input CNF formula, and the preprocessing path flattens ordinary two-dimensional formulas. Under an equal 64-candidate budget, a behavior-level reconstruction of that formula-agnostic policy showed no reliable held-out advantage over uniform random search.

## What is included

- `audit_upstream.py`: reproduces the preprocessing shape issue and verifies that the preserved model uses a constant all-ones input instead of the formula.
- `benchmark.py`: generates deterministic train/test splits, trains a formula-agnostic Bernoulli policy, evaluates equal-budget random search, and computes exact optima for small formulas.
- `analyze_results.py`: calculates paired policy-minus-random differences and 95% confidence intervals.
- `tests/`: correctness and split-isolation tests.
- `results/`: raw runs, aggregate summaries, paired differences, run metadata, and the upstream audit result.
- `BENCHMARK_REPORT.md`: methodology, tables, interpretation, limitations, and the smallest defensible redesign.

## Reproduce

From this `benchmark` directory after cloning the repository:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python audit_upstream.py
python benchmark.py --output-dir results --data-seed 523 --datasets 5 --runs 3 --update-budget 30000
python analyze_results.py
```

## Protocol

For each of `(8 variables, 32 clauses)`, `(10, 40)`, and `(12, 48)`, the harness generates 160 training formulas and 40 held-out test formulas. It repeats the protocol across five independently generated datasets and three stochastic run seeds. Clauses use three distinct variables with independently sampled signs, and train/test formulas are fingerprinted to detect overlap.

The reported methods are:

- exact enumeration, feasible here only because the largest search space contains 4,096 assignments;
- uniform random search with candidate budgets of 1 and 64;
- the formula-agnostic policy-gradient baseline with the same candidate budgets.

Metrics include satisfaction ratio, solved rate, exact-optimal rate, regret to exact, evaluation time, paired differences, and 95% confidence intervals.

## Main finding

At 64 candidates per formula, every paired 95% confidence interval for policy-minus-random satisfaction ratio and solved rate includes zero. The current policy class therefore does not demonstrate formula-specific learning.

See [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) for the complete result and methodological limitations.
