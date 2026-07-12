# Policy-Gradient Solver for Boolean Satisfiability

An experimental PyTorch implementation that uses a stochastic policy to assign truth values in randomly generated CNF formulas.

The model learns assignment probabilities and is rewarded by the fraction of clauses satisfied by a sampled assignment. This makes the repository an exploration of reinforcement learning for Max-SAT-style objectives; it is not an exact SAT solver and does not prove that a formula is satisfiable or unsatisfiable.

## Approach

For each CNF formula, the pipeline:

1. Generates clauses with positive and negated literals.
2. Preprocesses the generated formula.
3. Uses a neural policy to produce variable-assignment probabilities.
4. Samples a Boolean assignment and records its log probability.
5. Computes the fraction of clauses satisfied.
6. Updates the policy with an Adam-based policy-gradient training loop.
7. Verifies sampled assignments clause by clause in the example solver path.

## Repository map

| File | Purpose |
|---|---|
| [`model.py`](model.py) | Policy network, assignment sampling, satisfiability reward, and loss |
| [`data_utils.py`](data_utils.py) | Random CNF generation, preprocessing, and dataset creation |
| [`train.py`](train.py) | Policy-gradient training loop and reward/loss logging |
| [`solver.py`](solver.py) | Wrapper that generates a formula, samples an assignment, and returns its reward |
| [`formula_utils.py`](formula_utils.py) | Human-readable formula output and assignment interpretation |
| [`example_run.py`](example_run.py) | End-to-end generated-data training and evaluation example |
| [`example_solver.py`](example_solver.py) | Small clause-by-clause verification example |

The checked-in training example uses 10 variables, 50 clauses, three literals per clause, and 100 randomly generated formulas, then evaluates one newly generated formula.

## What the score means

The reported reward is the fraction of clauses satisfied by one sampled assignment:

```text
reward = satisfied clauses / total clauses
```

A score of `1.0` means that the sampled assignment satisfied every clause in that formula. A lower score does not establish unsatisfiability; another assignment may perform better.

## Reproducible audit and benchmark

The [`benchmark`](benchmark) directory audits the checked-in implementation and evaluates its observable formula-agnostic policy against equal-budget uniform random search and exact enumeration on small random 3-CNF formulas.

The audit reproduces two blocking issues: preprocessing flattens ordinary two-dimensional formulas, and the model passes a constant all-ones vector to its policy instead of encoding the input clauses. Across five independently generated datasets and three run seeds per dataset, the reconstructed policy showed no reliable advantage over random search at a 64-candidate budget; all paired 95% confidence intervals included zero.

See the [benchmark report](benchmark/BENCHMARK_REPORT.md) for the full protocol, negative result, limitations, and smallest defensible redesign.

## Current limitations

- The method optimizes a stochastic satisfaction ratio. It does not implement the completeness guarantees of a conventional SAT solver.
- Both the original experiment and the added benchmark use randomly generated formulas; no standard SAT benchmark suite is included.
- The benchmark compares uniform random search and exact enumeration on small instances, but does not yet include local search, DPLL/CDCL, or industrial formulas.
- The original model path does not contain a trained checkpoint or a fixed benchmark corpus.
- The original example evaluates a single generated test formula; the added benchmark supplies separated, fingerprinted synthetic splits instead.
- [`example_solver.py`](example_solver.py) imports `SAT_solver`, while the checked-in model implementation is named [`model.py`](model.py). That import must be reconciled before the verification example is reproducible as checked in.

## Attribution

This README intentionally describes only the implementation present in the repository. Add collaborators or institutional affiliations only with an exact public source and agreed attribution.

## License

The repository includes an [MIT License](LICENSE).
