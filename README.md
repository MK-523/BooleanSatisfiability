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

## Current limitations

- The method optimizes a stochastic satisfaction ratio. It does not implement the completeness guarantees of a conventional SAT solver.
- Training and evaluation use randomly generated formulas; no standard SAT benchmark suite is included.
- There are no comparisons against random assignment, local search, DPLL/CDCL, or another exact/approximate baseline.
- The repository does not contain a dependency manifest, fixed benchmark corpus, trained checkpoint, or aggregate result table.
- The example evaluates a single generated test formula rather than a separated benchmark distribution.
- [`example_solver.py`](example_solver.py) imports `SAT_solver`, while the checked-in model implementation is named [`model.py`](model.py). That import must be reconciled before the verification example is reproducible as checked in.

## Attribution

This README intentionally describes only the implementation present in the repository. Add collaborators or institutional affiliations only with an exact public source and agreed attribution.

## License

The repository includes an [MIT License](LICENSE).

