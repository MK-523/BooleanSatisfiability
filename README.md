# SATRL Exact

SATRL Exact is a dependency-free Python toolkit for reading, generating, and
exactly solving Boolean satisfiability problems in conjunctive normal form.
Its supported solver is deterministic DPLL with unit propagation, pure-literal
elimination, subsumption-aware preprocessing, and a formula-dependent branching
heuristic.

The name preserves the repository's history, but the supported package does not
claim to be a learned solver. The original policy-gradient prototype ignored
its input formula and is preserved under [`legacy/original`](legacy/original)
for reproducible audit work. Its seven original files also remain byte-for-byte
at the repository root so historical links and imports keep their paths; new
code should use `satrl`.

## What works end to end

- strict DIMACS CNF parsing and serialization;
- shape-preserving clause normalization with explicit validation;
- exact SAT/UNSAT results when no search-node budget is configured;
- verified satisfying assignments;
- `UNKNOWN` rather than a guess when a node budget is exhausted;
- deterministic random and planted-solution k-CNF generation;
- small-instance exhaustive enumeration as a correctness oracle;
- JSON/CSV benchmark reports with raw per-instance records;
- unit, integration, CLI, and randomized property tests;
- continuous tests across Python 3.10–3.12.

## Quick start

The supported solver has no third-party runtime dependencies.

```bash
python -m pip install -e .
python -m satrl solve examples/satisfiable.cnf
python -m satrl solve examples/pigeonhole_3_2.cnf --json
```

A human-readable SAT result contains a verified DIMACS-style assignment:

```text
s SAT
v 1 -2 3 4 0
c verified=true ...
```

Generate a reproducible instance:

```bash
satrl generate \
  --variables 10 \
  --clauses 40 \
  --clause-size 3 \
  --seed 523 \
  --output instance.cnf
```

Use `--planted` when a known satisfying witness is useful during development.
The witness is not written into the DIMACS output.

## CLI

```text
satrl solve INPUT [--json] [--max-nodes N] [--no-preprocess]
satrl generate --variables N --clauses M --clause-size K --seed S
satrl benchmark [--variables N] [--clauses M] [--instances R] [--output report.json]
```

`INPUT` may be `-` to read DIMACS from standard input. A bounded search exits
with status code `3` if it reaches the node limit and prints `s UNKNOWN`.
Malformed input and invalid arguments exit with status code `2`.

## Python API

```python
from satrl import CNFFormula, SolveStatus, solve

formula = CNFFormula.from_clauses(
    [[1, 2], [-1, 3], [-2, 3]],
    num_variables=3,
)
result = solve(formula)

assert result.status is SolveStatus.SAT
assert result.verified
print(result.assignment)
```

## Reproducible validation

Run the maintained package tests:

```bash
python -m unittest discover -s tests -v
```

The randomized property suite generates many small formulas and requires DPLL
to agree with exhaustive enumeration. It separately checks that preprocessing
preserves satisfiability and that planted generators retain their witnesses.

Run a benchmark without overwriting a checked-in claim:

```bash
mkdir -p benchmark-output
satrl benchmark \
  --variables 10 \
  --clauses 40 \
  --clause-size 3 \
  --instances 20 \
  --seed 523 \
  --output benchmark-output/dpll-10v-40c.json
```

See [results methodology](docs/results-methodology.md) before interpreting
runtime numbers. This repository does not claim that this educational DPLL
implementation is competitive with production CDCL solvers.

## Repository map

| Path | Purpose |
|---|---|
| [`satrl/formula.py`](satrl/formula.py) | Immutable CNF model, preprocessing, and evaluation |
| [`satrl/dimacs.py`](satrl/dimacs.py) | Validated DIMACS input/output |
| [`satrl/solver.py`](satrl/solver.py) | Exact DPLL solver and search statistics |
| [`satrl/generator.py`](satrl/generator.py) | Seeded random and planted k-CNF generation |
| [`satrl/baselines.py`](satrl/baselines.py) | Small-instance exhaustive oracle |
| [`satrl/benchmarking.py`](satrl/benchmarking.py) | Per-instance benchmark records and reports |
| [`satrl/cli.py`](satrl/cli.py) | `solve`, `generate`, and `benchmark` commands |
| [`tests`](tests) | Unit, CLI, and randomized correctness tests |
| [`examples`](examples) | Small documented DIMACS instances |
| [`docs/architecture.md`](docs/architecture.md) | Data flow and solver design |
| Root prototype files | Byte-for-byte historical paths retained for compatibility |
| [`legacy/original`](legacy/original) | Unmodified historical PyTorch prototype |
| [`benchmark`](benchmark) | Preserved audit of the historical policy |

## Historical policy audit

The historical benchmark remains intentionally separate from the exact solver.
It reproduces two blocking issues in the old prototype: CNF preprocessing could
flatten formulas, and the policy used a constant input rather than the formula.
Its formula-agnostic policy showed no reliable held-out advantage over uniform
random search under the recorded protocol.

Those negative results are evidence about the preserved prototype only. They
are not presented as a comparison between DPLL and a learned solver.

## Scope and limitations

- DPLL is exact but has exponential worst-case complexity.
- The implementation is designed for clarity, reproducibility, and small to
  medium experiments; it does not implement clause learning, watched literals,
  restarts, or industrial preprocessing.
- Exhaustive enumeration is restricted to small formulas and is used only as a
  correctness oracle.
- Generated random k-CNF instances do not represent industrial SAT workloads.
- No formula-conditioned learned model is included. The requirements for one
  are described in the [architecture document](docs/architecture.md).

## License

MIT. See [`LICENSE`](LICENSE).
