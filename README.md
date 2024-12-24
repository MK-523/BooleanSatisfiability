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

