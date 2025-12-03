# Results methodology

This document defines how to produce and report solver evidence. It contains no
checked-in performance claim because runtime depends on the machine, Python
version, formula distribution, and exact revision being evaluated.

## Correctness gate

Performance results are interpretable only after all of these pass:

1. known SAT instances return assignments that independently satisfy every
   clause;
2. known UNSAT instances are reported as UNSAT;
3. DPLL agrees with exhaustive enumeration on generated small formulas;
4. preprocessing and unprocessed formulas have identical exhaustive SAT status;
5. DIMACS serialization round-trips without changing the canonical formula;
6. a bounded incomplete search reports UNKNOWN.

Run the gate with:

```bash
python -m unittest discover -s tests -v
```

## Benchmark protocol

The `satrl benchmark` command uses consecutive, published seeds beginning at
the requested seed. Each record stores:

- instance index and seed;
- canonical DIMACS SHA-256 fingerprint;
- variables, clauses, and clause size;
- SAT/UNSAT status and assignment-verification result;
- exhaustive-oracle agreement when the variable count is within the configured
  oracle limit;
- nodes, decisions, conflicts, backtracks, and elapsed milliseconds.

JSON output contains environment and configuration metadata, raw records, and a
summary. CSV output contains one row per instance. Raw records should be
retained; a summary alone cannot reveal outliers or a seed-specific failure.
