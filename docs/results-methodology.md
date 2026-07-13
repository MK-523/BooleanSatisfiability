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

## Comparisons

For solver comparisons:

- use the same immutable DIMACS files for every solver;
- identify the dataset, generator parameters, seeds, hardware, Python version,
  dependency versions, and git revision;
- separate SAT and UNSAT instances when reporting runtime;
- report median, mean, dispersion, timeouts, and solved count;
- count UNKNOWN or timeout separately rather than as UNSAT;
- use wall-time and memory limits consistently;
- include an established CDCL solver before making competitiveness claims.

The historical `benchmark/` has a different question and objective: it compares
a reconstruction of the formula-agnostic policy with uniform random candidate
search on Max-SAT-style satisfaction ratio. Its results must not be merged with
exact DPLL timing or presented as a learned-versus-exact solver comparison.

## Suggested report structure

1. exact command and revision;
2. environment and resource limits;
3. instance manifest with hashes;
4. correctness-gate output;
5. raw machine-readable records;
6. aggregate table with uncertainty or dispersion;
7. failures, timeouts, and limitations;
8. conclusion limited to the evaluated distribution and configuration.
