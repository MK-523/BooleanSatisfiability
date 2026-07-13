# Contributing

Keep correctness changes small and testable. A solver optimization should:

1. preserve the DIMACS and `CNFFormula` contracts;
2. include a focused regression test;
3. extend the exhaustive-oracle property tests when it changes search logic;
4. report `UNKNOWN`, not a guessed result, when a configured limit is reached;
5. avoid performance claims without a saved, reproducible benchmark report.

Run both maintained test suites before submitting a change:

```bash
python -m unittest discover -s tests -v
(cd benchmark && python -m unittest discover -s tests -v)
```
