# Example DIMACS instances

- `satisfiable.cnf` has a verified satisfying assignment.
- `unsatisfiable.cnf` excludes all four assignments to two variables.
- `pigeonhole_3_2.cnf` encodes three pigeons, two holes, and at most one
  pigeon per hole; it is unsatisfiable.

Run an example from the repository root:

```bash
python -m satrl solve examples/pigeonhole_3_2.cnf
```
