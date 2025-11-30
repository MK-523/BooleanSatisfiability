# Architecture

## Supported data flow

```mermaid
flowchart TD
    A["DIMACS or generated clauses"] --> B["Validate header and literals"]
    B --> C["Canonical CNFFormula"]
    C --> D["DPLL search"]
    D --> E{"Result"}
    E -->|SAT| F["Complete and verify assignment"]
    E -->|UNSAT| G["Exhausted exact search"]
    E -->|Node limit| H["UNKNOWN"]
```

### Formula layer

`CNFFormula` stores clauses as immutable tuples of signed integers. Variable
identifiers are one-based to match DIMACS. Construction validates literal
ranges, and parsing validates the declared clause count before preprocessing.

Preprocessing performs satisfiability-preserving transformations:

1. remove repeated literals within a clause;
2. remove tautological clauses containing both `x` and `-x`;
3. remove repeated clauses;
4. remove a clause when a retained shorter clause subsumes it;
5. retain empty clauses as explicit UNSAT evidence.

Unlike the historical tensor mask, these operations always preserve clause
boundaries.

### Exact solver

Every search step operates on the current formula, so solver behavior is
formula-conditioned by construction. DPLL applies:

- unit propagation until a fixed point;
- pure-literal elimination;
- a deterministic Jeroslow-Wang-style literal score;
