# Security and resource limits

This project does not execute code embedded in CNF files. The DIMACS parser
accepts integers, comments, and one problem header, and validates declared
variable and clause counts.

SAT is NP-complete, so a valid input can still require exponential time. When
solving untrusted or unexpectedly large formulas, set a search budget:

```bash
satrl solve untrusted.cnf --max-nodes 100000
```

If the budget is reached, the solver reports `UNKNOWN`; it never converts an
incomplete search into a SAT or UNSAT claim. Process-level CPU, memory, and wall
time limits remain appropriate for public services.

Please report suspected vulnerabilities privately to the repository owner
before opening a public issue.
