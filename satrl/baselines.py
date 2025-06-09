"""Small-instance reference algorithms used for validation."""

from __future__ import annotations

from itertools import product

from .formula import Assignment, CNFFormula, is_satisfied


def brute_force_solve(
    formula: CNFFormula, *, variable_limit: int = 24
) -> Assignment | None:
    """Return the first satisfying assignment by exhaustive enumeration.

    This is intentionally limited to small formulas and serves as an oracle for
    tests and methodology checks, not as a scalable solver.
    """

    if formula.num_variables > variable_limit:
        raise ValueError(
            f"brute-force validation is limited to {variable_limit} variables"
        )
    for values in product((False, True), repeat=formula.num_variables):
        assignment = {
            variable: values[variable - 1] for variable in formula.variables
        }
        if is_satisfied(formula, assignment):
            return assignment
    return None
