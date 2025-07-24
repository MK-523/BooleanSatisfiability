"""Deterministic random k-CNF generation for examples and benchmarks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .formula import Assignment, CNFFormula


@dataclass(frozen=True, slots=True)
class GeneratedCNF:
    formula: CNFFormula
    seed: int
    planted_assignment: Assignment | None = None


def generate_random_cnf(
    num_variables: int,
    num_clauses: int,
    clause_size: int,
    *,
    seed: int,
    planted: bool = False,
) -> GeneratedCNF:
    """Generate a unique-clause random k-CNF formula.

    When ``planted`` is true, a deterministic assignment is sampled first and
    every generated clause is forced to be satisfied by that assignment.
    """

    if num_variables <= 0:
        raise ValueError("num_variables must be positive")
    if num_clauses < 0:
        raise ValueError("num_clauses must be non-negative")
    if clause_size <= 0 or clause_size > num_variables:
        raise ValueError("clause_size must be between 1 and num_variables")
    maximum_unique = math.comb(num_variables, clause_size) * (2**clause_size)
    if planted:
        maximum_unique -= math.comb(num_variables, clause_size)
    if num_clauses > maximum_unique:
        raise ValueError(
            f"requested {num_clauses} unique clauses, but at most "
            f"{maximum_unique} are available"
        )

    rng = random.Random(seed)
    assignment = (
        {variable: bool(rng.getrandbits(1)) for variable in range(1, num_variables + 1)}
        if planted
        else None
    )
    clauses: set[tuple[int, ...]] = set()
    while len(clauses) < num_clauses:
        variables = sorted(rng.sample(range(1, num_variables + 1), clause_size))
        clause = tuple(
            variable if rng.getrandbits(1) else -variable for variable in variables
        )
        if assignment is not None and not any(
            assignment[abs(literal)] == (literal > 0) for literal in clause
        ):
            position = rng.randrange(clause_size)
            variable = abs(clause[position])
            replacement = variable if assignment[variable] else -variable
            clause = clause[:position] + (replacement,) + clause[position + 1 :]
        clauses.add(clause)

    formula = CNFFormula.from_clauses(
        sorted(clauses), num_variables=num_variables, preprocess=True
    )
    return GeneratedCNF(formula, seed, assignment)
