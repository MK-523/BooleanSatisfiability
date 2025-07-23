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
