"""Reproducible small-instance DPLL benchmark support."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

from .baselines import brute_force_solve
from .dimacs import to_dimacs
from .formula import is_satisfied
from .generator import generate_random_cnf
from .solver import SolveStatus, solve


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    instance: int
    seed: int
    variables: int
    clauses: int
    clause_size: int
    formula_sha256: str
    status: str
    verified: bool
    oracle_agrees: bool | None
    nodes: int
    decisions: int
    conflicts: int
    backtracks: int
    elapsed_ms: float


def run_benchmark(
    *,
    variables: int,
    clauses: int,
    clause_size: int,
    instances: int,
    seed: int,
    oracle_variable_limit: int = 16,
) -> list[BenchmarkRecord]:
    if instances <= 0:
        raise ValueError("instances must be positive")

    records: list[BenchmarkRecord] = []
    for instance in range(instances):
        instance_seed = seed + instance
        generated = generate_random_cnf(
            variables,
            clauses,
            clause_size,
            seed=instance_seed,
        )
        formula_sha256 = hashlib.sha256(
            to_dimacs(generated.formula).encode("utf-8")
        ).hexdigest()
        result = solve(generated.formula)
        oracle_agrees: bool | None = None
        if variables <= oracle_variable_limit:
            oracle_assignment = brute_force_solve(
                generated.formula, variable_limit=oracle_variable_limit
            )
            oracle_sat = oracle_assignment is not None
            oracle_agrees = oracle_sat == (result.status is SolveStatus.SAT)
        verified = result.verified and (
            result.assignment is None
            or is_satisfied(generated.formula, result.assignment)
        )
        records.append(
            BenchmarkRecord(
                instance=instance,
                seed=instance_seed,
                variables=variables,
                clauses=clauses,
