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
