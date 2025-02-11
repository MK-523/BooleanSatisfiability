#!/usr/bin/env python3
"""Reproducible benchmark for the public BooleanSatisfiability project.

The upstream model's network is never conditioned on the CNF formula.  Its
observable policy is therefore a vector of independent Bernoulli probabilities
shared by every formula.  This file reconstructs that policy class directly in
NumPy and trains it with the same uncentered REINFORCE objective used upstream.

This is intentionally *not* presented as a corrected neural SAT solver.  It is
an evaluation harness that makes the current method's limitations measurable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Config:
    num_variables: int
    num_clauses: int
    clause_size: int = 3
    train_formulas: int = 160
    test_formulas: int = 40

    @property
    def name(self) -> str:
        return f"n{self.num_variables}_m{self.num_clauses}_k{self.clause_size}"


@dataclass
class Result:
    config: str
    num_variables: int
    num_clauses: int
    clause_size: int
    train_formulas: int
    test_formulas: int
    data_seed: int
    run_seed: int | str
    method: str
    candidate_budget: int | str
    mean_satisfaction_ratio: float
    solved_rate: float
    optimal_rate: float
    mean_regret_to_exact: float
    runtime_ms_total: float
    runtime_ms_per_formula: float
    training_ms: float
    policy_prob_min: float | str = ""
    policy_prob_mean: float | str = ""
    policy_prob_max: float | str = ""


def formula_fingerprint(formula: np.ndarray) -> str:
    """Stable identity used to prove that the split has no overlap."""
    return hashlib.sha256(formula.astype(np.int16).tobytes()).hexdigest()


def generate_formula(
    rng: np.random.Generator,
    num_variables: int,
    num_clauses: int,
    clause_size: int,
) -> np.ndarray:
    """Generate a standard random k-CNF with distinct variables per clause.

    Variables are sampled without replacement inside a clause, so tautologies
    and repeated literals cannot occur.  Canonicalized clauses are deduplicated.
    This avoids relying on the upstream preprocessing bug.
