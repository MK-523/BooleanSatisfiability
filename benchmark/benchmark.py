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
    """
    if clause_size > num_variables:
        raise ValueError("clause_size cannot exceed num_variables")

    clauses: set[tuple[int, ...]] = set()
    while len(clauses) < num_clauses:
        variables = rng.choice(
            np.arange(1, num_variables + 1), size=clause_size, replace=False
        )
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=clause_size)
        literals = variables * signs
        canonical = tuple(int(v) for v in literals[np.argsort(np.abs(literals))])
        clauses.add(canonical)
    return np.asarray(sorted(clauses), dtype=np.int16)


def generate_split(config: Config, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    formulas: list[np.ndarray] = []
    seen: set[str] = set()
    total = config.train_formulas + config.test_formulas
    while len(formulas) < total:
        formula = generate_formula(
            rng, config.num_variables, config.num_clauses, config.clause_size
        )
        identity = formula_fingerprint(formula)
        if identity not in seen:
            formulas.append(formula)
            seen.add(identity)
    return formulas[: config.train_formulas], formulas[config.train_formulas :]


def satisfaction_ratios(formula: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Return the fraction of satisfied clauses for each Boolean assignment."""
    assignments = np.asarray(assignments, dtype=bool)
    if assignments.ndim == 1:
        assignments = assignments[None, :]
    variable_values = assignments[:, np.abs(formula) - 1]
    literal_values = np.where(formula[None, :, :] > 0, variable_values, ~variable_values)
    return literal_values.any(axis=2).mean(axis=1)


def exact_optimum(formula: np.ndarray, num_variables: int) -> tuple[float, bool]:
    """Enumerate every assignment; suitable here because n <= 12."""
    ids = np.arange(1 << num_variables, dtype=np.uint32)
    assignments = ((ids[:, None] >> np.arange(num_variables)) & 1).astype(bool)
    ratios = satisfaction_ratios(formula, assignments)
    best = float(ratios.max())
    return best, math.isclose(best, 1.0, abs_tol=1e-12)


class FormulaAgnosticPolicyGradient:
    """Behavior-level reconstruction of the upstream formula-agnostic policy.

    Upstream's MLP receives ``torch.ones(num_variables)`` for every formula, so
    it can only learn one global Bernoulli probability per variable.  Direct
    logits represent the same policy family without claiming bit-for-bit parity
    with PyTorch's MLP parameterization or initialization.
    """

    def __init__(self, num_variables: int, seed: int, learning_rate: float = 0.001):
        self.num_variables = num_variables
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.logits = np.zeros(num_variables, dtype=np.float64)

    @property
    def probabilities(self) -> np.ndarray:
        clipped = np.clip(self.logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def sample(self, count: int = 1) -> np.ndarray:
        return self.rng.random((count, self.num_variables)) < self.probabilities

    def train(self, formulas: Sequence[np.ndarray], update_budget: int) -> None:
        """Apply the upstream uncentered REINFORCE update.

        The reward is the satisfied-clause fraction.  As in the public code,
        there is no baseline, entropy term, minibatching, or formula input.
        """
        order = np.arange(len(formulas))
        position = len(formulas)
        for _ in range(update_budget):
            if position == len(formulas):
                self.rng.shuffle(order)
                position = 0
            formula = formulas[int(order[position])]
            position += 1
            assignment = self.sample(1)[0]
            reward = float(satisfaction_ratios(formula, assignment)[0])
            # For Bernoulli logits, grad(log pi(a)) = a - p.  Gradient descent
            # on -reward * log pi is therefore this ascent update.
            self.logits += self.learning_rate * reward * (
                assignment.astype(np.float64) - self.probabilities
            )


def evaluate_sampler(
    formulas: Sequence[np.ndarray],
    sampler,
    candidate_budget: int,
    exact_scores: Sequence[float],
) -> tuple[float, float, float, float, float]:
    start = time.perf_counter()
    scores = []
    solved = []
    optimal = []
    regrets = []
    for formula, exact_score in zip(formulas, exact_scores):
        assignments = sampler(candidate_budget)
        score = float(satisfaction_ratios(formula, assignments).max())
        scores.append(score)
        solved.append(math.isclose(score, 1.0, abs_tol=1e-12))
        optimal.append(math.isclose(score, exact_score, abs_tol=1e-12))
        regrets.append(exact_score - score)
    elapsed_ms = (time.perf_counter() - start) * 1_000.0
    return (
        float(np.mean(scores)),
        float(np.mean(solved)),
        float(np.mean(optimal)),
