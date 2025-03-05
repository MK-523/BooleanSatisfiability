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
        float(np.mean(regrets)),
        elapsed_ms,
    )


def run_benchmark(
    configs: Sequence[Config],
    data_seeds: Sequence[int],
    run_seeds: Sequence[int],
    budgets: Sequence[int],
    update_budget: int,
) -> tuple[list[Result], dict]:
    results: list[Result] = []
    split_manifest: dict[str, dict] = {}

    for base_data_seed in data_seeds:
      for config in configs:
        effective_data_seed = base_data_seed + config.num_variables
        train, test = generate_split(config, effective_data_seed)
        train_ids = {formula_fingerprint(f) for f in train}
        test_ids = {formula_fingerprint(f) for f in test}
        if train_ids & test_ids:
            raise AssertionError("train/test overlap detected")

        split_manifest[f"{config.name}_seed{effective_data_seed}"] = {
            **asdict(config),
            "data_seed": effective_data_seed,
            "train_fingerprints": sorted(train_ids),
            "test_fingerprints": sorted(test_ids),
        }

        exact_start = time.perf_counter()
        exact_scores = []
        for formula in test:
            score, _ = exact_optimum(formula, config.num_variables)
            exact_scores.append(score)
        exact_ms = (time.perf_counter() - exact_start) * 1_000.0
        exact_solved = float(np.mean(np.isclose(exact_scores, 1.0)))
        results.append(
            Result(
                config=config.name,
                num_variables=config.num_variables,
                num_clauses=config.num_clauses,
                clause_size=config.clause_size,
                train_formulas=config.train_formulas,
                test_formulas=config.test_formulas,
                data_seed=effective_data_seed,
                run_seed="deterministic",
                method="exact_enumeration",
                candidate_budget=1 << config.num_variables,
                mean_satisfaction_ratio=float(np.mean(exact_scores)),
                solved_rate=exact_solved,
                optimal_rate=1.0,
                mean_regret_to_exact=0.0,
                runtime_ms_total=exact_ms,
                runtime_ms_per_formula=exact_ms / len(test),
                training_ms=0.0,
            )
        )

        for run_seed in run_seeds:
            policy = FormulaAgnosticPolicyGradient(config.num_variables, run_seed)
            train_start = time.perf_counter()
            policy.train(train, update_budget)
            training_ms = (time.perf_counter() - train_start) * 1_000.0

            for budget in budgets:
                random_rng = np.random.default_rng(run_seed + 1_000_003 * budget)

                def random_sampler(count: int, *, rng=random_rng, n=config.num_variables):
                    return rng.random((count, n)) < 0.5

                metrics = evaluate_sampler(test, random_sampler, budget, exact_scores)
                results.append(
                    Result(
                        config=config.name,
                        num_variables=config.num_variables,
                        num_clauses=config.num_clauses,
                        clause_size=config.clause_size,
                        train_formulas=config.train_formulas,
                        test_formulas=config.test_formulas,
                        data_seed=effective_data_seed,
                        run_seed=run_seed,
                        method="uniform_random",
                        candidate_budget=budget,
                        mean_satisfaction_ratio=metrics[0],
                        solved_rate=metrics[1],
                        optimal_rate=metrics[2],
                        mean_regret_to_exact=metrics[3],
                        runtime_ms_total=metrics[4],
                        runtime_ms_per_formula=metrics[4] / len(test),
                        training_ms=0.0,
                    )
                )

                metrics = evaluate_sampler(test, policy.sample, budget, exact_scores)
                probabilities = policy.probabilities
                results.append(
                    Result(
                        config=config.name,
                        num_variables=config.num_variables,
                        num_clauses=config.num_clauses,
                        clause_size=config.clause_size,
                        train_formulas=config.train_formulas,
                        test_formulas=config.test_formulas,
                        data_seed=effective_data_seed,
                        run_seed=run_seed,
                        method="formula_agnostic_policy_gradient",
                        candidate_budget=budget,
                        mean_satisfaction_ratio=metrics[0],
                        solved_rate=metrics[1],
                        optimal_rate=metrics[2],
                        mean_regret_to_exact=metrics[3],
                        runtime_ms_total=metrics[4],
                        runtime_ms_per_formula=metrics[4] / len(test),
                        training_ms=training_ms,
                        policy_prob_min=float(probabilities.min()),
                        policy_prob_mean=float(probabilities.mean()),
                        policy_prob_max=float(probabilities.max()),
                    )
                )

    return results, split_manifest


def aggregate_results(results: Sequence[Result]) -> list[dict]:
    groups: dict[tuple, list[Result]] = {}
    for row in results:
        key = (row.config, row.method, row.candidate_budget)
        groups.setdefault(key, []).append(row)

    summary = []
    for (config, method, budget), rows in sorted(groups.items()):
        def mean_std(field: str) -> tuple[float, float]:
            values = np.asarray([float(getattr(row, field)) for row in rows], dtype=float)
            return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

        satisfaction_mean, satisfaction_std = mean_std("mean_satisfaction_ratio")
        solved_mean, solved_std = mean_std("solved_rate")
        optimal_mean, optimal_std = mean_std("optimal_rate")
        regret_mean, regret_std = mean_std("mean_regret_to_exact")
        runtime_mean, runtime_std = mean_std("runtime_ms_per_formula")
        training_mean, training_std = mean_std("training_ms")
        first = rows[0]
        summary.append(
            {
                "config": config,
                "num_variables": first.num_variables,
                "num_clauses": first.num_clauses,
                "clause_size": first.clause_size,
                "train_formulas": first.train_formulas,
                "test_formulas": first.test_formulas,
                "method": method,
                "candidate_budget": budget,
                "runs": len(rows),
                "mean_satisfaction_ratio": satisfaction_mean,
                "std_satisfaction_ratio": satisfaction_std,
                "mean_solved_rate": solved_mean,
                "std_solved_rate": solved_std,
                "mean_optimal_rate": optimal_mean,
