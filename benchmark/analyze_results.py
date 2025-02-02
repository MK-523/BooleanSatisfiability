#!/usr/bin/env python3
"""Create paired policy-minus-random confidence intervals from raw run rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


KEY_FIELDS = ("config", "data_seed", "run_seed", "candidate_budget")
METRICS = ("mean_satisfaction_ratio", "solved_rate", "optimal_rate")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/benchmark_runs.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/paired_differences.csv")
    )
    args = parser.parse_args()

    methods: dict[str, dict[tuple[str, ...], dict[str, str]]] = defaultdict(dict)
    for row in read_rows(args.runs):
        method = row["method"]
        if method not in {"formula_agnostic_policy_gradient", "uniform_random"}:
            continue
        key = tuple(row[field] for field in KEY_FIELDS)
        methods[method][key] = row

