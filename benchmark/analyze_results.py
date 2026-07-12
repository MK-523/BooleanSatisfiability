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

    policy = methods["formula_agnostic_policy_gradient"]
    random = methods["uniform_random"]
    if policy.keys() != random.keys():
        raise ValueError("policy and random run keys do not match")

    differences: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key in sorted(policy):
        config, _, _, budget = key
        for metric in METRICS:
            differences[(config, budget, metric)].append(
                float(policy[key][metric]) - float(random[key][metric])
            )

    output_rows = []
    for (config, budget, metric), values in sorted(differences.items()):
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        std = float(array.std(ddof=1))
        critical = float(t.ppf(0.975, len(array) - 1))
        half_width = critical * std / math.sqrt(len(array))
        output_rows.append(
            {
                "config": config,
                "candidate_budget": budget,
                "metric": metric,
                "paired_runs": len(array),
                "mean_policy_minus_random": mean,
                "sample_std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
