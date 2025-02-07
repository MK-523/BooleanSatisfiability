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


