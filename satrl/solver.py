"""A deterministic exact DPLL SAT solver with observable search statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from time import perf_counter
from typing import Iterable

from .formula import Assignment, CNFFormula, is_satisfied


class SolveStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class SolverStats:
    nodes: int = 0
    decisions: int = 0
    propagations: int = 0
    pure_literal_assignments: int = 0
    conflicts: int = 0
    backtracks: int = 0
    max_depth: int = 0
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class SolveResult:
    status: SolveStatus
    assignment: Assignment | None
    stats: SolverStats
    verified: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
