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
            "status": self.status.value,
            "assignment": (
                {str(variable): value for variable, value in sorted(self.assignment.items())}
                if self.assignment is not None
                else None
            ),
            "verified": self.verified,
            "reason": self.reason,
            "stats": asdict(self.stats),
        }


class _SearchLimitReached(RuntimeError):
    pass


class DPLLSolver:
    """Solve CNF formulas exactly with DPLL.

    Unit propagation, pure-literal elimination, and a deterministic
    Jeroslow-Wang-style branching score reduce the search space. With no node
    limit the result is exact. If ``max_nodes`` is reached, the solver returns
    ``UNKNOWN`` rather than making an unsupported SAT/UNSAT claim.
    """

    def __init__(self, *, max_nodes: int | None = None) -> None:
        if max_nodes is not None and max_nodes < 0:
            raise ValueError("max_nodes must be non-negative or None")
        self.max_nodes = max_nodes
        self.stats = SolverStats()

    def solve(self, formula: CNFFormula) -> SolveResult:
        self.stats = SolverStats()
        started = perf_counter()
        try:
            assignment = self._search(formula.clauses, {}, depth=0)
        except _SearchLimitReached:
            self.stats.elapsed_ms = (perf_counter() - started) * 1000
            return SolveResult(
                SolveStatus.UNKNOWN,
