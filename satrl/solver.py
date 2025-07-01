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
                None,
                self.stats,
                verified=False,
                reason=f"search node limit {self.max_nodes} reached",
            )

        self.stats.elapsed_ms = (perf_counter() - started) * 1000
        if assignment is None:
            return SolveResult(
                SolveStatus.UNSAT,
                None,
                self.stats,
                verified=True,
            )

        completed = {
            variable: assignment.get(variable, False) for variable in formula.variables
        }
        verified = is_satisfied(formula, completed)
        if not verified:
            raise AssertionError("internal error: DPLL returned an invalid assignment")
        return SolveResult(SolveStatus.SAT, completed, self.stats, verified=True)

    def _search(
        self,
        clauses: tuple[tuple[int, ...], ...],
        assignment: Assignment,
        *,
        depth: int,
    ) -> Assignment | None:
        if self.max_nodes is not None and self.stats.nodes >= self.max_nodes:
            raise _SearchLimitReached
        self.stats.nodes += 1
        self.stats.max_depth = max(self.stats.max_depth, depth)

        reduced, propagated_assignment, conflict = self._propagate(
            clauses, assignment
        )
        if conflict:
            self.stats.conflicts += 1
            return None
        if not reduced:
            return propagated_assignment

        variable, preferred_value = self._choose_branch(reduced)
        self.stats.decisions += 1
        for value in (preferred_value, not preferred_value):
            child_assignment = dict(propagated_assignment)
            child_assignment[variable] = value
            child_clauses = self._apply_literal(
                reduced, variable if value else -variable
            )
            result = self._search(child_clauses, child_assignment, depth=depth + 1)
            if result is not None:
                return result
            self.stats.backtracks += 1
        return None

    def _propagate(
        self,
        clauses: tuple[tuple[int, ...], ...],
        assignment: Assignment,
    ) -> tuple[tuple[tuple[int, ...], ...], Assignment, bool]:
        reduced = clauses
        current = dict(assignment)
        while True:
            if any(not clause for clause in reduced):
                return reduced, current, True
            if not reduced:
                return reduced, current, False

            unit_literals = sorted(
                (clause[0] for clause in reduced if len(clause) == 1),
                key=lambda literal: (abs(literal), literal < 0),
            )
            if unit_literals:
                literal = unit_literals[0]
                variable = abs(literal)
                value = literal > 0
                existing = current.get(variable)
                if existing is not None and existing != value:
                    return reduced, current, True
                current[variable] = value
                self.stats.propagations += 1
                reduced = self._apply_literal(reduced, literal)
                continue

            polarities: dict[int, set[bool]] = {}
            for clause in reduced:
                for literal in clause:
                    polarities.setdefault(abs(literal), set()).add(literal > 0)
            pure = sorted(
                (
                    (variable, next(iter(values)))
                    for variable, values in polarities.items()
                    if len(values) == 1 and variable not in current
                ),
                key=lambda item: item[0],
            )
            if pure:
                variable, value = pure[0]
                current[variable] = value
                self.stats.pure_literal_assignments += 1
                reduced = self._apply_literal(
                    reduced, variable if value else -variable
                )
                continue
            return reduced, current, False

    @staticmethod
    def _apply_literal(
        clauses: Iterable[tuple[int, ...]], literal: int
    ) -> tuple[tuple[int, ...], ...]:
        opposite = -literal
        reduced: list[tuple[int, ...]] = []
        for clause in clauses:
            if literal in clause:
                continue
            if opposite in clause:
                reduced.append(tuple(item for item in clause if item != opposite))
