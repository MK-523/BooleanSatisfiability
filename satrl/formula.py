"""CNF data structures, normalization, and assignment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


Literal = int
Clause = tuple[Literal, ...]
Assignment = dict[int, bool]


class CNFError(ValueError):
    """Raised when a formula violates the CNF data contract."""


def _literal_sort_key(literal: Literal) -> tuple[int, bool]:
    return abs(literal), literal < 0


def normalize_clause(literals: Iterable[int]) -> Clause | None:
    """Canonicalize a clause, returning ``None`` for a tautology.

    Duplicate literals are removed. A clause containing both ``x`` and ``-x``
    is always true and can be removed from a conjunction without changing its
    satisfiability.
    """

    unique: set[int] = set()
    for raw_literal in literals:
        literal = int(raw_literal)
        if literal == 0:
            raise CNFError("literal 0 is reserved as the DIMACS clause terminator")
        if -literal in unique:
            return None
        unique.add(literal)
    return tuple(sorted(unique, key=_literal_sort_key))


def preprocess_clauses(
    clauses: Iterable[Iterable[int]], *, remove_subsumed: bool = True
) -> tuple[Clause, ...]:
    """Return an equivalent, deterministic tuple of clauses.

    The transformation removes tautologies, repeated literals, repeated
    clauses, and optionally clauses subsumed by a shorter clause. Empty clauses
    are retained because they make the formula unsatisfiable.
    """

    normalized: set[Clause] = set()
    for clause in clauses:
        canonical = normalize_clause(clause)
        if canonical is not None:
            normalized.add(canonical)

    ordered = sorted(normalized, key=lambda clause: (len(clause), clause))
    if not remove_subsumed:
        return tuple(ordered)

    kept: list[Clause] = []
    kept_sets: list[frozenset[int]] = []
    for clause in ordered:
        literals = frozenset(clause)
        if any(existing.issubset(literals) for existing in kept_sets):
            continue
        kept.append(clause)
        kept_sets.append(literals)
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class CNFFormula:
    """An immutable CNF formula using DIMACS-style signed integers."""

    num_variables: int
    clauses: tuple[Clause, ...]

    def __post_init__(self) -> None:
        if self.num_variables < 0:
