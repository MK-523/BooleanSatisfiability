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


