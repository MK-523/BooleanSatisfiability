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
            raise CNFError("num_variables must be non-negative")
        for clause_index, clause in enumerate(self.clauses, start=1):
            for literal in clause:
                if literal == 0:
                    raise CNFError(
                        f"clause {clause_index} contains literal 0; use an empty "
                        "tuple for an empty clause"
                    )
                if abs(literal) > self.num_variables:
                    raise CNFError(
                        f"literal {literal} in clause {clause_index} exceeds "
                        f"declared variable count {self.num_variables}"
                    )

    @classmethod
    def from_clauses(
        cls,
        clauses: Iterable[Iterable[int]],
        *,
        num_variables: int | None = None,
        preprocess: bool = True,
        remove_subsumed: bool = True,
    ) -> "CNFFormula":
        materialized = tuple(tuple(int(literal) for literal in clause) for clause in clauses)
        inferred = max(
            (abs(literal) for clause in materialized for literal in clause),
            default=0,
        )
        variable_count = inferred if num_variables is None else int(num_variables)
        processed = (
            preprocess_clauses(materialized, remove_subsumed=remove_subsumed)
            if preprocess
            else materialized
        )
        return cls(variable_count, processed)

    @property
    def variables(self) -> range:
        return range(1, self.num_variables + 1)

    @property
    def is_trivially_unsatisfiable(self) -> bool:
        return any(not clause for clause in self.clauses)

    @property
    def is_empty(self) -> bool:
        return not self.clauses


def clause_value(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool | None:
    """Evaluate a clause under a partial assignment.

    ``True`` means the clause is satisfied, ``False`` means every literal is
    assigned false, and ``None`` means its value is still undecided.
    """

    undecided = False
    for literal in clause:
        variable = abs(literal)
        if variable not in assignment:
            undecided = True
            continue
        value = assignment[variable]
        if value == (literal > 0):
            return True
    return None if undecided else False


def formula_value(formula: CNFFormula, assignment: Mapping[int, bool]) -> bool | None:
    """Evaluate a formula under a partial assignment."""

    undecided = False
    for clause in formula.clauses:
        value = clause_value(clause, assignment)
        if value is False:
            return False
        if value is None:
            undecided = True
    return None if undecided else True


def is_satisfied(formula: CNFFormula, assignment: Mapping[int, bool]) -> bool:
    return formula_value(formula, assignment) is True


def unsatisfied_clause_indices(
    formula: CNFFormula, assignment: Mapping[int, bool]
) -> tuple[int, ...]:
    return tuple(
        index
        for index, clause in enumerate(formula.clauses)
        if clause_value(clause, assignment) is not True
    )
