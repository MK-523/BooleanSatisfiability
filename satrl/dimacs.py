"""Strict DIMACS CNF parsing and serialization."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .formula import CNFError, CNFFormula


def parse_dimacs(
    text: str, *, preprocess: bool = True, remove_subsumed: bool = True
) -> CNFFormula:
    """Parse one DIMACS CNF document.

    Clauses may span lines, but every clause must end with ``0``. The declared
    variable and clause counts are validated before optional preprocessing.
    """

    num_variables: int | None = None
    declared_clauses: int | None = None
    clauses: list[tuple[int, ...]] = []
    current_clause: list[int] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            if num_variables is not None:
                raise CNFError(f"line {line_number}: duplicate problem header")
            parts = line.split()
            if len(parts) != 4 or parts[0] != "p" or parts[1].lower() != "cnf":
                raise CNFError(
                    f"line {line_number}: expected 'p cnf <variables> <clauses>'"
                )
            try:
                num_variables = int(parts[2])
                declared_clauses = int(parts[3])
            except ValueError as exc:
                raise CNFError(f"line {line_number}: header counts must be integers") from exc
            if num_variables < 0 or declared_clauses < 0:
                raise CNFError(f"line {line_number}: header counts must be non-negative")
            continue
        if line.startswith("%"):
            break
        if num_variables is None:
            raise CNFError(f"line {line_number}: clause encountered before problem header")

        for token in line.split():
            try:
                literal = int(token)
            except ValueError as exc:
                raise CNFError(
                    f"line {line_number}: invalid integer token {token!r}"
                ) from exc
            if literal == 0:
                clauses.append(tuple(current_clause))
                current_clause.clear()
                continue
            if abs(literal) > num_variables:
                raise CNFError(
                    f"line {line_number}: literal {literal} exceeds declared "
                    f"variable count {num_variables}"
                )
            current_clause.append(literal)

    if num_variables is None or declared_clauses is None:
        raise CNFError("missing DIMACS problem header")
    if current_clause:
        raise CNFError("final clause is not terminated by 0")
    if len(clauses) != declared_clauses:
        raise CNFError(
            f"header declares {declared_clauses} clauses but document contains "
            f"{len(clauses)}"
        )
    return CNFFormula.from_clauses(
        clauses,
        num_variables=num_variables,
        preprocess=preprocess,
