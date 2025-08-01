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
