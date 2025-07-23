"""Exact, reproducible tools for Boolean satisfiability experiments."""

from .dimacs import parse_dimacs, read_dimacs, to_dimacs, write_dimacs
from .formula import CNFError, CNFFormula, preprocess_clauses
from .solver import DPLLSolver, SolveResult, SolveStatus, SolverStats, solve

__all__ = [
    "CNFError",
    "CNFFormula",
    "DPLLSolver",
    "SolveResult",
    "SolveStatus",
    "SolverStats",
    "parse_dimacs",
    "preprocess_clauses",
    "read_dimacs",
    "solve",
    "to_dimacs",
    "write_dimacs",
]

__version__ = "0.1.0"
