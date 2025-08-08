"""Command-line interface for parsing, solving, generating, and benchmarking CNF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmarking import run_benchmark, summarize, write_csv_report, write_json_report
from .dimacs import parse_dimacs, read_dimacs, to_dimacs
from .formula import CNFError
from .generator import generate_random_cnf
from .solver import SolveStatus, solve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satrl",
        description="Parse, generate, and exactly solve DIMACS CNF formulas.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="solve a DIMACS CNF file")
    solve_parser.add_argument("input", help="path to a DIMACS file, or '-' for stdin")
    solve_parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="return UNKNOWN after this many search nodes",
    )
    solve_parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="disable tautology, duplicate, and subsumption preprocessing",
    )
    solve_parser.add_argument("--json", action="store_true", help="emit JSON")
    solve_parser.set_defaults(handler=_handle_solve)

    generate_parser = subparsers.add_parser(
