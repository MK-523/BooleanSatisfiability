import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from satrl.cli import main
from satrl.dimacs import read_dimacs


class CliTests(unittest.TestCase):
    def test_solve_json(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["solve", "examples/satisfiable.cnf", "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "SAT")
        self.assertTrue(payload["verified"])

    def test_generate_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.cnf"
            second = Path(directory) / "second.cnf"
            arguments = [
                "generate",
                "--variables",
                "6",
                "--clauses",
                "12",
                "--clause-size",
                "3",
                "--seed",
                "523",
            ]
            self.assertEqual(main([*arguments, "--output", str(first)]), 0)
            self.assertEqual(main([*arguments, "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(len(read_dimacs(first).clauses), 12)
