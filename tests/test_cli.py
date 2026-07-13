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

    def test_benchmark_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "benchmark",
                        "--variables",
                        "6",
                        "--clauses",
                        "12",
                        "--instances",
                        "3",
                        "--seed",
                        "9",
                        "--output",
                        str(report),
                    ]
                )
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["summary"]["instances"], 3)
            self.assertTrue(payload["summary"]["all_oracle_checks_agree"])
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["records"][0]["formula_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
