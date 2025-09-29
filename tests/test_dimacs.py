import tempfile
import unittest
from pathlib import Path

from satrl.dimacs import parse_dimacs, read_dimacs, to_dimacs, write_dimacs
from satrl.formula import CNFError, CNFFormula


class DimacsTests(unittest.TestCase):
    def test_parse_comments_and_multiline_clause(self):
        formula = parse_dimacs(
            """c example
p cnf 3 2
1 -2
3 0
-1 0
"""
        )
        self.assertEqual(formula.num_variables, 3)
        self.assertEqual(formula.clauses, ((-1,), (1, -2, 3)))

    def test_round_trip(self):
        original = CNFFormula.from_clauses([[1, -2], [2, 3]], num_variables=3)
        parsed = parse_dimacs(to_dimacs(original))
        self.assertEqual(parsed, original)

    def test_read_and_write_file(self):
        formula = CNFFormula.from_clauses([[1], [-1, 2]], num_variables=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.cnf"
            write_dimacs(formula, path, comment="test")
            self.assertEqual(read_dimacs(path), formula)

    def test_header_clause_count_is_enforced(self):
        with self.assertRaisesRegex(CNFError, "declares 2 clauses"):
            parse_dimacs("p cnf 2 2\n1 0\n")

    def test_clause_must_end_with_zero(self):
        with self.assertRaisesRegex(CNFError, "not terminated"):
            parse_dimacs("p cnf 2 1\n1 -2\n")
