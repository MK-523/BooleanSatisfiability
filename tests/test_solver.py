import unittest

from satrl.dimacs import read_dimacs
from satrl.formula import CNFFormula, is_satisfied
from satrl.solver import DPLLSolver, SolveStatus, solve


class SolverTests(unittest.TestCase):
    def test_solves_sample_sat_formula(self):
        formula = read_dimacs("examples/satisfiable.cnf")
        result = solve(formula)
        self.assertIs(result.status, SolveStatus.SAT)
        self.assertIsNotNone(result.assignment)
        self.assertTrue(result.verified)
        self.assertTrue(is_satisfied(formula, result.assignment or {}))

    def test_proves_sample_unsat_formula(self):
        formula = read_dimacs("examples/unsatisfiable.cnf")
        result = solve(formula)
        self.assertIs(result.status, SolveStatus.UNSAT)
        self.assertIsNone(result.assignment)
        self.assertTrue(result.verified)

    def test_empty_formula_is_sat(self):
        result = solve(CNFFormula(3, ()))
        self.assertIs(result.status, SolveStatus.SAT)
        self.assertEqual(result.assignment, {1: False, 2: False, 3: False})

    def test_empty_clause_is_unsat(self):
        result = solve(CNFFormula(0, ((),)))
        self.assertIs(result.status, SolveStatus.UNSAT)

    def test_unit_propagation(self):
        formula = CNFFormula.from_clauses([[1], [-1, 2], [-2, 3]], num_variables=3)
        result = solve(formula)
        self.assertEqual(result.assignment, {1: True, 2: True, 3: True})
        self.assertGreaterEqual(result.stats.propagations, 3)

    def test_node_limit_returns_unknown(self):
        formula = CNFFormula.from_clauses([[1, 2], [-1, -2]], num_variables=2)
        result = DPLLSolver(max_nodes=0).solve(formula)
        self.assertIs(result.status, SolveStatus.UNKNOWN)
        self.assertFalse(result.verified)
        self.assertIn("node limit", result.reason or "")

    def test_solver_instance_can_be_reused(self):
        solver = DPLLSolver()
        sat = solver.solve(CNFFormula.from_clauses([[1]], num_variables=1))
        unsat = solver.solve(CNFFormula.from_clauses([[1], [-1]], num_variables=1))
        self.assertIs(sat.status, SolveStatus.SAT)
        self.assertIs(unsat.status, SolveStatus.UNSAT)


if __name__ == "__main__":
    unittest.main()
