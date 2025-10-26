import math
import random
import unittest

from satrl.baselines import brute_force_solve
from satrl.formula import CNFFormula, is_satisfied
from satrl.generator import generate_random_cnf
from satrl.solver import SolveStatus, solve


def _brute_force_status(formula: CNFFormula) -> SolveStatus:
    return (
        SolveStatus.SAT
        if brute_force_solve(formula, variable_limit=10) is not None
        else SolveStatus.UNSAT
    )


class SolverPropertyTests(unittest.TestCase):
    def test_dpll_agrees_with_exhaustive_oracle_across_random_formulas(self):
        for seed in range(80):
            variables = 1 + seed % 7
            clause_size = 1 + seed % variables
            maximum = min(
                20,
                math.comb(variables, clause_size) * (2**clause_size),
            )
            clauses = seed % (maximum + 1)
            formula = generate_random_cnf(
                variables,
                clauses,
                clause_size,
                seed=seed,
            ).formula
            with self.subTest(seed=seed, formula=formula):
                result = solve(formula)
                self.assertIs(result.status, _brute_force_status(formula))
                if result.assignment is not None:
                    self.assertTrue(is_satisfied(formula, result.assignment))

