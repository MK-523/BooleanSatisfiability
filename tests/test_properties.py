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

    def test_preprocessing_preserves_satisfiability(self):
        for seed in range(60):
            rng = random.Random(seed)
            variables = 1 + seed % 6
            raw_clauses: list[list[int]] = []
            for _ in range(1 + seed % 14):
                clause: list[int] = []
                for _ in range(seed % 5):
                    variable = rng.randint(1, variables)
                    clause.append(variable if rng.getrandbits(1) else -variable)
                raw_clauses.append(clause)
            raw = CNFFormula.from_clauses(
                raw_clauses,
                num_variables=variables,
                preprocess=False,
            )
            processed = CNFFormula.from_clauses(
                raw_clauses,
                num_variables=variables,
                preprocess=True,
            )
            with self.subTest(seed=seed):
                self.assertIs(_brute_force_status(raw), _brute_force_status(processed))

    def test_planted_generator_always_preserves_its_witness(self):
        for seed in range(30):
            generated = generate_random_cnf(7, 18, 3, seed=seed, planted=True)
            with self.subTest(seed=seed):
                self.assertIsNotNone(generated.planted_assignment)
                self.assertTrue(
                    is_satisfied(
                        generated.formula,
                        generated.planted_assignment or {},
                    )
                )


if __name__ == "__main__":
    unittest.main()
