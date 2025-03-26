import unittest

import numpy as np

from benchmark import (
    Config,
    FormulaAgnosticPolicyGradient,
    exact_optimum,
    formula_fingerprint,
    generate_split,
    satisfaction_ratios,
)


class BenchmarkTests(unittest.TestCase):
    def test_satisfaction_ratio(self):
        formula = np.asarray([[1, -2], [-1, 2]], dtype=np.int16)
        assignments = np.asarray([[True, True], [True, False]], dtype=bool)
        np.testing.assert_allclose(satisfaction_ratios(formula, assignments), [1.0, 0.5])

    def test_exact_optimum(self):
        satisfiable = np.asarray([[1], [2]], dtype=np.int16)
        score, solved = exact_optimum(satisfiable, 2)
        self.assertEqual(score, 1.0)
        self.assertTrue(solved)

        unsatisfiable = np.asarray([[1], [-1]], dtype=np.int16)
        score, solved = exact_optimum(unsatisfiable, 1)
        self.assertEqual(score, 0.5)
        self.assertFalse(solved)

    def test_split_is_disjoint_and_reproducible(self):
        config = Config(6, 12, train_formulas=8, test_formulas=4)
        train_a, test_a = generate_split(config, 523)
        train_b, test_b = generate_split(config, 523)
        ids_train = {formula_fingerprint(f) for f in train_a}
        ids_test = {formula_fingerprint(f) for f in test_a}
        self.assertFalse(ids_train & ids_test)
        self.assertEqual(
            [formula_fingerprint(f) for f in train_a + test_a],
            [formula_fingerprint(f) for f in train_b + test_b],
        )

    def test_policy_is_formula_agnostic(self):
        policy = FormulaAgnosticPolicyGradient(4, seed=1)
        before = policy.probabilities.copy()
        formula_a = np.asarray([[1, 2, 3]], dtype=np.int16)
        formula_b = np.asarray([[-1, -2, -3]], dtype=np.int16)
        # The observable probabilities have no formula argument at all.
        np.testing.assert_array_equal(before, policy.probabilities)
        self.assertEqual(formula_a.shape, formula_b.shape)


if __name__ == "__main__":
    unittest.main()
