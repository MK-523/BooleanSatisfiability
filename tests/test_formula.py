import unittest

from satrl.formula import (
    CNFError,
    CNFFormula,
    clause_value,
    formula_value,
    is_satisfied,
    preprocess_clauses,
)


class FormulaTests(unittest.TestCase):
    def test_preprocessing_preserves_clause_rows(self):
        clauses = [[1, 2, 3], [-1, 2, 3], [1, -2, 3]]
        processed = preprocess_clauses(clauses, remove_subsumed=False)
        self.assertEqual(len(processed), 3)
        self.assertTrue(all(isinstance(clause, tuple) for clause in processed))
        self.assertTrue(all(len(clause) == 3 for clause in processed))

    def test_preprocessing_removes_tautologies_duplicates_and_subsumed_clauses(self):
        formula = CNFFormula.from_clauses(
            [[1, -1, 2], [1, 1, 2], [2, 1], [1], [1, 2, 3]],
            num_variables=3,
        )
        self.assertEqual(formula.clauses, ((1,),))

    def test_empty_clause_is_retained(self):
        formula = CNFFormula.from_clauses([[], [1]], num_variables=1)
        self.assertEqual(formula.clauses, ((),))
        self.assertTrue(formula.is_trivially_unsatisfiable)

    def test_literal_range_is_validated(self):
        with self.assertRaises(CNFError):
            CNFFormula.from_clauses([[3]], num_variables=2, preprocess=False)

    def test_partial_assignment_evaluation(self):
        formula = CNFFormula.from_clauses([[1, -2], [2]], num_variables=2)
        self.assertIsNone(clause_value((1, -2), {}))
        self.assertIsNone(formula_value(formula, {1: True}))
        self.assertFalse(is_satisfied(formula, {1: True}))
        self.assertTrue(is_satisfied(formula, {1: True, 2: True}))


if __name__ == "__main__":
    unittest.main()
