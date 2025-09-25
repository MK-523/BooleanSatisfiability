import torch
from SAT_solver import SATSolverRL, generate_random_formula, preprocess_data
from formula_utils import display_formula

def solve_sat_problem(model, num_variables, num_clauses, clause_size):
    #generate random SAT formula
    formula = generate_random_formula(num_variables, num_clauses, clause_size)
    
    #display formula
    display_formula(formula)
    
    #preprocess formula (remove duplicates, tautologies, clamp literals)
    preprocessed_formula = preprocess_data(formula, num_variables)
    
    #use trained or untrained model to sample variable assignments
    with torch.no_grad():
        sampled_variables, _ = model(preprocessed_formula)
        #compute SAT reward for sampled assignment
        sat_reward = model.evaluate_satisfiability(preprocessed_formula, sampled_variables)
    
    return formula, sampled_variables, sat_reward
