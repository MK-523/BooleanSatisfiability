import torch
from model import SATSolverRL
from data_utils import generate_dataset, generate_random_formula, preprocess_data
from train import train_rl

#parameters for model and dataset
num_variables = 10
num_clauses = 50
clause_size = 3
num_formulas = 100

#initialize SATSolverRL model
model = SATSolverRL(num_variables, num_clauses, clause_size)

#generate dataset
dataset = generate_dataset(num_formulas, num_variables, num_clauses, clause_size)

#train model using reinforcement learning
trained_model = train_rl(model, dataset)

#evaluate trained model on new test formula
with torch.no_grad():
    test_formula = generate_random_formula(num_variables, num_clauses, clause_size)
    test_formula = preprocess_data(test_formula, num_variables)
    sampled_variables, _ = trained_model(test_formula)
    sat_reward = trained_model.evaluate_satisfiability(test_formula, sampled_variables)
    print(f"Final SAT Reward on test formula: {sat_reward.item():.4f}")
