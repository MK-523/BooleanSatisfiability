import torch

def generate_random_formula(num_variables, num_clauses, clause_size):
    #generate random CNF formula with integers 1..num_variables
    clauses = torch.randint(1, num_variables + 1, (num_clauses, clause_size))
    #randomly flip signs (-1 or 1) to create negated literals
    negations = torch.randint(0, 2, (num_clauses, clause_size)) * 2 - 1
    return clauses * negations

def preprocess_data(clauses, num_variables):
    #remove duplicate clauses
    clauses = torch.unique(clauses, dim=0)
    #remove tautologies (clauses containing x and ~x)
    non_tautology_mask = torch.sum(clauses.unsqueeze(2) == -clauses.unsqueeze(1), dim=2) == 0
    clauses = clauses[non_tautology_mask]
    #clamp literals to valid variable range
    clauses = torch.clamp(clauses, -num_variables, num_variables)
    return clauses

def generate_dataset(num_formulas, num_variables, num_clauses, clause_size):
    dataset = []
    for _ in range(num_formulas):
        formula = generate_random_formula(num_variables, num_clauses, clause_size)
        preprocessed_formula = preprocess_data(formula, num_variables)
        dataset.append(preprocessed_formula)
    return dataset
