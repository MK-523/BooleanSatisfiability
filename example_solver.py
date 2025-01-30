import torch
from SAT_solver import SATSolverRL
from solver import solve_sat_problem
from formula_utils import interpret_solution

if __name__ == "__main__":
    #model instance with random specifications
    num_variables = 3
    num_clauses = 2  # reduced for readability
    clause_size = 3
    model = SATSolverRL(num_variables, num_clauses, clause_size)
    
    #solve SAT problem using the model
    formula, solution, reward = solve_sat_problem(model, num_variables, num_clauses, clause_size)
    
    #print interpreted solution
    print("Solution:")
    interpreted_solution = interpret_solution(solution)
    for var, val in interpreted_solution.items():
        print(f"  {var} = {val}")
    print(f"\nsatisfiability score: {reward.item():.4f}")

    #verify solution clause by clause
    satisfied_clauses = 0
    for clause in formula:
        #check if any literal in clause satisfies the clause
        if any((interpreted_solution[f"x{abs(lit)}"] != (lit < 0)) for lit in clause):
            satisfied_clauses += 1
    
    #print verification results
    print(f"\nclauses satisfied: {satisfied_clauses} out of {num_clauses}")
    print(f"verification score: {satisfied_clauses / num_clauses:.4f}")
