def display_formula(formula):
    #print SAT formula in human-readable format
    print("SAT Formula:")
    for i, clause in enumerate(formula):
        #convert clause to string with literals x1, x2,... using ∨ operator
        clause_str = " ∨ ".join([f"{'-' if lit < 0 else ''}x{abs(lit)}" for lit in clause])
        print(f"  Clause {i+1}: ({clause_str})")
    print()

def interpret_solution(solution):
    #convert sampled variable tensor to dictionary of boolean assignments
    return {f"x{i+1}": bool(val) for i, val in enumerate(solution)}
