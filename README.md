# BooleanSatisfiability
solving the Boolean Satisfiability Problem with MIT Deep Learning Lab pHD candidate Nikolaos Karalias

This attempt to solve the boolean satisfiability problem uses the reinforcement learning (RL). It defines a neural network policy 
that learns to assign truth values to variables in a CNF formula in order 
to maximize the number of satisfied clauses. The system includes data 
generation, preprocessing, training, and evaluation components.

1. SATSolverRL model (model.py)
   - Policy network: maps variables to assignment probabilities
   - forward pass: samples assignments and computes log probabilities
   - evaluate_satisfiability: computes fraction of clauses satisfied
   - loss: merges policy gradient with SAT reward for training

2. Data utilities (data_utils.py)
   - generate_random_formula: creates random CNF formulas with negations
   - preprocess_data: removes duplicates, and clamps literals
   - generate_dataset: produces dataset of preprocessed formulas

3. Training loop (train.py)
   - train_rl: performs RL training using policy gradient
   - updates model parameters with Adam optimizer
   - records average loss and SAT reward periodically

4. Example run (example_run.py)
   - initiates model and generates dataset
   - trains RL agent on dataset
   - evaluates trained model on test formula
   - prints final SAT reward corresponding to test formula
  
5. Formula display & solution utilities (formula_utils.py)
   - display_formula: prints CNF formula in human-readable format with ∨ operator
   - interpret_solution: converts tensor of sampled variables into dictionary of boolean assignments

6. SAT problem solver wrapper (solver.py)
   - solve_sat_problem: generates random formula, preprocesses it, samples assignments using SATSolverRL
   - computes SAT reward for the sampled assignment
   - returns original formula, sampled variables, and reward
   - modular function separating solving logic from main execution

7. Example SAT problem run (example_solver_run.py)
   - sets model parameters for number of variables, clauses, and clause size
   - instantiates SATSolverRL model
   - solves SAT problem using solve_sat_problem
   - prints interpreted solution and satisfiability score
   - verifies solution clause by clause and prints verification score

