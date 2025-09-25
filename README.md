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

