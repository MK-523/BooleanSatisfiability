import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class SATSolverRL(nn.Module):
    def __init__(self, num_variables, num_clauses, clause_size):
        super(SATSolverRL, self).__init__()
        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.clause_size = clause_size

        #define policy network: input=num_variables, output=num_variables probabilities
        self.policy = nn.Sequential(
            nn.Linear(num_variables, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_variables),
            nn.Sigmoid() #output probabilities between 0 and 1
        )

    def forward(self, clauses):
        #compute probabilities from policy network
        probabilities = self.policy(torch.ones(self.num_variables))
        distribution = Bernoulli(probabilities)
        sampled_variables = distribution.sample().bool() #sample binary assignment
        #return sampled assignment and log probabilities for policy gradient
        return sampled_variables, distribution.log_prob(sampled_variables.float())

    def evaluate_satisfiability(self, clauses, sampled_variables):
        #ensure clauses have batch dimension
        if clauses.dim() == 1:
            clauses = clauses.unsqueeze(0)

        #expand sampled_variables to match clause batch size
        sampled_variables = sampled_variables.view(1, -1).expand(clauses.size(0), -1)

        #mask positive and negative literals
        positive_mask = clauses > 0
        negative_mask = clauses < 0
