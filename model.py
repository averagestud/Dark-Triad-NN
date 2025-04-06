import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ANFIS Model
class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, num_mfs=2):
        super(FuzzyLayer, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.centers = nn.Parameter(torch.rand(input_dim, num_mfs))
        self.sigmas = nn.Parameter(torch.rand(input_dim, num_mfs))
    
    def forward(self, x):
        x_expanded = x.unsqueeze(2)  # After this it becomes (batch, input_dim, 1)
        membership = torch.exp(- (x_expanded - self.centers)**2 / (2 * self.sigmas**2 + 1e-6))
        return membership  # Final output is: (batch, input_dim, num_mfs)

class RuleLayer(nn.Module):
    def __init__(self):
        super(RuleLayer, self).__init__()
    
    def forward(self, x):
        batch_size = x.shape[0]
        flat = x.view(batch_size, -1)
        return flat

class ConsequentLayer(nn.Module):
    def __init__(self, num_rules, output_dim=3): 
        super(ConsequentLayer, self).__init__()
        self.num_rules = num_rules
        self.output_dim = output_dim
        self.consequents = nn.Parameter(torch.rand(num_rules, output_dim))
    
    def forward(self, x):
        norm_firing = x / (torch.sum(x, dim=1, keepdim=True) + 1e-6)
        output = torch.matmul(norm_firing, self.consequents)
        return output

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_mfs=2, output_dim=3):
        super(ANFIS, self).__init__()
        self.fuzzy = FuzzyLayer(input_dim, num_mfs)
        self.rule = RuleLayer()
        self.num_rules = input_dim * num_mfs # num_rules = input_dim * num_mfs
        self.consequent = ConsequentLayer(self.num_rules, output_dim)
    
    def forward(self, x):
        fuzzy_out = self.fuzzy(x)       # (batch, input_dim, num_mfs)
        rule_out = self.rule(fuzzy_out)  
        output = self.consequent(rule_out) # (batch, output_dim)
        return output
