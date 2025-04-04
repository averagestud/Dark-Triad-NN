import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Define the ANFIS model architecture in PyTorch (Multi-output version)
# -------------------------------
class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, num_mfs=2):
        super(FuzzyLayer, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        # Trainable parameters for each input: centers and sigmas.
        self.centers = nn.Parameter(torch.rand(input_dim, num_mfs))
        self.sigmas = nn.Parameter(torch.rand(input_dim, num_mfs))
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x_expanded = x.unsqueeze(2)  # (batch, input_dim, 1)
        membership = torch.exp(- (x_expanded - self.centers)**2 / (2 * self.sigmas**2 + 1e-6))
        return membership  # (batch, input_dim, num_mfs)

class RuleLayer(nn.Module):
    def __init__(self):
        super(RuleLayer, self).__init__()
    
    def forward(self, x):
        batch_size = x.shape[0]
        flat = x.view(batch_size, -1)
        return flat

class ConsequentLayer(nn.Module):
    def __init__(self, num_rules, output_dim=3):  # output_dim=3 for three traits
        super(ConsequentLayer, self).__init__()
        self.num_rules = num_rules
        self.output_dim = output_dim
        # Consequent parameters: one weight per rule for each output.
        self.consequents = nn.Parameter(torch.rand(num_rules, output_dim))
    
    def forward(self, x):
        # x shape: (batch_size, num_rules)
        norm_firing = x / (torch.sum(x, dim=1, keepdim=True) + 1e-6)
        output = torch.matmul(norm_firing, self.consequents)
        return output

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_mfs=2, output_dim=3):
        super(ANFIS, self).__init__()
        self.fuzzy = FuzzyLayer(input_dim, num_mfs)
        self.rule = RuleLayer()
        # Total number of rules = input_dim * num_mfs (simplified approach)
        self.num_rules = input_dim * num_mfs
        self.consequent = ConsequentLayer(self.num_rules, output_dim)
    
    def forward(self, x):
        fuzzy_out = self.fuzzy(x)       # (batch, input_dim, num_mfs)
        rule_out = self.rule(fuzzy_out)   # (batch, input_dim * num_mfs)
        output = self.consequent(rule_out) # (batch, output_dim)
        return output