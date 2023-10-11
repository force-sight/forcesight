#!/usr/bin/env python3

import torch
import torch.nn as nn
from utils.config_utils import *
import numpy as np

class GripForceMLP(nn.Module):
    def __init__(self, num_inputs, hidden_dim=128, num_outputs=1):
        super(GripForceMLP, self).__init__()
        self.mlp_force = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        force = self.mlp_force(x)
        return force