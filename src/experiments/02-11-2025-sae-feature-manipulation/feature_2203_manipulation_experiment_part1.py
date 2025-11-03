#!/usr/bin/env python3
# SAE Feature 2203 Manipulation Experiment
# Part 1: Imports and SAE class

import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer
from arguments import ModelArguments
from training_args import TrainingArguments
from model_utils import create_model


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int = 2048, n_features: int = 8192, l1_coefficient: float = 0.001):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.l1_coefficient = l1_coefficient
        self.encoder = torch.nn.Linear(input_dim, n_features, bias=True)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=True)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.t()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.relu(self.encoder(x))
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
