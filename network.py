import torch
from torch import nn


def build_model(input_dim, hidden_dim=64, dropout=0.2):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
