import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super().__init__()

        layers = []
        prev_size = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h

        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)