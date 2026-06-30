"""Neural network approximating temperature field T(x, y, t)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatPINN(nn.Module):
    """MLP mapping (x, y, t) -> temperature T."""

    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_width: int = 64,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        act = activations[activation]
        layers: list[nn.Module] = [nn.Linear(3, hidden_width), act()]

        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), act()])

        layers.append(nn.Linear(hidden_width, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)
