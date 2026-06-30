"""Training loop for the heat transfer PINN."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .model import HeatPINN
from .physics import initial_condition, pde_residual, sample_collocation


@dataclass
class TrainConfig:
  """Hyperparameters and physical constants."""

  alpha: float = 0.01  # thermal diffusivity (m^2/s in normalized units)
  domain: tuple[float, float, float, float, float, float] = (
      0.0,
      1.0,
      0.0,
      1.0,
      0.0,
      1.0,
  )
  n_interior: int = 4096
  n_boundary: int = 1024
  n_initial: int = 1024
  hidden_layers: int = 4
  hidden_width: int = 64
  activation: str = "tanh"
  epochs: int = 5000
  lr: float = 1e-3
  weight_pde: float = 1.0
  weight_bc: float = 10.0
  weight_ic: float = 10.0
  log_every: int = 500
  device: str = "cpu"
  ic_sigma: float = 0.1
  ic_center: tuple[float, float] = (0.5, 0.5)


@dataclass
class TrainHistory:
  pde_loss: list[float] = field(default_factory=list)
  bc_loss: list[float] = field(default_factory=list)
  ic_loss: list[float] = field(default_factory=list)
  total_loss: list[float] = field(default_factory=list)


def train(config: TrainConfig | None = None) -> tuple[HeatPINN, TrainHistory]:
  """Train the PINN and return the model plus loss history."""
  cfg = config or TrainConfig()
  device = torch.device(cfg.device)

  model = HeatPINN(
      hidden_layers=cfg.hidden_layers,
      hidden_width=cfg.hidden_width,
      activation=cfg.activation,
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
  mse = nn.MSELoss()
  history = TrainHistory()

  points = sample_collocation(
      n_interior=cfg.n_interior,
      n_boundary=cfg.n_boundary,
      n_initial=cfg.n_initial,
      domain=cfg.domain,
      device=device,
  )

  ic_target = initial_condition(
      points["x_ic"],
      points["y_ic"],
      x_center=cfg.ic_center[0],
      y_center=cfg.ic_center[1],
      sigma=cfg.ic_sigma,
  )

  for epoch in range(1, cfg.epochs + 1):
    optimizer.zero_grad()

    residual = pde_residual(
        model,
        points["x_int"],
        points["y_int"],
        points["t_int"],
        cfg.alpha,
    )
    loss_pde = mse(residual, torch.zeros_like(residual))

    t_bc_pred = model(points["x_bc"], points["y_bc"], points["t_bc"])
    loss_bc = mse(t_bc_pred, torch.zeros_like(t_bc_pred))

    t_ic_pred = model(points["x_ic"], points["y_ic"], points["t_ic"])
    loss_ic = mse(t_ic_pred, ic_target)

    loss = (
        cfg.weight_pde * loss_pde
        + cfg.weight_bc * loss_bc
        + cfg.weight_ic * loss_ic
    )
    loss.backward()
    optimizer.step()
    scheduler.step()

    history.pde_loss.append(loss_pde.item())
    history.bc_loss.append(loss_bc.item())
    history.ic_loss.append(loss_ic.item())
    history.total_loss.append(loss.item())

    if epoch % cfg.log_every == 0 or epoch == 1:
      print(
          f"Epoch {epoch:5d} | "
          f"total={loss.item():.4e} | "
          f"pde={loss_pde.item():.4e} | "
          f"bc={loss_bc.item():.4e} | "
          f"ic={loss_ic.item():.4e}"
      )

  return model, history
