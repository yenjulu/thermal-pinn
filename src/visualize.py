"""Visualization utilities for trained heat PINN solutions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import HeatPINN


def predict_grid(
    model: HeatPINN,
    t: float,
    resolution: int = 100,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Evaluate temperature on a 2D grid at fixed time t."""
  device = device or next(model.parameters()).device
  model.eval()

  x = np.linspace(0.0, 1.0, resolution)
  y = np.linspace(0.0, 1.0, resolution)
  xx, yy = np.meshgrid(x, y)

  x_t = torch.tensor(xx.ravel(), dtype=torch.float32, device=device).reshape(-1, 1)
  y_t = torch.tensor(yy.ravel(), dtype=torch.float32, device=device).reshape(-1, 1)
  t_t = torch.full_like(x_t, t)

  with torch.no_grad():
    temperature = model(x_t, y_t, t_t).cpu().numpy().reshape(resolution, resolution)

  return xx, yy, temperature


def plot_temperature_snapshots(
    model: HeatPINN,
    times: list[float] | None = None,
    output_path: str | Path | None = None,
    resolution: int = 100,
) -> None:
  """Plot temperature field at multiple time slices."""
  times = times or [0.0, 0.1, 0.3, 0.6, 1.0]
  n_cols = len(times)
  fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5), constrained_layout=True)

  if n_cols == 1:
    axes = [axes]

  for ax, t in zip(axes, times):
    xx, yy, temp = predict_grid(model, t, resolution=resolution)
    contour = ax.contourf(xx, yy, temp, levels=50, cmap="inferno")
    ax.set_title(f"t = {t:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(contour, ax=ax, label="T")

  fig.suptitle("2D Transient Heat Conduction (PINN)", fontsize=14)

  if output_path:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")
  else:
    plt.show()

  plt.close(fig)


def plot_loss_history(
    history,
    output_path: str | Path | None = None,
) -> None:
  """Plot training loss curves."""
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.semilogy(history.total_loss, label="total", linewidth=2)
  ax.semilogy(history.pde_loss, label="PDE", alpha=0.8)
  ax.semilogy(history.bc_loss, label="BC", alpha=0.8)
  ax.semilogy(history.ic_loss, label="IC", alpha=0.8)
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Loss")
  ax.set_title("PINN Training Loss")
  ax.legend()
  ax.grid(True, alpha=0.3)

  if output_path:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved loss plot to {output_path}")
  else:
    plt.show()

  plt.close(fig)
