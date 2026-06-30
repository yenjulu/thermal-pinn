"""
Train a Physics-Informed Neural Network for 2D transient heat conduction.

Governing equation:
    dT/dt = alpha * (d2T/dx2 + d2T/dy2)

Problem setup:
    - Unit square domain [0,1] x [0,1]
    - Dirichlet BC: T = 0 on all boundaries
    - IC: Gaussian hot spot at center at t = 0

Usage:
    python main.py
    python main.py --epochs 10000 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.train import TrainConfig, train
from src.visualize import plot_loss_history, plot_temperature_snapshots


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Heat transfer PINN trainer")
  parser.add_argument("--epochs", type=int, default=5000)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--alpha", type=float, default=0.01, help="Thermal diffusivity")
  parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
  parser.add_argument("--output-dir", type=str, default="outputs")
  parser.add_argument("--no-plot", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

  print("=" * 60)
  print("Heat Transfer PINN - 2D Transient Conduction")
  print("=" * 60)
  print(f"Device: {device}")
  print(f"Thermal diffusivity (alpha): {args.alpha}")
  print()

  config = TrainConfig(
      alpha=args.alpha,
      epochs=args.epochs,
      lr=args.lr,
      device=device,
  )

  model, history = train(config)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  checkpoint = output_dir / "heat_pinn.pt"
  torch.save(
      {
          "model_state_dict": model.state_dict(),
          "config": config.__dict__,
      },
      checkpoint,
  )
  print(f"\nModel saved to {checkpoint}")

  if not args.no_plot:
    plot_temperature_snapshots(
        model,
        times=[0.0, 0.1, 0.25, 0.5, 1.0],
        output_path=output_dir / "temperature_snapshots.png",
    )
    plot_loss_history(history, output_path=output_dir / "loss_history.png")


if __name__ == "__main__":
  main()
