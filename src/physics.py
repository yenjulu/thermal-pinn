"""PDE residual and boundary/initial conditions for heat conduction."""

from __future__ import annotations

import torch

from .model import HeatPINN


def pde_residual(
    model: HeatPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Transient 2D heat equation residual:
        dT/dt - alpha * (d2T/dx2 + d2T/dy2) = 0
    """
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)

    temperature = model(x, y, t)
    ones = torch.ones_like(temperature)

    dT_dx = torch.autograd.grad(temperature, x, ones, create_graph=True)[0]
    dT_dy = torch.autograd.grad(temperature, y, ones, create_graph=True)[0]
    dT_dt = torch.autograd.grad(temperature, t, ones, create_graph=True)[0]

    d2T_dx2 = torch.autograd.grad(dT_dx, x, torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y, torch.ones_like(dT_dy), create_graph=True)[0]

    return dT_dt - alpha * (d2T_dx2 + d2T_dy2)


def initial_condition(
    x: torch.Tensor,
    y: torch.Tensor,
    x_center: float = 0.5,
    y_center: float = 0.5,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Gaussian hot spot at t = 0."""
    return torch.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma**2))


def sample_collocation(
    n_interior: int,
    n_boundary: int,
    n_initial: int,
    domain: tuple[float, float, float, float, float, float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Sample training points for interior PDE, boundary BC, and initial IC.

    Domain: (x_min, x_max, y_min, y_max, t_min, t_max)
    """
    x_min, x_max, y_min, y_max, t_min, t_max = domain

    # Interior collocation points
    x_int = torch.rand(n_interior, 1, device=device) * (x_max - x_min) + x_min
    y_int = torch.rand(n_interior, 1, device=device) * (y_max - y_min) + y_min
    t_int = torch.rand(n_interior, 1, device=device) * (t_max - t_min) + t_min

    # Boundary: four sides at random times
    n_per_side = n_boundary // 4
    t_bc = torch.rand(n_per_side * 4, 1, device=device) * (t_max - t_min) + t_min

    x_left = torch.full((n_per_side, 1), x_min, device=device)
    y_left = torch.rand(n_per_side, 1, device=device) * (y_max - y_min) + y_min
    t_left = t_bc[:n_per_side]

    x_right = torch.full((n_per_side, 1), x_max, device=device)
    y_right = torch.rand(n_per_side, 1, device=device) * (y_max - y_min) + y_min
    t_right = t_bc[n_per_side : 2 * n_per_side]

    y_bottom = torch.full((n_per_side, 1), y_min, device=device)
    x_bottom = torch.rand(n_per_side, 1, device=device) * (x_max - x_min) + x_min
    t_bottom = t_bc[2 * n_per_side : 3 * n_per_side]

    y_top = torch.full((n_per_side, 1), y_max, device=device)
    x_top = torch.rand(n_per_side, 1, device=device) * (x_max - x_min) + x_min
    t_top = t_bc[3 * n_per_side :]

    x_bc = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
    y_bc = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
    t_bc_all = torch.cat([t_left, t_right, t_bottom, t_top], dim=0)

    # Initial condition at t = 0
    x_ic = torch.rand(n_initial, 1, device=device) * (x_max - x_min) + x_min
    y_ic = torch.rand(n_initial, 1, device=device) * (y_max - y_min) + y_min
    t_ic = torch.full((n_initial, 1), t_min, device=device)

    return {
        "x_int": x_int,
        "y_int": y_int,
        "t_int": t_int,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "t_bc": t_bc_all,
        "x_ic": x_ic,
        "y_ic": y_ic,
        "t_ic": t_ic,
    }
