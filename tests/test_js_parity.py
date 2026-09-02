"""The browser port in docs/bb_core.js must agree with the Python.

Tolerances, with their measured sources:

- Best response and baselines: 1e-6 relative. The only difference is the
  JS normal CDF (an erfc approximation with ~1.2e-7 fractional error);
  measured discrepancies are at the 1e-8 level.
- Lagrangian objective: 1e-4 relative. The JS uses vanilla Frank-Wolfe
  with exact line search where the Python is fully corrective (SLSQP);
  both converge to the same concave optimum. Measured: 2e-5 relative.
- Budget-constrained objective: 3e-3 absolute. On top of the above, the
  two bisections stop at slightly different path-times within the 2e-3
  budget tolerance, worth about lambda * 2e-3 in objective.
  Measured: 1.2e-3 absolute.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from brownianbandit import (
    BrownianGrid,
    best_one_shot_screening_baseline,
    point_mass_initial_intensity,
    solve_for_budget,
    solve_lagrangian,
    solve_linearized_best_response,
    static_random_thinning_baseline,
)

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def test_js_port_matches_python() -> None:
    runner = Path(__file__).with_name("parity_runner.js")
    js = json.loads(
        subprocess.run(
            [NODE, str(runner)], capture_output=True, text=True, check=True
        ).stdout
    )

    n_steps = 40
    grid = BrownianGrid.build(
        x_min=-6.0, x_max=6.0, n_space=301, dt=1.0 / n_steps, sigma=1.0
    )
    initial = point_mass_initial_intensity(grid, total_intensity=100.0, x0=0.0)

    reward = np.maximum(grid.x - 1.5, 0.0)
    policy = solve_linearized_best_response(
        terminal_reward=reward,
        lambda_path_time=0.02,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
    )
    assert js["best_response_path_time"] == pytest.approx(
        policy.path_time, rel=1e-6
    )
    assert js["best_response_linear_value"] == pytest.approx(
        policy.linear_value, rel=1e-6
    )

    lag = solve_lagrangian(
        lambda_path_time=0.08,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
        tolerance=1e-7,
        max_iterations=200,
    )
    assert js["lagrangian_objective"] == pytest.approx(lag.objective, rel=1e-4)

    solution = solve_for_budget(
        target_path_time=10.0,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
        budget_tolerance=2e-3,
        frank_wolfe_tolerance=1e-6,
        max_frank_wolfe_iterations=100,
    )
    assert js["budget_objective"] == pytest.approx(solution.objective, abs=3e-3)
    assert js["budget_path_time"] == pytest.approx(10.0, abs=0.03)

    static = static_random_thinning_baseline(
        target_path_time=10.0,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
    )
    one_shot = best_one_shot_screening_baseline(
        target_path_time=10.0,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
    )
    assert js["static_objective"] == pytest.approx(static.objective, rel=1e-6)
    assert js["one_shot_objective"] == pytest.approx(one_shot.objective, rel=1e-6)
