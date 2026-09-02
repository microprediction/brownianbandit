"""Reproduce the numerical example behind the README and the site.

Writes the summary JSON and diagnostic CSVs to examples/output/, and the
figures to docs/assets/images/ where index.html displays them.

Requires the demo extras: pip install -e ".[demo]"
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brownianbandit import (
    BrownianGrid,
    best_one_shot_screening_baseline,
    point_mass_initial_intensity,
    simulate_poisson_terminal_maximum,
    solve_for_budget,
    static_random_thinning_baseline,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets" / "images"


def main() -> None:
    horizon = 1.0
    n_steps = 100
    total_intensity = 100.0
    optional_path_time_budget = 10.0
    fallback = 0.0

    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    grid = BrownianGrid.build(
        x_min=-6.0,
        x_max=6.0,
        n_space=601,
        dt=horizon / n_steps,
        sigma=1.0,
    )
    initial = point_mass_initial_intensity(
        grid,
        total_intensity=total_intensity,
        x0=0.0,
    )

    solution = solve_for_budget(
        target_path_time=optional_path_time_budget,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=fallback,
        budget_tolerance=8e-4,
        frank_wolfe_tolerance=1e-6,
        max_frank_wolfe_iterations=100,
    )
    static = static_random_thinning_baseline(
        target_path_time=optional_path_time_budget,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=fallback,
    )
    one_shot = best_one_shot_screening_baseline(
        target_path_time=optional_path_time_budget,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=fallback,
    )
    monte_carlo = simulate_poisson_terminal_maximum(
        terminal_mass=solution.terminal_mass,
        x=grid.x,
        fallback=fallback,
        n_trials=100_000,
        seed=20260901,
    )

    active = [
        {
            "weight": float(weight),
            "path_time": float(policy.path_time),
            "terminal_survivor_intensity": float(policy.terminal_mass.sum()),
        }
        for weight, policy in zip(solution.weights, solution.policies)
        if weight > 1e-8
    ]

    summary = {
        "model": "Poissonized IID Brownian cloud with deterministic fallback",
        "horizon": horizon,
        "initial_optional_intensity": total_intensity,
        "optional_path_time_budget": optional_path_time_budget,
        "fallback": fallback,
        "n_time_steps": n_steps,
        "n_space_points": grid.n_space,
        "optimal_expected_maximum": solution.objective,
        "optimal_expected_path_time": solution.path_time,
        "shadow_price_lambda": solution.lambda_path_time,
        "frank_wolfe_dual_gap": solution.dual_gap,
        "expected_terminal_survivor_intensity": float(solution.terminal_mass.sum()),
        "static_thinning_expected_maximum": static.objective,
        "one_shot_expected_maximum": one_shot.objective,
        "one_shot_screening_time": one_shot.screening_time,
        "one_shot_retained_intensity": one_shot.retained_intensity,
        "gain_over_static_fraction": solution.objective / static.objective - 1.0,
        "gain_over_one_shot_fraction": solution.objective / one_shot.objective - 1.0,
        "monte_carlo_expected_maximum": monte_carlo.mean,
        "monte_carlo_standard_error": monte_carlo.standard_error,
        "budget_solution_uses_bracket_mixture": solution.budget_mixture,
        "active_policy_atoms": active,
    }
    (OUTPUT_DIR / "mean_field_demo_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    times = np.arange(n_steps + 1) * grid.dt
    survivor_frame = pd.DataFrame(
        {
            "time": times,
            "mass_before_decision": solution.predecision_count_curve,
            "mass_kept": np.r_[solution.alive_count_curve, np.nan],
        }
    )
    survivor_frame.to_csv(OUTPUT_DIR / "survivor_curve.csv", index=False)

    terminal_frame = pd.DataFrame(
        {
            "x": grid.x,
            "terminal_intensity": solution.terminal_mass,
            "tail_intensity": solution.terminal_stats.tail_intensity,
            "marginal_pivotal_value": solution.terminal_stats.gradient,
        }
    )
    terminal_frame.to_csv(OUTPUT_DIR / "terminal_wavefront.csv", index=False)

    cutoff_data = {"time": np.arange(n_steps) * grid.dt}
    active_number = 0
    for weight, policy in zip(solution.weights, solution.policies):
        if weight <= 1e-8 or policy.path_time == 0:
            continue
        active_number += 1
        cutoff_data[f"cutoff_{active_number}"] = policy.cutoffs
        cutoff_data[f"weight_{active_number}"] = np.full(n_steps, weight)
    pd.DataFrame(cutoff_data).to_csv(OUTPUT_DIR / "cutoff_atoms.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(times[:-1], solution.alive_count_curve, label="Expected paths kept")
    plt.xlabel("Time")
    plt.ylabel("Expected alive optional paths")
    plt.title("Adaptive Brownian pruning: survivor schedule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "survivors.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for weight, policy in zip(solution.weights, solution.policies):
        if weight <= 1e-8 or policy.path_time == 0:
            continue
        finite_cutoff = policy.cutoffs.copy()
        finite_cutoff[~np.isfinite(finite_cutoff)] = np.nan
        plt.plot(
            times[:-1],
            finite_cutoff,
            label=f"atom weight={weight:.3f}",
        )
    plt.xlabel("Time")
    plt.ylabel("Kill below cutoff")
    plt.title("Active cutoff-policy atoms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cutoffs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(grid.x, solution.terminal_mass, label="Terminal survivor intensity")
    front_index = int(
        np.argmin(np.abs(solution.terminal_stats.tail_intensity - 1.0))
    )
    plt.axvline(
        grid.x[front_index],
        linestyle="--",
        label=f"tail intensity ≈ 1 at x={grid.x[front_index]:.2f}",
    )
    plt.xlabel("Terminal path level")
    plt.ylabel("Poisson intensity")
    plt.title("Terminal survivor cloud and leader front")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "terminal_intensity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        grid.x,
        solution.terminal_stats.gradient,
        label="Marginal future pivotal value",
    )
    plt.xlabel("Candidate terminal level")
    plt.ylabel("Incremental expected maximum")
    plt.title("Terminal wavefront option value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "pivotal_value.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
