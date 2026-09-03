from __future__ import annotations

import numpy as np

from brownianbandit import (
    BrownianGrid,
    best_one_shot_screening_baseline,
    finite_difference_gradient_check,
    point_mass_initial_intensity,
    propagate_without_pruning,
    simulate_poisson_terminal_maximum,
    solve_for_budget,
    solve_lagrangian,
    solve_linearized_best_response,
    static_random_thinning_baseline,
    terminal_maximum_stats,
)


def small_problem(n_steps: int = 40, n_space: int = 301):
    grid = BrownianGrid.build(
        x_min=-6.0,
        x_max=6.0,
        n_space=n_space,
        dt=1.0 / n_steps,
        sigma=1.0,
    )
    initial = point_mass_initial_intensity(
        grid,
        total_intensity=100.0,
        x0=0.0,
    )
    return grid, initial, n_steps


def test_transition_is_stochastic_and_matches_brownian_variance() -> None:
    grid, _, n_steps = small_problem()
    row_sums = np.asarray(grid.transition.sum(axis=1)).ravel()
    assert np.max(np.abs(row_sums - 1.0)) < 1e-12
    assert grid.transition.data.min() >= 0.0

    unit_mass = point_mass_initial_intensity(grid, total_intensity=1.0, x0=0.0)
    terminal = propagate_without_pruning(unit_mass, grid.transition, n_steps)
    mean = float(np.dot(terminal, grid.x))
    variance = float(np.dot(terminal, (grid.x - mean) ** 2))
    assert abs(terminal.sum() - 1.0) < 1e-11
    assert abs(mean) < 1e-10
    assert abs(variance - 1.0) < 0.015


def test_terminal_poisson_max_gradient() -> None:
    grid, _, _ = small_problem(n_steps=30, n_space=241)
    rng = np.random.default_rng(7)
    mass = rng.gamma(shape=1.5, scale=0.01, size=grid.n_space)
    error = finite_difference_gradient_check(
        terminal_mass=mass,
        x=grid.x,
        fallback=0.0,
        epsilon=1e-6,
    )
    assert error < 3e-7

    stats = terminal_maximum_stats(mass, grid.x, fallback=0.0)
    assert np.all(np.diff(stats.gradient) >= -1e-13)
    assert np.all(np.diff(stats.gradient, n=2) >= -1e-12)


def test_linearized_best_response_is_an_upper_cutoff_policy() -> None:
    grid, initial, n_steps = small_problem(n_steps=35, n_space=281)
    # A convex increasing terminal reward, representative of a wavefront option.
    terminal_reward = np.maximum(grid.x - 1.5, 0.0)
    policy = solve_linearized_best_response(
        terminal_reward=terminal_reward,
        lambda_path_time=0.08,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
    )

    for mask in policy.keep_mask:
        transitions = np.diff(mask.astype(int))
        assert np.sum(transitions == 1) <= 1
        assert np.sum(transitions == -1) == 0
    assert policy.path_time >= 0.0
    assert policy.terminal_mass.sum() <= initial.sum() + 1e-10


def test_lagrangian_cost_decreases_with_shadow_price() -> None:
    grid, initial, n_steps = small_problem(n_steps=35, n_space=281)
    prices = [0.02, 0.08, 0.25, 1.0]
    solutions = [
        solve_lagrangian(
            lambda_path_time=price,
            grid=grid,
            initial_mass=initial,
            n_steps=n_steps,
            fallback=0.0,
            tolerance=3e-6,
            max_iterations=80,
        )
        for price in prices
    ]
    costs = np.asarray([solution.path_time for solution in solutions])
    assert np.all(np.diff(costs) <= 1e-6)
    assert solutions[-1].path_time < 1e-10
    assert max(solution.dual_gap for solution in solutions) < 2e-5


def test_budget_solver_beats_simple_baselines() -> None:
    grid, initial, n_steps = small_problem(n_steps=50, n_space=351)
    budget = 10.0
    solution = solve_for_budget(
        target_path_time=budget,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
        budget_tolerance=2e-3,
        frank_wolfe_tolerance=3e-6,
        max_frank_wolfe_iterations=80,
    )
    static = static_random_thinning_baseline(
        target_path_time=budget,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
    )
    one_shot = best_one_shot_screening_baseline(
        target_path_time=budget,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
    )

    assert abs(solution.path_time - budget) <= 0.03
    assert solution.objective > one_shot.objective + 0.15
    assert one_shot.objective > static.objective + 0.05
    assert solution.terminal_mass.sum() > 0.0


def test_terminal_maximum_matches_poisson_monte_carlo() -> None:
    grid, initial, n_steps = small_problem(n_steps=45, n_space=321)
    solution = solve_for_budget(
        target_path_time=8.0,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
        budget_tolerance=3e-3,
        frank_wolfe_tolerance=5e-6,
        max_frank_wolfe_iterations=70,
    )
    estimate = simulate_poisson_terminal_maximum(
        terminal_mass=solution.terminal_mass,
        x=grid.x,
        fallback=0.0,
        n_trials=60_000,
        seed=20260901,
    )
    discrepancy = abs(estimate.mean - solution.objective)
    assert discrepancy < 4.0 * estimate.standard_error + grid.dx


def test_finer_grid_is_numerically_consistent() -> None:
    values = []
    for n_steps, n_space in [(35, 281), (70, 441)]:
        grid, initial, _ = small_problem(n_steps=n_steps, n_space=n_space)
        solution = solve_for_budget(
            target_path_time=10.0,
            grid=grid,
            initial_mass=initial,
            n_steps=n_steps,
            fallback=0.0,
            budget_tolerance=4e-3,
            frank_wolfe_tolerance=5e-6,
            max_frank_wolfe_iterations=70,
        )
        values.append(solution.objective)
    assert abs(values[1] - values[0]) < 0.08
    assert values[1] >= values[0] - 0.02


def test_budget_certificate_bounds_feasible_improvements() -> None:
    grid, initial, n_steps = small_problem(n_steps=50, n_space=301)
    baseline = best_one_shot_screening_baseline(
        target_path_time=10.0,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
    )

    # A deliberately crippled bisection returns a poor mixture; the
    # certificate must be large enough to cover any feasible policy's
    # advantage over it, the one-shot baseline included.
    crippled = solve_for_budget(
        target_path_time=10.0,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=0.0,
        max_bisection_iterations=0,
    )
    assert crippled.path_time <= 10.0 + 1e-9
    assert crippled.certificate is not None
    assert crippled.certificate >= baseline.objective - crippled.objective

    # The normal solve is feasible by construction and certified tight.
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
    assert solution.path_time <= 10.0 + 1e-9
    assert solution.certificate is not None
    assert 0.0 <= solution.certificate < 1e-3
    assert solution.objective >= baseline.objective - solution.certificate
