"""Numerical solver for the mean-field limit of a budgeted Brownian race.

Model
-----
A Poisson cloud of IID Brownian paths starts with a supplied intensity measure.
At each decision time, every alive path may either be killed or kept for one
more time step. Keeping one path for dt costs dt. Killed paths never return.
At the terminal time, the payoff is the maximum of a deterministic fallback
level and all surviving paths.

The code solves the *Poissonized, expected-budget, discrete-time* version of
this problem. It is the natural finite-difference approximation to the
continuous mean-field obstacle/Fokker--Planck system. It is not yet the
finite-n, pathwise-hard-budget control problem.

The optimization is over *per-path, population-blind* controls: each path's
keep-or-kill decision depends on its own state and time, plus independent
randomization, never on the realized population, so the surviving cloud
stays Poisson by independent thinning. A population-aware controller can do
better on the same expected budget. One step, Poisson(100) paths at zero,
budget one, fallback zero: independent thinning to intensity one pays about
0.3469, while keeping exactly one path whenever any exist pays
1/sqrt(2*pi) ~ 0.3989. The values computed here are optimal within the
population-blind class only.

For terminal survivor intensity m on ordered grid points x, let

    R_j = sum_{k >= j} m_k.

Then, exactly for this grid-supported Poisson cloud,

    J(m) = fallback + sum_{x_j > fallback}
                         Delta_j [1 - exp(-R_j)],

where Delta_j is the distance from the preceding attainable maximum level.
Its derivative is the future pivotal value of one additional path ending at
x_j:

    g_j = dJ/dm_j = E[(x_j - M)^+].

For a Lagrange multiplier lambda on path-time, the linearized best response is
an obstacle dynamic program:

    v_N = g,
    v_k(x) = max(0, E[v_{k+1}(X_{k+1}) | X_k=x] - lambda*dt).

Its keep set at each step is an upper tail of the state grid: the *wavefront
policy*, which retains a path while its propagated future pivotal value
exceeds its carrying cost.

The outer nonlinear problem is concave. A fully-corrective Frank--Wolfe
(column-generation) method repeatedly:

  1. computes g from the current terminal intensity;
  2. solves the obstacle best response;
  3. adds that cutoff policy to an active set; and
  4. re-optimizes the mixture of active policies.

A bisection in lambda enforces a requested expected path-time budget.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import ndtr

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BrownianGrid:
    """Uniform state grid and one-step Brownian transition matrix.

    `transition[j, k]` is the probability of moving from grid point j to grid
    point k in one time step. Probability leaving the numerical domain is
    absorbed into the two edge cells, so every row sums to one.
    """

    x: FloatArray
    dt: float
    sigma: float
    transition: sparse.csr_matrix

    @property
    def n_space(self) -> int:
        return int(self.x.size)

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @classmethod
    def build(
        cls,
        *,
        x_min: float,
        x_max: float,
        n_space: int,
        dt: float,
        sigma: float = 1.0,
        tail_sd: float = 8.0,
    ) -> "BrownianGrid":
        if n_space < 5:
            raise ValueError("n_space must be at least 5")
        if x_max <= x_min:
            raise ValueError("x_max must exceed x_min")
        if dt <= 0 or sigma <= 0:
            raise ValueError("dt and sigma must be positive")

        x = np.linspace(float(x_min), float(x_max), int(n_space), dtype=float)
        transition = _build_transition_matrix(x, dt, sigma, tail_sd)
        return cls(x=x, dt=float(dt), sigma=float(sigma), transition=transition)


def _build_transition_matrix(
    x: FloatArray,
    dt: float,
    sigma: float,
    tail_sd: float,
) -> sparse.csr_matrix:
    """Construct a positive, row-stochastic, sparse Gaussian transition."""

    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid")
    dxs = np.diff(x)
    dx = float(dxs[0])
    if not np.allclose(dxs, dx, rtol=1e-11, atol=1e-13):
        raise ValueError("x must be uniformly spaced")

    n = int(x.size)
    step_sd = float(sigma * np.sqrt(dt))
    half_band = int(np.ceil(tail_sd * step_sd / dx)) + 2

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for origin, x0 in enumerate(x):
        destinations: set[int] = {0, n - 1}
        destinations.update(
            range(max(0, origin - half_band), min(n - 1, origin + half_band) + 1)
        )

        row_cols: list[int] = []
        row_probs: list[float] = []
        for dest in sorted(destinations):
            lower = -np.inf if dest == 0 else x[dest] - 0.5 * dx
            upper = np.inf if dest == n - 1 else x[dest] + 0.5 * dx
            probability = float(
                ndtr((upper - x0) / step_sd) - ndtr((lower - x0) / step_sd)
            )
            if probability > 1e-16:
                row_cols.append(dest)
                row_probs.append(probability)

        probs = np.asarray(row_probs, dtype=float)
        probs /= probs.sum()  # removes only numerically omitted far-tail mass
        rows.extend([origin] * len(row_cols))
        cols.extend(row_cols)
        data.extend(probs.tolist())

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if np.max(np.abs(row_sums - 1.0)) > 1e-12:
        raise RuntimeError("transition construction failed to preserve probability")
    if matrix.data.min(initial=0.0) < 0:
        raise RuntimeError("transition matrix contains a negative probability")
    return matrix


def point_mass_initial_intensity(
    grid: BrownianGrid,
    *,
    total_intensity: float,
    x0: float = 0.0,
) -> FloatArray:
    """Put a Poisson intensity at x0, linearly split across adjacent cells."""

    if total_intensity < 0:
        raise ValueError("total_intensity must be non-negative")
    x = grid.x
    if not (x[0] <= x0 <= x[-1]):
        raise ValueError("x0 lies outside the numerical grid")

    mass = np.zeros_like(x)
    right = int(np.searchsorted(x, x0, side="left"))
    if right == 0:
        mass[0] = total_intensity
    elif right == x.size:
        mass[-1] = total_intensity
    elif np.isclose(x[right], x0):
        mass[right] = total_intensity
    else:
        left = right - 1
        weight_right = (x0 - x[left]) / (x[right] - x[left])
        mass[left] = total_intensity * (1.0 - weight_right)
        mass[right] = total_intensity * weight_right
    return mass


@dataclass(frozen=True)
class TerminalMaximumStats:
    """Value and differential quantities for the terminal Poisson maximum."""

    value: float
    gradient: FloatArray
    tail_intensity: FloatArray
    interval_widths: FloatArray
    strict_max_cdf: FloatArray


def terminal_maximum_stats(
    terminal_mass: FloatArray,
    x: FloatArray,
    *,
    fallback: float = 0.0,
) -> TerminalMaximumStats:
    """Return exact discrete Poisson-max value and its gradient.

    The optional terminal paths form independent Poisson counts at grid points
    `x[j]`, with mean `terminal_mass[j]`. The payoff is the maximum of those
    points and `fallback`.

    The gradient at x[j] is the expected improvement from adding an
    infinitesimal Poisson path at x[j], namely E[(x[j] - M)^+].
    """

    mass = np.asarray(terminal_mass, dtype=float)
    x = np.asarray(x, dtype=float)
    if mass.shape != x.shape:
        raise ValueError("terminal_mass and x must have the same shape")
    if np.any(mass < -1e-12):
        raise ValueError("terminal_mass must be non-negative")
    mass = np.maximum(mass, 0.0)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    tail = np.cumsum(mass[::-1])[::-1]

    # Exact layer-cake increments for a variable supported on {fallback} U x.
    widths = np.zeros_like(x)
    eligible = np.flatnonzero(x > fallback)
    previous = float(fallback)
    for j in eligible:
        widths[j] = x[j] - previous
        previous = float(x[j])

    no_point_at_or_above = np.exp(-tail)
    value = float(fallback + np.dot(widths, 1.0 - no_point_at_or_above))
    gradient = np.cumsum(widths * no_point_at_or_above)

    return TerminalMaximumStats(
        value=value,
        gradient=gradient,
        tail_intensity=tail,
        interval_widths=widths,
        strict_max_cdf=no_point_at_or_above,
    )


@dataclass
class PureCutoffPolicy:
    """One deterministic obstacle-policy atom."""

    terminal_mass: FloatArray
    path_time: float
    linear_value: float
    value_function: FloatArray
    continuation_value: FloatArray
    keep_mask: BoolArray
    cutoffs: FloatArray
    mass_before_decision: FloatArray
    alive_mass: FloatArray

    @classmethod
    def kill_all(
        cls,
        *,
        initial_mass: FloatArray,
        x: FloatArray,
        n_steps: int,
    ) -> "PureCutoffPolicy":
        n_space = x.size
        mass_history = np.zeros((n_steps + 1, n_space), dtype=float)
        mass_history[0] = initial_mass
        return cls(
            terminal_mass=np.zeros(n_space, dtype=float),
            path_time=0.0,
            linear_value=0.0,
            value_function=np.zeros((n_steps + 1, n_space), dtype=float),
            continuation_value=np.zeros((n_steps, n_space), dtype=float),
            keep_mask=np.zeros((n_steps, n_space), dtype=bool),
            cutoffs=np.full(n_steps, np.inf, dtype=float),
            mass_before_decision=mass_history,
            alive_mass=np.zeros((n_steps, n_space), dtype=float),
        )


def _suffix_mask_and_cutoff(
    continuation: FloatArray,
    x: FloatArray,
    *,
    decision_tolerance: float,
    monotonicity_tolerance: float,
) -> tuple[BoolArray, float]:
    """Convert an increasing continuation value to an upper-tail keep set."""

    min_increment = float(np.min(np.diff(continuation), initial=0.0))
    scale = max(1.0, float(np.max(np.abs(continuation), initial=0.0)))
    if min_increment < -monotonicity_tolerance * scale:
        raise RuntimeError(
            "continuation value is not increasing; enlarge/refine the grid"
        )

    # Remove harmless floating-point wiggles before locating the zero crossing.
    monotone = np.maximum.accumulate(continuation)
    positive = np.flatnonzero(monotone > decision_tolerance)
    if positive.size == 0:
        return np.zeros_like(monotone, dtype=bool), np.inf
    first = int(positive[0])
    mask = np.zeros_like(monotone, dtype=bool)
    mask[first:] = True
    if first == 0:
        cutoff = -np.inf
    else:
        cutoff = 0.5 * (x[first - 1] + x[first])
    return mask, float(cutoff)


def solve_linearized_best_response(
    *,
    terminal_reward: FloatArray,
    lambda_path_time: float,
    grid: BrownianGrid,
    initial_mass: FloatArray,
    n_steps: int,
    decision_tolerance: float = 1e-14,
    monotonicity_tolerance: float = 1e-10,
) -> PureCutoffPolicy:
    """Solve the single-particle obstacle problem and propagate the population."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if lambda_path_time < 0:
        raise ValueError("lambda_path_time must be non-negative")
    reward = np.asarray(terminal_reward, dtype=float)
    initial = np.asarray(initial_mass, dtype=float)
    if reward.shape != grid.x.shape or initial.shape != grid.x.shape:
        raise ValueError("reward and initial_mass must match the grid")

    n_space = grid.n_space
    value = np.empty((n_steps + 1, n_space), dtype=float)
    continuation = np.empty((n_steps, n_space), dtype=float)
    keep = np.empty((n_steps, n_space), dtype=bool)
    cutoffs = np.empty(n_steps, dtype=float)
    value[n_steps] = reward

    for k in range(n_steps - 1, -1, -1):
        raw_continuation = (
            grid.transition @ value[k + 1]
            - lambda_path_time * grid.dt
        )
        mask, cutoff = _suffix_mask_and_cutoff(
            raw_continuation,
            grid.x,
            decision_tolerance=decision_tolerance,
            monotonicity_tolerance=monotonicity_tolerance,
        )
        continuation[k] = raw_continuation
        keep[k] = mask
        cutoffs[k] = cutoff
        value[k] = np.where(mask, np.maximum(raw_continuation, 0.0), 0.0)

    mass_before = np.zeros((n_steps + 1, n_space), dtype=float)
    alive_mass = np.zeros((n_steps, n_space), dtype=float)
    mass_before[0] = initial
    path_time = 0.0

    for k in range(n_steps):
        alive_mass[k] = mass_before[k] * keep[k]
        path_time += grid.dt * float(alive_mass[k].sum())
        mass_before[k + 1] = grid.transition.T @ alive_mass[k]

    terminal_mass = mass_before[n_steps].copy()
    linear_value = float(
        np.dot(reward, terminal_mass) - lambda_path_time * path_time
    )
    bellman_value = float(np.dot(initial, value[0]))
    if abs(linear_value - bellman_value) > 5e-8 * max(1.0, abs(linear_value)):
        raise RuntimeError(
            "forward propagation and backward Bellman values are inconsistent"
        )

    return PureCutoffPolicy(
        terminal_mass=terminal_mass,
        path_time=float(path_time),
        linear_value=linear_value,
        value_function=value,
        continuation_value=continuation,
        keep_mask=keep,
        cutoffs=cutoffs,
        mass_before_decision=mass_before,
        alive_mass=alive_mass,
    )


@dataclass(frozen=True)
class FrankWolfeIteration:
    iteration: int
    objective: float
    path_time: float
    lagrangian_value: float
    dual_gap: float
    active_atoms: int


@dataclass
class LagrangianSolution:
    """Optimal relaxed population policy for a fixed path-time price."""

    lambda_path_time: float
    objective: float
    path_time: float
    terminal_mass: FloatArray
    terminal_stats: TerminalMaximumStats
    dual_gap: float
    weights: FloatArray
    policies: list[PureCutoffPolicy]
    history: list[FrankWolfeIteration] = field(default_factory=list)
    budget_mixture: bool = False

    @property
    def active_policy_indices(self) -> FloatArray:
        return np.flatnonzero(self.weights > 1e-9)

    @property
    def alive_count_curve(self) -> FloatArray:
        n_steps = self.policies[0].alive_mass.shape[0]
        result = np.zeros(n_steps, dtype=float)
        for weight, policy in zip(self.weights, self.policies):
            if weight > 0:
                result += weight * policy.alive_mass.sum(axis=1)
        return result

    @property
    def predecision_count_curve(self) -> FloatArray:
        n_steps_plus_one = self.policies[0].mass_before_decision.shape[0]
        result = np.zeros(n_steps_plus_one, dtype=float)
        for weight, policy in zip(self.weights, self.policies):
            if weight > 0:
                result += weight * policy.mass_before_decision.sum(axis=1)
        return result


def _is_duplicate_policy(
    candidate: PureCutoffPolicy,
    policies: Sequence[PureCutoffPolicy],
    initial_total: float,
) -> bool:
    for policy in policies:
        if abs(policy.path_time - candidate.path_time) > 1e-10:
            continue
        distance = float(
            np.sum(np.abs(policy.terminal_mass - candidate.terminal_mass))
        )
        if distance <= 1e-8 * max(1.0, initial_total):
            return True
    return False


def _fully_correct_weights(
    *,
    policies: Sequence[PureCutoffPolicy],
    initial_weights: FloatArray,
    x: FloatArray,
    fallback: float,
    lambda_path_time: float,
) -> FloatArray:
    terminal_atoms = np.column_stack([p.terminal_mass for p in policies])
    costs = np.asarray([p.path_time for p in policies], dtype=float)

    def objective(weights: FloatArray) -> float:
        terminal_mass = terminal_atoms @ weights
        stats = terminal_maximum_stats(terminal_mass, x, fallback=fallback)
        return -(stats.value - lambda_path_time * float(costs @ weights))

    def jacobian(weights: FloatArray) -> FloatArray:
        terminal_mass = terminal_atoms @ weights
        gradient = terminal_maximum_stats(
            terminal_mass, x, fallback=fallback
        ).gradient
        return -(terminal_atoms.T @ gradient - lambda_path_time * costs)

    equality = {
        "type": "eq",
        "fun": lambda weights: float(weights.sum() - 1.0),
        "jac": lambda weights: np.ones_like(weights),
    }
    result = minimize(
        objective,
        np.asarray(initial_weights, dtype=float),
        jac=jacobian,
        bounds=[(0.0, 1.0)] * len(policies),
        constraints=equality,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"active-set optimization failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    weights /= weights.sum()
    return weights


def solve_lagrangian(
    *,
    lambda_path_time: float,
    grid: BrownianGrid,
    initial_mass: FloatArray,
    n_steps: int,
    fallback: float = 0.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    prune_weight: float = 1e-11,
) -> LagrangianSolution:
    """Solve max J(m_T) - lambda * expected_path_time.

    This is a fully-corrective Frank--Wolfe/column-generation method. Each
    generated atom is a deterministic upper-cutoff policy. Their mixture is a
    feasible pathwise randomization for the Poisson population.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    initial = np.asarray(initial_mass, dtype=float)
    if initial.shape != grid.x.shape or np.any(initial < 0):
        raise ValueError("initial_mass must be non-negative and match the grid")

    zero_policy = PureCutoffPolicy.kill_all(
        initial_mass=initial,
        x=grid.x,
        n_steps=n_steps,
    )
    policies: list[PureCutoffPolicy] = [zero_policy]
    weights = np.asarray([1.0], dtype=float)
    history: list[FrankWolfeIteration] = []
    initial_total = float(initial.sum())

    for iteration in range(max_iterations):
        terminal_atoms = np.column_stack([p.terminal_mass for p in policies])
        costs = np.asarray([p.path_time for p in policies], dtype=float)
        terminal_mass = terminal_atoms @ weights
        path_time = float(costs @ weights)
        stats = terminal_maximum_stats(
            terminal_mass, grid.x, fallback=fallback
        )

        candidate = solve_linearized_best_response(
            terminal_reward=stats.gradient,
            lambda_path_time=lambda_path_time,
            grid=grid,
            initial_mass=initial,
            n_steps=n_steps,
        )
        dual_gap = float(
            np.dot(stats.gradient, candidate.terminal_mass - terminal_mass)
            - lambda_path_time * (candidate.path_time - path_time)
        )
        lagrangian_value = stats.value - lambda_path_time * path_time
        history.append(
            FrankWolfeIteration(
                iteration=iteration,
                objective=stats.value,
                path_time=path_time,
                lagrangian_value=lagrangian_value,
                dual_gap=dual_gap,
                active_atoms=len(policies),
            )
        )

        scale = max(1.0, abs(lagrangian_value))
        if dual_gap <= tolerance * scale:
            break

        if not _is_duplicate_policy(candidate, policies, initial_total):
            policies.append(candidate)
            weights = np.append(weights, 0.0)

        weights = _fully_correct_weights(
            policies=policies,
            initial_weights=weights,
            x=grid.x,
            fallback=fallback,
            lambda_path_time=lambda_path_time,
        )

        if len(policies) > 30:
            retained = np.flatnonzero(weights > prune_weight)
            policies = [policies[int(i)] for i in retained]
            weights = weights[retained]
            weights /= weights.sum()
    else:
        # Returning an approximate answer is preferable to hiding convergence.
        pass

    terminal_atoms = np.column_stack([p.terminal_mass for p in policies])
    costs = np.asarray([p.path_time for p in policies], dtype=float)
    terminal_mass = terminal_atoms @ weights
    path_time = float(costs @ weights)
    stats = terminal_maximum_stats(terminal_mass, grid.x, fallback=fallback)
    candidate = solve_linearized_best_response(
        terminal_reward=stats.gradient,
        lambda_path_time=lambda_path_time,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
    )
    dual_gap = float(
        np.dot(stats.gradient, candidate.terminal_mass - terminal_mass)
        - lambda_path_time * (candidate.path_time - path_time)
    )

    return LagrangianSolution(
        lambda_path_time=float(lambda_path_time),
        objective=stats.value,
        path_time=path_time,
        terminal_mass=terminal_mass,
        terminal_stats=stats,
        dual_gap=dual_gap,
        weights=weights,
        policies=policies,
        history=history,
    )


def _combine_budget_bracket(
    *,
    lower_price_solution: LagrangianSolution,
    higher_price_solution: LagrangianSolution,
    target_path_time: float,
    x: FloatArray,
    fallback: float,
) -> LagrangianSolution:
    """Mix two nearby feasible policies to hit the budget exactly."""

    high_cost = lower_price_solution.path_time
    low_cost = higher_price_solution.path_time
    if not (high_cost >= target_path_time >= low_cost):
        raise ValueError("solutions do not bracket the requested path-time")
    if high_cost <= low_cost + 1e-14:
        return higher_price_solution

    high_weight = (target_path_time - low_cost) / (high_cost - low_cost)
    low_weight = 1.0 - high_weight

    policies = (
        list(lower_price_solution.policies)
        + list(higher_price_solution.policies)
    )
    weights = np.concatenate(
        [
            high_weight * lower_price_solution.weights,
            low_weight * higher_price_solution.weights,
        ]
    )
    retained = np.flatnonzero(weights > 1e-13)
    policies = [policies[int(i)] for i in retained]
    weights = weights[retained]
    weights /= weights.sum()

    terminal_mass = sum(
        (weight * policy.terminal_mass for weight, policy in zip(weights, policies)),
        start=np.zeros_like(x),
    )
    stats = terminal_maximum_stats(terminal_mass, x, fallback=fallback)
    lambda_mid = 0.5 * (
        lower_price_solution.lambda_path_time
        + higher_price_solution.lambda_path_time
    )

    return LagrangianSolution(
        lambda_path_time=lambda_mid,
        objective=stats.value,
        path_time=float(target_path_time),
        terminal_mass=terminal_mass,
        terminal_stats=stats,
        dual_gap=max(
            lower_price_solution.dual_gap,
            higher_price_solution.dual_gap,
        ),
        weights=weights,
        policies=policies,
        history=(
            lower_price_solution.history + higher_price_solution.history
        ),
        budget_mixture=True,
    )


def solve_for_budget(
    *,
    target_path_time: float,
    grid: BrownianGrid,
    initial_mass: FloatArray,
    n_steps: int,
    fallback: float = 0.0,
    budget_tolerance: float = 1e-3,
    max_bisection_iterations: int = 30,
    frank_wolfe_tolerance: float = 1e-6,
    max_frank_wolfe_iterations: int = 100,
) -> LagrangianSolution:
    """Solve the constrained expected-path-time problem by dual bisection."""

    if target_path_time < 0:
        raise ValueError("target_path_time must be non-negative")
    initial = np.asarray(initial_mass, dtype=float)
    full_path_time = float(initial.sum()) * n_steps * grid.dt

    lower_price = 0.0
    lower_solution = solve_lagrangian(
        lambda_path_time=lower_price,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=fallback,
        tolerance=frank_wolfe_tolerance,
        max_iterations=max_frank_wolfe_iterations,
    )
    if target_path_time >= lower_solution.path_time:
        return lower_solution

    higher_price = 1.0
    higher_solution = solve_lagrangian(
        lambda_path_time=higher_price,
        grid=grid,
        initial_mass=initial,
        n_steps=n_steps,
        fallback=fallback,
        tolerance=frank_wolfe_tolerance,
        max_iterations=max_frank_wolfe_iterations,
    )
    while higher_solution.path_time > target_path_time:
        higher_price *= 2.0
        if higher_price > 1e8:
            raise RuntimeError("failed to bracket the budget multiplier")
        higher_solution = solve_lagrangian(
            lambda_path_time=higher_price,
            grid=grid,
            initial_mass=initial,
            n_steps=n_steps,
            fallback=fallback,
            tolerance=frank_wolfe_tolerance,
            max_iterations=max_frank_wolfe_iterations,
        )

    best = min(
        [lower_solution, higher_solution],
        key=lambda solution: abs(solution.path_time - target_path_time),
    )
    absolute_tolerance = budget_tolerance * max(1.0, target_path_time)

    for _ in range(max_bisection_iterations):
        midpoint = 0.5 * (lower_price + higher_price)
        middle_solution = solve_lagrangian(
            lambda_path_time=midpoint,
            grid=grid,
            initial_mass=initial,
            n_steps=n_steps,
            fallback=fallback,
            tolerance=frank_wolfe_tolerance,
            max_iterations=max_frank_wolfe_iterations,
        )
        if abs(middle_solution.path_time - target_path_time) < abs(
            best.path_time - target_path_time
        ):
            best = middle_solution
        if abs(middle_solution.path_time - target_path_time) <= absolute_tolerance:
            return middle_solution

        if middle_solution.path_time > target_path_time:
            lower_price = midpoint
            lower_solution = middle_solution
        else:
            higher_price = midpoint
            higher_solution = middle_solution

    if abs(best.path_time - target_path_time) <= absolute_tolerance:
        return best

    # Discrete time/state can make the dual cost curve jump. Per-particle
    # randomization between the two bracketing policies is feasible and hits
    # the expected budget exactly.
    return _combine_budget_bracket(
        lower_price_solution=lower_solution,
        higher_price_solution=higher_solution,
        target_path_time=target_path_time,
        x=grid.x,
        fallback=fallback,
    )


@dataclass(frozen=True)
class BaselineResult:
    name: str
    objective: float
    path_time: float
    terminal_mass: FloatArray
    screening_time: float | None = None
    retained_intensity: float | None = None


def propagate_without_pruning(
    mass: FloatArray,
    transition: sparse.csr_matrix,
    n_steps: int,
) -> FloatArray:
    result = np.asarray(mass, dtype=float).copy()
    for _ in range(n_steps):
        result = transition.T @ result
    return result


def static_random_thinning_baseline(
    *,
    target_path_time: float,
    grid: BrownianGrid,
    initial_mass: FloatArray,
    n_steps: int,
    fallback: float = 0.0,
) -> BaselineResult:
    """Randomly retain a fixed fraction at time zero and never prune again."""

    horizon = n_steps * grid.dt
    maximum_cost = float(initial_mass.sum()) * horizon
    if maximum_cost <= 0:
        fraction = 0.0
    else:
        fraction = min(1.0, max(0.0, target_path_time / maximum_cost))
    retained = fraction * np.asarray(initial_mass, dtype=float)
    terminal = propagate_without_pruning(retained, grid.transition, n_steps)
    stats = terminal_maximum_stats(terminal, grid.x, fallback=fallback)
    return BaselineResult(
        name="static random thinning",
        objective=stats.value,
        path_time=fraction * maximum_cost,
        terminal_mass=terminal,
        screening_time=0.0,
        retained_intensity=float(retained.sum()),
    )


def _retain_top_intensity(mass: FloatArray, amount: float) -> FloatArray:
    """Retain exactly `amount` of the upper tail, fractionally at one cell."""

    target = min(max(float(amount), 0.0), float(mass.sum()))
    retained = np.zeros_like(mass, dtype=float)
    remaining = target
    for j in range(mass.size - 1, -1, -1):
        take = min(float(mass[j]), remaining)
        retained[j] = take
        remaining -= take
        if remaining <= 1e-13:
            break
    return retained


def best_one_shot_screening_baseline(
    *,
    target_path_time: float,
    grid: BrownianGrid,
    initial_mass: FloatArray,
    n_steps: int,
    fallback: float = 0.0,
) -> BaselineResult:
    """Keep all until one screening date, then retain the upper tail to T."""

    initial_total = float(initial_mass.sum())
    horizon = n_steps * grid.dt
    current_mass = np.asarray(initial_mass, dtype=float).copy()
    best: BaselineResult | None = None

    for screen_step in range(n_steps):
        screen_time = screen_step * grid.dt
        remaining_horizon = horizon - screen_time
        path_time_spent = initial_total * screen_time
        remaining_budget = target_path_time - path_time_spent
        if remaining_budget < -1e-12:
            break

        retained_intensity = min(
            float(current_mass.sum()),
            max(0.0, remaining_budget / remaining_horizon),
        )
        retained = _retain_top_intensity(current_mass, retained_intensity)
        terminal = propagate_without_pruning(
            retained,
            grid.transition,
            n_steps - screen_step,
        )
        actual_cost = path_time_spent + retained_intensity * remaining_horizon
        stats = terminal_maximum_stats(terminal, grid.x, fallback=fallback)
        candidate = BaselineResult(
            name="best one-shot screening",
            objective=stats.value,
            path_time=actual_cost,
            terminal_mass=terminal,
            screening_time=screen_time,
            retained_intensity=retained_intensity,
        )
        if best is None or candidate.objective > best.objective:
            best = candidate

        current_mass = grid.transition.T @ current_mass

    if best is None:
        terminal = np.zeros_like(initial_mass)
        return BaselineResult(
            name="best one-shot screening",
            objective=fallback,
            path_time=0.0,
            terminal_mass=terminal,
        )
    return best


@dataclass(frozen=True)
class MonteCarloMaximumEstimate:
    mean: float
    standard_error: float
    n_trials: int


def simulate_poisson_terminal_maximum(
    *,
    terminal_mass: FloatArray,
    x: FloatArray,
    fallback: float = 0.0,
    n_trials: int = 50_000,
    seed: int = 12345,
) -> MonteCarloMaximumEstimate:
    """Monte Carlo check of the terminal Poisson maximum formula."""

    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    mass = np.asarray(terminal_mass, dtype=float)
    total = float(mass.sum())
    rng = np.random.default_rng(seed)
    maxima = np.full(n_trials, float(fallback), dtype=float)
    if total > 0:
        counts = rng.poisson(total, size=n_trials)
        total_draws = int(counts.sum())
        if total_draws > 0:
            trial_ids = np.repeat(np.arange(n_trials), counts)
            sampled_indices = rng.choice(
                mass.size,
                size=total_draws,
                p=mass / total,
            )
            np.maximum.at(maxima, trial_ids, x[sampled_indices])
    return MonteCarloMaximumEstimate(
        mean=float(maxima.mean()),
        standard_error=float(maxima.std(ddof=1) / np.sqrt(n_trials)),
        n_trials=int(n_trials),
    )


def finite_difference_gradient_check(
    *,
    terminal_mass: FloatArray,
    x: FloatArray,
    fallback: float = 0.0,
    indices: Iterable[int] | None = None,
    epsilon: float = 1e-6,
) -> float:
    """Return the largest relative error in the analytic terminal gradient."""

    stats = terminal_maximum_stats(terminal_mass, x, fallback=fallback)
    if indices is None:
        indices = np.linspace(0, x.size - 1, 9, dtype=int)
    largest = 0.0
    for index in indices:
        j = int(index)
        base = np.asarray(terminal_mass, dtype=float)
        if base[j] > epsilon:
            bumped_up = base.copy()
            bumped_down = base.copy()
            bumped_up[j] += epsilon
            bumped_down[j] -= epsilon
            value_up = terminal_maximum_stats(
                bumped_up, x, fallback=fallback
            ).value
            value_down = terminal_maximum_stats(
                bumped_down, x, fallback=fallback
            ).value
            finite_difference = (value_up - value_down) / (2.0 * epsilon)
        else:
            bumped_up = base.copy()
            bumped_up[j] += epsilon
            value_up = terminal_maximum_stats(
                bumped_up, x, fallback=fallback
            ).value
            finite_difference = (value_up - stats.value) / epsilon
        denominator = max(1.0, abs(finite_difference), abs(stats.gradient[j]))
        largest = max(
            largest,
            abs(finite_difference - stats.gradient[j]) / denominator,
        )
    return float(largest)


__all__ = [
    "BaselineResult",
    "BrownianGrid",
    "FrankWolfeIteration",
    "LagrangianSolution",
    "MonteCarloMaximumEstimate",
    "PureCutoffPolicy",
    "TerminalMaximumStats",
    "best_one_shot_screening_baseline",
    "finite_difference_gradient_check",
    "point_mass_initial_intensity",
    "propagate_without_pruning",
    "simulate_poisson_terminal_maximum",
    "solve_for_budget",
    "solve_lagrangian",
    "solve_linearized_best_response",
    "static_random_thinning_baseline",
    "terminal_maximum_stats",
]
