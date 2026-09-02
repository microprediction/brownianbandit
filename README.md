# brownianbandit (view as [web page](https://brownianbandit.microprediction.org))

*Wavefront pruning of budgeted Brownian races.*

In a *budgeted Brownian race*, a controller watches a cloud of Brownian
paths, pays per unit time for every path kept alive, may irreversibly
prune, and spends a finite path-time budget to maximize the expected
terminal maximum. The distinctive object is the *wavefront*: the narrow
region near the future head of the race where one more path can still be
terminally pivotal. The optimal policy retains a path while its
propagated future pivotal value exceeds its carrying cost.

This package solves the Poissonized, expected-budget, discrete-time
version of the problem: the mean-field relaxation of the race, not yet
the finite-n, pathwise-hard-budget control problem. Core depends only on
numpy and scipy.

[![CI](https://github.com/microprediction/brownianbandit/workflows/CI/badge.svg)](https://github.com/microprediction/brownianbandit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

    pip install git+https://github.com/microprediction/brownianbandit.git

or clone and `pip install -e ".[test]"`.

## Quick start

A hundred paths, a path-time budget of ten, one call.

```python
from brownianbandit import (
    BrownianGrid, point_mass_initial_intensity, solve_for_budget,
)

grid = BrownianGrid.build(x_min=-6, x_max=6, n_space=601, dt=0.01, sigma=1.0)
initial = point_mass_initial_intensity(grid, total_intensity=100.0, x0=0.0)

solution = solve_for_budget(
    target_path_time=10.0,
    grid=grid,
    initial_mass=initial,
    n_steps=100,
    fallback=0.0,
)
solution.objective        # expected terminal maximum
solution.path_time        # expected path-time actually spent
solution.lambda_path_time # shadow price of one unit of path-time
```

## The model

- Optional paths form a Poisson cloud with a supplied initial intensity
  measure; each alive path follows IID Brownian motion.
- At every time step a path is either killed irreversibly or kept for
  one more step. Keeping one path for `dt` costs `dt`.
- The budget constrains **expected aggregate path-time**.
- The terminal payoff is the maximum of a deterministic fallback and the
  surviving paths.

One reading of the fallback: a reserve path or outside option has
already been funded. If a reserve path costs one unit per unit time, add
the horizon `T` to the optional budget reported here.

## The method

For terminal survivor intensity `m`, the expected maximum `J(m)` is
concave, and its gradient at level `x` is the future pivotal value
`E[(x - M)^+]` of placing one more infinitesimal path there. For a fixed
shadow price `lambda` on path-time, that gradient defines a one-particle
obstacle problem whose Bellman recursion generates an upper-cutoff
wavefront policy. A fully-corrective Frank–Wolfe (column-generation)
method repeatedly adds such policies and re-optimizes their mixture, and
bisection in `lambda` enforces the requested budget. This is a discrete
counterpart of the coupled obstacle/Fokker–Planck KKT system.

## The demo

`python examples/run_mean_field_demo.py` reproduces the committed
numerical example (T=1, initial intensity 100, budget 10, fallback 0,
100 time steps, 601 state points). From
[examples/output/mean_field_demo_summary.json](examples/output/mean_field_demo_summary.json):

| policy | expected terminal maximum |
| --- | --- |
| wavefront pruning (this solver) | 1.960 |
| best one-shot screening | 1.643 |
| static random thinning | 1.509 |

The one-shot baseline keeps everyone until a single screening date and
then retains an upper tail; the static baseline thins at random at time
zero. A 100,000-trial Monte Carlo puts the adaptive policy's expected
maximum at 1.9609 with standard error 0.0021, against the solver's
1.9601. The Frank–Wolfe dual gap at termination is below 1e-7.

## Tests

`pytest -q` runs checks that can fail for the right reason: the
transition matrix is stochastic and reproduces Brownian variance; the
analytic pivotal-value gradient matches finite differences to 3e-7; the
best response is an upper-cutoff policy at every step; spent path-time
is monotone in the shadow price; the solver beats both baselines by a
stated margin; the closed-form expected maximum agrees with Poisson
Monte Carlo; and refining the grid moves the objective by less than
0.08.

## Cite

    @software{cotton2026brownianbandit,
      author = {Cotton, Peter},
      title  = {brownianbandit: wavefront pruning of budgeted Brownian races},
      year   = {2026},
      url    = {https://github.com/microprediction/brownianbandit}
    }
