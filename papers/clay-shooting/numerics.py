"""Numerical certificates for "Clay Shooting: Equilibrium Effort Solving
Outstanding Problems".

Every number quoted in the paper is produced here. Run

    python papers/clay-shooting/numerics.py

to regenerate papers/clay-shooting/numerics.json.

Depends only on numpy (and the standard library).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

SEED = 20260909


# ---------------------------------------------------------------- reduction

def width(V: float, c: float, sigma: float, rho: float) -> float:
    """Continuation width b - a of the optimal abandonment boundary."""
    if rho <= 0.0:
        return sigma * math.sqrt(V / c)
    kappa = math.sqrt(2.0 * rho) / sigma
    return math.acosh(1.0 + rho * V / c) / kappa


def reduced_form(V: float, c: float, sigma: float, d: float, rho: float = 0.0):
    """(p, k, v) of one optimally run attempt started at distance d from b."""
    L = width(V, c, sigma, rho)
    if L <= d:
        return 0.0, 0.0, 0.0
    p = 1.0 - d / L
    k = c * d * (L - d) / sigma ** 2
    v = c * (L - d) ** 2 / sigma ** 2 if rho <= 0.0 else None
    return p, k, v


def monte_carlo_attempt(V, c, sigma, d, n_paths=50_000, dt=2e-3, rng=None):
    """Simulate the driftless attempt to the optimal boundary, Euler scheme.

    Independent of the closed forms: the only inputs are the boundary
    location and the increment law.
    """
    rng = rng or np.random.default_rng(SEED)
    L = width(V, c, sigma, rho=0.0)
    a, b, x0 = 0.0, L, L - d
    x = np.full(n_paths, x0)
    idx = np.arange(n_paths)
    solved = np.zeros(n_paths, dtype=bool)
    elapsed = np.zeros(n_paths)
    step = sigma * math.sqrt(dt)
    while idx.size:
        x[idx] += step * rng.standard_normal(idx.size)
        elapsed[idx] += dt
        hit = x[idx] >= b
        solved[idx[hit]] = True
        idx = idx[(~hit) & (x[idx] > a)]
    return {
        "n_paths": n_paths,
        "dt": dt,
        "p_hat": float(solved.mean()),
        "p_se": float(solved.std(ddof=1) / math.sqrt(n_paths)),
        "k_hat": float(c * elapsed.mean()),
        "k_se": float(c * elapsed.std(ddof=1) / math.sqrt(n_paths)),
        "value_hat": float(V * solved.mean() - c * elapsed.mean()),
    }


# --------------------------------------------------------- universal entry

def phi(w: float, n: int) -> float:
    """sum_{j<n} w^j - n w; nonnegative iff q = 1 is an equilibrium."""
    if abs(w - 1.0) < 1e-12:
        return float(n) - float(n) * w
    return (1.0 - w ** n) / (1.0 - w) - n * w


def universal_entry_threshold(n: int) -> float | None:
    """p*(n): the least p at which everybody entering is an equilibrium."""
    if n <= 3:
        return 0.0
    lo, hi = 1e-15, 1.0 - 1e-15
    if phi(lo, n) <= 0.0:
        return None
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if phi(mid, n) > 0.0:
            lo = mid
        else:
            hi = mid
    return 1.0 - 0.5 * (lo + hi)


# ------------------------------------------------------- free-entry volume

def y_star(p: float) -> float:
    """Unique y > 0 with (1 - exp(-y)) / y = 1 - p."""
    target = 1.0 - p
    lo, hi = 1e-14, 1.0
    while (1.0 - math.exp(-hi)) / hi > target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("no bracket")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if (1.0 - math.exp(-mid)) / mid > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def free_entry(p: float) -> dict:
    y = y_star(p)
    return {
        "p": p,
        "y_star": y,
        "entrants": y / p,
        "solve_probability": 1.0 - math.exp(-y),
        "spend_over_V": (1.0 - math.exp(-y)),
    }


def finite_n_entrants(p: float, n: int) -> float:
    """Expected entrants n q at the interior symmetric equilibrium, exact in n."""
    def g(q):
        return (1.0 - (1.0 - p * q) ** n) - n * q * p * (1.0 - p)

    if g(1.0) >= 0.0:
        return float(n)
    lo, hi = 1e-14, 1.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return n * 0.5 * (lo + hi)


# ------------------------------------------------------------ purse design

def purses(budget: float, zeta: float, m_max: int = 4000) -> dict:
    """Split committed capital `budget` into M purses of V = budget / M."""
    best = None
    for M in range(1, m_max + 1):
        V = budget / M
        if V <= zeta ** 2:
            continue
        p = 1.0 - zeta / math.sqrt(V)
        solved = M * (1.0 - math.exp(-y_star(p)))
        if best is None or solved > best["expected_solved"]:
            best = {
                "M": M,
                "purse": V,
                "purse_over_minimum": V / zeta ** 2,
                "p": p,
                "expected_solved": solved,
                "solved_per_problem": solved / M,
            }
    approx_M = 4.0 * budget / (9.0 * zeta ** 2)
    best["long_odds_M"] = approx_M
    best["long_odds_purse_over_minimum"] = 2.25
    best["long_odds_expected_solved"] = 8.0 * budget / (27.0 * zeta ** 2)
    return best


# -------------------------------------------------------------- Clay dials

MILLENNIUM_PROBLEMS = 7
MILLENNIUM_SOLVED = 1
MILLENNIUM_YEARS = 2026 - 2000


def rivalry_share(h: float, r: float) -> float:
    return h / (r + h)


def poisson_interval(count: int, exposure: float, level: float = 0.95):
    """Exact (Garwood) two-sided interval for a Poisson rate."""
    from scipy.stats import chi2
    alpha = 1.0 - level
    lo = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2.0
    hi = chi2.ppf(1.0 - alpha / 2, 2 * (count + 1)) / 2.0
    return lo / exposure, hi / exposure


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED}

    # 1. the reduction, checked against Monte Carlo
    V, c, sigma, d = 100.0, 1.0, 1.0, 4.0
    p, k, v = reduced_form(V, c, sigma, d)
    zeta = d * math.sqrt(c) / sigma
    out["reduction"] = {
        "V": V, "c": c, "sigma": sigma, "d": d,
        "zeta": zeta,
        "V_min": zeta ** 2,
        "width": width(V, c, sigma, 0.0),
        "p": p,
        "k": k,
        "k_identity_p_1_minus_p_V": p * (1.0 - p) * V,
        "value": v,
        "value_identity_V_p_squared": V * p * p,
        "value_identity_sqrt_form": (math.sqrt(V) - zeta) ** 2,
        "monte_carlo": monte_carlo_attempt(V, c, sigma, d, rng=rng),
    }

    # 2. when does everybody enter
    rows = []
    for n in [2, 3, 4, 5, 6, 8, 10, 20, 50, 100, 1000]:
        pstar = universal_entry_threshold(n)
        closed = ((1.0 + math.sqrt(1.0 - 4.0 / n)) / 2.0
                  if n >= 4 else None)
        rows.append({
            "n": n,
            "p_star": pstar,
            "p_star_closed_form": closed,
            "closed_form_error": (None if closed is None
                                  else abs(closed - pstar)),
            "asymptote": 1.0 - 1.0 / n - 1.0 / n ** 2,
        })
    out["universal_entry"] = rows

    # 3. free-entry volume
    vol = []
    for pp in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        row = free_entry(pp)
        row["entrants_expansion_2_plus_4p_over_3"] = 2.0 + 4.0 * pp / 3.0
        row["entrants_finite_n_50"] = finite_n_entrants(pp, 50)
        row["entrants_finite_n_1000"] = finite_n_entrants(pp, 1000)
        vol.append(row)
    out["free_entry"] = vol

    # 4. depth is logarithmic in the purse
    depth = []
    c_d, sigma_d, rho_d = 1.0, 1.0, 0.05
    base = width(1.0, c_d, sigma_d, rho_d)
    for mult in [1, 2, 10, 100, 1000, 10_000]:
        depth.append({
            "V": float(mult),
            "width_rho_positive": width(float(mult), c_d, sigma_d, rho_d),
            "width_ratio": width(float(mult), c_d, sigma_d, rho_d) / base,
            "width_rho_zero": width(float(mult), c_d, sigma_d, 0.0),
        })
    out["depth"] = {"c": c_d, "sigma": sigma_d, "rho": rho_d, "rows": depth}

    # 5. purse design
    out["purse_design"] = purses(budget=100.0, zeta=1.0)

    # 6. the Clay dials
    h_hat = MILLENNIUM_SOLVED / (MILLENNIUM_PROBLEMS * MILLENNIUM_YEARS)
    shares = []
    for r in [0.03, 0.05, 0.10]:
        shares.append({
            "r": r,
            "millennium": rivalry_share(h_hat, r),
            "warm_10yr": rivalry_share(0.1, r),
            "hot_3yr": rivalry_share(1.0 / 3.0, r),
        })
    implied = []
    for k_att in [0.6e6, 1.0e6, 1.2e6]:
        for pp in [0.005, 0.01, 0.02]:
            implied.append({
                "attempt_cost": k_att,
                "p": pp,
                "implied_V": k_att / (pp * (1.0 - pp)),
                "purse_share_of_implied_V": 1.0e6 / (k_att / (pp * (1.0 - pp))),
            })
    exposure = MILLENNIUM_PROBLEMS * MILLENNIUM_YEARS
    h_lo, h_hi = poisson_interval(MILLENNIUM_SOLVED, exposure)
    out["clay"] = {
        "hazard_ci95": [h_lo, h_hi],
        "rivalry_share_at_ci_upper_r05": rivalry_share(h_hi, 0.05),
        "rivalry_share_at_ci_lower_r05": rivalry_share(h_lo, 0.05),
        "problems": MILLENNIUM_PROBLEMS,
        "solved": MILLENNIUM_SOLVED,
        "years": MILLENNIUM_YEARS,
        "problem_years": MILLENNIUM_PROBLEMS * MILLENNIUM_YEARS,
        "hazard_per_year": h_hat,
        "rivalry_shares": shares,
        "implied_prize": implied,
        "diffusive_years_funded_by_1M_at_300k": 1.0e6 / 3.0e5,
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "numerics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
