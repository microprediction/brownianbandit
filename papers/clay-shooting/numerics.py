"""Numerical certificates for "Clay Shooting: Equilibrium Effort Solving
Outstanding Problems".

Every number quoted in the paper is produced here. Run

    python papers/clay-shooting/numerics.py

to regenerate papers/clay-shooting/numerics.json.

The sections mirror the paper: the price of a success probability and its
Monte Carlo check; the equilibrium attempt size with endogenous widths,
finite and large-pool, with a unilateral-deviation certificate; the
committed-width benchmark it replaces, including the profitable deviation
that condemns it and the pure-equilibrium selection problem; the exact
purse design; and the Millennium calibration.

Depends on numpy and scipy.
"""

from __future__ import annotations

import json
import math
import os
from math import comb

import numpy as np

SEED = 20260909


# --------------------------------------------------------------- utilities

def bisect(f, lo, hi, iters: int = 400) -> float:
    """Bisection on a sign change, no bracket check (callers supply one)."""
    f_lo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f_lo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def G(y: float) -> float:
    """(1 - exp(-y)) / y, strictly decreasing from 1 to 0."""
    return (1.0 - math.exp(-y)) / y


def H(y: float) -> float:
    return 1.0 - math.exp(-y)


# --------------------------------- the price of a success probability

def price_of_probability(p: float, zeta_sq: float) -> float:
    """k(p) = zeta^2 p / (1 - p): the cost of buying success probability p."""
    return zeta_sq * p / (1.0 - p)


def attempt_from_width(L: float, c: float, sigma: float, d: float):
    """(p, k) of an attempt of continuation width L, by gambler's ruin."""
    if L <= d:
        return 0.0, 0.0
    return 1.0 - d / L, c * d * (L - d) / sigma ** 2


def optimal_size(A: float, zeta_sq: float):
    """Best response to effective reward A: p, k, and value."""
    if A <= zeta_sq:
        return 0.0, 0.0, 0.0
    p = 1.0 - math.sqrt(zeta_sq / A)
    return p, p * (1.0 - p) * A, A * p * p


def monte_carlo_attempt(A, c, sigma, d, n_paths=50_000, dt=2e-3, rng=None):
    """Euler simulation of the attempt sized optimally against reward A."""
    rng = rng or np.random.default_rng(SEED)
    L = sigma * math.sqrt(A / c)
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
        "n_paths": n_paths, "dt": dt,
        "p_hat": float(solved.mean()),
        "p_se": float(solved.std(ddof=1) / math.sqrt(n_paths)),
        "k_hat": float(c * elapsed.mean()),
        "k_se": float(c * elapsed.std(ddof=1) / math.sqrt(n_paths)),
    }


# ------------------------------------- equilibrium with endogenous widths

def effective_reward(p: float, n: int, V: float) -> float:
    """A = V E[1/(1+K)], K ~ Binomial(n-1, p), all n researchers working."""
    if p <= 0.0:
        return V
    return V * (1.0 - (1.0 - p) ** n) / (n * p)


def equilibrium_finite(n: int, V: float, zeta_sq: float) -> float:
    """The unique symmetric equilibrium success probability."""
    def phi(p):
        return optimal_size(effective_reward(p, n, V), zeta_sq)[0] - p
    if phi(1e-15) <= 0.0:
        return 0.0
    return bisect(phi, 1e-15, 1.0 - 1e-15)


def deviation_gain(p_eq: float, n: int, V: float, zeta_sq: float) -> float:
    """Largest unilateral gain from re-sizing against the equilibrium rivals.

    A Nash profile has zero gain; this is the diagnostic the companion paper
    asks for, and it is a real test because the grid search does not know
    the closed-form best response.
    """
    A = effective_reward(p_eq, n, V)
    util = lambda p: A * p - price_of_probability(p, zeta_sq)
    grid = np.linspace(0.0, 1.0 - 1e-9, 2_000_001)
    return float(np.max(A * grid - zeta_sq * grid / (1.0 - grid)) - util(p_eq))


def large_pool(V_over_entry: float) -> dict:
    """G(y*) = V_entry / V: the large-pool equilibrium."""
    if V_over_entry <= 1.0:
        return {"y_star": 0.0, "solve_probability": 0.0}
    y = bisect(lambda y: G(y) - 1.0 / V_over_entry, 1e-14, 1e6)
    return {"V_over_entry": V_over_entry, "y_star": y,
            "solve_probability": H(y)}


def with_setup_cost(V: float, zeta: float, K: float) -> dict:
    """Entry margin restored by a setup cost K: A* = (zeta + sqrt(K))^2."""
    V_entry = (zeta + math.sqrt(K)) ** 2
    y = bisect(lambda y: G(y) - V_entry / V, 1e-14, 1e6)
    p = math.sqrt(K) / (zeta + math.sqrt(K))
    A = V * G(y)
    k = zeta * math.sqrt(K)
    N = y / p
    return {
        "V": V, "zeta_sq": zeta ** 2, "K": K, "V_entry": V_entry,
        "y_star": y, "p_star": p, "entrants": N,
        "A_star": A, "A_star_predicted": V_entry,
        "attempt_cost": p * (1 - p) * A, "attempt_cost_predicted": k,
        "entrant_payoff": A * p * p - K,
        "aggregate_effort": N * (k + K),
        "expected_payout": V * H(y),
    }


# --------------------------------------- the committed-width benchmark

def committed_width_deviation(V, c, sigma, d):
    """The reviewer's example: the solo width is not a best response."""
    L_solo = sigma * math.sqrt(V / c)
    p_solo, _ = attempt_from_width(L_solo, c, sigma, d)
    A = V * (1.0 - p_solo / 2.0)          # n = 2, one rival at p_solo
    util = lambda L: A * attempt_from_width(L, c, sigma, d)[0] \
        - attempt_from_width(L, c, sigma, d)[1]
    L_br = sigma * math.sqrt(A / c)
    return {
        "V": V, "c": c, "sigma": sigma, "d": d,
        "L_solo": L_solo, "p_solo": p_solo, "A": A,
        "L_best_response": L_br,
        "payoff_at_solo_width": util(L_solo),
        "payoff_at_best_response": util(L_br),
        "gain": util(L_br) - util(L_solo),
    }


def pure_entrant_payoff(m: int, p: float, V: float = 1.0) -> float:
    """Payoff to one of m entrants in the committed-width lottery."""
    return V * (1.0 - (1.0 - p) ** m) / m - p * (1.0 - p) * V


def potential(m: int, p: float, V: float = 1.0) -> float:
    """The companion paper's exact potential at m symmetric entrants."""
    harmonic = lambda j: sum(1.0 / l for l in range(1, j + 1))
    e_h = sum(comb(m, j) * p ** j * (1 - p) ** (m - j) * harmonic(j)
              for j in range(m + 1))
    return V * e_h - m * p * (1.0 - p) * V


def committed_entrants(p: float, n: int) -> float:
    """Expected entrants n q at the interior symmetric mixed equilibrium."""
    g = lambda q: (1.0 - (1.0 - p * q) ** n) - n * q * p * (1.0 - p)
    if g(1.0) >= 0.0:
        return float(n)
    return n * bisect(g, 1e-13, 1.0)


# ------------------------------------------------------------ purse design

def purse_design(power: int) -> dict:
    """Maximise Psi = B H(y)^(power+1) / (V_e^? y^power).

    power = 1 is the endogenous-width model (V proportional to y/H),
    power = 2 the committed-width model (V proportional to (y/H)^2).
    The first-order condition (power+1)H'/H = power/y reduces to
    exp(y) = 1 + (1 + 1/power) y, and then V/V_e = (y + 1/slope)^power.
    """
    slope = 1.0 + 1.0 / power
    y = bisect(lambda y: math.exp(y) - 1.0 - slope * y, 1e-9, 20.0)
    return {
        "power": power,
        "condition": "exp(y) = 1 + %.1f y" % slope,
        "y_star": y,
        "purse_over_break_even": (y / H(y)) ** power,
        "purse_closed_form": (y + 1.0 / slope) ** power,
        "solve_probability": H(y),
        "problems_per_capital": (H(y) / y) ** power,     # times B / V_e
        "solved_per_capital": H(y) ** (power + 1) / y ** power,
    }


def purse_design_bruteforce(power: int, n_grid: int = 2_000_000) -> float:
    y = np.linspace(1e-6, 12.0, n_grid)
    obj = H_vec = 1.0 - np.exp(-y)
    obj = H_vec ** (power + 1) / y ** power
    return float(y[int(np.argmax(obj))])


# -------------------------------------------------------------- Millennium

PROBLEMS = 7
SOLVED = 1
POSTED = 2000
POINCARE = 2003
NOW = 2026


def poisson_interval(count, exposure, level=0.95):
    from scipy.stats import chi2
    alpha = 1.0 - level
    lo = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2.0
    hi = chi2.ppf(1.0 - alpha / 2, 2 * (count + 1)) / 2.0
    return lo / exposure, hi / exposure


def identify(k_attempt: float, p: float) -> dict:
    """Illustrative arithmetic for the surplus identity, NOT an estimate.

    (k, p) pins zeta^2 = k(1-p)/p by gambler's ruin alone; if the attempt
    was also sized optimally then A = k/(p(1-p)) and the surplus over the
    walk-away threshold is A - zeta^2 = k(2-p)/(1-p) >= 2k. The point of
    the arithmetic is that the surplus scales with the cost of ONE attempt,
    whatever the level of the reward.
    """
    zeta_sq = k_attempt * (1.0 - p) / p
    A = k_attempt / (p * (1.0 - p))
    return {
        "attempt_cost": k_attempt, "p": p,
        "zeta_sq": zeta_sq, "V_min": zeta_sq,
        "A": A, "A_over_V_min": A / zeta_sq,
        "surplus": A - zeta_sq,
        "surplus_identity": k_attempt * (2.0 - p) / (1.0 - p),
        "surplus_over_attempt_cost": (A - zeta_sq) / k_attempt,
        "purse_over_surplus": 1.0e6 / (A - zeta_sq),
        "purse_over_twice_cost": 1.0e6 / (2.0 * k_attempt),
        "purse_over_level": 1.0e6 / A,
    }


def purse_counterfactual(k_attempt: float, p: float,
                         purse: float = 1.0e6) -> dict:
    """Expected solvers with and without the cash purse in the reward."""
    zeta_sq = k_attempt * (1.0 - p) / p
    A = k_attempt / (p * (1.0 - p))
    out = {}
    for tag, V in (("with_purse", A), ("without_purse", A - purse)):
        if V <= zeta_sq:
            out[tag] = {"y_star": None, "solve_probability": None,
                        "note": "below the walk-away threshold"}
        else:
            y = bisect(lambda y: G(y) - zeta_sq / V, 1e-14, 1e6)
            out[tag] = {"y_star": y, "solve_probability": H(y)}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED}

    # ---- 1. the price of a success probability
    c, sigma, d = 1.0, 1.0, 4.0
    zeta = d * math.sqrt(c) / sigma
    zeta_sq = zeta ** 2
    widths = []
    for L in (5.0, 6.0, 8.0, 10.0, 20.0):
        p, k = attempt_from_width(L, c, sigma, d)
        widths.append({"L": L, "p": p, "k": k,
                       "price_formula": price_of_probability(p, zeta_sq),
                       "residual": k - price_of_probability(p, zeta_sq)})
    A0 = 100.0
    p0, k0, v0 = optimal_size(A0, zeta_sq)
    out["price"] = {
        "c": c, "sigma": sigma, "d": d, "zeta": zeta, "zeta_sq": zeta_sq,
        "widths": widths,
        "optimal_against_A": {
            "A": A0, "p": p0, "k": k0,
            "k_identity_p_1_minus_p_A": p0 * (1 - p0) * A0,
            "value": v0, "value_identity_A_p_sq": A0 * p0 ** 2,
            "value_identity_sqrt_form": (math.sqrt(A0) - zeta) ** 2,
            "max_spend_share_of_reward": 0.25,
        },
        "monte_carlo": monte_carlo_attempt(A0, c, sigma, d, rng=rng),
    }

    # ---- 2. equilibrium with endogenous widths
    rows = []
    for V_ratio in (1.5, 2.0, 4.0, 10.0, 100.0):
        V = V_ratio * zeta_sq
        for n in (1, 2, 3, 5, 10, 100, 1000, 10_000):
            p = equilibrium_finite(n, V, zeta_sq)
            A = effective_reward(p, n, V)
            rows.append({
                "V_over_V_min": V_ratio, "n": n, "p_star": p,
                "n_p": n * p, "A_star": A, "A_over_V_min": A / zeta_sq,
                "payoff": A * p * p,
                "solve_probability": 1.0 - (1.0 - p) ** n,
            })
    out["equilibrium_finite"] = rows
    out["equilibrium_limit"] = [large_pool(v)
                                for v in (1.02, 1.5, 2.0, 4.0, 10.0, 100.0)]
    out["deviation_certificate"] = [
        {"n": n, "V_over_V_min": vr,
         "p_star": equilibrium_finite(n, vr * zeta_sq, zeta_sq),
         "max_unilateral_gain": deviation_gain(
             equilibrium_finite(n, vr * zeta_sq, zeta_sq), n,
             vr * zeta_sq, zeta_sq)}
        for n, vr in ((2, 4.0), (3, 2.0), (10, 10.0))
    ]
    out["setup_cost"] = [with_setup_cost(100.0, 1.0, K)
                         for K in (1.0, 0.01, 1e-4)]

    # ---- 3. the committed-width benchmark it replaces
    out["committed_width"] = {
        "profitable_deviation": committed_width_deviation(100., 1., 1., 5.),
        "pure_equilibria": [
            {"p": p,
             "incumbent_of_three": pure_entrant_payoff(3, p),
             "p_cubed_over_three": p ** 3 / 3.0,
             "fourth_entrant": pure_entrant_payoff(4, p),
             "fourth_formula": p * p * (-0.5 + p - p * p / 4.0),
             "potential_argmax": max(range(12),
                                     key=lambda m: potential(m, p))}
            for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9)
        ],
        "pure_threshold_2_minus_sqrt2": 2.0 - math.sqrt(2.0),
        "mixed_entrants_small_p": [
            {"n": n, "n_q_at_p_1e_3": committed_entrants(1e-3, n),
             "limit_2n_over_n_minus_1": 2.0 * n / (n - 1.0)}
            for n in (4, 10, 50, 1000)
        ],
    }

    # ---- 4. purse design
    out["purse_design"] = {
        "endogenous": {**purse_design(1),
                       "brute_force_y": purse_design_bruteforce(1)},
        "committed": {**purse_design(2),
                      "brute_force_y": purse_design_bruteforce(2)},
    }

    # ---- 5. Millennium
    exposure_naive = PROBLEMS * (NOW - POSTED)
    exposure = (PROBLEMS - SOLVED) * (NOW - POSTED) + (POINCARE - POSTED)
    h_hat = SOLVED / exposure
    lo, hi = poisson_interval(SOLVED, exposure)
    shares = []
    for r in (0.03, 0.05, 0.10):
        row = {"r": r,
               "aggregate_share": h_hat / (r + h_hat),
               "aggregate_share_ci_upper": hi / (r + hi)}
        for N in (2, 3, 5):
            hr = h_hat * (N - 1) / N
            row["rival_share_N%d" % N] = hr / (r + hr)
        shares.append(row)
    out["millennium"] = {
        "exposure_naive_7x26": exposure_naive,
        "exposure_corrected": exposure,
        "hazard_per_year": h_hat,
        "hazard_ci95": [lo, hi],
        "shares": shares,
        "surplus_identity_illustration": [identify(k, p)
                                          for k, p in ((1.0e6, 0.01),
                                                       (0.2e6, 0.01),
                                                       (1.0e7, 0.01),
                                                       (1.0e6, 0.5))],
        "counterfactual": {
            "k_%.1gM_p_%.3g" % (k / 1e6, p): purse_counterfactual(k, p)
            for k, p in ((0.2e6, 0.01), (1.0e6, 0.01),
                         (1.0e6, 0.005), (1.0e7, 0.01))
        },
        "purse_rationalises_at_most": 1.0e6 / 4.0,
    }

    # ---- 6. sensitivity to the purse
    out["sensitivity"] = [
        {"p": p, "multiple": m,
         "p_new": 1.0 - (1.0 - p) / math.sqrt(m),
         "dp_dlogV": (1.0 - p) / 2.0}
        for p in (0.01, 0.1, 0.3) for m in (2, 10)
    ]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "numerics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
