/* Wavefront pruning of budgeted Brownian races: browser port.
   Mirrors brownianbandit/mean_field.py. Differences from the Python:
   - the normal CDF uses an erfc approximation good to ~1.2e-7
     fractional error, where scipy's ndtr is exact to double precision;
   - the Lagrangian is solved by vanilla Frank-Wolfe with exact line
     search, where the Python is fully corrective (SLSQP over the
     active set). Both converge to the same concave optimum; the
     measured objective discrepancy on the parity configuration is
     recorded in tests/test_js_parity.py. */

const BB = (function () {
  'use strict';

  /* erfc via the Chebyshev-fitted exponential (Numerical Recipes),
     fractional error below 1.2e-7 everywhere. */
  function erfc(x) {
    const z = Math.abs(x);
    const t = 1.0 / (1.0 + 0.5 * z);
    const ans = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 +
      t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
      t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? ans : 2.0 - ans;
  }
  function ndtr(z) { return 0.5 * erfc(-z / Math.SQRT2); }

  /* Uniform grid and banded row-stochastic Brownian transition.
     rows[j] = {start, probs} covering the in-band destinations, with
     edge cells 0 and n-1 always present via the edge extras. */
  function buildGrid(xMin, xMax, nSpace, dt, sigma) {
    const n = nSpace;
    const x = new Float64Array(n);
    const dx = (xMax - xMin) / (n - 1);
    for (let j = 0; j < n; j++) x[j] = xMin + j * dx;
    const stepSd = sigma * Math.sqrt(dt);
    const halfBand = Math.ceil(8.0 * stepSd / dx) + 2;

    const rows = new Array(n);
    for (let origin = 0; origin < n; origin++) {
      const lo = Math.max(0, origin - halfBand);
      const hi = Math.min(n - 1, origin + halfBand);
      const x0 = x[origin];
      const cols = [];
      const probs = [];
      const dests = [];
      if (lo > 0) dests.push(0);
      for (let d = lo; d <= hi; d++) dests.push(d);
      if (hi < n - 1) dests.push(n - 1);
      let total = 0;
      for (const dest of dests) {
        const lower = dest === 0 ? -Infinity : x[dest] - 0.5 * dx;
        const upper = dest === n - 1 ? Infinity : x[dest] + 0.5 * dx;
        const p = ndtr((upper - x0) / stepSd) - ndtr((lower - x0) / stepSd);
        if (p > 1e-16) { cols.push(dest); probs.push(p); total += p; }
      }
      for (let k = 0; k < probs.length; k++) probs[k] /= total;
      rows[origin] = { cols, probs };
    }
    return { x, dt, sigma, n, dx, rows };
  }

  /* y = T v (backward expectation step). */
  function applyT(grid, v) {
    const y = new Float64Array(grid.n);
    for (let j = 0; j < grid.n; j++) {
      const { cols, probs } = grid.rows[j];
      let s = 0;
      for (let k = 0; k < cols.length; k++) s += probs[k] * v[cols[k]];
      y[j] = s;
    }
    return y;
  }

  /* y = T' v (forward mass push). */
  function applyTt(grid, v) {
    const y = new Float64Array(grid.n);
    for (let j = 0; j < grid.n; j++) {
      const vj = v[j];
      if (vj === 0) continue;
      const { cols, probs } = grid.rows[j];
      for (let k = 0; k < cols.length; k++) y[cols[k]] += probs[k] * vj;
    }
    return y;
  }

  function pointMassInitial(grid, totalIntensity, x0) {
    const mass = new Float64Array(grid.n);
    const t = (x0 - grid.x[0]) / grid.dx;
    const left = Math.max(0, Math.min(grid.n - 2, Math.floor(t)));
    const w = t - left;
    mass[left] = totalIntensity * (1 - w);
    mass[left + 1] = totalIntensity * w;
    return mass;
  }

  /* Exact layer-cake value and pivotal-value gradient of the Poisson
     terminal maximum. */
  function terminalMaximumStats(mass, x, fallback) {
    const n = x.length;
    const tail = new Float64Array(n);
    let acc = 0;
    for (let j = n - 1; j >= 0; j--) { acc += Math.max(mass[j], 0); tail[j] = acc; }
    const widths = new Float64Array(n);
    let previous = fallback;
    for (let j = 0; j < n; j++) {
      if (x[j] > fallback) { widths[j] = x[j] - previous; previous = x[j]; }
    }
    const noPoint = new Float64Array(n);
    let value = fallback;
    const gradient = new Float64Array(n);
    let g = 0;
    for (let j = 0; j < n; j++) {
      noPoint[j] = Math.exp(-tail[j]);
      value += widths[j] * (1 - noPoint[j]);
      g += widths[j] * noPoint[j];
      gradient[j] = g;
    }
    return { value, gradient, tail, widths };
  }

  /* One-particle obstacle problem for a fixed shadow price, plus the
     population forward pass. Keep sets are upper tails (wavefront
     policies); cutoffs[k] is the kill-below level at step k. */
  function bestResponse(reward, lambda, grid, initial, nSteps) {
    const n = grid.n;
    let v = Float64Array.from(reward);
    const keepFirst = new Int32Array(nSteps);   // first kept index, n = keep nobody
    const cutoffs = new Float64Array(nSteps);
    for (let k = nSteps - 1; k >= 0; k--) {
      const cont = applyT(grid, v);
      for (let j = 0; j < n; j++) cont[j] -= lambda * grid.dt;
      // monotone accumulate to shave floating-point wiggles
      let m = -Infinity;
      let first = n;
      for (let j = 0; j < n; j++) {
        if (cont[j] > m) m = cont[j];
        cont[j] = m;
        if (first === n && m > 1e-14) first = j;
      }
      keepFirst[k] = first;
      cutoffs[k] = first === n ? Infinity :
        (first === 0 ? -Infinity : 0.5 * (grid.x[first - 1] + grid.x[first]));
      const vNew = new Float64Array(n);
      for (let j = first; j < n; j++) vNew[j] = Math.max(cont[j], 0);
      v = vNew;
    }
    let mass = Float64Array.from(initial);
    let pathTime = 0;
    const aliveCurve = new Float64Array(nSteps);
    for (let k = 0; k < nSteps; k++) {
      const alive = new Float64Array(n);
      let total = 0;
      for (let j = keepFirst[k]; j < n; j++) { alive[j] = mass[j]; total += mass[j]; }
      aliveCurve[k] = total;
      pathTime += grid.dt * total;
      mass = applyTt(grid, alive);
    }
    return { terminalMass: mass, pathTime, cutoffs, aliveCurve, keepFirst };
  }

  /* Population forward pass under a given wavefront policy, so a demo
     can display the exact expectations its sampled races should hit. */
  function propagateUnderCutoffs(keepFirst, grid, initial, nSteps) {
    const n = grid.n;
    let mass = Float64Array.from(initial);
    let pathTime = 0;
    const aliveCurve = new Float64Array(nSteps);
    for (let k = 0; k < nSteps; k++) {
      const alive = new Float64Array(n);
      let total = 0;
      for (let j = keepFirst[k]; j < n; j++) { alive[j] = mass[j]; total += mass[j]; }
      aliveCurve[k] = total;
      pathTime += grid.dt * total;
      mass = applyTt(grid, alive);
    }
    return { terminalMass: mass, pathTime, aliveCurve };
  }

  function combine(a, b, theta) {
    const n = a.terminalMass.length;
    const terminalMass = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      terminalMass[j] = (1 - theta) * a.terminalMass[j] + theta * b.terminalMass[j];
    }
    const aliveCurve = new Float64Array(a.aliveCurve.length);
    for (let k = 0; k < aliveCurve.length; k++) {
      aliveCurve[k] = (1 - theta) * a.aliveCurve[k] + theta * b.aliveCurve[k];
    }
    return {
      terminalMass,
      pathTime: (1 - theta) * a.pathTime + theta * b.pathTime,
      aliveCurve,
      cutoffs: theta > 0.5 ? b.cutoffs : a.cutoffs,
      keepFirst: theta > 0.5 ? b.keepFirst : a.keepFirst,
    };
  }

  /* max over theta in [0,1] of the Lagrangian along the segment, by
     golden section (the objective is concave). */
  function lineSearch(current, candidate, x, fallback, lambda) {
    const L = function (theta) {
      const mix = combine(current, candidate, theta);
      return terminalMaximumStats(mix.terminalMass, x, fallback).value -
        lambda * mix.pathTime;
    };
    const phi = (Math.sqrt(5) - 1) / 2;
    let a = 0, b = 1;
    let c = b - phi * (b - a), d = a + phi * (b - a);
    let fc = L(c), fd = L(d);
    for (let i = 0; i < 40; i++) {
      if (fc > fd) { b = d; d = c; fd = fc; c = b - phi * (b - a); fc = L(c); }
      else { a = c; c = d; fc = fd; d = a + phi * (b - a); fd = L(d); }
    }
    const theta = 0.5 * (a + b);
    // the endpoints are candidates too
    const ends = [0, theta, 1];
    let best = theta, bestVal = -Infinity;
    for (const t of ends) { const val = L(t); if (val > bestVal) { bestVal = val; best = t; } }
    return best;
  }

  /* Two atoms are the same if they have the same effect on this initial
     mass: equal path-time and (nearly) equal terminal intensity. Policies
     differing only where no mass ever goes collapse together. */
  function sameEffect(a, candidate, initialTotal) {
    if (Math.abs(a.pathTime - candidate.pathTime) > 1e-10) return false;
    let distance = 0;
    for (let j = 0; j < a.terminalMass.length; j++) {
      distance += Math.abs(a.terminalMass[j] - candidate.terminalMass[j]);
    }
    return distance <= 1e-8 * Math.max(1, initialTotal);
  }

  /* Frank-Wolfe with exact line search for a fixed shadow price. The
     solution carries its atom decomposition: deterministic wavefront
     policies with weights, so a sampled race can randomize over them
     per path and reproduce the mixture exactly in distribution. */
  function solveLagrangian(lambda, grid, initial, nSteps, fallback, tol, maxIter) {
    let current = {
      terminalMass: new Float64Array(grid.n),
      pathTime: 0,
      aliveCurve: new Float64Array(nSteps),
      cutoffs: new Float64Array(nSteps).fill(Infinity),
      keepFirst: new Int32Array(nSteps).fill(grid.n),
    };
    let initialTotal = 0;
    for (let j = 0; j < grid.n; j++) initialTotal += initial[j];
    let atoms = [{
      keepFirst: current.keepFirst,
      cutoffs: current.cutoffs,
      terminalMass: current.terminalMass,
      pathTime: 0,
      weight: 1,
    }];
    let dualGap = Infinity;
    for (let iter = 0; iter < maxIter; iter++) {
      const stats = terminalMaximumStats(current.terminalMass, grid.x, fallback);
      const candidate = bestResponse(stats.gradient, lambda, grid, initial, nSteps);
      let gap = -lambda * (candidate.pathTime - current.pathTime);
      for (let j = 0; j < grid.n; j++) {
        gap += stats.gradient[j] * (candidate.terminalMass[j] - current.terminalMass[j]);
      }
      dualGap = gap;
      const scale = Math.max(1, Math.abs(stats.value - lambda * current.pathTime));
      if (gap <= tol * scale) break;
      const theta = lineSearch(current, candidate, grid.x, fallback, lambda);
      const mixed = combine(current, candidate, theta);
      mixed.cutoffs = candidate.cutoffs; // representative wavefront policy
      mixed.keepFirst = candidate.keepFirst;
      current = mixed;
      for (const a of atoms) a.weight *= (1 - theta);
      let merged = false;
      for (const a of atoms) {
        if (sameEffect(a, candidate, initialTotal)) {
          a.weight += theta;
          merged = true;
          break;
        }
      }
      if (!merged && theta > 0) {
        atoms.push({
          keepFirst: candidate.keepFirst,
          cutoffs: candidate.cutoffs,
          terminalMass: candidate.terminalMass,
          pathTime: candidate.pathTime,
          weight: theta,
        });
      }
      atoms = atoms.filter(a => a.weight > 1e-9);
      let totalWeight = 0;
      for (const a of atoms) totalWeight += a.weight;
      for (const a of atoms) a.weight /= totalWeight;
    }
    // Recompute the gap at the final iterate: the loop's last gap belongs
    // to the iterate before its last update, so it would be stale here.
    const stats = terminalMaximumStats(current.terminalMass, grid.x, fallback);
    const finalCandidate = bestResponse(stats.gradient, lambda, grid, initial, nSteps);
    dualGap = -lambda * (finalCandidate.pathTime - current.pathTime);
    for (let j = 0; j < grid.n; j++) {
      dualGap += stats.gradient[j] * (finalCandidate.terminalMass[j] - current.terminalMass[j]);
    }
    dualGap = Math.max(dualGap, 0);
    const scale = Math.max(1, Math.abs(stats.value - lambda * current.pathTime));
    return {
      lambda,
      objective: stats.value,
      pathTime: current.pathTime,
      terminalMass: current.terminalMass,
      aliveCurve: current.aliveCurve,
      cutoffs: current.cutoffs,
      keepFirst: current.keepFirst,
      atoms,
      dualGap,
      converged: dualGap <= tol * scale,
      stats,
    };
  }

  /* Make a solution feasible for the budget and attach a valid
     suboptimality certificate. Feasibility: if the returned mixture
     costs more than the budget, thin it by mixing in the kill-all
     policy, so the returned expected cost is at most the budget.
     Certificate: for any price lambda >= 0 and any feasible m,
     V(B) - J(m) <= gap(m; lambda) + lambda * (B - C(m)), where gap is
     the fresh Frank-Wolfe gap at m. The slackness term is not
     optional: the gap alone bounds nothing at the requested budget. */
  function finalize(sol, target, grid, initial, nSteps, fallback, tol) {
    let s = sol;
    if (s.pathTime > target) {
      const theta = target / s.pathTime;
      const atoms = s.atoms.map(a => Object.assign({}, a, { weight: a.weight * theta }));
      atoms.push({
        keepFirst: new Int32Array(nSteps).fill(grid.n),
        cutoffs: new Float64Array(nSteps).fill(Infinity),
        terminalMass: new Float64Array(grid.n),
        pathTime: 0,
        weight: 1 - theta,
      });
      s = Object.assign({}, s, {
        terminalMass: Float64Array.from(s.terminalMass, v => theta * v),
        aliveCurve: Float64Array.from(s.aliveCurve, v => theta * v),
        atoms,
        pathTime: target,
      });
    }
    const stats = terminalMaximumStats(s.terminalMass, grid.x, fallback);
    const candidate = bestResponse(stats.gradient, s.lambda, grid, initial, nSteps);
    let gap = -s.lambda * (candidate.pathTime - s.pathTime);
    for (let j = 0; j < grid.n; j++) {
      gap += stats.gradient[j] * (candidate.terminalMass[j] - s.terminalMass[j]);
    }
    gap = Math.max(gap, 0);
    const budgetSlack = s.lambda * Math.max(0, target - s.pathTime);
    // `converged` reports whether the inner Frank-Wolfe solves reached
    // their tolerance; `certificate` is the accuracy statement for the
    // returned policy at the requested budget, and can be materially
    // larger than tol (a mixture of two prices is not stationary at
    // either, and an iteration-capped solve carries its residual gap).
    return Object.assign({}, s, {
      objective: stats.value,
      stats,
      dualGap: gap,
      budgetSlack,
      certificate: gap + budgetSlack,
      converged: s.converged !== false,
    });
  }

  /* Dual bisection on the shadow price to hit the requested expected
     path-time budget; mixes the bracketing solutions if the dual cost
     curve jumps over the target. The returned policy costs at most the
     budget, and `certificate` bounds its suboptimality at that budget. */
  function solveForBudget(target, grid, initial, nSteps, fallback, opts) {
    const o = Object.assign({ tol: 1e-5, maxIter: 60, budgetTol: 2e-3, bisections: 22 }, opts);
    const done = function (sol) {
      return finalize(sol, target, grid, initial, nSteps, fallback, o.tol);
    };
    const solve = function (lambda) {
      return solveLagrangian(lambda, grid, initial, nSteps, fallback, o.tol, o.maxIter);
    };
    let lowPrice = 0;
    let lowSol = solve(0);
    if (lowSol.pathTime <= target) return done(lowSol);
    let highPrice = 1;
    let highSol = solve(1);
    while (highSol.pathTime > target) {
      highPrice *= 2;
      if (highPrice > 1e8) throw new Error('failed to bracket the budget multiplier');
      highSol = solve(highPrice);
    }
    let best = Math.abs(lowSol.pathTime - target) < Math.abs(highSol.pathTime - target) ?
      lowSol : highSol;
    const absTol = o.budgetTol * Math.max(1, target);
    for (let i = 0; i < o.bisections; i++) {
      const mid = 0.5 * (lowPrice + highPrice);
      const midSol = solve(mid);
      if (Math.abs(midSol.pathTime - target) < Math.abs(best.pathTime - target)) best = midSol;
      if (Math.abs(midSol.pathTime - target) <= absTol) return done(midSol);
      if (midSol.pathTime > target) { lowPrice = mid; lowSol = midSol; }
      else { highPrice = mid; highSol = midSol; }
    }
    if (Math.abs(best.pathTime - target) <= absTol) return done(best);
    // mix the bracketing solutions to hit the expected budget exactly;
    // the certificate is recomputed for the mixture itself in finalize
    // (the endpoint gaps do not certify the mixture)
    const span = lowSol.pathTime - highSol.pathTime;
    const theta = span > 1e-14 ? (lowSol.pathTime - target) / span : 0;
    const mix = combine(lowSol, highSol, theta);
    const atoms = [];
    for (const a of lowSol.atoms) {
      atoms.push(Object.assign({}, a, { weight: a.weight * (1 - theta) }));
    }
    for (const a of highSol.atoms) {
      atoms.push(Object.assign({}, a, { weight: a.weight * theta }));
    }
    return done({
      lambda: 0.5 * (lowSol.lambda + highSol.lambda),
      objective: NaN,
      pathTime: mix.pathTime,
      terminalMass: mix.terminalMass,
      aliveCurve: mix.aliveCurve,
      cutoffs: mix.cutoffs,
      keepFirst: mix.keepFirst,
      atoms: atoms.filter(a => a.weight > 1e-9),
      converged: lowSol.converged && highSol.converged,
    });
  }

  function propagateWithoutPruning(mass, grid, nSteps) {
    let m = Float64Array.from(mass);
    for (let k = 0; k < nSteps; k++) m = applyTt(grid, m);
    return m;
  }

  function staticThinningBaseline(target, grid, initial, nSteps, fallback) {
    const horizon = nSteps * grid.dt;
    let total = 0;
    for (let j = 0; j < grid.n; j++) total += initial[j];
    const maxCost = total * horizon;
    const fraction = maxCost <= 0 ? 0 : Math.min(1, Math.max(0, target / maxCost));
    const retained = Float64Array.from(initial, function (v) { return fraction * v; });
    const terminal = propagateWithoutPruning(retained, grid, nSteps);
    const stats = terminalMaximumStats(terminal, grid.x, fallback);
    return { objective: stats.value, pathTime: fraction * maxCost };
  }

  function oneShotScreeningBaseline(target, grid, initial, nSteps, fallback) {
    let total = 0;
    for (let j = 0; j < grid.n; j++) total += initial[j];
    const horizon = nSteps * grid.dt;
    let current = Float64Array.from(initial);
    let best = null;
    for (let screenStep = 0; screenStep < nSteps; screenStep++) {
      const screenTime = screenStep * grid.dt;
      const remainingHorizon = horizon - screenTime;
      const spent = total * screenTime;
      const remainingBudget = target - spent;
      if (remainingBudget < -1e-12) break;
      let alive = 0;
      for (let j = 0; j < grid.n; j++) alive += current[j];
      const retainedIntensity = Math.min(alive, Math.max(0, remainingBudget / remainingHorizon));
      const retained = new Float64Array(grid.n);
      let remaining = retainedIntensity;
      for (let j = grid.n - 1; j >= 0 && remaining > 1e-13; j--) {
        const take = Math.min(current[j], remaining);
        retained[j] = take;
        remaining -= take;
      }
      const terminal = propagateWithoutPruning(retained, grid, nSteps - screenStep);
      const stats = terminalMaximumStats(terminal, grid.x, fallback);
      const candidate = {
        objective: stats.value,
        pathTime: spent + retainedIntensity * remainingHorizon,
        screeningTime: screenTime,
      };
      if (!best || candidate.objective > best.objective) best = candidate;
      current = applyTt(grid, current);
    }
    return best || { objective: fallback, pathTime: 0, screeningTime: null };
  }

  return {
    buildGrid,
    pointMassInitial,
    terminalMaximumStats,
    bestResponse,
    propagateUnderCutoffs,
    solveLagrangian,
    solveForBudget,
    staticThinningBaseline,
    oneShotScreeningBaseline,
    ndtr,
  };
})();
if (typeof window !== 'undefined') window.BB = BB;
if (typeof module !== 'undefined') module.exports = BB;
