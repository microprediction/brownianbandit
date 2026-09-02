/* Runs the JS port on the parity configuration and prints JSON.
   Invoked by tests/test_js_parity.py; compare against the Python. */
const path = require('path');
const BB = require(path.join(__dirname, '..', 'docs', 'bb_core.js'));

const nSteps = 40;
const grid = BB.buildGrid(-6, 6, 301, 1 / nSteps, 1.0);
const initial = BB.pointMassInitial(grid, 100.0, 0.0);

const reward = Float64Array.from(grid.x, x => Math.max(x - 1.5, 0));
const policy = BB.bestResponse(reward, 0.02, grid, initial, nSteps);
let linearValue = -0.02 * policy.pathTime;
for (let j = 0; j < grid.n; j++) linearValue += reward[j] * policy.terminalMass[j];

const lag = BB.solveLagrangian(0.08, grid, initial, nSteps, 0.0, 1e-7, 200);

const sol = BB.solveForBudget(10.0, grid, initial, nSteps, 0.0,
  { tol: 1e-6, maxIter: 100, budgetTol: 2e-3, bisections: 25 });
const st = BB.staticThinningBaseline(10.0, grid, initial, nSteps, 0.0);
const os = BB.oneShotScreeningBaseline(10.0, grid, initial, nSteps, 0.0);

console.log(JSON.stringify({
  best_response_path_time: policy.pathTime,
  best_response_linear_value: linearValue,
  lagrangian_objective: lag.objective,
  budget_objective: sol.objective,
  budget_path_time: sol.pathTime,
  static_objective: st.objective,
  one_shot_objective: os.objective,
}));
