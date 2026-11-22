# Next-session prompt (copy-paste to continue RaoRocketSim)

Continue RaoRocketSim. Analyze the repo state first per project rules:
read the head STATUS paragraphs of JAX_DIFFERENTIABLE_PLAN.md (the
2026-06-11h characteristic-pairing correction + the follow-through) and
the session memory — they carry the full record. Where we are:

- THE MILESTONE LANDED: after the characteristic-pairing correction
  (the stack enforced the C− relation along the C+ CE/DE + a spurious
  cosμ/cos(θ±μ) source factor; corrected per Anderson §11.4 and
  oracle-proven — tests/test_characteristic_pairing.py), FULL D-state
  continuity is the DEFAULT (pin_d_theta=pin_d_mach=True), the J4
  residual gate passes at the default config, the BDE region closes
  exactly at the commanded exit, and the bell-shape regression is
  grounded in the smooth stationary-DE existence root
  (θ_B=25.5659°, kdf=0.15216, exit θ=11.1193°).  Full suite:
  **790 passed / 5 xfailed / 0 failed.**  Position-only attachment is
  a demoted diagnostic.  The Guderley-jump hypothesis is shelved.
  Defaults: solver_backend="jax", formulation="characteristic", ladder
  (1,10,30,100).  Chart values (θ_N=30°/θ_E=15.5° at ε=10/L80) are
  Rao-1960 PARABOLA-FIT parameters — exact-variational deltas vs the
  charts are expected and documented, not bugs.

TODAY'S QUEUE (DIRECTION order, tool-first):
1. J5 de-circularization: report the SOLVED θ_B as sol.theta_N under
   the characteristic formulation (today theta_N is still the chart
   lookup via _design_angles_rad — circular in the benchmark); keep
   the chart as comparison data.  Then host re-run
   scripts/j5_chart_sweep.py: expect the low-ε θ_E gap to shrink
   (it was the shadow of the wrong-relation/shallow-θ_B solutions)
   and document the systematic exact-vs-parabola-chart deltas as the
   benchmark's finding, not its failure.  Revisit
   test_rao_chart_benchmark_plan_targets (xfail) tolerances
   accordingly — the right ground truth question is now "what does
   the EXACT variational solution say across the grid".
2. Regenerate deliverable figures on host: scripts/generate_contour.py
   (default config now = full-continuity gate config),
   scripts/plot_reference_topology.py, scripts/j4_gate_wall_probe.py
   (update its hardcoded pin_d_theta=False to the new default or
   delete it in favour of the suite).
3. J3b — the lax.scan differentiable kernel march (θ_B as a solved
   unknown inside the BVP instead of the seed's secant; unlocks J6 v2:
   total design derivatives d(outputs)/d(Rt, ε, L%, γ) via optimistix
   IFT with args-lifted assembly constants, and dCf on bell-wall
   nodes through a differentiable calc_bde_region).
4. Phase 13 #8 plot_net_diagnostics (needs RaoResidualReport link
   plumbing); then the §11.7→12.7 topology wiring (RaoTopology as the
   internal representation in solve_rao_bvp's export path).
5. raosim/jax absorption (DIRECTION 1) only after the core is in its
   fullest form.

Sandbox notes: pip install --break-system-packages -r requirements.txt
pytest matplotlib; env PYTHONPATH=. from repo root; bash ≤42s per call,
NO background jobs (bwrap per-call --unshare-pid; pgrep -f self-matches
— use file markers); JAX solves are HOST-ONLY (user runs and pastes).
git: use `git --no-optional-locks`; user must `rm .git/index.lock`
before committing (sandbox git leaves locks it cannot remove).
M3.5Perf oracle suites (test_nasa_kernel_march_parity, test_nasa_port)
+ tests/test_characteristic_pairing.py must stay green through ANY
numerics change.  Project rule: ground every equation in
propulsion_texts (Rao 1961 review = RaoRecentDevinRockNozConfig.pdf)
before trusting or changing it.
