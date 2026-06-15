# Next-session prompt (copy-paste to continue RaoRocketSim)

Continue RaoRocketSim. Analyze the repo state first per project rules:
read the STATUS blocks at the top of JAX_DIFFERENTIABLE_PLAN.md
(STATUS-HEAD, θ_N RECONCILIATION 2026-06-11f, DIRECTION) and the session
memory — they carry the full record. Quick context:

- Defaults are FLIPPED (DIRECTION 2 complete): solver_backend="jax",
  formulation="characteristic", pin_d_theta=False, ladder (1,10,30,100).
  Full suite: 781 passed / 9 xfailed. J4 gate 7.50e-4. J5 loose gate
  passes (record: builds/j5_chart_sweep.json). J6 v1 shipped
  (raosim/jax/sensitivities.py; known-sign gate dCf/dpa =
  −(r_E²−r_D²)/Rt² pinned).
- θ_N RECONCILIATION (read that STATUS paragraph first — it unifies the
  flare, the 5.7e-2 full-pin floor, and the J5 θ_E gap): chart θ_N=30°
  at ε=10/L80 is the optimum (Rao ARS J. 1961 pp. 1490-91, γ=1.23
  charts, contour γ-insensitive at fixed (ε,L)); the fixed-end seed
  freezes the kernel at the sub-optimal 25.5° and the inner set_theta_b
  secant overrides any outer θ_B iteration — that's why full-continuity
  stalled at 5.7e-2 regardless of θ_B guess. The new
  `RaoSolverConfig.theta_b_freeze_deg` bypasses the inner secant.

TODAY'S CRITICAL PATH — run and interpret the decisive experiment:
1. On this Mac (sandbox can't run JAX solves; bash calls cap ~42s and
   detached jobs get reaped — have the USER run host commands and paste
   output):
       PYTHONPATH=. python scripts/theta_b_freeze_sweep.py
   Prediction (from the reconciliation): the full-pin floor collapses
   near θ_B ≈ 30°, the 2e-3 gate closes WITH full D-state continuity,
   the BDE wall peaks ≈30° just after the throat (mid-bell flare gone),
   exit angle → chart θ_E 15.5°.
2. If the prediction holds:
   a. Wire it in: outer θ_B secant (on the inner floor) around the BVP
      as a config-driven option (pattern: scripts/theta_b_picard_probe.py
      + the freeze knob), or fold θ_B into the iteration properly (J3b).
   b. De-xfail test_bde_wall_is_bell_shaped (already re-grounded to
      chart values via lookup_angles).
   c. Report solved θ_N (= converged frozen θ_B) instead of the chart
      lookup in solve_rao_bvp under the characteristic formulation —
      de-circularizes the J5 benchmark; then re-run
      scripts/j5_chart_sweep.py and check the low-ε θ_E gap closed;
      plan-target gate (1.5°/3°) may now be reachable
      (test_rao_chart_benchmark_plan_targets xfail).
   d. Regenerate figures: scripts/generate_contour.py (gate-passing
      config), scripts/plot_reference_topology.py.
3. If the floor does NOT collapse anywhere in 26-31°: the Guderley
   branch is live — the fixed-(L,ε) optimum may be genuinely
   discontinuous at D. Before any further smoothing work, check
   propulsion_texts for Guderley & Hantsch 1955 ("Beste Formen für
   achsensymmetrische Überschallschubdüsen") and the ~1968 short-nozzle
   results (also: 978-3-7091-4745-0_18.pdf and the EUCASS/variational
   PDFs in propulsion_texts may cover it). Then decide: model the
   discontinuity explicitly (corner expansion at D) vs fixed-length
   transversality blocks.
4. Queue after that (DIRECTION order, tool-first): J3b lax.scan
   differentiable march (θ_B as solved unknown; also unlocks J6 v2
   design totals + dCf on bell-wall nodes), Phase 13 #8
   plot_net_diagnostics, raosim/jax absorption (DIRECTION 1) only after
   full form.

Sandbox notes: pip install --break-system-packages -r requirements.txt
pytest matplotlib; env PYTHONPATH=. from repo root; bash ≤42s per call,
NO detached/background jobs (bwrap per-call --unshare-pid since
2026-06-11 — verify with a touch-marker probe before trusting either
mode); pgrep -f gives false ALIVE (matches its own wrapper cmdline).
.git/index.lock: any sandbox git status leaves one the sandbox cannot
remove — use `git --no-optional-locks`, and have the user
`rm .git/index.lock` before committing. Working tree may carry
uncommitted work — check `git --no-optional-locks status` first.
M3.5Perf oracle suites (tests/test_nasa_kernel_march_parity.py,
tests/test_nasa_port.py) must stay green through ANY numerics change.
