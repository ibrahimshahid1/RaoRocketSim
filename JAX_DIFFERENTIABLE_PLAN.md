# RaoRocketSim — Differentiable (JAX) Rao Nozzle Tool

> **J3b COMPLETE — BIT-PARITY MARCH + θ_B IN THE BVP; Phase 13 10/10;
> §12.7 topology export wired (2026-06-12, continuation of the J3b-1
> session below).**  The "consistency defect" was structural and is
> FIXED: the all-special prescribed-grid assumption was wrong — NASA's
> march mixes ~95% special insertions with ~5% RAW rows (at the
> reference: marched-row indices 2/9/21/39/65, wall θ 0.56°–16.7°)
> where prev[1]'s C+ **terminates on the arc** and becomes the wall
> point, shifting the interior pairing by one.  Termination is
> physics (every C+ ends on the wall); skipping it starves the
> wall-adjacent strip — measured ν_w−θ_w → −0.67° at B vs the
> oracle's +0.68° (axisymmetric wall expansion must run AHEAD of
> planar PM), and an 8% converged-to-wrong s15.  **Landed march
> (`raosim/jax/moc_kernel.py`):** NASA dθ-form unit-process ports
> (`interior_point_nasa` with the on-axis coefficient substitution,
> `axis_point_nasa` one-sided source, `special_wall_point_nasa`,
> `arc_wall_point_raw_nasa` — all parity ≤1e-12 on oracle rows; note
> the C++'s SIGNED `B1 <= R1` selection in the raw point, ported
> as-is) + the exact raw/special row policy under static padded
> shapes with TRACED valid width and pairing offset.  **Result: BD
> bit-parity with `build_kernel` — max|ΔM| 4.8e-10 over all 120
> nodes; d(BD)/dθ_B exact (jacfwd vs FD ≤1.1e-8, sensitivity flows
> only through the final clamped row; smooth within a 0.25° grid
> window, re-centred per rung like kernel_bd re-seeding).**
> `tests/test_jax_moc_kernel.py`: 9 passed (xfail acceptance FLIPPED
> to a hard 1e-8 parity gate).  **J3b-2 LANDED:**
> `raosim/jax/theta_b_solve.py` — θ_B as a live BVP unknown: the
> residual recomputes BD in-graph per evaluation; wiring exploits
> `StaticParams._replace` with traced BD arrays through the untouched
> J2-parity assembly (padded tails are ds=0 segments the flux rules
> already mask; the arc-length cumsum plateaus so kdf parametrises
> the valid arc).  Gates: live-vs-static residual parity at the seed
> ≤1e-6 (automatic via BD bit-parity), d(residual)/dθ_B FD-exact,
> end-to-end LM smoke (29 s coarse).  Production opt-in
> `RaoSolverConfig.solve_theta_b=True` (JAX backend): θ_B joins the
> unknown vector with ±dθ_limit/4 bounds, the kernel is re-frozen at
> the LM-solved angle with provenance `"bvp_solved"` (flows into the
> J5 θ_N reporting automatically); verified end-to-end in-sandbox
> (7:08, @slow: solved θ_N 25.57° ± window at the reference).
> **The decisive physics experiment is now runnable on host:** full
> D-state pins + `solve_theta_b=True` at the reference — does the
> stationarity floor collapse with θ_B live, or does the Guderley
> branch stand?  **Phase 13 #8 LANDED:** `plot_net_diagnostics`
> (spec plot 10/10) — per-link C± residuals via the existing
> `characteristic_net_links`/`characteristic_net_compatibility_
> residuals` plumbing + CE-chain fallback for `evaluate_moc=False`;
> worst-5% + 3×RMS links red, indices to stdout,
> `fig.net_diagnostics` dict; §12.5 gate test passes (flags bad links
> on the max_nfev=0 mis-converged run).  **§11.7→12.7 LANDED:** the
> BDE wall path lifts the SOLVED state into the full-form
> `moc_topology.RaoTopology` — `sol.topology_solved` (+
> `construction_diagnostics["topology_closure"]` floats); seams exact
> at the reference, full_wall ends on the commanded exit;
> `sol.topology` remains the nasa-flavour SEED object.  Sandbox
> suite: pairing 5/5, jax kernel 9, theta_b fast 2, plotting 13,
> topology 5(+slow) — all green.
> J5 de-circularization): the differentiable kernel-march SKELETON
> exists with exact smooth d/dθ_B; BD state accuracy vs the oracle is
> the open gate.**  `raosim/jax/moc_kernel.py` (stub → real):
> Anderson-form unit processes ported from the corrected `raosim.moc`
> (fixed-iteration `fori_loop`s; parity with the NumPy processes at
> ``tol=0.0`` measured ≤ 2e-15 — float64 round-off), plus the march:
> θ_B-scaled prescribed wall grid (θ_w,k = θ_B·k/n_rows; the last scan
> carry IS BD at exactly θ_B), NASA's grow-by-one fan in static PADDED
> arrays (width n0+n_rows; per-row valid prefix is a function of the
> static row index; padded slots filled with the row's axis state as
> benign finite sentinels — no where-NaN-grad), prescribed wall point =
> NASA `CalcSpecialWallPoint` semantics in bounded angle form (C− 
> tangent-line foot through the prev wall point + x-ratio state
> interpolation + corrected C+ compatibility).  **Design history, all
> measured in `.venv-jax` (py3.12 + pinned jax — note: the sandbox CAN
> run small jax marches/solves; only the long LM ladders stay
> host-only):** (1) static-width forward march — wall steps explode
> 0.56°→2.3°→5.4°→NaN (nothing limits the step once the row is
> coarse); (2) static-width inverse march + per-row re-gridding —
> weakly unstable, axis Mach 1.63→6.75 vs oracle ~3.3, NaN by row ~98
> (fan starved); (3) padded fan + chord-foot wall point — stable/finite
> but wall chain over-accumulates ν (+0.034°/row, B Mach +4.4%);
> (4) LANDED: padded fan + NASA-faithful tangent-line foot — wall chain
> tracks the oracle within |ΔM| ≤ 0.033 across the arc.  **Proven (
> `tests/test_jax_moc_kernel.py`, 7 passed / 1 xfailed, 5s):** unit
> parity 1e-12 gates; march finite; wall trace exactly on the grid and
> on the arc; BD satisfies the corrected C− invariant (RMS 2.8e-5;
> oracle's own BD 4.1e-6); `jacfwd` d(BD)/dθ_B finite and matching
> central FD to <1e-4 rel (measured ~1e-6).  **Open (xfail
> `test_bd_matches_oracle_acceptance`):** BD interior/axis states —
> B Mach −1.7%, M(15% arc ≈ D region) +7.7%, axis +8.7%, and the
> deltas GROW when resolution doubles (nasa converged 3.384→3.382 at
> s15 while jax moves 3.64→3.75): a chain consistency defect, not
> discretization.  Root-cause candidates, in suspicion order: (a) the
> near-axis C+ source in the last interior cell (NASA substitutes the
> off-axis node's coefficients; the midpoint-r regularisation may
> differ at O(1) along the marched limit); (b) foot-state
> interpolation space in the wall process (NASA interpolates
> tan-slopes linearly in x — materially different from angle-space on
> near-vertical TT' segments); (c) the special-vs-regular row mix at
> finer resolution (grow-by-one verified only at n_kernel=24, where
> NASA's special point fires every row: lens 24→120 over 102 rows,
> α-steps of dθ_limit/2 = 0.25°).  Next session: chase (a)-(c) against
> MOC_GridCalc_BDE.cpp + Zucrow & Hoffman Ch. 17 (project rule: ground
> in the texts first), THEN wire θ_B as a BVP unknown (assembly
> args-lift; n_rows re-frozen per ladder rung like kernel_bd today).
> `build_start_line` (differentiable Kliegel–Levine, needed for
> d/d(Rt, γ) totals) stays a documented stub.

> **J5 DE-CIRCULARIZED (2026-06-12, sandbox-verified; host record run
> queued).**  Under the characteristic formulation `sol.theta_N` is now
> the SOLVED kernel arc-end angle θ_B (with provenance —
> `MOCKernel.theta_b_provenance`: `fixed_end_secant` / `frozen_override`
> / `seed_guess` — surfaced as
> `construction_diagnostics["design_angles"]["theta_N_source"]` and as
> a per-row benchmark column) and `sol.theta_E` is the solved CE exit
> flow angle `ce.theta[-1]`; the Rao-1960 parabola-fit chart pair is
> demoted to comparison data in the same diagnostics block.  Legacy
> formulation reporting is unchanged.  **Discovery that sharpens the
> old J5 record's reading: the pre-J5 "solved θ_E" column was the
> chart-N→exit straight CHORD** — with `evaluate_moc=False` the export
> wall degenerates to two points, and pure geometry reproduces the
> entire recorded signature (~21.1° @L70 / ~18.6° @L80 / ~16.6° @L90,
> ε-independent) to ~0.1° — so the "low-ε θ_E gap" was chart-vs-chord,
> not solver physics (and not only the shallow-θ_B shadow as previously
> hypothesised).  err_* columns are now exact-variational vs
> parabola-fit DELTAS — the benchmark's documented finding (expected at
> the reference: θ_B 25.57 vs chart 30, θ_E 11.12 vs 15.5).  Tests:
> three new fast reporting gates on the `max_nfev=0` path
> (sandbox-green, no JAX needed); the corner test additionally guards
> secant provenance; the full-grid slow test now asserts completion +
> physical bands + ≥80% secant-sourced rows and RECORDS deltas (the
> half-vacuous 3/6° chart gate is retired); the plan-target xfail
> reason is now *definitional*.  `to_contour_dict` draws the throat arc
> to the exported wall's actual start (was: reported θ_N — would gap
> post-J5).  Scripts: `j5_chart_sweep.py` re-documented + JSON gains
> `theta_n_source_counts`; `generate_contour.py` / `j4_gate_wall_probe.py`
> now rely on package defaults (= the gate config), print the solved
> angles, stale 21.87°/22° guess echoes removed;
> `plot_reference_topology.py` re-run in-sandbox (θ_B 25.5619°,
> kdf 0.1522, seams 0.0, peak 26.26°, exit 11.20°).  Sandbox suite:
> pairing 5/5, nasa_moc_topology, moc fast, rao_residuals, smoke +
> 3 new gates all green (jax-import tests are host-only as ever).
> **HOST QUEUE (in order): (1) fast suite + slow chart sweep regression
> (`pytest -q -m "not slow"` then `pytest tests/test_rao_chart_benchmark.py -q`),
> (2) `PYTHONPATH=. python scripts/j5_chart_sweep.py` — expect
> systematic θ_N deltas ~4-5° (solved θ_B below the parabola chart) and
> θ_E deltas ~3-4.5°, sources all `fixed_end_secant`; then pin
> per-corner delta baselines into the full-grid test (its TODO),
> (3) `scripts/j4_gate_wall_probe.py` + `scripts/generate_contour.py`
> for fresh deliverable figures, (4) full-suite re-record (was 790/5).**

> **FORMULATION DEFECT FOUND AND FIXED (2026-06-11h, identified by
> ibrahim; supersedes the Guderley reading of the freeze sweep).**
> The residual stack enforced the WRONG characteristic invariant: along
> the C+ (θ+μ) CE/DE it imposed d(θ+ν) — the C− family's relation —
> and the axisymmetric source carried a spurious cosμ/cos(θ±μ) factor.
> The mirrored pairing ran through `rao_residuals.py`, `moc.py`'s three
> unit processes, `jax/residuals.py`, and `jax/assembly.py`.  Correct
> relations (Anderson MCF §11.4; Zucrow & Hoffman Vol. 2 Ch. 17), nodes
> downstream, S = sinθ sinμ/r:  **C−: d(θ+ν) = +S ds;  C+: d(θ−ν) =
> −S ds.**  Empirical proof against the NASA oracle: on kernel-RRC
> (true C−) segments the corrected C− form holds at RMS 2.3e-6 (old
> forms 2.7e-4/4.9e-4); on a verbatim-NASA-``Deriv`` LRC the corrected
> C+ form holds at RMS 8.8e-8 (old pairing 3.4e-3).  ``nasa_deriv`` is
> the independent reference: its closed-form (dM/dr, dθ/dr) already
> encodes the stationary-DE system, which is why the fixed-end closure
> IS the smooth stationary solution.  **Consequences, re-read:** the
> freeze sweep measured the defective residual (its rising floor and
> kdf→B collapse = LM fleeing an incorrectly paired equation, and its
> "wall" never closed the endpoint — unlike the J4 gate wall, which
> did); the position-only 7.5e-4 is two released variables compensating
> for the wrong relation, NOT evidence of a jump (a physical jump obeys
> PM-realizability, not two free dofs); ibrahim's solver-independent RK
> existence scan closes all three targets smoothly at **θ_B =
> 25.5659°, kdf = 0.15216, D(M 3.40145, θ 18.5182°), E(M 3.47655,
> θ 11.1193°)** — pinned as `test_smooth_existence_root_regression`.
> The Guderley-jump hypothesis is SHELVED unless the corrected
> full-pin BVP refuses to close.  **Also fixed:** the CE
> Mach-monotonicity penalty (1.0 per 0.05 Mach) is zeroed under the
> characteristic formulation (CE crosses streamlines; the smooth DE
> decelerates slightly near D — the penalty displaced the correct
> branch).  **Landed + sandbox-verified:** corrected
> rao_residuals/moc/jax-residuals/assembly (J2 parity 10/10 holds —
> both sides changed identically), penalty gates (sizes preserved),
> `tests/test_characteristic_pairing.py` (5 oracle-grounded tests:
> RRC, LRC, planar invariants both directions, oracle-node
> reconstruction by the corrected unit process, existence root), two
> legacy tests re-baselined to the correct invariants
> (test_moc.test_planar_invariants, test_rao_residuals planar pair);
> oracle suites untouched and green. Current full-suite record after the
> correction and topology cleanup: **790 passed / 5 xfailed**.
> **Verified after re-baseline:** full D-state continuity is now the
> default (`pin_d_theta=True`, `pin_d_mach=True`).  The reference
> full-pin solve passes the 2e-3 gate, the BDE region closes exactly at
> the commanded exit, and the bell-shape regression is grounded in the
> smooth stationary-DE root (θ_B=25.5659°, θ_E=11.1193°), not the
> parabola-fit chart angles.  The relaxed position-only branch remains
> an explicit diagnostic and drives D toward B under the corrected
> equations.



> **SUPERSEDED HISTORICAL STATUS (pre-characteristic correction).**
>
> **STATUS-HEAD (2026-06-11): Phase 12.4 LANDED — the ~24.2° "march cap"
> was never the unit process.**  Root cause: at Rao's sharp radius
> (Rd = 0.382·Rt) the RRC slope tan(θ−μ) changes sign mid-row once the
> wall passes ~24.3° (θ crosses μ; the characteristic climbs in r before
> descending — a *benign fold*, verified non-crossing with neighbouring
> RRCs, so not a limit line).  `calc_massflow_along_rrc`'s
> `dr <= 1e-15` guard zeroed those segments — the C++ integrates
> `fabs(mdot_a)·fabs(da)` straight through them
> (MOC_GridCalc_BDE.cpp:3217-3228) — so the first folded row read −13.6%
> mass and `build_kernel`'s 5% sanity check halted the march at
> θ_w ≈ 24.2°, masquerading as a unit-process cap.  Fixed with the
> algebraically identical regularised product form
> `0.5·π·(r₁+r₂)·|ρu_sum·dr + ρv_sum·dx|` (finite at dr = 0; monotone
> rows bit-identical to the oracle-validated path).  Consequences, all
> test-pinned (march-parity +3, nasa_port +2 slow):
> the march reaches θ_B = 30.000° at n_kernel 24/60/101 (M_B
> 2.165 ± 0.001, mass drift −0.11%); **`set_theta_b` now closes the
> fixed-(L, ε) topology exactly** at the Rao reference — θ_B = 25.54°,
> |ΔL/L| ≈ 1e-6, |Δr_E/r_E| ≈ 1e-7, kdf = 0.152 (the "~9% long" item is
> resolved); and the **seed-topology wall is a true bell at the
> commanded (ε=10, L80)**: slope peaks 26.3° at 5.4% length (θ_B + ~0.7°
> arc/BFE discretisation overshoot), decreases monotonically to 11.2° at
> the exit, endpoint exactly on (L, Re) — the 35.6° mid-bell flare is
> absent on this path.  (Caveat: fixed-end ≠ Rao-stationary TOP; the
> chart values θ_N ≈ 21.9° / θ_E ≈ 8.3° describe the *optimal* family,
> so the fixed-end bell turns harder.  Treat it as the first
> geometrically-sane Rao-case contour, not yet the optimum.)
> **Full-continuity probe** (pin_d_theta + pin_d_mach + ladder + θ_B
> Picard refresh on kernel-stationarity at the solved (kdf, log_C)):
> the Picard converges θ_B 21.87 → 28.10 → 28.17° in two refreshes —
> the cap no longer binds — but the inner floor stays at ~5.7e-2
> (all-node stationarity ramp, kdf drifting to 0.088, i.e. D crawling
> toward B), *θ_B-insensitive* (identical floor at 21.87° and 28.10°).
> Reading: with the kernel/θ_B frozen per solve, fixed-(L, ε) + full
> D-continuity stays overdetermined — DE is fully determined by D and
> the stationarity+C⁺ ODE pair, so hitting (r_E, L) needs kdf *and*
> θ_B live inside the iteration.  Next levers: J3b differentiable march
> (θ_B as a solved unknown), an outer secant on the *inner floor* (the
> kernel-stationarity invariant refresh measured ineffective), or the
> fixed-length transversality/multiplier blocks — and check Guderley
> (~1968) before forcing smoothness at D: for short nozzles the optimum
> is genuinely discontinuous, so the position-only solution's ΔM jump
> at D may be physics, not artifact.  Sandbox could not run the J4 gate
> test / solved-CE wall probe (JAX solves exceed the per-call budget);
> run on host: `pytest tests/test_jax_convergence.py -q -m "not slow"`
> and `PYTHONPATH=. python scripts/j4_gate_wall_probe.py`, then
> `scripts/generate_contour.py` for fresh latex-report figures.
>
> **Phase 12.6 LANDED (2026-06-11b, same session as the DIRECTION
> block):** `raosim/moc_topology.py` formalises the §11.7 object —
> `RaoTopology(TT_prime, B, BF, D, BD, DE, E, streamline_BE, theta_B,
> mass_BD, mass_DE)` on CharPoints, plus provenance (`d_fraction`,
> `arc_wall`, diagnostics), `full_wall()` (throat arc + streamline_BE,
> monotone-x enforced), `closure_report()`, and the one-call
> `build_reference_topology(...)`.  Reference-case closure metrics:
> every seam (B/D/E attachments, wall start/end) at 0.0 exactly, mass
> pair 1.7e-10, exit on the commanded station, bell peak 26.26° /
> exit 11.20° (`tests/test_moc_topology.py`, 4 tests).
> `_construct_wall_from_ce` deletion stays staged with the flip
> (DIRECTION 2d) — the legacy default path still uses it.  **Phase 13
> increment:** `plot_characteristic_net` / `plot_flowfield_mach`
> already existed and match the §12.2 outline; added `plot_topology`
> (spec plot #9, works on the 12.6 object) and the missing
> `tests/test_plotting.py` (5 smoke tests, Agg).  Reference artifacts:
> `scripts/plot_reference_topology.py` →
> `builds/reference_topology.png` + `builds/reference_wall.csv`.
> In-sandbox verification: 21-test lite sweep + 2 slow topology tests
> green; JAX-solve work (gate re-confirm → flip) still host-gated.

> **J5 / J6 / Phase-13 batch (2026-06-11c, post-flip session).**
> **J5:** the chart sweep runs and passes the loose gate (3°/6°) under
> the JAX defaults — confirmed by the post-flip host suite
> (`test_rao_chart_benchmark_full_grid` in the 769-pass run).  Record
> script `scripts/j5_chart_sweep.py` (weight 1.0 + JSON dump, host-run)
> landed.  **Record run (2026-06-11, host, weight 1.0, n=10/nk=10/
> nfev=300): 33/33 completed, 0 raised; θ_E RMS 2.67° / max 5.72°
> (loose gate PASSES; plan gate 1.5°/3° fails on θ_E).**  Diagnostic
> signature worth chasing before tightening: the solved θ_E is nearly
> a function of length_pct alone (~21.0° @L70, ~18.5° @L80, ~16.6°
> @L90 across ALL ε) while the chart's θ_E rises with ε — so the error
> concentrates at the grid edges (5.2-5.7° at ε=6; ≤0.6° in the ε=25-50
> mid-band; ~3° again at ε≥40/L70).  Candidates: the tiny sweep budget
> (n_control=10, max_nfev=300 — re-run a low-ε case at n=24 to
> separate resolution from physics), the pa_over_p0=0 vacuum
> convention, or the θ_E extraction itself.  Many high-ε/L70 corners
> flag `invalid_short_nozzle_region` (§2.E) — consistent with the
> chart extrapolating where Guderley says the smooth optimum thins.  The plan-target gate (1.5°/3°) stays xfail and is now
> *definitionally* blocked, not solver-blocked: reported θ_N is still
> the chart lookup (`_design_angles_rad`) — circular — and the repo
> carries TWO disagreeing θ_N sources (`nozzle_geometry._THETA_N_TABLE`
> ~21.9° vs `benchmarks` tables 30° at ε=10/L80).  Reconcile both
> against Rao 1958 / NASA SP-8120 before wiring a solved θ_N (candidate:
> the converged fixed-end topology θ_B) into the report path.
> **J6 v1 LANDED:** `raosim/jax/sensitivities.py` —
> `rao_sensitivities(config)` (api stub now delegates): exact
> reverse-mode node tolerance fields (dCf/dM, dCf/dθ, dCf/dr per CE/DE
> node + dCf/dkdf through the D-slide chain), explicit design partials
> at fixed u* (dCf/dpa, dCf/dγ), and `jacfwd` Jacobian conditioning
> (σ_max/σ_min).  Cf functional is a line-for-line jnp port of
> `surface_thrust_coefficient` (parity 1e-10 on a real fixed-end DE);
> the J6 known-sign gate is the analytic identity
> **dCf/dpa = −(r_E²−r_D²)/Rt²** (telescoped trapezoid), pinned to
> 1e-10 in `tests/test_jax_sensitivities.py` along with an FD
> cross-check of the node gradient.  v2 deferrals documented in the
> module docstring: IFT design totals (needs assembly arg-lifting +
> J3b for kernel-reaching params), bell-wall-node map (differentiable
> BDE march), Hessian.  End-to-end smoke is @slow (host).
> `plot_sensitivity_field` (§6 tolerance map) added to plotting.py.
> **Phase 13: 9 of 10 spec plots now exist** — added #1 geometry,
> #4 pressure, #5 theta, #6 wall distributions, #7 exit plane,
> #10 NASA overlay (+ the J6 sensitivity map); pre-existing #2 net /
> #3 mach; #9 topology from 12.6.  Remaining: #8 `plot_net_diagnostics`
> (needs RaoResidualReport link plumbing).  `tests/test_plotting.py`
> now 11 smokes.  In-sandbox verification: 29-test sweep + flip-default
> tests green.  **Host record (2026-06-11): full suite 781 passed /
> 9 xfailed in 12:53 — zero failures; the J6 end-to-end @slow smoke
> passed (15s including its solve).**  Open next, in order: θ_N-table
> reconciliation (gates the J5 plan target AND informs the flare fork),
> the low-ε θ_E gap (see the J5 record paragraph), then J3b (unlocks
> full D-continuity and the J6 v2 design totals).
> `scripts/theta_b_picard_probe.py` remains optional (full-continuity
> record).

> **θ_N RECONCILIATION (2026-06-11f) — literature-grounded, and it
> unifies every open accuracy item.**  Primary source check
> (propulsion_texts/RaoRecentDevinRockNozConfig.pdf — Rao, ARS J.
> 31(11), 1961): the classic TOP angle charts are Rao's 1960 ARS J.
> parabola-fit charts (his ref 22), computed for **γ = 1.23**, and at
> fixed (ε, L) the optimal *contour* is nearly **γ-insensitive** (Rao,
> p. 1490: "differences in nozzle contours are negligible"; only Cf
> depends strongly on γ) — so the in-repo tables (provenance comment
> added in nozzle_geometry.py) are valid γ=1.4 geometry targets.
> Optimal wall angles per Rao pp. 1490-1491: ~28-30° after the throat,
> ~10-14° at the exit.  Findings: (1) there is ONE in-repo chart family
> — `nozzle_geometry` tables, consumed by `benchmarks` — giving
> **θ_N = 30.0° / θ_E = 15.5° at ε=10/L80** (my earlier "two
> disagreeing tables" note was wrong); (2) the "chart θ_N ≈ 21.9°"
> claims in test docstrings were MISLABELS — 21.87° is the
> kernel-stationarity refresh diagnostic, not a chart value (comments
> fixed); (3) the three angles now form a coherent picture:
> chart 30° = the optimum; Picard full-continuity 28.17° = the solver
> driving TOWARD the optimum; fixed-end 25.56° = the sub-optimal
> utility closure — and the 5.7e-2 full-pin floor was θ_B-insensitive
> *by construction*, because the seed's inner `set_theta_b` secant
> rebuilds the kernel at the fixed-end angle on every solve, overriding
> the outer Picard.  (4) The θ_E sweep signature (solved θ_E ≈ f(L%)
> only, worst at low ε) is the same root cause's shadow: a too-shallow
> frozen θ_B forces a steeper exit angle at fixed (L, ε); at high ε the
> chart θ_E rises to meet it — exactly the observed error pattern.
> **Landed:** `RaoSolverConfig.theta_b_freeze_deg` (bypasses the inner
> secant; kernel built at exactly the frozen angle, D/DE still seeded
> by the fixed-end walk with r_E pinned; unit-tested), the bell xfail
> test re-grounded to chart values via `lookup_angles` (peak ≈ θ_N ±
> 2.5° near the throat, exit ≈ θ_E ± 2.5°), mislabeled comments fixed.
> **Host experiment queued (the decisive one):** outer θ_B
> Picard/secant over `theta_b_freeze_deg` ∈ [26°, 31°] with FULL
> D-state pins (pin_d_theta + pin_d_mach) at the reference point —
> prediction: the stationarity floor collapses near θ_B ≈ 30° (chart),
> the gate closes with full continuity, the BDE wall peaks ≈ 30° at
> the throat (flare gone), and θ_E drops toward 15.5° — closing the
> J5 plan-target gap at low ε as a corollary.  If the floor does NOT
> collapse, the Guderley-discontinuity branch becomes the live
> hypothesis (then check propulsion_texts for Guderley/Hantsch 1955 +
> the ~1968 short-nozzle results before any further smoothing effort).
>
> **RESULT (2026-06-11g, host): PREDICTION FALSIFIED — the Guderley
> branch is live.**  Sweep (full pins, n=24, ladder): floor RISES
> monotonically with θ_B — 5.92e-2 @26° / 6.53e-2 @27° / 7.21e-2 @28° /
> 7.93e-2 @29° / 8.65e-2 @30° / 6.30e-2 @31° (different basin) — while
> kdf collapses monotonically toward B (0.076 → 0.010): the solver
> flees the kernel state at every θ_B, hardest at the chart angle.
> Wall at "best" (26°) is degenerate (slope = θ_B all the way to a
> 100%-length "peak"; DE never turned the flow).  Full D-state
> continuity is unsatisfiable across the whole band — θ_B was NOT the
> missing degree of freedom.  Counting argument now favours a genuine
> **state discontinuity at D**: freeing (θ₀, M₀) at fixed D-position
> (the position-only attachment) adds exactly the two dofs of a jump,
> and the system then closes to 7.5e-4 — i.e. the gate-passing solution
> may BE the optimum-with-corner (Guderley-type), with the 30.6° "flare"
> being the BDE back-march *rendering the jump as a resolved wave fan*
> rather than treating D as a centred-wave origin.  Caveat against
> over-reading: the repo's §2.E valid-region check calls ε=10/L80
> "valid_shock_free_region", so either that check is too coarse
> (quasi-1D) or the non-closure is a formulation defect — discriminate
> BEFORE building discontinuity machinery.  Next discriminating
> experiment (sandbox-feasible, numpy-only, no LM in the loop): direct
> **stationary-DE shooting/existence scan** — integrate the
> (C⁺ compatibility + d(stationarity)=0) ODE pair from D(kdf, θ_B)
> by RK and map which (kdf, θ_B) hit (mass, r_E, L); if the smooth
> 3-target intersection is empty over the physical rectangle, the
> jump is established independent of the solver.  Ground the ODE pair
> in Rao 1958 / the propulsion_texts variational PDFs first; then the
> Guderley & Hantsch 1955 / Guderley ~1968 literature check decides
> corner-modelling vs fixed-length transversality.

> **DIRECTION (2026-06-11, set by ibrahim).**
> 1. **End state: `raosim/jax` becomes the only core package.**  The
>    NumPy mirror modules get absorbed/deleted once the JAX core is in
>    its fullest form (all residual blocks, march, topology, wall).
>    Deferred until then — the §3 port boundary stays as-is meanwhile;
>    the imperative shell (I/O, CEA, plotting, export) is unaffected.
> 2. **Flip the default backend to JAX once J4 passes** — meaning: once
>    the gate is *re-confirmed on the post-12.4 seed* (the march fix
>    changed `set_theta_b`'s convergence, so the 2026-06-10 pass needs
>    re-measuring; sandbox cannot run JAX solves).  Flip checklist:
>      a. ✅ (2026-06-11, host) `scripts/j4_gate_wall_probe.py`:
>         **max_scaled = 7.5039e-4** ≤ 2e-3, converged, mass −9.4e-10,
>         len −1.5e-10, kdf 0.2271 — *better* than the pre-12.4 pass
>         (1.157e-3).  **Wall verdict (fixed probe, same run): the
>         solved-CE BDE flare PERSISTS** — slope rises through the arc
>         to ~25.5° (= the seed θ_B, correct), eases to 24.7° at 31%
>         length, then climbs to a **30.57° peak at 64.4%** and crashes
>         to 3.62° at the exit (was 35.6° @ 60% / 4.6° exit pre-12.4 —
>         shrunk by the healthier kdf/seed, not removed).  Exit lands
>         exactly on (L, Re).  Confirms the diagnosis: the flare is
>         intrinsic to the position-only ΔM jump at D being rendered as
>         a fictitious wave system by the (correct) BDE back-march —
>         the geometric bell still comes only from the topology path.
>         `test_bde_wall_is_bell_shaped` stays xfail; the decision
>         between full-continuity (needs θ_B in-solve, J3b), fixed-L
>         transversality blocks, or a genuinely discontinuous Guderley
>         optimum at D is now THE open shape question.
>      b. ✅ (2026-06-11, host) Full suite post-flip: **769 passed /
>         1 failed / 9 xfailed in 13:13** — even the pre-existing
>         `test_wall_monotonic` failure cleared (12.4 seed and/or the
>         new defaults; mechanism not isolated).  The single failure was
>         `test_ablation_matrix...`: `rao_residual_ablation_matrix`
>         sweeps FD-reference block subsets that the JAX assembly
>         rejects by design (`SUPPORTED_BLOCKS`), so the runner now
>         pins `solver_backend="numpy"` explicitly (was implicit
>         pre-flip).  Fixed + verified; expected suite state:
>         770 passed / 9 xfailed.
>      c. ✅ (2026-06-11b) Bundle applied in `RaoSolverConfig`:
>         `solver_backend="jax"`, `formulation="characteristic"`,
>         `pin_d_theta=False`, `jax_constraint_weight_ladder=
>         (1.0, 10.0, 30.0, 100.0)`.  Default-pinning tests rewritten
>         (`test_jax_backend_is_default`,
>         `test_characteristic_formulation_is_default`, + opt-in
>         legacy/numpy coverage).  Note: the
>         `jax_characteristic_weight1` fixture now inherits
>         `pin_d_theta=False` from the default (was True) — its floor
>         should improve from ~3e-3 toward the gate value.
>      d. Re-baseline per (b); keep the legacy stack importable behind
>         explicit opt-in until the §3 absorption (directive 1) lands.
>         `_construct_wall_from_ce` deletion happens with the
>         absorption, not the flip (the opt-in legacy path still calls
>         it).
> 3. **Every fix validates against the M3.5Perf reference** —
>    `outputs_M3.5Perf` stays the binding oracle
>    (`tests/test_nasa_kernel_march_parity.py`, `tests/test_nasa_port.py`
>    must stay green through any numerics change, as in Phase 12.4).
> 4. **Tool first until J6 ships.**  No §11 novelty work (Guderley
>    boundary mapping, real-gas hooks) until the tool exports a
>    mathematically accurate contour and `rao_sensitivities` (J6) is
>    done.  The Guderley *literature check* stays in scope as it bears
>    directly on D-attachment correctness (tool accuracy, not novelty).

> **STATUS (2026-06-09, J0–J4 spike landed).**
> J0 ✅ (skeleton, pinned deps, `solver_backend="jax"` wired into
> `solve_rao_bvp`); J1 ✅; J2 ✅ **assembled**-residual parity at ~1e-15 vs
> `_scaled_rao_bvp_residual` on real Phase-6 states
> (`raosim/jax/assembly.py`, `tests/test_jax_assembly_parity.py`);
> J3 ✅ (Optimistix LM + exact jacfwd/jacrev, FD-verified on the real
> system).  **J4: the §10 diagnosis branch fired, productively.**  Exact
> Jacobians cut the stall 8 → ~2.8 and exposed the true blocker: the
> NASA kernel march never advanced past TT' (`rrcs == 1`, silent
> arc+sonic-line BD fallback).  Root causes found & fixed, oracle-validated
> against `outputs_M3.5Perf` (`tests/test_nasa_kernel_march_parity.py`):
> (1) the "visible source" KLThroat port transcribed C++ **integer
> division** `5/8` as `0.625` (binary ran with the term dropped — TT'
> now matches `TT'.out` to 5e-7, march reproduces the full 58-row grid,
> BD matches `LastKernel.out`); (2) the KL start line was fed the
> *downstream* throat radius where the C++ passes `rUp` — fixed via
> `build_kernel(Ru=...)` + `RaoSolverConfig.throat_upstream_radius_factor`
> (Rao convention 1.5).  With a real kernel: max_scaled ≈ **0.5** on
> the ε=10/L80/w=1.0 reference (from 8).
>
> **Topology seed landed (same session).**  The degenerate `calc_lrc_de`/
> `set_theta_b` D≈E collapse was structural: the Python port only
> implemented NASA's free-exit RAO inner branch, so the fixed-(L, ε)
> composite was overdetermined.  Fixed by porting the **FIXEDEND** branch
> (C++ lines 1560-1610): `calc_lrc_de(end_condition="fixed_end")` walks D
> along the marched BD until DE's endpoint pins r_E (secant to 1e-7;
> mass_BD == mass_DE exactly), and `set_theta_b` secants θ_B on the
> *length* mismatch with `ThetaBTooLow/High` fail-bracketing (C++
> SEC_FAIL semantics).  `find_point_e` honours `n_steps` via a per-step
> mass cap (C++ `dMdot = mdotMatch/nRRCPlus` intent).  The BVP seeds its
> CE from the resampled DE (`RaoSolverConfig.ce_seed="auto"`).  Result:
> max_scaled ≈ **0.49** with the Rao physics blocks (stationarity, C±,
> CE↔wall C+) at ~**3e-2** — misfit now concentrates in
> length/wall-endpoint.  The remaining J4 gap is genuine variational
> tension, not infrastructure: the march's unit-process edge caps
> θ_B ≈ 24° for Rd = 0.382·Rt, the fixed-end topology there runs ~9%
> long, and at weight 1.0 LM trades length against stationarity.  Next
> levers: length continuation from the topology's natural length, march
> robustness past the θ-cap (Phase 12.4 CalcRRCsAlongArc completion), or
> multiplier/transversality blocks pinning the fixed-length optimum.
> J3b (lax.scan march port) and J5/J6 remain open; J3b is *not* required
> for J4 (the kernel BD is static during the solve).
>
> **Formulation fix (opt-in) — max_scaled 0.5 → 0.062.**  Root cause of
> the residual-shuffling stalls: two scaffold blocks are structurally
> unsatisfiable at the converged Rao topology, because the refactored CE
> is a C+ characteristic *by construction* — ``moc_cminus`` applied the
> C− relation along those C+ segments, and ``cplus_ce_to_wall`` /
> ``wall_intersection`` paired CE nodes to the wall along C+ slopes (the
> C+ through a DE point *is* DE; it meets the wall only at E).  Rao's DE
> closure carries C+ relations + stationarity + mass/length only (Rao
> 1958; AIAA 99-2584; NASA FindPointE integrates only the LRC system);
> the wall belongs to the BDE-region march (``calc_bde_region``).
> Landed as ``RaoSolverConfig.formulation="characteristic"``
> (``CHARACTERISTIC_RAO_RESIDUAL_BLOCKS``) plus a constraint-weight
> ladder in the JAX solver (``jax_constraint_weight_ladder``; mass/
> length/endpoint pins are single elements drowned by O(n) physics
> elements in plain least squares).  Reference case, weight 1.0:
> **max_scaled = 0.062** — mass 5e-3, C+ compat 8e-3, remaining floor in
> stationarity-at-D (~6e-2) / regularizer / length (3e-2).  Legacy stack
> stays the default until Phase 6/7 re-baseline.
>
> **D-closure probe (frozen-BD floor identified).**  Pinning M_ce[0] to
> D's kernel Mach (full state continuity at D — classically how D is
> *selected*) made things worse on the frozen BD (cp 8e-3 → 0.78): with
> ``config.kernel_bd`` fixed by the seed's θ_B-capped kernel, D's
> (r, θ, M) is a 1-parameter curve that cannot satisfy the fixed-(L, ε)
> optimum.  Landed as ``RaoSolverConfig.pin_d_mach`` (default False)
> with the failure mode documented.  Per-node diagnostic at the 0.062
> solution: stationarity is a smooth monotone ramp (−0.062 at D →
> +0.015 at E) — the frozen-BD inconsistency spread along the CE — and
> kdf drifts to ~0.02 (near-streamline DE closing mass trivially).
> Conclusion: the remaining closure requires **θ_B/BD inside the
> iteration** — either an outer BD-refresh loop around the BVP, or the
> J3b differentiable march making θ_B a solved unknown.  That promotes
> J3b from "nice for J6 gradients" to the J4 critical path.
>
> **BDE wall path wired.**  ``wall_method="bde"`` builds the wall from
> the *solved* CE via NASA's region march (``_wall_from_bde_region``:
> solved kdf → D, solved CE → DE, ``calc_bde_region``; wall = kernel
> throat-arc wall + BFE wall contour).  Supporting knobs:
> ``kernel_d_fraction_min`` (guards the degenerate kdf→0.02 topology)
> and a graceful export guard (endpoint mismatch downgrades reliability
> instead of raising).  Status: the BFE mesh completes and the wall
> lands exactly on r_E, but x overshoots (~+20% at L87) — the same
> frozen-BD length tension; and the forward-MOC audit needs alignment
> (it marches a SplineWall + chart-θ_N starting line, inconsistent with
> a BDE-built wall).  Meanwhile the *seed-topology* contour
> (``set_theta_b`` fixed_end → ``calc_bde_region``) is complete and
> closed at the kernel's natural length (≈L87 for ε=10): wall ends
> exactly at E.  Bottom line: every remaining gap — J4's 2e-3, exact
> length at arbitrary (ε, L%), and a closed BDE wall — reduces to
> **BD-in-the-loop** (outer kernel refresh or the J3b differentiable
> march).
>
> **Length-bookkeeping fix — max_scaled 0.062 → 0.0030 (gate 2e-3).**
> The dominant "frozen-BD" symptom was a consistency bug: the length
> residual used Σdx = x_E − x_D while ce_geometry pinned x_E = L —
> contradictory unless x_D → 0 (hence kdf → 0.02; len ≈ −x_D/L).  Rao's
> constraint is the *exit station* L = z_C + ∫cot(φ)dr (length_integrand
> docstring).  Fixed characteristic-gated in both backends + report
> layer.  Result: mass 5e-9, length 3e-9, kdf interior at 0.273 unaided,
> only stationarity left at ~3.0e-3 — resolution-independent AND
> θ_B-insensitive: the cheap θ_B-refresh extraction (secant on
> kernel-state stationarity at the solved kdf/log_C) recovers
> θ_B = 21.87° ≈ the chart θ_N (21.9°!) yet re-solving there doesn't
> move the floor.  The frozen-BD hypothesis is resolved.
>
> **J4 GATE PASSED (2026-06-10): max_scaled = 1.157e-3 ≤ 2e-3.**
> The last 1.5× was the θ_D start pin: the node dump showed an M
> discontinuity at the attachment (kernel M_D = 3.78 vs CE M₀ = 3.12,
> uniform-M start region) — pinning the CE start angle to the
> *interpolated approximate* kernel value imports kernel discretization
> error into the stationarity chain at full weight.  Position-only D
> attachment (``pin_d_theta=False``; r pinned, θ/M free; D's position
> and the B→D mass budget stay enforced) closes the gate:
> mass 2e-9, length 6e-10, kdf interior at 0.304, converged=True
> (``test_j4_gate_passes_with_position_only_attachment``).  Full state
> continuity (pin_d_mach) remains blocked by the ~24.2° march cap
> (Picard-θ_B drives into it) — Phase 12.4 territory, no longer
> gate-blocking.  Next: flip the characteristic/pin defaults and
> re-baseline Phase 6/7, J5 chart sweep under JAX, BDE wall closure at
> the converged solution (kdf now interior + length consistent), J6
> ``rao_sensitivities``.


**Architecture & implementation plan.** This document is the build plan for
re-expressing the Rao optimum-thrust contour solver as an end-to-end
differentiable program in JAX, with two goals, in priority order:

1. **Tool first.** Fix the standing convergence blocker
   (`solve_rao_bvp` sits at `max_scaled ≈ 8`; the gate is `≤ 2e-3`) by
   replacing scipy's finite-difference Jacobian with exact autodiff
   Jacobians, and ship a robust contour generator.
2. **Novelty later, for free.** Exact gradients are the headline feature
   of the tool (sensitivity fields, manufacturing-tolerance maps) and the
   enabling machinery for the open research questions parked in §11.

This plan is subordinate to `REWRITE_PLAN.md`: it does not change the
physics that plan specifies, it changes the *numerics and software
substrate* underneath it. Every phase, residual, and reliability tier in
`REWRITE_PLAN.md` still applies. References to "§2.x / Phase N" below point
into that document unless stated otherwise.

---

## 0. Honest status of the idea (read this first)

We checked the literature before committing. Two corrections to earlier
assumptions:

- **"Differentiable MOC" is *not* clearly novel as a headline, but it is
  sparse.** Adjoint and surrogate optimization of MOC/CFD nozzles is
  established (chemically-reacting adjoint, 2012; MOC + free-form-deformation
  shape optimization, CEAS Space Journal 2023; NN surrogate of SERN flow with
  backprop, arXiv 2409.12707 — note that last one is a *CFD surrogate*, not
  MOC-in-a-network, contrary to how it is sometimes summarized). A fully
  autodifferentiable MOC that backpropagates *through the marching grid
  construction itself* and through the **Rao variational closure** appears
  largely unshipped, but we are **not** staking the project on that claim.
- **The interesting physics is old but under-mapped.** The fact that the
  smooth Rao optimum ceases to exist for short / large-radius nozzles and is
  replaced by a *discontinuous* solution was established by Guderley and
  co-workers (~1968). What is missing is a modern, open, numerically
  resolved characterization of *how* the optimum degrades at that boundary,
  and how real-gas γ(T) moves the boundary. Those are the §11 hooks.

Conclusion: build the differentiable tool because it is the right
engineering substrate and it fixes convergence. Treat novelty as an
application built on top, decided later.

---

## 1. Why JAX, why now

The current solver (`raosim/rao_variational.py`) builds its residual vector
from per-segment scalar functions —
`residual_Cplus_axisym`, `residual_Cminus_axisym`,
`residual_left_mach_geometry`, `residual_wall_tangency`,
`residual_cplus_child_position`, the algebraic Rao stationarity residual
(`mstar_from_M`, `gas_dynamics.py:168`), mass closure, and the valid-region
inequality — assembled in `_scaled_rao_bvp_residual` (line 2704) and
`_coupled_wall_residuals` (line 2183), then handed to
`scipy.optimize.least_squares` (line 2965 / 3559). scipy estimates the
Jacobian by **finite differences**: for a coupled-wall unknown vector with
CE + wall + characteristic nodes plus `log_C`, `λ_mdot`, `λ_length`, that is
hundreds of extra residual evaluations per iteration, each noisy near the
sonic and Mach-line singularities. That noise is the most likely reason
`max_scaled` stalls around 8 instead of descending to 2e-3.

Every one of those residual functions is **pure, composed scalar
arithmetic** (`sin`, `cos`, `atan2`, `sqrt`, `asin(1/M)`, `hypot`,
linear interpolation on arc length). That is the ideal case for reverse-mode
autodiff. Re-expressing them in `jax.numpy` gives:

- **Exact Jacobians** via `jax.jacfwd` / `jax.jacrev`, fed to a
  Gauss–Newton / Levenberg–Marquardt least-squares solver. Exact > FD for
  both convergence rate and robustness.
- **`jit`** compilation of the whole residual, so the inner solve is fast
  enough to sweep the (ε, length_pct, γ) chart.
- **Gradients of any scalar output** (Cf, Isp, wall-tangency RMS, mass
  residual) with respect to *every* design input and *every* node — the
  tool's value-add, at no extra derivation cost.

Solver library: **Optimistix** (`optimistix.LevenbergMarquardt`,
`optimistix.least_squares`, `optimistix.root_find`). It is the current,
maintained JAX nonlinear-solver library (Equinox-based; implicit-function-
theorem differentiation is the default for both forward- and reverse-mode).
`jaxopt` has broader scope but is being wound down into optax/optimistix, so
we standardize on Optimistix. (Refs in §12.)

---

## 2. Design principles

1. **Two layers, one boundary.** A pure-functional **differentiable core**
   (JAX) that owns everything between throat conditions and the emitted
   contour + residuals; and an **imperative shell** (existing NumPy/Python)
   that owns I/O, CEA, plotting, export, trajectory, CLI. Gradients only
   need to flow through the core.
2. **Oracle parity is non-negotiable.** The JAX core must reproduce the
   existing NumPy closed-form results to ~1e-10 before we trust any gradient.
   The current test suite (260 tests) is the regression oracle. This is also
   how we honor the project rule that equations be empirically grounded: the
   JAX `prandtl_meyer`, `mstar_from_M`, isentropic ratios, and Q± terms must
   match the already-literature-checked NumPy versions bit-for-bit in the
   constant-γ limit.
3. **Differentiate the *solution*, not the *iterations*.** For the converged
   BVP and for inner secant loops (θ_B, point D), use the implicit function
   theorem (Optimistix default), not backprop-through-unrolled-iterations.
   Cheaper, exact, and avoids accumulating autodiff state across hundreds of
   marching steps.
4. **No silent post-processing — enforced by construction.** The JAX core
   never rescales x_ce, never forces endpoints, never clamps θ_N. Those are
   the `REWRITE_PLAN.md` §2.F / Phase 1 sins; a differentiable rewrite is the
   natural moment to delete them for good, because endpoint closure becomes a
   residual the solver actually satisfies.
5. **Incremental, not big-bang.** Land the core behind a feature flag
   (`solver_backend="jax"`) next to the existing path. The NumPy path stays
   the reference until the JAX path passes the same gates.

---

## 3. The port boundary — what moves to JAX vs stays NumPy

| Module / concern | Move to JAX core? | Rationale |
|---|---|---|
| `gas_dynamics.py` (isentropic ratios, `prandtl_meyer`, `mach_from_prandtl_meyer`, `mstar_from_M`, `mach_angle`, area–Mach) | **Yes** (mirror module `gas_dynamics_jax.py`) | These are the leaf primitives every residual differentiates through. |
| `rao_residuals.py` (C±, left-Mach, wall-tangency, intersection) | **Yes** | The residual leaves; trivial, pure scalar math. |
| Rao algebraic stationarity + differential consistency (§2.B) | **Yes** | Core of the optimum condition; needs gradients. |
| Mass closure `curve_mass_flux` / kernel-BD integral (§2.D) | **Yes** | Integrand differentiable; D becomes a solved unknown. |
| Valid-region inequality (§2.E) | **Yes** (as a smooth diagnostic) | Used for gating + the §11 cliff study (needs ∂/∂design). |
| `_pack_bvp` / `_unpack_bvp` / `_coupled_wall_residuals` / `_scaled_rao_bvp_residual` | **Yes** (re-expressed as flat array ops) | The residual assembly is the function we autodiff + jit. |
| BVP solve (currently scipy `least_squares`) | **Yes** (Optimistix LM) | The whole point. |
| `transonic_kernel.py` (Kliegel–Levine start line) | **Yes** | Feeds the seed; differentiable so start-line sensitivity is available (§11). |
| `moc.py` unit processes (`solve_interior_point`, `solve_axis_point`, `solve_wall_point`) | **Port the math, keep the marching as `lax.scan`/`while_loop`** | Sequential, data-dependent length — see §5. |
| `nasa_moc.py` (NASA port, kernel/topology, `set_theta_b`, secant loops) | **Partial** — port the numerics, wrap secant loops with implicit diff | Reference + topology; gradients only needed through the converged result. |
| `cea.py`, `propellants.py`, `physics.py` (Bartz), `thermo.py` (Phase 10) | **No (call as host functions)** | CEA is an external equilibrium solver; not differentiable. Use frozen γ inside the core; consult CEA outside it. Real-gas hook in §11. |
| `legacy_io.py`, `export.py`, `plotting.py`, `trajectory.py`, `altitude_performance.py`, `trade_study.py`, CLI (`main.py`, `design.py`) | **No** | I/O and orchestration; no gradients needed. They consume the core's output dict. |
| `benchmarks.py`, test suite | **No** (but extend) | Stay as the oracle; add JAX-path parity + convergence tests. |

Net: a focused new subpackage `raosim/jax/` holding ~6 files
(`primitives.py`, `moc_kernel.py`, `residuals.py`, `bvp.py`, `pack.py`,
`api.py`). Everything else is untouched or lightly adapted.

---

## 4. Differentiable core architecture

```
raosim/jax/
    primitives.py   # gas dynamics in jax.numpy, sonic-safe
    moc_kernel.py   # transonic start line + characteristic march (lax.scan)
    residuals.py    # all Rao/MOC residual blocks, vectorized over nodes
    pack.py         # flat-vector <-> structured state (pytree) packing
    bvp.py          # Optimistix LM solve + implicit-diff wrapper
    api.py          # solve_rao_bvp_jax(config) -> RaoSolution (same dict shape)
```

### 4.1 Primitives (`primitives.py`)

Reimplement, in `jax.numpy`, exactly the closed forms already in
`gas_dynamics.py`: `isentropic_{temperature,pressure,density}_ratio`,
`prandtl_meyer`, `mstar_from_M`, `mach_angle`, `area_mach_relation`. The
constant-γ Prandtl–Meyer and M* forms are the ones already cross-checked
against `prmeyer.pdf` / `rao1999.pdf`, so parity to 1e-10 with the NumPy
versions is the acceptance test (Phase J1).

**Sonic-safety.** `mach_angle = asin(1/M)` and the `sqrt(M²−1)` in
Prandtl–Meyer are singular/branch-sensitive at M→1. Use guarded forms
(`jnp.sqrt(jnp.maximum(M*M - 1, 0))`, clamp `1/M` to `(0,1]`) and a
`jax.custom_jvp` on the throat-region primitives so the derivative stays
finite at M=1. `mach_from_prandtl_meyer` (currently a Newton iterate at
`gas_dynamics.py:198`) becomes either a fixed `lax.scan` of Newton steps or
an Optimistix `root_find` with implicit diff.

### 4.2 MOC kernel & march (`moc_kernel.py`)

Port the throat start line (`transonic_kernel.py`, Kliegel–Levine, with the
documented NASA `KLThroat` typo correction noted in `REWRITE_PLAN.md` §8.1)
and the axisymmetric unit processes from `moc.py`. The Anderson Q± source
terms (`moc.py:154-160`) are already the literature ground truth and port
verbatim. The march itself — variable number of characteristic rows — is the
one genuinely sequential, data-dependent piece; handle per §5 with a fixed
`lax.scan` over a padded grid.

### 4.3 Residual stack (`residuals.py`)

One pure function:

```python
def rao_residual(state: BVPState, params: DesignParams) -> jax.Array:
    """Flat residual vector r(u; p). Blocks, in REWRITE_PLAN priority order:
      - endpoint closure        (ce[-1]=wall[-1]=(L,Re); wall[0]=(Nx,Ny))
      - Rao algebraic stationarity at each CE node      (§2.B)
      - left-Mach-line geometry on CE                   (§2.C)
      - axisymmetric C+/C- compatibility on CE & wall   (§2.A)
      - mass closure CE vs kernel BD                    (§2.D)
      - wall tangency + flow-angle match at wall        (§2.F)
      - characteristic-intersection closure             (§2.F)
      - (smoothness regularizer, tiny weight 0.02)      (§2.A)
    """
```

This is the existing `_coupled_wall_residuals` + `_scaled_rao_bvp_residual`
content, but vectorized with `vmap` over node pairs instead of Python list
comprehensions, and returning a single concatenated array. Identical scaling
(`L_scale`, `Re_scale`, `theta_scale = radians(1°)`) so residual magnitudes
match the current run and the 2e-3 gate is directly comparable.

### 4.4 BVP solve (`bvp.py`)

```python
solver = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-10)
sol = optimistix.least_squares(
    lambda u, p: rao_residual(unpack(u, p), p),
    solver, y0=u_seed, args=params, max_steps=...,
)
```

Exact Jacobian comes from Optimistix's Gauss–Newton/LM step using
`jax.jacfwd` internally. Implicit-function-theorem differentiation of the
converged `u*(p)` w.r.t. `params` is automatic — that is what powers the
gradient API in §6.

### 4.5 Inner iterations via implicit diff

`set_theta_b` (outer secant on θ_B) and the point-D mass-matching secant
(`calc_lrc_de` / `find_point_e`) are root-finds. Wrap each as
`optimistix.root_find`; their gradients then come from the implicit function
theorem, so we never backprop through the secant iterations. This is the
single most important trick for keeping the autodiff graph small and the
derivatives exact.

---

## 5. Autodiff hazards & mitigations

| Hazard | Where | Mitigation |
|---|---|---|
| **Data-dependent loop length** (number of characteristic rows depends on geometry) | MOC march, kernel build | Fix a maximum grid size; `lax.scan` over a padded array with a validity mask. Masked nodes contribute zero residual. JAX requires static shapes under `jit`. |
| **Non-smooth post-processing** (monotonic-x filter, endpoint forcing, θ_N clamp) | `_construct_wall_from_ce`, `resample_wall_for_export`, `to_contour_dict` | **Delete them** in the JAX path (they are `REWRITE_PLAN.md` defects anyway). Endpoint closure becomes a residual, not a fixup. |
| **Sonic singularities** (`asin(1/M)`, `sqrt(M²−1)`) | primitives near throat | Guarded ops + `custom_jvp` finite-derivative overrides (§4.1). |
| **`atan2`, `hypot` at coincident nodes** (ds→0) | residual segments | `eps`-floors already present in NumPy (`max(..., 1e-9)`, `1e-12`); keep them, they are differentiable. |
| **`min`/`max` in valid-region inequality** (non-smooth at the active segment) | §2.E gate | For *gating*, keep hard `min`. For the §11 *sensitivity study*, use a smooth-min (log-sum-exp) variant so ∂(boundary)/∂p is defined. |
| **Branch selection** (`mach_from_area_ratio` sub/supersonic branch) | wall seed | Select branch outside the differentiated region (in the seed builder), pass the chosen branch as a static flag. |
| **NaN poisoning** (one bad node NaNs the whole Jacobian) | everywhere | `jax.debug` checks in tests; `jnp.where` guards; unit parity tests per primitive before integration. |

---

## 6. The gradient API (the tool's headline feature)

Once `u*(params)` is differentiable, expose:

```python
# raosim/jax/api.py
sensitivities = rao_sensitivities(config)   # one call, all gradients

sensitivities.dCf_dparams      # ∂Cf/∂(ε, length_pct, Rt, γ, θ_B, ...)
sensitivities.dCf_dwall        # ∂Cf/∂r at every wall node  -> tolerance field
sensitivities.dIsp_dgamma      # real-gas sensitivity (scalar)
sensitivities.jacobian         # full ∂r/∂u (for diagnostics / conditioning)
sensitivities.hessian_thrust   # ∂²(thrust)/∂(wall)²  (for §11 soft-mode study)
```

Immediate tool features that fall out:

- **Manufacturing-tolerance map.** Paint `|∂Cf/∂r_node|` onto the contour:
  which millimeters of the bell actually move performance. Add
  `plot_sensitivity_field(solution)` to `plotting.py` (Phase 13 already wants
  flowfield plots; this slots in).
- **Gradient-based design.** `maximize Cf s.t. (length, ε)` becomes a
  handful of Optimistix steps with exact gradients instead of the current
  scipy SLSQP wall optimizer (`rao_optimizer.py`, slated for deprecation in
  `REWRITE_PLAN.md` Phase 8).
- **Exact ∂/∂γ.** Sets up the Phase 10 real-gas work cleanly.

---

## 7. Validation strategy

Gates, in order — each must pass before the next:

1. **J1 — primitive parity.** `gas_dynamics_jax` matches `gas_dynamics` to
   1e-10 across M∈[1.01, 6], γ∈{1.14…1.4}. (`tests/test_jax_primitives.py`)
2. **J2 — residual parity.** `rao_residual` on a fixed state equals the
   NumPy `_scaled_rao_bvp_residual` + `_coupled_wall_residuals` to 1e-8.
   This proves the port changed *nothing physical*. (`tests/test_jax_residual_parity.py`)
3. **J3 — Jacobian sanity.** `jax.jacfwd(rao_residual)` matches a
   central-difference Jacobian to 1e-5 on a few states. Confirms autodiff is
   wired right. (`tests/test_jax_jacobian.py`)
4. **J4 — convergence (the payoff).** On the ε=10 / length_pct=80 / γ=1.4
   reference case, Optimistix LM drives `max_scaled` **below 2e-3** with
   `moc_compatibility_preserved=True` and `postprocessed=False`. This is the
   `REWRITE_PLAN.md` Phase 6 gate that currently xfails
   (`test_phase6_coupled_wall.py:378`). (`tests/test_jax_convergence.py`)
5. **J5 — chart benchmark.** Re-run Phase 7: θ_N/θ_E across the
   (ε, length_pct) grid within RMS 1.5° / max 3°
   (`test_rao_chart_benchmark_plan_targets`, currently xfail). Must use
   `angle_boundary_mode="free"` (REWRITE_PLAN §13.1) so it isn't circular.
6. **J6 — NASA oracle (later).** Feed the JAX path through
   `scripts/compare_nasa_reference.py` against `outputs_M3.5Perf/` once the
   kernel port matches (Phase 12 territory).

The whole existing 260-test suite must stay green throughout — the JAX path
is additive until it earns the default.

---

## 8. Phasing (maps onto REWRITE_PLAN phases)

| JAX phase | Deliverable | REWRITE_PLAN tie-in | Gate |
|---|---|---|---|
| **J0** | `raosim/jax/` skeleton, deps, `solver_backend` flag, CI job | infra | imports + smoke |
| **J1** | `primitives.py` + parity tests | substrate for all | J1 |
| **J2** | `residuals.py` + `pack.py`, residual parity | §2.A–F re-expressed | J2 |
| **J3** | `bvp.py` Optimistix LM, exact Jacobian, implicit diff | Phase 6 numerics | J3 |
| **J4** | **Convergence on reference case** | **closes Phase 6 xfail** | **J4** |
| **J5** | chart sweep under JAX | closes Phase 7 plan-target xfail | J5 |
| **J6** | gradient API + `plot_sensitivity_field` | Phase 13 + tool value | smoke + a known-sign sensitivity |
| **J7** | (optional) real-gas ∂/∂γ, NASA oracle | Phase 10 / 12 | J6 / parity |

J0–J4 is the spike that proves the thesis (exact Jacobians fix
convergence). J5–J6 turn it into the tool. J7 is the on-ramp to the §11
research.

---

## 9. Dependencies & environment

Add to `requirements.txt`: `jax`, `jaxlib`, `optimistix`, `equinox`.
(`jax[cpu]` is sufficient; no GPU needed at this problem size.)

Note: the checked-in `.venv` references Python 3.14 and is not portable —
stand up a fresh environment (`python -m venv .venv && pip install -r
requirements.txt`) for the JAX work. Pin JAX/jaxlib versions in
`requirements.txt` since their API moves; record the exact versions in the
J0 commit.

---

## 10. Risks & mitigations

- **Risk: exact Jacobians converge but to a *different* (non-Rao) point.**
  Mitigation: J2 residual parity guarantees identical physics; J5 chart
  benchmark guarantees the converged point is the published Rao optimum.
- **Risk: the marching grid's static-shape requirement bloats memory or
  hurts accuracy via padding.** Mitigation: size the padded grid from the
  known `lastKernelJ ≈ 57` of the M3.5Perf reference; mask is exact, padding
  only costs compute.
- **Risk: scope creep into a full JAX rewrite.** Mitigation: the §3 boundary
  is deliberately narrow; CEA/export/plotting/trajectory stay NumPy forever.
- **Risk: convergence still stalls (Jacobian wasn't the problem).** Then the
  blocker is the *seed* / kernel TT′ mismatch (`docs/legacy_code_audit.md`:
  Python axis-point M≈1.27 vs NASA 1.5), not the optimizer. J4 failing would
  *diagnose* this cleanly because exact Jacobians remove the FD-noise
  confound — itself a useful result.

---

## 11. Preserved novelty hooks (build later, on top of the tool)

The differentiable core makes these cheap; none are required for the tool.

1. **Soft-mode at the Guderley cliff.** Use `hessian_thrust` to track the
   spectrum of ∂²(thrust)/∂(contour)² as (ε, length_pct) approaches the
   valid-region boundary (§2.E). Hypothesis: a near-zero eigenvalue with a
   characterizable spatial shape appears as the smooth optimum loses
   existence — a variational bifurcation the 1968 theory predicts but never
   computed at this resolution. *Honest framing:* the boundary is classically
   known (Guderley); the **sensitivity/stability structure across it** is the
   open part.
2. **Real-gas migration of the cliff.** With ∂/∂γ available and the Phase 10
   `thermo.py` in place, map how the existence boundary itself moves between
   frozen and shifting-equilibrium chemistry for kerolox/methalox. The
   classical boundary is perfect-gas only.
3. **Start-line provenance propagation.** Exact ∂Cf/∂(start-line model
   params) quantifies how much Sauer vs Hall vs Kliegel–Levine actually
   changes the final optimum — a long-hand-waved modeling choice, now
   measurable.

---

## 12. Open decisions for you

1. **Where does `raosim/jax/` live** — new subpackage (recommended) vs
   methods inside existing modules?
2. **Default switch policy** — keep NumPy default until J5 passes (safe), or
   flip to JAX once J4 passes (faster iteration, more risk)?
3. **Padded-grid size** — fix from the M3.5Perf reference, or make it a
   config field?
4. **Pursue any §11 hook now**, or strictly tool-first until J6 ships?

---

### References

Repo: `REWRITE_PLAN.md` (governing physics plan, §2.A–F, Phases 6–7, §8.1,
§13.1); `docs/legacy_code_audit.md` (TT′ seed blocker); `raosim/gas_dynamics.py`
(`mstar_from_M:168`, `prandtl_meyer:152`); `raosim/moc.py:154-160` (Anderson Q±);
`raosim/rao_variational.py` (`_coupled_wall_residuals:2183`,
`_scaled_rao_bvp_residual:2704`, scipy `least_squares:2965`).

Literature (`propulsion_texts/`): Rao, *Exhaust Nozzle Contour for Optimum
Thrust*, Jet Propulsion 1958 (`RaoRecentDevinRockNozConfig.pdf` lineage);
Rao/Beck/Booth AIAA 99-2584 (`rao1999.pdf`); Östlund, KTH 2002
(`fulltext01.pdf`); Rice, JHU/APL RTDC-TPS-481 2003 (`20030067852.pdf`);
*Contoured Rocket Nozzles* (`978-3-7091-4745-0_18.pdf`).

External (verified June 2026): Guderley et al., *Continuous and discontinuous
solutions for optimum thrust nozzles of given length*, JOTA —
https://link.springer.com/article/10.1007/BF00934781 ; Optimistix (Rader,
Lyons, Kidger), arXiv 2402.09983, docs https://docs.kidger.site/optimistix/ ;
MOC + free-form-deformation shape optimization, CEAS Space Journal 2023 —
https://link.springer.com/article/10.1007/s12567-023-00511-1 ; adjoint design
in reacting flow —
https://www.sciencedirect.com/science/article/abs/pii/S0045793012001880 .

*Document owner: (you). Update phase gates as J0–J7 land. Keep §0's honesty
note — it is why the project is scoped tool-first.*
