# `raosim/moc.py` forward march — topology audit and remediation plan

> **STATUS (2026-09-21, later the same day): PHASES 0–2 LANDED; Oracle B
> PASSES.** `raosim/moc_bounded.py` (new) marches the bounded net RRC-by-RRC
> on the NASA/JHU topology (wall point = C− parent of `[1][j]`, previous
> axis point = C+ parent, `iLast[j] = iLast[j−1] + specialFlag`), reusing
> `nasa_moc._calc_interior_mesh_point` / `_calc_axial_mesh_point` and the
> arc unit processes verbatim (throat-arc rows are **bit-identical** to
> `build_kernel`, incl. NASA's axis closure of an r<0-truncated row). A
> general dθ-form wall point (`_calc_wall_point_raw_general`) + a
> geometry-factored NASA special point (`nasa_moc._calc_special_wall_point_at`)
> handle any bell wall; `intersect_wall_strict` never clamps; the exit lip is
> placed by the special-point process when the C+ leaves through the exit;
> each RRC is cropped to `x_E` (NASA `CropNozzleToLength`). **Oracle B
> (M3.5Perf wall, NASA-visible KL line, DTHETAB 0.5°, 0.5 s):** wall Mach vs
> `wall.out` RMS 3.1e-4; axis Mach vs `center.out` RMS 8.9e-3 (halves at
> 0.25°); exit plane **M = 3.5009 ± 0.0017, θ_rms 0.003°**; stream thrust
> **2517.72 vs 2518.05 lbf**, mass **35.989 vs 35.998 lbm/s**, initial-vs-exit
> drift 0.74 % vs NASA's 0.754 % (`summary.out` exit-plane block — its
> "2-D Gross Thrust 1971.99 lbf" is the THROAT figure). Gates in
> `tests/test_moc_bounded_net.py` (17). **Physics found by the audit:** the
> chart Bézier/TOP bell's first bell RRC crosses BD — a compression
> coalescence (internal shock) ~20 mm past N at Rt = 20 mm whose location
> converges with resolution (x 24.9→22.9 mm from 0.5°→0.1°); the march
> stops with `stop_reason="rrc_coalescence"` + `fold_location` instead of
> integrating through it (literature: TOP internal shock, Hagemann 1998 /
> Frey & Hagemann 2000). Consequently `rao_optimizer` seeds coalesce → its
> objective now penalises incompleteness and `moc_bell_nozzle` exports NaN
> exit metrics + `moc_net_complete=False` (experimental, unreachable from
> CLI). Also landed: `solve_interior_point(on_upstream="keep"|"raise"|"clamp")`
> (clamp no longer default; JAX kernel copy untouched pending its own
> validation), `march_coupled_net` deprecated (DeprecationWarning, pinned
> in tests), `CharRow.parents` explicit parent map consumed by
> `rao_variational.characteristic_net_links` (crossing/residual audits now
> follow the true topology), `rao_variational` `coupled`/`legacy` forward
> audit switched to `solve_flowfield` (`construction_diagnostics["forward_net"]`;
> incomplete net ⇒ `moc_compatibility_preserved=False`; 6 JAX net tests
> green, 12 min), `nasa_moc._make_throat_initial_line` raises on unknown
> methods (the old `'area_ratio'`/`'hall'` silently gave a flat M=1.001
> line — `set_theta_b` default fixed, `test_transonic_kernel` baseline
> re-pinned absolutely at 1.11e-4), registry `nozzle.moc` notes, docstrings
> (F9). Suites: numpy 74 + 17 + 45, JAX 136 + 14 + 6 — green. **Open
> (Phase 3/4):** `rao_variational._construct_wall_from_ce` still hard-codes
> `'area_ratio'` for its own arc line (unchanged); `build_kernel` cannot
> build the NASA-visible line at n>101 (its M=1.5 cap loop degenerates —
> pre-existing); M\* vs M convention flag not added; KL 1969 literature
> request; JAX port of the bounded march; promotion of `nozzle.moc` to
> full-net `reference_matched` awaits a second oracle (M4Perfect) and a
> shock-free Rao BVP wall run on host.

> **STATUS (2026-09-21): ANALYSIS + PLAN ONLY, NOTHING IMPLEMENTED.**
> An external review of `raosim/moc.py` claimed `march_coupled_net` is not a
> coupled net. This audit (6 parallel readers + 23 adversarial verifications +
> first-hand instrumented runs) **confirms every item and finds it materially
> worse**: the "starting line" is the downstream throat arc itself, the net lives
> entirely outside the wall, every wall point of every row is the single clamped
> point N, the net dies after `n_char` rows at ~30 % of the nozzle length, and the
> "exit plane" is a two-point interpolation (Cf 0.81 where ε=10/γ=1.2 gives ≈1.65).
> **Blast radius is small**: the MDO never uses this march (its contour is the
> fixed-topology Bézier; its Cf is 1-D isentropic), the CLI `rao-bvp` path and the
> B3 validator default to `wall_method="bde"` (NASA-port march, oracle-matched),
> and `rao_optimizer`/`method='moc'` is unreachable from any CLI. The defect
> reaches: (i) the `net_report` "forward MOC compatibility audit" under
> `wall_method` `coupled`/`legacy` (RaoSolverConfig default is `coupled`), (ii)
> `bell_nozzle_contour(method='moc')`, (iii) `flow_viz` when fed these rows.
> No prior repo assessment identified any of it; existing tests **enshrine** the
> shrinking triangle. Plan below: Phase 0 (honesty + failing gates) is a
> half-day; Phase 1 (bounded net on the NASA topology) reuses `nasa_moc`
> primitives; promotion gates are numeric and oracle-anchored.

## 1. How the contour is built today (data pipeline)

**MDO inner loop (`lrekit run`, `execute_engine_job`)** — `raosim/mdo/engine.py:271-335`
`solve_engine` closes an Optimistix Newton root on `[Rt, ṁ]` whose thrust term is
`jt.ambient_thrust_coefficient(eps, γ, Pc, Pa)` (engine.py:157-158, 306-307) — a
**1-D isentropic C_F × `mission.eta_CF`**; no wall geometry enters the closure.
After the root, `evaluate_contour` (engine.py:325-328) → `raosim/mdo/nozzle.py:103-126`
(single registered provider `rao_top_bezier_fixed_topology`, nozzle.py:30,
132-151; job default job.py:433) → `build_station_grid` (`raosim/mdo/grid.py:349-452`):
barrel/shoulder/cone/upstream arc, downstream arc `Rd = throat_rd_factor·Rt` swept to
θ_n, **quadratic Bézier** N→E with the tangent-intersection control point
(grid.py:400-423), `L_div = length_pct·(Re−Rt)/tan15°` (grid.py:365), θ_n/θ_e from
a JAX bilinear of the Rao chart tables (`nozzle_geometry.py:25-77`), station Mach
from `mach_from_area_ratio` (grid.py:435-440). The grid feeds cooling, mass,
collapse/buckling, envelope, chart-domain and monotonic-wall constraints only.
`raosim/moc.py`, `rao_variational.py`, `raosim/jax/moc_kernel.py` are **never
imported** by `mdo/engine.py`, `grid.py`, `nozzle.py`.

**B3 post-solve validator** — `raosim/mdo/contour_validation.py:934-1130`, opt-in
(`postsolve_contour_validator=None`, job.py:434-435; needs the B5
`rao_convergence_domain` artifact + four tolerances, job.py:481-530). Defaults
`solver_backend="jax"`, `wall_method="bde"`, 24×24 → 32×32 refinement, 0.25° kernel
step (contour_validation.py:177-217). Result is a verdict only (fail →
CANDIDATE_VIOLATES, unknown → INDETERMINATE; execute.py:237-261, 585-590); **nothing
feeds back into the geometry or NLP**. If the B5 lookup returns
`DOMAIN_REFUSED_INVALID_RAO_REGION` it fails *without* solving (1082-1183).

**`solve_rao_bvp`** — `raosim/rao_variational.py:5833-6706`. Seed:
`_initial_ce_from_kernel` (2712+) → `nasa_moc.set_theta_b`/`build_kernel`
(`nasa_moc.py:1469`: `_make_throat_initial_line` KL line at `Ru=1.5Rt`, then the
NASA RRC-by-RRC kernel march to θ_B; last RRC = BD). Unknowns (`_pack_bvp`
2957-3011): `[M_ce, θ_ce, r_ce, (wall if couple_wall), λ2, λ3, log C,
kernel_d_fraction, (pair_fractions)]`, x_ce reconstructed from the left-Mach ODE
(3014-3112). Residual groups (3944-4086) under `formulation="characteristic"`
(default, L666): mass, length, moc_cplus, ce_geometry, regularization, penalties,
algebraic_stationarity, wall_endpoint, wall_tangency (`CHARACTERISTIC_RAO_RESIDUAL_BLOCKS`
783-793). Backends (5937-6027): **`jax`** (default, L640) → `raosim/jax/api.py:53-132`
Optimistix LM, exact autodiff Jacobian on `raosim/jax/assembly.py:369 make_residual`
(a JAX re-expression of the same groups; NumPy kernel BD frozen in `StaticParams`),
sigmoid box bijection, weight ladder (1,10,30,100); with `solve_theta_b=True`
(L714) BD is recomputed in-graph by `theta_b_solve.py:92-136` + `moc_kernel.march_kernel`.
**`numpy`** → `scipy.optimize.least_squares` with FD Jacobians (6004-6020).
`max_nfev<=0` → residual only. **Wall** (6086-6162): `bde` →
`_wall_from_bde_region` (4192-4243): kernel arc wall points + `nasa_moc.calc_bde_region`
(2866) BFE rows with mass-matched wall points (`_calc_wall_contour_rows` 2783),
audited by `_bde_measured_topology_report` (4272). `coupled` →
`solve_wall_from_ce_coupled` (4560+, scipy strip solve of C− chords from the CE with
chart end angles); `legacy` → `construct_wall_from_ce_raw` (4488 → 2215, which
hard-codes `starting_line_method='area_ratio'` at 2301-2303). For `coupled`/`legacy`
the built wall is wrapped in `SplineWall` and **`moc.march_coupled_net` is run as a
forward audit** producing `net_report` / `moc_compatibility_preserved` (6117-6153).
Cf: `full_control_surface_thrust` (6258) else `cf_de`; ladder can never reach
`BENCHMARK_VALIDATED` (`BENCHMARK_VALIDATED_AT_RELEASE=False`, L861).

**`raosim/jax/moc_kernel.py`** ports the NASA dθ-form unit processes
(`interior_point_nasa` 388, `axis_point_nasa` 487, `arc_wall_point_raw_nasa` 519,
`special_wall_point_nasa` 580) and `march_kernel` (723-820, padded `lax.scan` to BD).
It marches **only the throat-arc kernel**; no BDE region, no wall-reflected net;
`build_start_line` is a stub (823-828). BD bit-parity with `build_kernel` 4.8e-10.
It also carries the moc.py downstream clamp verbatim (200-207).

**CLI** — `scripts/run_nozzle.py` → `raosim/run_nozzle.py:_solve` (1407-1450):
default Bézier; `--contour-method rao-bvp` → `wall_method="bde"`; in MDO mode it is
handed only to the traditional re-evaluation (1806-1849 → `design_nozzle_v2`),
never to the MDO core. `bell_nozzle_contour` (`nozzle_geometry.py:151-405`)
default `'bezier'`; `'moc'` → `rao_optimizer.moc_bell_nozzle` (→ `moc.solve_flowfield`);
`'rao_variational_moc'` → BVP with `rao_moc_wall_method="coupled"` (L173).
`rao_optimizer` is reachable only via `method='moc'`, `DesignInput.method="moc"`,
`benchmarks.run_benchmark(method="moc")` — no CLI/MDO path.

## 2. Findings (each verified; file:line as of HEAD 6c8ded3)

| # | Finding | Evidence |
|---|---|---|
| F1 | **Boundary points are dead ends.** `parent_start=1` (moc.py:512), interior pairs `(prev[j+1],prev[j])` for `j∈[1,n−3]` (513-516), axis from `prev[1]` (500-501), wall from `prev[-2]` (520). `prev[0]` (axis) and `prev[-1]` (wall) are never parents. | id()-instrumented run: 0/78 boundary points used as parents at n=40; every interior point used exactly twice. Wall dead-end dates from the first commit 11e6f07 (2026-03-06); the docstring "wall point at row k is a parent for row k+1" (L11, L476) has never been true. Axis exclusion added in c9261ae (2026-06-15, "march loop fix") with the comment at 507-511. |
| F2 | **Net shrinks by one per row and never reaches the exit.** Loop ends at `len(prev_pts)<3` (495); exit test (527) never fires. | Row lengths `[40,40,39,…,3,2]`; farthest point x=0.039 m of x_end=0.129 m (30 %). |
| F3 | **The "starting line" is the wall arc**, not an axis-to-wall data line: `x=Rd·sin(ang), r=Rt+Rd(1−cos ang)` (432-434). | 100 % of interior points lie **outside** the wall in 54 configurations (γ 1.2/1.4 × 3 methods × 3 geometries × n 10/20/40). With KL, θ is the KL field angle, not the wall angle (up to 16.8° off; ~5.5 % of choked ṁ "crosses" the wall). |
| F4 | **Every wall point of every row is the same point N.** The C+ from `prev[-2]` hits the wall upstream of `x_start`; `SplineWall.intersect_char` clamps `x_guess` to `[x_start,x_end]` (`wall_model.py:160`). | Distinct `wall.x` per run = 1 in all 54 configurations; θ_w=30° by construction, so the wall-tangency "residual" is trivially zero. |
| F5 | **Downstream clamp returns a non-solution** (moc.py:191-195, from 7d25621 2026-04-16; ported to `jax/moc_kernel.py:200-207`). | Fires 0× for KL/area_ratio but 18/171 (n=20) – 106/741 (n=40) for `sauer_modified`; sample: point moved 0.35 mm off the C−, returned ν=1.146° vs 0.949° from compatibility at the clamped location. |
| F6 | **`sample_exit_plane` finds zero genuine intersections**; the 30-sample profile is the two-point branch (654-668) between the nearest axis point (x=0.039 m) and N; `Re=max wall r`=Ny=0.021 m, not the real 0.063 m (562). | Cf = 0.81 / 0.81 (n=20/40) for ε=10, γ=1.2 (quasi-1D M_e=3.28, vacuum Cf≈1.65). This is the objective `rao_optimizer.objective` maximises (163-176). |
| F7 | **Tests enshrine the defect.** `test_moc.py:210-234` asserts one-fewer-interior-per-row; 167-208 check only `is not None` and `θ_w == wall.theta(x)`; 298-319 exit bounds (θ_max<30°, M_std<2.2) are met by the 2-point interpolation; `rao_variational.characteristic_net_links` (4906-4990) copies the parent rule verbatim (`parent_start` 4953, `prev_pts[-2]` 4970); `moc_net_compatibility_report` (5042-5110) never checks wall containment or exit reach. All pass today. | In-session pytest. |
| F8 | **Never assessed before.** No docs/*.md, KG record, or commit message treats the shrink/dead-end as a defect; KG `basis raosim/moc.py:march_coupled_net` records the c9261ae comment as a *validity statement*; REWRITE_PLAN.md:67-68 endorses the march; JAX plan L574-576 only notes start-line inconsistency. Related precedent: REWRITE_PLAN.md:35 endorsed the pre-c9261ae Q± form that c9261ae later found spurious. | history reader, git `log -S`, `blame`. |
| F9 | **Docstring/default drift.** `solve_flowfield` default `'kliegel_levine'` (813) vs doc "area_ratio default" (828-829); same in `nozzle_geometry.py:196-197`, `rao_optimizer.py:15-20` (whose `optimize_wall`/`moc_bell_nozzle` in fact still default to `'area_ratio'`, L119/306); `nasa_moc.build_kernel` docstring "TT' at x=0" (1493-1494) is false (axis point at x/Rt≈0.52); `nasa_moc.set_theta_b` default `'hall'` (3086) is unrecognised by `_make_throat_initial_line` → flat M=1.001 line (704-707; latent, all callers pass a method). | |
| F10 | **`sauer_modified` is not Sauer's formula** (radial-offset polynomial on the arc; M=1.000 at the throat wall point where KL gives 1.131; Young p.57 eq 3.4-3.6 has α multiplying x*, not a radial offset). In `nasa_moc` it gives M=1.0001 for every r≤Rt. | moc.py:423-427, 446-451; nasa_moc.py:696-703. |
| F11 | **KL kernel reports M\* as M.** Young App. A eq A.2 and p.57 eq 3.7: `U=ũ/a*, V=ṽ/a*`, `M=√((1+u)²+v²)` — the critical Mach number; NASA `KLThroat` C++ 3167-3168 assigns `mach = Q`; `transonic_kernel.py:294` copies it. Anderson eq 3.37: at Q=1.131 (Rc/Rt=1.5, γ=1.4) true M=1.164 (+3 %); at NASA's cap Q=1.5, M=1.73. `p_over_p0` (296-301) is then evaluated at the wrong M. Inherited from the lineage — oracle parity *requires* keeping it as a mode. The only M\*→M attempt, `nasa_moc._hall_throat_line` (356-421), is dead code and inverted (L400). | Verified against `Young_Thesis.md` PDF 114-115 / eq 3.7, C++ 3160-3170. |
| F12 | **KL validation on record is thin.** `test_transonic_kernel.py`: one throat-plane wall Mach vs NASA `wall.out` row 0 (1.17779, at x=0 where all four typo-corrected z-terms vanish), 1-D limit, monotonicity, Cuffel-Back-Massier rounded narrative anchors (±0.03/±0.2), and "KL not worse than 2.5× area_ratio BVP residual". No comparison to `TT'.out`, no mass-flow closure, no sonic-line shape, no coefficient check (KL 1969 not in corpus). `nasa_visible_kliegel_levine` reproduces `TT'.out` to \|ΔM\| 4.9e-6; the corrected branch differs by \|ΔM\| 0.36, \|Δθ\| 2°, \|Δx\| 0.23 Rt (different cap logic + corrected z-terms). Mass-flow closure: NASA's own TT' 0.985 of 1-D choked; port KL line 1.009 (stored M) / 1.005 (M\*→M). | transonic reader + `docs/nasa_tt_prime_provenance.md:98-135`. |

Verification note: the 23 adversarial votes that completed all confirmed the
pipeline claims (§1) — 5 "refutations" were precision refinements (e.g. `formulation`
default at L666 not 677; B3 fails without solving when B5 refuses; `solve_theta_b`
sub-path). The march claims F1-F7 were confirmed independently by three routes
(first-hand instrumented run; march reader's 54-configuration census; NASA reader's
scratch run of NASA's own row 0 through `march_coupled_net`: 32 rows shrinking
101→70, final axis point x=1.11 vs NASA BD axis end x=3.83).

## 3. How our MOC compares to the NASA MOC in the repo

**NASA C++ topology** (`MOC_Grid_BDE/MOC_GridCalc_BDE.cpp`): net indexed `[i][j]`,
j = RRC (C−), i = 0 at the wall → axis. `CalcRRCsAlongArc` (1030-1119) per RRC:
`CalcArcWallPoint` (835-947; C+ from `[1][j−1]` terminates on the arc, θ = arc angle;
`CalcSpecialWallPoint` 2360-2454 inserted when Δθ > DTHETAB) → `iLast[j] =
iLast[j−1] + specialFlag` (1069) → `CalcInteriorMeshPoints` (2466-2686: point
`[i][j]` = C+ from `[i+1][j−1]` ∩ C− from `[i−1][j]`; **at i=1 the C− parent is the
wall point just computed; the previous axis point is a C+ parent**, with the
off-axis coefficient substitution 2521-2536) → `CalcAxialMeshPoint` (2262-2310,
single parent `[iEnd−1][j]`). Point count is **constant** (+1 per special row; +1 per
RRC in `CalcRemainingMesh` 1132), reduced only by r<0 truncation (2662-2666,
1137-1155). Every RRC's ṁ is checked against the start line (2 % abort, 1077-1089).
Downstream of B there is **no wall BC**: `CalcBDERegion` (3258-3319) marches
backward from DE, `CalcWallContour` (1167-1330) places wall points by mass-flow
matching, `CropNozzleToLength` (1341-1429) interpolates each RRC to x_E, exit
plane = trapezoidal ρu cosθ and stream thrust with initial-vs-exit ṁ % reported
(IO 189-240). The "point behind previous RRC" crossing check is commented out
(2671-2674).

**Python port `raosim/nasa_moc.py`** reproduces this exactly: `_rrc_march_step`
(1404-1466) wall-first, `_calc_interior_mesh_point` (1195-1358, `ii=i/i+1`,
`p2 = new_wall_first[i−1]` → wall point is the C− parent), `_calc_axial_mesh_point`
(1360-1402); per-RRC counts `[101, 94×17, 95 … 121]` identical to `TT'BF_Kernel.out`.
It never calls `march_coupled_net`; it reuses `moc.solve_interior_point`/`solve_axis_point`
only inside `_calc_remaining_mesh_row` (2634, 2672) with NASA's topology.

**Oracle test status (run 2026-09-20; 49 passed, 0 failed, 2 slow deselected):**
`test_nasa_kernel_march_parity` 8 (TT' \|Δx,Δr\|<5e-6, \|ΔM\|<5e-5; BD vs
`LastKernel.out` gate 5e-4/2e-3, measured 1.4e-5 / 8.7e-5 over all 58 rows);
`test_nasa_port` 2 (wall r RMS vs `wall.out` gate 1e-3, measured 1.84e-4, max 4.79e-4;
exit (12.5357, 2.5954) vs NASA (12.5363, 2.5955); BD ṁ within 2 %);
`test_characteristic_pairing` 5 (C− RMS on kernel RRCs 2.3e-6; C+ on derived LRC <1e-6);
`test_nasa_kernel` 16(+1 jax); `test_nasa_moc_topology` 2; `test_compare_nasa_reference`
2 (bookkeeping only, `--skip-solve`); `test_jax_moc_kernel` 13 (BD parity 4.8e-10).

**Verdict:** the repo's *NASA-port* march is oracle-matched at the level of every
kernel node and the BFE wall. The repo's *own* march (`moc.py`) has **never** been
compared to NASA on any full-net quantity — only its unit process
`solve_interior_point` is checked against one oracle node
(`test_characteristic_pairing.py:109-146`, rel 2e-3 M / 0.2° θ) — and when fed NASA's
own row 0 it diverges immediately (row-1 axis x 0.7266/M 1.5013 vs 0.6863/1.4780; B
wall Mach 1.683 vs 1.639). `compare_nasa_reference.py`'s `center.out` comparison of
`solution.characteristic_net` has never been run without `--skip-solve`.
`MODEL_NOZZLE_MOC` (`model_registry.py:456-465`, `reference_matched_subset`) is
honest only if "subset" is read as *the nasa_moc port + unit processes*, which is
what its verification list names; it does not cover `march_coupled_net`,
`sample_exit_plane`, `solve_flowfield`.

## 4. What the literature prescribes (corpus, page-anchored)

Zucrow & Hoffman Vol. 2 is **not in the corpus** (cited only in reference lists);
nothing below relies on it.

- **Compatibility form** — Anderson MCF 3rd ed. eqs 11.52/11.53 (PDF p.422, book
  p.406): `d(θ+ν) = dr/[r(√(M²−1)−cotθ)]` on C−, `d(θ−ν) = −dr/[r(√(M²−1)+cotθ)]` on
  C+; identical to `d(θ±ν) = ±(sinθ sinμ/r)ds` (MIT 16.512 Lecture 4-5 PDF p.16
  writes exactly this Δm form). `moc.py:121-135` already uses this pairing (oracle
  validated 2026-06-11). **Keep.**
- **Unit processes** — Anderson §11.5 (PDF 409-412): interior 3 = C− from 1 ∩ C+
  from 2, averaged slopes; wall point 5 from one interior parent 4, θ_5 = wall
  slope, ν_5 from the invariant. Axis: Anderson handles by symmetry (PDF 415);
  explicit process in Lecture 4-5 (PDF 17): single parent, θ=0, r=0,
  `x = x_a + r_a·tanα`, source term at the parent — the NASA `CalcAxialMeshPoint`.
- **Bounded-net topology** — Rice/NASA manual `20030067852.md` PDF 7-9, 17: an LRC
  from initial-line point (1,0) reaches the wall giving (0,1); an RRC launches from
  it; (1,1) = that RRC ∩ LRC from (2,0); "as a LRC reaches the nozzle wall it is
  reflected toward the centerline"; centerline by symmetry. Count per RRC does not
  shrink. Anderson's shrinking count in Table 11.1 (PDF 418) is *wave cancellation*
  in a design (straightening) nozzle, not a bounded reflecting net.
- **Bad intersections** — never patched. Young PDF 60-63 (eq 3.10) counts crossings
  incl. `x_{i+1,j} < x_{i,j}` and rejects the solution (fitness weights Table 3.2,
  PDF 76); Rice manual PDF 21: refine DTHETAB / number of starting characteristics;
  NASA accepts a non-converged iterate only if min ΔM ≤ 5e-4 else SEC_FAIL (2643-2659).
- **Exit plane / thrust** — Rao 1958 eqs [1],[2] (PDF 2): ṁ and F integrated along
  an *oblique* control surface; Young eqs 3.25-3.29 (PDF 81-82): along a
  characteristic + wall-pressure term; NASA: `CalcMassFlowAndThrustAlongMesh`
  (3183-3236) per RRC, `CropNozzleToLength` interpolates each RRC to x_E, trapezoidal
  `Σρu cosθ dA` and `Σ(ρu²cosθ+p)dA` with initial-vs-exit ṁ % (IO 189-240).
- **Start line** — Anderson PDF 411-412, 419, 470: initial-data line at/downstream
  of the limiting characteristic; curved sonic line preferred for rapid-expansion
  throats. SP-8120 PDF 24, 79: series transonic methods accurate to Ru/Rt≈1.0,
  Kliegel-Levine to 0.6 (Cuffel), recommend Ru/Rt>0.6, ≈1.0, constant γ. Young PDF
  55-59: cannot start on the sonic line; start on the Sauer constant-M line from the
  geometric-throat wall point. NASA/Rice PDF 19 + C++ 2833-2865, 3103-3120: modified
  Hall/KL line walked along an RRC from the wall, M capped at 1.5, r ∝ sin^1.5.
- **Does Rao's design need a wall-reflected net?** Only in the kernel over the
  prescribed arc TB (Rao 1958 PDF 2 line 86); downstream of B the control surface
  DE, the BDE characteristic IVP and the streamline through B,E replace the wall
  (PDF 4-5, Fig. 4; Rao 1999 PDF 2; Young PDF 66-68). A wall-reflected *analysis*
  net is needed only to **verify** a finished contour (Young PDF 35, SP-8120 PDF 35)
  — which is exactly the job `march_coupled_net` is used for.

## 5. Plan

Principles: (a) the `wall_method="bde"` design path is already the NASA topology and
is oracle-matched — do not touch it or the oracle suites
(`test_nasa_kernel_march_parity`, `test_nasa_port`); (b) `tests/test_characteristic_pairing.py`
stays green through every step; (c) the NumPy march is validated before anything is
ported to JAX; (d) every numerical-method choice cites §4.

### Phase 0 — Honesty + failing gates (no numerics change; ~half day)

0.1 **Documentation truth**: rewrite `moc.py` module docstring (5-10, 29-36) and
`march_coupled_net` docstring (469-478) to state what it does; fix `solve_flowfield`
828-829, `nozzle_geometry.py:196-197`, `rao_optimizer.py:15-20`, `nasa_moc.build_kernel`
1493-1494; make `set_theta_b`'s `'hall'` default (3086) either a real branch or a
`ValueError` in `_make_throat_initial_line` 704-707.

0.2 **Registry + KG**: `MODEL_NOZZLE_MOC` (`model_registry.py:456-465`) gains
`known_limitations`: "forward analysis march (`march_coupled_net`/`solve_flowfield`/
`sample_exit_plane`) is not a bounded net; no full-net oracle comparison; Cf from it
is invalid". Re-harvest so `kg.sh basis raosim/moc.py:march_coupled_net` reports a
limitation, not a validity statement.

0.3 **Stop consuming the invalid audit**: in `rao_variational.py:6117-6153`, for
`wall_method in {"coupled","legacy"}` set `moc_compatibility_preserved=None` with
`warnings += ["forward net audit is not a bounded net (see MOC_MARCH_TOPOLOGY_
REMEDIATION_2026-09-21.md); use wall_method='bde'"]`; `VariationalPostSolveConfig.
__post_init__` (contour_validation.py:287) rejects non-`bde` until Phase 2 passes;
`bell_nozzle_contour(method='moc')` tags reliability `invalid_exit_plane` and
`rao_optimizer` refuses to report `Cf` (returns `None` with reason).

0.4 **Failing-by-design regression tests** (`tests/test_moc_bounded_net.py`,
`@pytest.mark.xfail(strict=True)` until Phase 1): (i) every axis and wall point of
row k is a parent of some point in row k+1; (ii) point count per characteristic is
non-decreasing except at r<0 truncation; (iii) all interior points satisfy
`r < wall.r(x)` (containment); (iv) the last wall point satisfies
`x ≥ wall.x_end − 1e-9` and distinct wall x count ≥ n_rows−1; (v) per-characteristic
ṁ within 2 % of the start-line ṁ (NASA rule, C++ 1077-1089); (vi) exit-plane M
within 5 % of `mach_from_area_ratio(ε)` at the axis and `θ_rms < 2°` for a
conical-free MLN test wall. Replace `test_triangular_row_topology_consistency`
(`test_moc.py:210-234`) with (ii). Delete the `x3 < x_min_parent` clamp branch from
the unit-process oracle test expectations once Phase 1 removes it.

### Phase 1 — Bounded net on the NASA topology (NumPy; ~2-3 days)

1.1 **New `raosim/moc.py::march_bounded_net(start_line, wall, gamma, *, dtheta_limit,
mass_tol=0.02, on_invalid="refine")`** marching **RRC by RRC, wall → axis**, reusing
`nasa_moc._calc_interior_mesh_point` (1195-1358; NASA dθ-form, on-axis coefficient
substitution) and `_calc_axial_mesh_point` (1360-1402). Rows are *characteristics*,
not zig-zag fronts. Parents per §4: wall point on RRC j from `[1][j−1]` via C+;
interior `[i][j]` from C+ of `[i+1][j−1]` (or `[i][j−1]` after a special insertion)
and C− of `[i−1][j]` (so `[0][j]`, the wall point, **is** a parent); axis from
`[iLast−1][j]`. Count per RRC = `iLast[j−1] + specialFlag`.

1.2 **Wall point for a general wall**: keep `solve_wall_point` (259-343; tangency BC,
`intersect_char`) but (a) make `SplineWall.intersect_char` (`wall_model.py:130-166`)
return `None`/raise `WallIntersectionOutOfRange` instead of clamping to
`[x_start, x_end]`; (b) when the C+ lands upstream of the previous wall point, insert
a NASA-style special wall point at `min(θ_prev + dθ_limit/2, …)` (C++ 922-931,
2360-2454 → port `nasa_moc._calc_special_wall_point` 1068 generalised to
`wall.theta(x)`); (c) on the throat arc use `_calc_arc_wall_point_with_special`
(1173-1192) unchanged so rows up to B are bit-identical to `build_kernel`.

1.3 **Start line**: `solve_flowfield` must start from an **axis-to-wall data line**
(`nasa_moc._make_throat_initial_line`, KL at `Ru`, r ∝ sin^1.5, M ≤ 1.5 cap — §4
NASA/Rice PDF 19) and treat the throat arc T→N as *wall* (RRC sources), not as
data. Rename the current `approximate_starting_line` to `throat_arc_wall_points`
(θ = wall angle, as `sauer_modified`/`area_ratio` already set) and deprecate the
name; `_construct_wall_from_ce` (2301-2303) honours `config.starting_line_method`.

1.4 **Remove the clamp** (moc.py:191-195 and `jax/moc_kernel.py:200-207`): on
`x3 < min(parent x)` return an `InvalidCell` sentinel; `march_bounded_net` responds
per `on_invalid`: `"refine"` halves the wall Δθ (insert special point) and retries
once, else `"reject"` raises `CharacteristicCrossing` (Young eq 3.10 / Rice PDF 21).
Also accept a non-converged corrector only if `|ΔM| ≤ 5e-4` (C++ 2643-2659).

1.5 **Exit plane and thrust** (`sample_exit_plane`, `compute_exit_thrust`): march
until the wall point passes `x_E`, then **interpolate each RRC to x = x_E**
(`CropNozzleToLength` 1341-1429) — the exit table is one point per RRC; thrust by
trapezoidal `Σ(ρu²cosθ + (p−p_a))dA` and ṁ by `Σρu cosθ dA` with the
initial-vs-exit ṁ % returned (IO 189-240; Rao eqs [1],[2] for the oblique
control-surface variant already in `full_control_surface_thrust`). Delete the
two-point `np.interp` branch (654-668) and the `Re = max wall r` anchor (562).

1.6 **Topology helpers**: rewrite `rao_variational.characteristic_net_links`
(4906-4990) and `characteristic_net_compatibility_residuals` to read the new
`[i][j]` parent map stored on the net (do not re-derive it from row order);
`moc_net_compatibility_report` (5042-5110) adds containment (iii), reach (iv) and
ṁ (v) gates. `flow_viz`/`moc_diagrams` consume the same structure.

### Phase 2 — Validation gates (all numeric; promotion criteria)

2.1 **Oracle A (kernel identity)**: `march_bounded_net` from NASA's `TT'` with the
throat `ArcWall` to θ_B must reproduce `build_kernel` rows **bit-for-bit** (max
\|ΔM\| ≤ 1e-12) — the arc-wall rows are the same primitives.

2.2 **Oracle B (full net of a known wall)**: build a `SplineWall` from
`outputs_M3.5Perf/wall.out` (throat arc + BFE contour), start from `TT'.out`
(`nasa_visible_kliegel_levine`), march to x_E; gates: axis Mach vs `center.out`
RMS ≤ 2e-3, wall Mach vs `wall.out` RMS ≤ 2e-3, exit-plane thrust vs `summary.out`
1971.99 lbf within 0.5 %, ṁ initial-vs-exit within 1 % (NASA's own figure is
reported in `summary.out`; match it, do not beat it). Persist via
`scripts/compare_nasa_reference.py` **without** `--skip-solve` so the `center.out`
comparison finally has a value.

2.3 **Perfect-nozzle uniformity**: `3D_MOC/outputs_M4Perfect` wall → exit
`θ_rms < 0.5°`, `M_std < 0.02`, mean M within 1 % of 4.0 (Rice manual PDF 36-38
reports 4.027-4.067 for its own run — use ±2 % of that band, not 4.000).

2.4 **Resolution convergence**: n_char 20/40/80 → exit Cf monotone, successive
change < 0.2 %; per-RRC ṁ drift < 2 % at every resolution.

2.5 Pairing suite green; oracle suites untouched; `test_moc_bounded_net` xfails
flip to pass (strict).

### Phase 3 — Transonic start line (independent of Phase 1; ~1 day)

3.1 **Pin the oracle line**: test `nasa_visible_kliegel_levine` vs `TT'.out`
(\|ΔM\| ≤ 1e-5, \|Δθ\| ≤ 1e-5°) — currently measured 4.9e-6 but unpinned.
3.2 **Physical gates for every method**: M>1 everywhere, θ=0 on axis, sonic line
concave-upstream (`x_sonic(axis) > x_sonic(wall)`), ṁ closure within 1.5 % of 1-D
choked (NASA's own line: 0.985).
3.3 **M\* vs M** (F11): add `transonic_kernel.kliegel_levine(..., mach_convention=
"nasa_visible"|"true_mach")`; default stays `nasa_visible` for oracle parity; the
BVP/Phase-1 march gets an explicit config flag; record the 3 %/15 % consequence in
the docstring; fix or delete the dead, inverted `_hall_throat_line` (356-421).
3.4 **Retire `sauer_modified`** (F10) or re-derive it from Young eqs 3.4-3.6 with
α on x\*; do not keep a formula that gives M=1.000 at the throat wall.
3.5 **Literature request**: Kliegel & Levine 1969 (AIAA J 7(7) 1375-1378) into
`propulsion_texts/` so the four coefficient corrections in `transonic_kernel.py:16-57`
can be checked against Table 1; add to `docs/LITERATURE_REQUESTS.md`. Range gate
from SP-8120 PDF 24/79: warn when `Ru/Rt < 0.6`.

### Phase 4 — Promotion and follow-through

4.1 `MODEL_NOZZLE_MOC.validation_level` → `reference_matched` (full net) only when
2.1-2.5 pass on host; `rao_optimizer` Cf re-enabled; `contour_validation` accepts
`coupled`/`legacy` again with the new audit; README "coupled wall feedback" wording
corrected.
4.2 JAX: port `march_bounded_net` only after 4.1 (padded static shapes as in
`march_kernel`; the special-point insertion is the traced-width problem already
solved for the kernel). Remove the JAX clamp in the same change.
4.3 `nasa_moc._hall_throat_line` and `calc_arc_wall_point` (public, uncalled) —
delete or test.

## 6. Risks

- Phase 1 changes the *forward analysis* only; the BDE design wall is untouched, but
  `rao_variational_moc_contour` defaults to `wall_method="coupled"` — Phase 0.3
  will surface `moc_compatibility_preserved=None` there until Phase 2.
- Oracle B depends on reading `wall.out` as a general wall; the NASA wall is a
  streamline, so tangency at its points is only as exact as the file (r RMS 1.8e-4).
  Gate 2.2 tolerances are set at NASA-port accuracy, not tighter.
- The special-wall-point insertion on a general spline wall has no NASA precedent
  (NASA only does it on the arc); Young's boundary-by-tangency (PDF 66-68) is the
  literature anchor; expect tuning of `dtheta_limit` on high-curvature bells.
- M\* convention (F11) is shared with the NASA oracle; flipping the default would
  break parity gates by design — keep it a flag with both values tested.
