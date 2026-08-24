---
title: H1 host evidence — production CEA tables and propellant survey
date: 2026-08-22
version: "1.1"
status: complete_for_four_certifiable_catalog_pairs
---

# H1 host evidence: held-out CEA tables and propellant survey

## Scope and environment

This is the host evidence for Completion Plan H1 / Track B1. It covers the four
catalog pairs whose RocketCEA reactant cards are sufficiently identified:
LOX/RP-1, LOX/LCH4, LOX/LH2, and N2O4/MMH. N2O/Ethanol was deliberately not
sampled: RocketCEA's `N2O` and `N2O_nbp` cards encode different inlet states,
and the catalog does not yet make that state choice consistently.

- Base Git HEAD: `71206464eee9963d5687ead9e46d3ee23ec6aaf9`
- Python: `3.12.13`
- RocketCEA: `1.2.3`
- Table schema: `1`; interpolator: `tensor-pchip-hermite-v1`
- `raosim/mdo/properties.py` worktree SHA-256:
  `f98f35f92b19aa9e5912ac590747d8cb226983c0d1ea045c0f431b22490691dd`
- `scripts/sample_cea_surface.py` worktree SHA-256:
  `de02ed257aa40c0419b5cab97ce8e6fdd7678f4493f94c36efc4d39e7d684cc5`

The worktree hashes are recorded because the B1 implementation was uncommitted
on top of the stated base HEAD when this evidence was generated. Default
field-specific tolerances were used unchanged. A tolerance ratio is
`abs(error)/(atol + rtol*abs(reference))`; every comparison must be `<= 1`.
`--holdout-count 0` checks one off-node point, including cell-local centered
backend differences on both axes, in every fit-grid cell.

## Production-table commands and results

Exact commands:

```bash
.venv-jax/bin/python scripts/sample_cea_surface.py --oxidizer LOX --fuel RP-1 --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 17 --of-min 1.6 --of-max 3.4 --n-of 21 --out builds/cea_tables/lox_rp1_frozen_v1.npz --holdout --holdout-count 0 --report builds/cea_tables/lox_rp1_frozen_v1.validation.json

.venv-jax/bin/python scripts/sample_cea_surface.py --oxidizer LOX --fuel LCH4 --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 17 --of-min 2.2 --of-max 4.4 --n-of 21 --out builds/cea_tables/lox_lch4_frozen_v1.npz --holdout --holdout-count 0 --report builds/cea_tables/lox_lch4_frozen_v1.validation.json

.venv-jax/bin/python scripts/sample_cea_surface.py --oxidizer LOX --fuel LH2 --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 17 --of-min 3.5 --of-max 8.0 --n-of 25 --out builds/cea_tables/lox_lh2_frozen_v1.npz --holdout --holdout-count 0 --report builds/cea_tables/lox_lh2_frozen_v1.validation.json

.venv-jax/bin/python scripts/sample_cea_surface.py --oxidizer N2O4 --fuel MMH --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 17 --of-min 1.3 --of-max 2.8 --n-of 21 --out builds/cea_tables/n2o4_mmh_frozen_v1.npz --holdout --holdout-count 0 --report builds/cea_tables/n2o4_mmh_frozen_v1.validation.json
```

All four commands exited `0`; no grid refinement was required.

| Pair | Grid | Cells checked | Table content SHA-256 | Worst tolerance ratio |
|---|---:|---:|---|---:|
| LOX/RP-1 | 17×21 | 320 | `a65fd05ca3a9a5254879037a96bddb5bd6854e7c5bef434b9a87fffa780e8fdd` | 0.361153078 (`Tc`, d/O-F) |
| LOX/LCH4 | 17×21 | 320 | `2d4ea43783bc99ac1eed43aacdedb82d33f9dcf4167e5c933c0ccb62b065192e` | 0.231320492 (`Tc`, d/O-F) |
| LOX/LH2 | 17×25 | 384 | `33ffca0b8693dc651efaa696fea296c3c1a5fd03d8428ecfcc0ef1db6fce709c` | 0.123881851 (`Tc`, d/O-F) |
| N2O4/MMH | 17×21 | 320 | `e039f44befc96484f74b76ad0696fea8e4e849517c79e3325c6c237612ffa9b7` | 0.436533661 (`Tc`, d/O-F) |

The detailed value, d/dPc, and d/dO-F results and the exact tolerances are in
the corresponding hash-linked `builds/cea_tables/*.validation.json` files.
RocketCEA rejects catalog spelling `LCH4`; the methalox run used its stock
liquid-methane card `CH4` while retaining `LOX/LCH4` as the hashed table
identity. The RP-1 run analogously resolved the registered `RP1` card.

## Distribution closure

The four accepted `.npz` files are packaged under `raosim/data/cea_tables/`.
`catalog_v1.json` pins each logical content hash and exact file hash along with
the pair, domain, grid, table schema/interpolator, RocketCEA version/cards, and
complete-cell validation status. The manifest also records N2O/Ethanol as
blocked with the unresolved `N2O` versus `N2O_nbp` state-selection reason.

`load_chamber_surfaces("catalog", expected_propellant=...)` resolves the
selected mission pair through `importlib.resources`, recomputes both hashes,
and checks the manifest against the table payload. The mission retains the
portable literal `cea_table_path="catalog"`; loaded-surface provenance names
the exact package resource. Explicit `.npz` paths are unchanged. An empty
table selection still uses the constant-property fallback, so packaging these
tables does not silently change default physics.

Distribution-closure worktree SHA-256 values:

- `raosim/mdo/properties.py`:
  `f98f35f92b19aa9e5912ac590747d8cb226983c0d1ea045c0f431b22490691dd`
- `pyproject.toml`:
  `34a2c24d585c52fd2bc504beccddc0ebe950cc1abbc7f4ddc265e612d925a58f`
- `raosim/data/cea_tables/catalog_v1.json`:
  `c99688ebe35597cd5c97dde2f1287f3b38ffd305f92c387e81805fa2a81912a4`

Packaging verification built `lrekit-0.1.1-py3-none-any.whl` with:

```bash
.venv-jax/bin/python -m build --wheel --no-isolation --outdir /tmp/raosim-cea-wheel.J4y0pb/dist
```

The wheel was installed with `--no-deps` into an isolated target and imported
from `/tmp`, outside the source checkout. All four `catalog` resolutions loaded
from the installed wheel and reproduced the content hashes in the table above.

## Located survey and rerun definition

The earlier feasibility survey was not a reusable script. It was recorded
manually in `docs/MDO_GUIDE.md` by commit `e57c93e`, including the old
LOX/LCH4-infeasible and approximately 183 krpm LH2 observations. The guide's
documented optimization baseline is a 13 kN, sea-level, single-point
epsilon-constraint solve. This rerun makes that baseline explicit:

- thrust `13,000 N`, sea-level ambient, CLI-default burn time `10 s`;
- `Isp >= 200 s`, default `min_dry_mass_partial` objective;
- O/F is optimized over the validated table domain;
- all manifest hard constraints are enforced;
- cryogenic screening runs use `--allow-incomplete-physics` only after the
  authoritative preflight result is recorded.

Authoritative commands:

```bash
.venv-jax/bin/lrekit --engine-mdo-optimize --isp-min 200 --target-thrust 13000 --mdo-propellant LOX/RP-1 --mdo-chamber-property-table builds/cea_tables/lox_rp1_frozen_v1.npz --mdo-of-mode optimize --cad none

.venv-jax/bin/lrekit --engine-mdo-optimize --isp-min 200 --target-thrust 13000 --mdo-propellant LOX/LCH4 --mdo-chamber-property-table builds/cea_tables/lox_lch4_frozen_v1.npz --mdo-of-mode optimize --cad none

.venv-jax/bin/lrekit --engine-mdo-optimize --isp-min 200 --target-thrust 13000 --mdo-propellant LOX/LH2 --mdo-chamber-property-table builds/cea_tables/lox_lh2_frozen_v1.npz --mdo-of-mode optimize --cad none

.venv-jax/bin/lrekit --engine-mdo-optimize --isp-min 200 --target-thrust 13000 --mdo-propellant N2O4/MMH --mdo-chamber-property-table builds/cea_tables/n2o4_mmh_frozen_v1.npz --mdo-of-mode optimize --cad none
```

Cryogenic screening-only commands:

```bash
.venv-jax/bin/lrekit --engine-mdo-optimize --isp-min 200 --target-thrust 13000 --mdo-propellant LOX/LCH4 --mdo-chamber-property-table builds/cea_tables/lox_lch4_frozen_v1.npz --mdo-of-mode optimize --allow-incomplete-physics --cad none

.venv-jax/bin/lrekit --engine-mdo-optimize --isp-min 200 --target-thrust 13000 --mdo-propellant LOX/LH2 --mdo-chamber-property-table builds/cea_tables/lox_lh2_frozen_v1.npz --mdo-of-mode optimize --allow-incomplete-physics --cad none
```

## Survey results

| Pair/mode | CLI exit | Workflow / optimizer / numerical / physics / requirements | Complete feasible | Main result |
|---|---:|---|---:|---|
| LOX/RP-1 authoritative | 0 | pass / pass / pass / pass / pass | yes | 66 iterations; max violation 0; mass 35.7287887283 kg; Isp 222.711323485 s |
| N2O4/MMH authoritative | 0 | pass / pass / pass / pass / pass | yes | 17 iterations; max violation 3.79658e-11; mass 49.4908027118 kg; Isp 242.015313052 s |
| LOX/LCH4 authoritative | 3 | stopped at model-coverage preflight | no | required methane real-fluid HTD surface unavailable |
| LOX/LCH4 screening | 3 | pass / pass / pass / unknown / pass | no | 16 iterations; max violation 8.25777e-10; mass 47.9690963118 kg; Isp 261.182082290 s |
| LOX/LH2 authoritative | 3 | stopped at model-coverage preflight | no | required hydrogen real-fluid HTD surface unavailable |
| LOX/LH2 screening | 4 | fail / fail / pass / fail / pass | no | 9 iterations; max violation 1.65840739; SLSQP: `Inequality constraints incompatible` |

Exact optimized designs from the API rerun of the same cases:

```json
{
  "LOX/RP-1": {
    "Pc_Pa": 2308172.8690153463,
    "OF": 2.9260116457492154,
    "eps": 4.080470576375355,
    "film_frac": 0.20439371221348673,
    "t_wall_m": 0.0004000000000000831,
    "N_rpm": 32998.81123978341
  },
  "N2O4/MMH": {
    "Pc_Pa": 3908074.520529444,
    "OF": 1.969409756630753,
    "eps": 5.676049652081119,
    "film_frac": 0.15609670187379207,
    "t_wall_m": 0.0004000000000000005,
    "N_rpm": 42103.90272338456
  },
  "LOX/LCH4_screening_only": {
    "Pc_Pa": 2681682.9391204044,
    "OF": 2.2876669474552864,
    "eps": 4.438227793749029,
    "film_frac": 0.0,
    "t_wall_m": 0.0004,
    "N_rpm": 44828.97035027501
  },
  "LOX/LH2_failed_screening_candidate": {
    "Pc_Pa": 5999999.901999294,
    "OF": 3.500000042655807,
    "eps": 7.881913048267268,
    "film_frac": 2.980013483700025e-09,
    "t_wall_m": 0.001999549897003423,
    "N_rpm": 93878.6099001359,
    "violations": {
      "chug": -0.046811207920288096,
      "land_fit": -6.462489472689207e-07,
      "pump_suction": -3.3168147798767507,
      "pump_tip_speed": -29.563515077927946
    }
  }
}
```

## Adjudication

1. Track B1 passes for these four table identities: every held-out value and
   gradient comparison passed at the initial production grid.
2. Kerolox and N2O4/MMH remain complete feasible results under the rerun.
3. The old methalox numerical-infeasibility conclusion moves: covered optimizer
   rows now pass with the real table. It still must be reported as `unknown`,
   not feasible, because the mandatory methane HTD model remains unavailable.
4. The hydrolox pump conclusion does not close. The current architecture
   reference is 187.757 krpm (consistent with the old approximately 183 krpm
   observation). Screening drives speed to its 93.879 krpm lower bound and
   still violates pump suction, pump tip speed, chug, and land fit. This failed
   candidate is not a design result.
5. The original generation outputs remain under the ignored `builds/` tree;
   byte-identical production copies are now versioned package resources. The
   content hashes above, not filesystem paths, remain their model identities.
6. The honest tracker split is now **B1 host gate: DONE** and **B1 table
   distribution: DONE for the four certified pairs**. A fresh wheel contains
   the hash-pinned resources, and callers can opt in portably with
   `chamber_property_table="catalog"` or
   `--mdo-chamber-property-table catalog`.
7. **N2O/Ethanol remains BLOCKED pending oxidizer-state selection.** The
   resolver refuses it with that reason and never substitutes a constant or a
   different N2O card.
