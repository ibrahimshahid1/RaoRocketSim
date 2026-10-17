# Lee inverter power-density remediation — 2026-10-14

Status: implemented; focused software verification recorded below. This is a
source-convention correction, not physical validation of an electric drive.

## Original-source evidence

The original article, *Performance Analysis and Mass Estimation of a
Small-Sized Liquid Rocket Engine with Electric-Pump Cycle*, DOI
`10.1007/s42405-020-00325-z`, was checked visually on PDF page 7 of
`propulsion_texts/10.1007_s42405-020-00325-z.pdf` (SHA-256
`00d2425ed35f3fe39b8debbfa852cdeb9a6f0ae006eca0296d9523ae7a499500`).
The page was rendered as `knowledge/.work/lee/page7.png` for inspection.
Generated Markdown was used only to locate the source.

Equations (26)-(27) define motor power density from motor shaft output and
identify motor output with pump shaft demand. Equations (28)-(29) define
inverter power density and mass as:

```text
delta_P,inv = P_inv,out / m_inv
m_inv = P_inv,out / delta_P,inv
      = P_m,out / (eta_m * delta_P,inv)
```

Inverter output equals motor electrical input. Inverter DC input additionally
includes inverter losses: `P_inv,in = P_inv,out / eta_inv`. Using DC input in
the output-density mass relation overestimates inverter mass by `1/eta_inv`
for the same numerical density and fixed pump demand.

## Implemented correction

`raosim/pumps.py::_drive_sizing`, `raosim/mdo/pump.py::electric_feed`, and
`raosim/mdo/assembly.py::readouts` now divide motor electrical input by the
inverter output-power density. Electrical input, current, battery demand and
inverter heat retain their loss dependence. Motor sizing is unchanged.

`ElectricDriveSpec`, `MissionSpec`, CLI help, the electric-pump basis document,
exported assumptions, and hardware BOM parameters now identify the density
basis explicitly. Drive serialization exposes inverter input and output power.
The model registry points to the exact PDF checked above and names the
relevant source equations. No numerical default density or efficiency changed.

The legacy combined `ElectricDriveSpec.power_density` override retains its
existing DC-input-power basis and mass allocation: the combined package mass
is carried in `motor_mass`, with zero separately allocated inverter mass.
Its exported BOM metadata identifies this inclusion rather than presenting
the inverter's zero allocation as an independent mass estimate.

Compatibility: separate inverter estimates decrease by `eta_inv` relative to
the earlier implementation when identical numerical densities are supplied.
For a vendor density defined from DC input, supply the corresponding output
density `delta_output = eta_inv * delta_input` at the same rated duty.
Existing generated artifacts are historical outputs and were not rewritten.

## Verification

The new regression failed against the original implementation: the Lee case
gave inverter mass `0.008897903989181881 kg` instead of the output-density
value `0.007563218390804598 kg`. The MDO checks likewise failed for efficiencies
0.85 and 0.96 and passed at unity, isolating the extra loss factor.

The corrected Lee check derives shaft power from the existing Table 4 motor
mass and Table 2 motor density (`0.4512 kg * 875 W/kg = 394.8 W`). At motor
efficiency 0.87 and inverter output density 60000 W/kg, the resulting inverter
mass is `7.563218390804598 g`. This value is calculated from source equations
and the existing benchmark inputs; it is not an independent measured inverter
mass or a Table 4 inverter result.

Checks cover efficiency-invariant inverter mass at fixed motor demand,
electrical power/current/heat dependence on inverter losses, the legacy
combined-density contract, BOM/export consistency, NumPy–JAX drive parity,
and the skeleton assembly ledger. Existing pump AD/finite-difference checks
remain part of the focused verification.

Initial focused command:

```text
.venv-jax/bin/python -m pytest tests/test_pumps.py::TestLiteratureBenchmarks tests/test_mdo_pump.py tests/test_mdo_pump_gate.py -q
18 passed in 2.43s
```

Final bounded implementation suite:

```text
.venv-jax/bin/python -m pytest tests/test_pumps.py tests/test_mdo_pump.py tests/test_mdo_pump_gate.py tests/test_mdo_skeleton.py -q
60 passed in 41.81s
```

`git diff --check` also passed. The full repository suite and physical
hardware tests were outside this focused remediation.

The repository's documented `.venv/bin/python` path is absent locally;
the available `.venv-jax` runtime was used for these implementation tests.

## Remaining limitations

The broader electric-drive model remains only partially supported by this
article: per-stream drive architecture, screening defaults, automatic bus
selection, ancillary gates and thermal screens remain repository extensions.
This correction establishes the density convention, not the appropriateness
of selected technologies, efficiencies or densities for an intended duty.
Measured component maps, rated-duty data, thermal and pulse evidence, and
hardware qualification remain required. No universal validity envelope was
inferred from the article's worked design point.
