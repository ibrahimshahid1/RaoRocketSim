# Spray-to-Cycle Coupling

The legacy pintle atomization block is a correlation screen, not a parcel
solver. It reports a liquid vaporized fraction from a Hinze diameter bound,
a fixed primary-breakup length, and a calibrated d-squared-law evaporation
constant. It does not infer mixing efficiency, chemical completion, or
characteristic-velocity efficiency.

`raosim.spray_coupling` provides an explicit, opt-in fixed point:

```text
eta_cstar
  -> mdot = Pc At / (cstar_ideal eta_cstar)
  -> pintle hydraulic sizing and atomization screen
  -> eta_vaporization
  -> eta_cstar = eta_vaporization eta_mixing eta_combustion
```

The update is relaxed and iterated until both efficiency and mass flow close.
`eta_mixing` and `eta_combustion` must be supplied independently. The solver
rejects gas, supercritical, transcritical, or otherwise inapplicable streams;
it never renormalizes vaporization efficiency over whichever stream happened
to remain inside the droplet-correlation domain.

## Python API

Attach a `SprayCStarCouplingSpec` to `DesignInput`. The final injector result,
engine performance, and report then share the same converged mass flow.

```python
from lrekit.design import DesignInput, ThermoSpec, design_nozzle_v2
from lrekit.injector import InjectorSpec
from lrekit.spray_coupling import SprayCStarCouplingSpec

result = design_nozzle_v2(DesignInput(
    thermo=ThermoSpec(
        mode="constant_gamma",
        propellant_name="LOX/RP-1",
    ),
    Pc=1.5e6,
    Rt=0.020,
    injector=InjectorSpec(type="pintle"),
    spray_cstar_coupling=SprayCStarCouplingSpec(
        enabled=True,
        eta_mixing=0.98,
        eta_combustion=0.99,
    ),
))
```

The d-squared-law constant must be calibrated for the propellant and gas state.
The repository default is only a screening value and often causes the fixed
point to reject the design as outside its accepted efficiency envelope.

## CLI

The same closure is available for a non-regenerative pintle run:

```bash
lrekit --propellant LOX/RP-1 --pc 1.5e6 --injector pintle \
  --spray-cstar-coupling \
  --spray-eta-mixing 0.98 --spray-eta-combustion 0.99 \
  --spray-evaporation-constant 1.0e-6
```

Regenerative coupling is supported only for the direct cycle-fuel path (all
fuel passes through the jacket and then the fuel injector). Every fixed-point
evaluation now derives coolant flow from the current fuel flow and re-solves
Bartz/Sieder-Tate wall cooling, jacket pressure loss, injector feed state, and
the feed/pump duty ledger. The final structural screen and the CLI cooling,
channel, cross-section, and structural-life artifacts are regenerated from the
final iterate. An independent coolant, bleed, or bypass remains rejected until
an explicit split and mixing model exists.

The serialized `scope` distinguishes
`injector_and_cycle_mass_flow_no_regen` from
`spray_cycle_regen_wall_feed_and_pump_duty`. Every iteration records fuel,
oxidizer, coolant, pressure-loss, cooling-margin, and pump-duty state, and the
final state is re-evaluated at the reported fixed point.

## What this is not

The default `source="legacy_screen"` model is not the Radhakrishnan, Lee, and
Koo Eulerian-to-Lagrangian CFD workflow. The repository now also contains an
opt-in, one-way Lagrangian mid-tier under `raosim.spray`; see
[`LAGRANGIAN_SPRAY_MODEL.md`](LAGRANGIAN_SPRAY_MODEL.md). Its typed cycle
handoff is deliberately fail-closed: current carrier, energy, phase,
convergence, geometry, and benchmark evidence cannot satisfy all coupling
gates, so it is not exposed as a CLI cycle source. Cold-flow droplet statistics
and hot-fire c-star data remain mandatory validation evidence.
