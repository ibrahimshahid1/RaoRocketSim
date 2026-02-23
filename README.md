# RaoRocketSim

A computational toolbox for **Rao Thrust-Optimized Parabolic (TOP) bell nozzle** design, engine performance estimation, and flight analysis.

## Features

### Core Design
- **Rao bell nozzle contour** — 3-section geometry (upstream arc + downstream arc + quadratic Bézier) with NASA SP-8120 / Rao empirical angle tables
- **Engine performance** — thrust, Isp, Cf, c*, mass flow from first-principles isentropic flow
- **Propellant database** — N₂O/Ethanol, LOX/RP-1, LOX/LCH₄, LOX/LH₂ + custom
- **Export** — CSV point coordinates + STL mesh (surface of revolution) for CAD

### Analysis (v2)
- **Batch CLI** — `python main.py --Rt 25 --Pc 60 --propellant LOX/LCH4 --epsilon 12 --output nozzle.csv` 
- **Parameter sweeps** — `python main.py --sweep epsilon 4 50 20` with multi-panel trade-study plots
- **Wall pressure distribution** — p(x) along the contour with monotonicity check per NASA SP-8120
- **Separation prediction** — Summerfield / Kalt-Badal / Schmucker criteria with location estimate
- **Altitude performance map** — thrust, Isp, Cf vs altitude with separation onset
- **Chamber geometry** — cylindrical chamber + convergent section parameterized by L* and contraction ratio

## Quick Start

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib

# Interactive mode
python main.py

# Batch mode (no prompts)
python main.py --propellant LOX/RP-1 --Pc 45 --Rt 20 --epsilon 10 \
    --output nozzle.csv --stl nozzle.stl --wall-pressure --separation --chamber

# Parameter sweep
python main.py --propellant LOX/LCH4 --Pc 60 --Rt 25 --sweep epsilon 4 50 20

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
raosim/
├── gas_dynamics.py          # isentropic relations, area-Mach, Cf, c*
├── propellants.py           # propellant database with derived R_gas, c*
├── nozzle_geometry.py       # Rao bell contour + angle tables
├── engine.py                # engine performance computation
├── wall_pressure.py         # wall pressure distribution + monotonicity
├── separation.py            # flow separation criteria
├── trade_study.py           # parameter sweeps
├── altitude_performance.py  # performance vs altitude map
├── chamber_geometry.py      # combustion chamber geometry
├── atmosphere.py            # ISA atmospheric model
├── trajectory.py            # vertical-ascent integrator
├── plotting.py              # 2D / 3D / curvature plots
└── export.py                # CSV + STL export
tests/                       # 45 unit tests
main.py                      # CLI (interactive / batch / sweep)
```

## Nozzle Geometry

| Section | Description | Radius |
|---------|-------------|--------|
| Upstream arc | Convergent-side circular fillet | 1.5 Rₜ |
| Downstream arc | Supersonic-side circular fillet | 0.382 Rₜ |
| Bell section | Quadratic Bézier (canted parabola) | N → E |

Wall angles (θₙ, θₑ) interpolated from Rao/NASA charts as f(ε, L%).

## References

- G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958
- NASA SP-8120, "Liquid Rocket Engine Nozzles," 1976
- R. Stark, "Flow Separation in Rocket Nozzles – An Overview," 2009
- J. M. Seitzman, Georgia Tech AE 6450 nozzle geometry notes
- J. D. Anderson, *Modern Compressible Flow*, McGraw-Hill

## License

MIT
