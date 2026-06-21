# Thermofluid and regenerative-cooling literature provenance

This document records the physical provenance, applicability limits, and
known omissions of RaoRocketSim's thermodynamic, heat-transfer, coolant-flow,
wall-sizing, and regenerative-CAD models.

## Corpus audit

The June 2026 addition to `propulsion_texts/` contained 54 PDF files. SHA-256
deduplication gives 46 unique documents and 1,484 PDF pages. Duplicate copies
were treated as one source. Text was extracted for search, and the governing
equations, assumptions, conclusions, and validation statements were reviewed
in the thermofluid-relevant papers. Page images remain the authority wherever
PDF text extraction damages an equation.

The literature does not turn the code into a qualification solver. It instead
lets the project distinguish:

- sourced equations implemented inside their stated range;
- sourced equations used as preliminary screens outside their original
  geometry;
- repository parameterizations that are clearly labeled as assumptions; and
- important physics that is still absent.

## Equation-to-code map

| Physical element | Current implementation | Primary local support | Status and limitation |
|---|---|---|---|
| Perfect-gas nozzle state | Constant-gamma isentropic temperature, pressure, density, area-Mach, Prandtl-Meyer, and thrust relations | SP-125; SP-8120; Rao (1958); standard gas dynamics | Preliminary. Frozen/equilibrium chemistry variation through the nozzle is not solved. |
| Gas transport estimates | `cp = gamma R/(gamma-1)`, Eucken Prandtl estimate, SP-125 viscosity power law | SP-125, equations 4-18 and 4-23 | Screening defaults. CEA or measured transport data should replace them. |
| Convective gas-side heating | Full Bartz (1957) coefficient, property factor, throat-curvature factor, area factor, and turbulent recovery temperature | `technical-notes-1957.pdf`; SP-125 | Appropriate for rapid preliminary estimates. Bose (1978), Wang (2006), Betti (2014), and Kim (2014) show why boundary-layer/CHT methods are needed for validation. |
| Gas radiation | Additive homogeneous multiband RTE screen, plus a Leccese-calibrated gray LOX/CH4 or LOX/H2 preset | Leccese et al. (2018), equations 3-7; Wang (2006) | Implemented as an opt-in engineering path. It is not the inhomogeneous DTM/ray trace used by Leccese. User bands require band weights and absorption coefficients; RP-1 soot radiation remains unresolved. |
| Wall thermal circuit | Gas film + normal liner conduction + coolant film; station heat integrated over true meridional area | SP-125; SP-8087; Atefi and Naraghi (2019) | One-dimensional through-wall screen. The optional 2-D/3-D solvers resolve more wall spreading but are not validated CHT. |
| Coolant bulk-energy march | Counterflow/coflow station march using `dT = q dA/(mdot cp)` | SP-125; SP-8087; Naraghi (2004); Pizzarelli et al. (2011) | Constant-property density, conductivity, and heat capacity remain crude unless overridden. |
| Turbulent coolant film | Sieder-Tate with bulk/wall viscosity ratio | SP-125; common preliminary regen practice | Screening correlation. Naraghi (2004) and Perakis et al. (2021) demonstrate strong result sensitivity to correlation choice. |
| Laminar coolant film | `Nu = 4.36` | classical fully developed circular duct, uniform heat flux | Explicit proxy only. Rectangular aspect ratio, heated-wall pattern, and developing flow are unresolved. |
| Fin/rib conductance | Adiabatic-tip rectangular fin efficiency and effective wetted area | SP-8087; Atefi and Naraghi (2019) | Supported for a preliminary channel-cell resistance model, especially high-aspect-ratio ribs. |
| High-aspect-ratio channels | Variable height search plus fin-area benefit; warnings above `h/w = 12` | Carlile and Quentmeyer (1992); Wadel (1997); Park (2013); Pizzarelli et al. (2013, 2014); Garcia et al. (2020) | Experiments support lower hot-wall temperature. Very high aspect ratios develop thermal stratification and diminishing benefit; no universal optimum is imposed. |
| Curved-channel heat transfer | Optional Niino-Kumakawa/Taylor multiplier `[Re(Dh/(2Rc))^2]^(+/-0.05)` | Pizzarelli et al. (2011); Torres, Stefanini, and Suslov (2009) | Sourced screen, disabled in automatic liquid sizing. SP-8087 recommends no liquid enhancement credit without experimental calibration. |
| Distributed pressure loss | Darcy-Weisbach along actual 3-D helical centerline | SP-125; SP-8087 | Implemented station by station. |
| Smooth turbulent friction | Blasius Darcy factor | standard smooth-pipe preliminary relation | Warned above its usual range. |
| Rough turbulent friction | Swamee-Jain with user-supplied mean roughness | Atefi and Naraghi (2019); Pizzarelli et al. (2011) supports roughness-sensitive treatment | Implemented. Manufacturing-process roughness must come from inspection or process data. |
| Local/manifold loss | Nonlinear graph containing every channel, both annular headers, discrete ports, channel entry/exit losses, and ring friction | Kang and Sun (2011); SP-8087 | Implemented as an opt-in 1-D network. It reports branch-flow maldistribution and source-to-sink pressure drop. Port/header K values and equivalent plenum geometry still require 3-D CFD or test calibration. |
| Coolant velocity | Peak velocity and margin to 61 m/s reported | SP-8087 | Recommendation, not a universal hard limit. |
| RP-1 chemistry | Coolant-side wall margin to 700 K; optional sizing gate | SP-8087 | Conservative coking screen. Residence time, sulfur, surface condition, and deposit growth are not modeled. |
| Real-fluid methane/hydrogen | Station-wise density, viscosity, conductivity, heat capacity, enthalpy, Prandtl number, and critical state from CoolProp HEOS | Bell et al. (2014); Pizzarelli et al. (2015, 2011); Kang and Sun (2011) | Implemented. Temperature and distributed channel pressure are iterated together; the final manifold offset is added after the graph solve. A full enthalpy-conservative march and supercritical heat-transfer-deterioration correlation remain required. |
| Boiling and CHF | CoolProp saturation/critical-state screen plus Zuber hydrodynamic saturated CHF reference | Zuber, AECU-4439 (1959); CoolProp HEOS | Implemented opt-in. Subcritical wall superheat and CHF margin are reported; supercritical flow is correctly identified as having no liquid-vapor CHF. Zuber is a conservative screening reference, not a forced-flow cryogenic rocket-channel qualification correlation. |
| Variable hot-wall thickness | Station-wise normal thickness from coupled thermal and SP-125 stress bounds | SP-8087 variable/tapered wall guidance; SP-125 equation 4-31 | Implemented as preliminary sizing. Manufacturing, erosion, coating, and defect floors remain user/design constraints. |
| Liner pressure/thermal stress | Station-wise coaxial-shell relation | SP-125 equation 4-31 | Implemented screen with temperature-independent material properties. |
| Tubular longitudinal buckling | SP-125 equation 4-29 with stress- and temperature-dependent cyclic tangent modulus where sourced | SP-125; NASA CR-134627; NASA 20060005216 | NARloy-Z and GRCop-84 now use source-derived cyclic Ramberg-Osgood curves. `h/2` remains an explicitly labeled equivalent-tube radius for milled channels; the equation does not gate by default. |
| Rib-bay liner collapse | Classical long-plate buckling screen under coolant-over-gas compression | classical plate theory; SP-8087 identifies inward collapse/crippling as failure modes | Separate engineering mapping, not SP-125 equation 4-29. Requires nonlinear imperfect-shell/panel FEA for qualification. |
| Thermal fatigue | Nominal `alpha DeltaT` strain; NARloy-Z Coffin-Manson/Basquin fit; GRCop-84 direct total-strain/life regressions | SP-125 equation 4-28; NASA CR-134627; NASA 20060005216; Miller (1974); Porowski et al. (1985); Dai and Ray (1995); Thiede et al. (2017); Pizzarelli (2020); Hötte et al. (2020) | Sourced catalog data now gate as preliminary screens. They remain isothermal coupon fits, not thermomechanical chamber qualification. Doghouse thinning, ratcheting, creep, and local cyclic plasticity are not solved. |
| Channel/manifold B-rep | Normal-offset liner, patterned positive ribs, jacket, plenums, ports, one-solid STEP round trip | SP-8087 geometry/manifold practice; Gradl papers; Kerstens (2021) | Real neutral B-rep, not native Inventor feature history or production drawing definition. |

## What the user inputs physically control

The principal paths are:

- `Rt` changes throat area, total mass flow at fixed `Pc/c*`, local
  circumference, channel pitch, Bartz diameter scaling, and the scale of
  pressure/stress loads.
- `epsilon` and `length-pct` change the gas expansion, heated wall area,
  coolant path length, downstream heat pickup, and distributed pressure loss.
- `gamma`, `Tc`, molecular weight, and `c*` control the perfect-gas state and
  Bartz transport/mass-flux terms. A user-supplied constant `gamma` does not
  reproduce equilibrium chemistry.
- `Pc` strongly raises Bartz convection and mass flow, while setting gas
  pressure and the minimum default coolant-outlet pressure.
- `O/F` divides total propellant flow into fuel coolant in the cycle-based
  sizing path. It does not by itself recompute combustion products unless the
  thermochemistry backend is also rerun.
- `max wall T` and thermal margin impose the gas-side metal-temperature limit.
- coolant identity selects CoolProp HEOS for methane/LH2 when available
  (`--coolant-property-backend auto|coolprop`) and otherwise the documented
  constant-property screen.
- channel `N`, width, height, helix, and roughness set pitch, fin area,
  hydraulic diameter, mass flux, Reynolds number, path length, and friction.
- pressure-drop budget includes distributed channel loss and, with
  `--hydraulic-network` (also enabled by `--regen-manifolds`), the solved
  inlet-port/header/channel/outlet-header/port graph.
- auto-size searches geometry and, in the wall solver, normal liner/jacket
  thickness. It does not yet change channel count by axial bifurcation.

## Unique-paper disposition

### Direct thermofluid and regen-model sources

- Bartz (1957), `technical-notes-1957.pdf`: implemented convective
  correlation.
- Huzel and Huang, `19710019929.pdf`: implemented gas properties, heat
  transfer, pressure loss, stress, and buckling relations.
- NASA SP-8087, `19730022965.pdf`: implemented/reported design constraints
  for passage tailoring, velocity, coking, manifolds, wall thickness, and
  structural failure modes.
- Bose (1978), `bose1978.pdf`: bounds the accuracy claim of simple Bartz
  estimates.
- Carlile and Quentmeyer (1992), `carlile1992.pdf`: experimental
  high-aspect-ratio evidence.
- Wadel (1997), `wadel1997.pdf`: high-aspect-ratio design trade and
  bifurcation evidence.
- Mirzamoghadam (1991), `mirzamoghadam1991.pdf`: advanced chamber-wall
  cooling criteria.
- Naraghi (2004), `naraghi2004.pdf`: coupled TDK/RTE design model and
  coolant-correlation sensitivity.
- Pizzarelli et al. (2011), `pizzarelli2011.pdf`: quasi-2-D rough,
  variable-property, entrance, and curved-channel model.
- Torres, Stefanini, and Suslov (2009), `eucass1p171.pdf`: curved-channel
  Dean-flow data and correlation comparison.
- Kang and Sun (2011), `kang2011.pdf`: conjugate regen CFD, curvature, and
  manifold/local-loss omissions.
- Park (2013), `park2013.pdf`: aspect-ratio-dependent turbulent heat transfer.
- Pizzarelli et al. (2013), `pizzarelli2013.pdf`: pressure-drop/power optimum
  and high-aspect-ratio thermal stratification.
- Pizzarelli et al. (2014), `pizzarelli2014.pdf`: aspect-ratio thermal
  behavior.
- Pizzarelli et al. (2015), `pizzarelli2015.pdf`: near-critical methane
  deterioration and roughness/pressure trade.
- Atefi and Naraghi (2019), `atefi2019.pdf`: variable channel optimization,
  fin resistance, and Swamee-Jain loss.
- Garcia et al. (2020), `garcia2020.pdf`: straight-versus-contoured
  high-aspect-ratio CHT.
- Betti, Pizzarelli, and Nasuti (2014), `betti2014.pdf`: coupled chamber heat
  transfer.
- Kim (2014), `kim2014.pdf`: multidisciplinary turbulent combustion/nozzle
  CHT.
- Wang (2006), `wang2006.pdf`: multidimensional convective and radiative
  nozzle heating.
- Leccese et al. (2018), `leccese2018.pdf`: radiative share and propellant
  dependence.
- Perakis, Preis, and Haidn (2021), `perakis2021.pdf`: inverse heat-flux
  evaluation and the need for validated coolant Nusselt models.

### Structural life and wall integrity

- NASA CR-134627, `materials_science/19740017910.pdf`: NARloy-Z
  elastic/plastic strain ranges, fatigue slopes, half-life stress ranges, and
  538 C modulus. The catalog intercepts are reconstructed from Table 2 with
  the report's Figure 10 slopes.
- NASA CR-134806, `materials_science/19750021165.pdf`: temperature-dependent
  copper-alloy modulus, stress-strain, cyclic stress-strain, and LCF context.
- NASA GRCop-84 handbook, `materials_science/20020070630.pdf`: process- and
  temperature-dependent LCF regression context.
- Lerch and Ellis, `materials_science/20060005216.pdf`: GRCop-84 direct
  total-strain/life and cyclic stress/plastic-strain range fits used by the
  catalog.

- Miller (1974), `miller1974.pdf`.
- Porowski et al. (1985), `porowski1985.pdf`.
- Dai and Ray (1995),
  `life-prediction-of-the-thrust-chamber-wall-of-a-reusable-rocket--1995.pdf`.
- Asraff et al. (2009), `s12666-010-0089-7.pdf`.
- Thiede, Riccius, and Reese (2017), `thiede2017.pdf`.
- Song et al. (2017), `song2017.pdf`.
- Pizzarelli (2020), `pizzarelli2020.pdf`.
- Hötte et al. (2020), `10.1016@j.ijfatigue.2020.105649.pdf`.
- Kuhl et al. (1998), `kuhl1998.pdf`.

These sources support the project's explicit distinction between elastic
screening, low-cycle fatigue, cyclic plasticity/ratcheting, creep, doghouse
thinning, and experimentally qualified life.

### Manufacturing, inspection, and CAD reality

- Gradl (2016), `gradl2016.pdf`.
- Gradl et al. (2017), `gradl2017.pdf`.
- Gradl et al. (2018), `gradl2018.pdf`.
- Gradl et al. (2018), `gradl2018 (1).pdf`.
- Gradl et al. (2019), `gradl2019.pdf`.
- Gradl et al. (2019), `gradl2019 (1).pdf`.
- Gradl (2020), `gradl2020.pdf`.
- Kerstens (2021), `kerstens2021.pdf`.
- Masuoka et al. (2011), `masuoka2011.pdf`.

These papers support variable integral channels, closeout/jacket process
choices, bimetallic construction, inspection needs, and the need to use
as-built roughness and wall thickness rather than nominal CAD alone.

### Nozzle contour, separation, and performance context

- Rao (1958), `rao1958.pdf`.
- NASA SP-8120, `19770009165.pdf`.
- Hagemann et al. (1998), `hagemann1998.pdf`.
- Frey and Hagemann (2000), `frey2000.pdf`.
- Schomberg et al. (2016), `schomberg2016.pdf`.
- Schomberg (2018), `schomberg2018.pdf`.

These sources govern contour, separation, and nozzle-performance portions of
the repository rather than coolant-channel heat transfer.

## Remaining high-priority physics

1. Add and validate a supercritical heat-transfer-deterioration correlation
   and make the real-fluid energy march enthalpy/pressure coupled.
2. Add composition-driven Planck/band coefficients and an inhomogeneous
   ray-traced radiation path; add a separate soot model for hydrocarbon fuels.
3. Calibrate port/header K values and annular-plenum geometry against 3-D CFD
   or manifold-flow tests, then couple the coldest branch to azimuthal CHT.
4. Add configurable turbulent coolant correlations (Petukhov-Gnielinski,
   Niino, Dipprey-Sabersky, and fluid-specific forms) with applicability
   checks and benchmark cases.
5. Replace the conservative Zuber reference with validated forced-flow
   cryogenic boiling/CHF correlations where applicable; add deposit growth.
6. Validate the 1-D, 2-D, and 3-D paths against the Carlile/Wadel/NASA chamber
   datasets before promoting any thermal result above preliminary sizing.
