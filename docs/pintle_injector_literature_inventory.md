# Pintle injector literature inventory

Status: first-pass literature map for using the `propulsion_texts/pintle_injector`
corpus to improve the pintle injector design CLI and simulation model.

## Corpus summary

- Local folder inspected: `propulsion_texts/pintle_injector`
- PDF files: 61
- Exact duplicate pairs: 7
- Unique PDFs after exact-duplicate collapse: 54
- Local helper outputs: `tmp/pintle_pdf_inventory.tsv`,
  `tmp/pintle_pdf_dois.json`, `tmp/ntrs_pintle_search_merged.json`

Exact duplicate pairs:

- `19730022965 (1).pdf` == `19730022965.pdf`
- `19750012398 (1).pdf` == `19750012398.pdf`
- `19760023196 (1).pdf` == `19760023196.pdf`
- `radhakrishnan2020 (1).pdf` == `radhakrishnan2020.pdf`
- `sakaki2015 (1).pdf` == `sakaki2015.pdf`
- `son2017 (1).pdf` == `son2017.pdf`
- `song2021 (1).pdf` == `song2021.pdf`

## Highest-priority sources for the CLI model

These are the papers/reports most directly useful for replacing pintle design
heuristics with literature-based calculations and gates.

| Source | Local file / link | Main use in RaoRocketSim |
|---|---|---|
| Gill and Nurick, *Liquid Rocket Engine Injectors*, NASA SP-8089, 1976 | `19760023196.pdf`; https://ntrs.nasa.gov/citations/19760023196 | Baseline injector hydraulics, pressure-drop practice, orifice/annulus/slot sizing, validation limits |
| Carter, *Extend the design criteria for the coaxial pintle injector to higher chamber pressures and to gaseous fuel*, NASA CR-107381, 1969 | `19700005097.pdf`; https://ntrs.nasa.gov/citations/19700005097 | Legacy coaxial pintle design criteria; pressure and gaseous-fuel extension |
| Hammock et al., *Apollo experience report: Descent propulsion system*, 1973 | `19730011150.pdf`; https://ntrs.nasa.gov/citations/19730011150 | Pintle flight heritage, throttle range, chamber/nozzle packaging, qualification lessons |
| Dressler and Bauer, *TRW pintle engine heritage and performance characteristics*, AIAA 2000-3871 | `dressler2000.pdf`; DOI https://doi.org/10.2514/6.2000-3871 | Heritage thrust ranges, throttling, stability claims, practical engine-level architecture |
| Gavitt and Mueller, *Testing of the 650 Klbf LOX/LH2 Low Cost Pintle Engine*, AIAA 2001-3987 | `gavitt2001.pdf`; DOI https://doi.org/10.2514/6.2001-3987 | Large-scale hot-fire testing, LCPE/TRL context, performance envelope |
| Mueller and Dressler, *TRW 40 klbf LOX/RP-1 low cost pintle engine test results*, AIAA 2000-3863 | `mueller2000.pdf`; DOI https://doi.org/10.2514/6.2000-3863 | LOX/RP-1 pintle hot-fire data and performance behavior |
| Austin, Heister, Anderson, *Characterization of pintle engine performance for nontoxic hypergolic bipropellants*, 2002/2005 | `austin2002.pdf`, `austin2005.pdf`; DOI https://doi.org/10.2514/1.7988 | Engine performance vs geometry/operating point; c-star efficiency calibration |
| Son et al., *Design procedure of a movable pintle injector for liquid rocket engines*, 2016/2017 | `son2016.pdf`, `son2017.pdf`; DOI https://doi.org/10.2514/1.B36301 | Movable sleeve geometry, throttle schedule, minimum areas, SMD/geometry workflow |
| Son et al., *Numerical study on the combustion characteristics of a fuel-centered pintle injector for methane rocket engines*, 2017 | `son2017 (2).pdf`; DOI https://doi.org/10.1016/j.actaastro.2017.02.005 | Fuel-centered LOX/methane combustion, chamber coupling, CFD calibration |
| Ninish et al., *Spray characteristics of liquid-liquid pintle injector*, 2018 | `ninish2018.pdf`; DOI https://doi.org/10.1016/j.expthermflusci.2018.03.033 | Liquid-liquid spray angle, droplet/mixing behavior, experimental nonreacting data |
| Radhakrishnan et al., *Effect of injection conditions on mixing performance of pintle injector for liquid rocket engines*, 2018 | `radhakrishnan2018.pdf`; DOI https://doi.org/10.1016/j.actaastro.2017.12.012 | Injection-condition sensitivity, mixing metrics, numerical/experimental comparison |
| Radhakrishnan et al., *Lagrangian approach to axisymmetric spray simulation of pintle injector for liquid rocket engines*, 2018 | `radhakrishnan2018 (1).pdf`; DOI https://doi.org/10.1615/AtomizSpr.2018022652 | Droplet breakup/spray simulation model candidate |
| Radhakrishnan et al., *A detailed modelling on spray atomisation and combustion of LOX/GCH4 in variable area pintle injector*, 2021 | `radhakrishnan2021.pdf`; DOI https://doi.org/10.1080/13647830.2021.1937702 | LOX/GCH4 atomization and combustion model structure |
| Freeberg et al., *Spray Cone Formation from Pintle-Type Injector Systems in Liquid Rocket Engines*, 2019 | `freeberg2019.pdf`; DOI https://doi.org/10.2514/6.2019-0152 | Spray-cone formation, TMR/blockage-factor comparison |
| Zhou and Shen, *Experimental study on the spray characteristics of a pintle injector element*, 2022 | `1-s2.0-S0094576522000741-main.pdf`; DOI https://doi.org/10.1016/j.actaastro.2022.02.019 | Liquid-liquid spray patterns, TMR/LMR angle correlations, block factor |
| Zhao et al., *Review of atomization and mixing characteristics of pintle injectors*, 2022 | add to corpus; DOI https://doi.org/10.1016/j.actaastro.2022.08.042 | Review paper; best starting point for correlation selection |
| Cheng et al., *On the prediction of spray angle of liquid-liquid pintle injectors*, 2017 | add to corpus; DOI https://doi.org/10.1016/j.actaastro.2017.05.037 | Literature spray-angle correlation; likely replacement for current vector estimate |
| Cheng et al., *Flow characteristics of a pintle injector element*, 2019 | add to corpus; DOI https://doi.org/10.1016/j.actaastro.2018.10.020 | Flow field and local injection physics for element-level modeling |
| Chen et al., *Experimental research on the spray characteristics of pintle injector*, 2019 | add to corpus; DOI https://doi.org/10.1016/j.actaastro.2019.06.032 | Spray experiment data for regression/validation |
| Lee et al., *Spray characteristics of a pintle injector based on annular orifice area*, 2020 | add to corpus; DOI https://doi.org/10.1016/j.actaastro.2019.11.008 | Annular-area effect; CLI should expose/solve annular opening constraints |
| Lee et al., *Effects of skip distance on the spray characteristics of a pintle injector*, 2021 | add to corpus; DOI https://doi.org/10.1016/j.actaastro.2020.09.043 | Skip/impingement distance sensitivity; should inform chamber wall-hit gate |
| Zhou et al., *Experimental and numerical investigations on the spray characteristics of liquid-gas pintle injector*, 2022 | add to corpus; DOI https://doi.org/10.1016/j.ast.2022.107354 | Liquid-gas pintle branch and gas/liquid momentum-ratio behavior |
| Zhang et al., *Experimental and numerical investigations on flow field characteristics of pintle injector*, 2020 | add to corpus; DOI https://doi.org/10.1016/j.ast.2020.105924 | Internal/external flow-field calibration and CFD comparison |
| Kang et al., *Design of pintle injector using Kerosene-LOx as propellant and solving the problem of pintle tip thermal damage in hot firing test*, 2022 | `kang2022.pdf`; DOI https://doi.org/10.1016/j.actaastro.2022.08.029 | Pintle tip thermal damage and design mitigation |
| Zhou et al., *Characterization of pintle-engine performance for GO2/kerosene propellants*, 2023 | `Characterization-of-pintle-engine-performance-for-GO2-kerosene-propellants.pdf`; DOI https://doi.org/10.1016/j.applthermaleng.2023.120421 | Deep throttling, chamber pressure range, heat flux, SMD trends, pressure-drop regulation |
| Lucchese et al., *Impact of chemical modeling on the numerical analysis of a LOx/GCH4 rocket engine pintle injector*, 2024 | `1-s2.0-S0094576524001103-main.pdf`; DOI https://doi.org/10.1016/j.actaastro.2024.02.038 | Chemical-kinetics sensitivity in CFD; warns against over-trusting simplified combustion chemistry |
| Scientific Reports, *Transient spray combustion characteristics in a gas-liquid pintle rocket engine under acoustic excitation*, 2024 | `s41598-024-64027-2.pdf`; DOI https://doi.org/10.1038/s41598-024-64027-2 | Acoustic forcing, transient spray/combustion coupling |
| Ahn et al., *Effects of injector recess on heat flux in a combustion chamber with cooling channels*, 2014 | `ahn2014.pdf`; DOI https://doi.org/10.1016/j.ast.2014.05.012 | Injector recess/chamber heat-flux coupling, wall cooling implications |
| Sakaki et al., ethanol/LOX planar pintle series, 2015-2018 | `sakaki2015.pdf`, `sakaki2016.pdf`, `sakaki2017.pdf`, `sakaki2018.pdf`; DOIs below | Optical diagnostics, c-star/performance, combustion instability, planar pintle combustor calibration |
| Casiano, Hulka, Yang, *Liquid-propellant rocket engine throttling: A comprehensive review*, 2010 | add to general corpus; DOI https://doi.org/10.2514/1.49791 | Throttle-system constraints, feed coupling, stability, mixture-ratio drift |
| Huzel and Huang, *Design of Liquid Propellant Rocket Engines*, NASA SP-125, 1967 | `19710019929 (1).pdf`; https://ntrs.nasa.gov/citations/19710019929 | Engine cycle, chamber geometry, L-star, mass flow, throat area |
| NASA SP-8120, *Liquid rocket engine nozzles*, 1976 | `19770009165.pdf`; https://ntrs.nasa.gov/citations/19770009165 | Nozzle geometry, throat/expansion constraints that set mdot and chamber coupling |
| NASA SP-8087, *Liquid rocket engine fluid-cooled combustion chambers*, 1972 | `19730022965.pdf`; https://ntrs.nasa.gov/citations/19730022965 | Regen cooling, wall heat transfer, chamber geometry, coolant pressure boundary |
| NASA turbopump monographs, SP-8052/SP-8107/SP-8112 family | `19710025474.pdf`, `19740020848.pdf`, `19750012398.pdf`, `19780023221.pdf` | Pump pressure/head/capacity boundaries feeding injector pressure drop |

## Local corpus inventory

| Local file | Identified title / role | Topic tag |
|---|---|---|
| `1-s2.0-S0094576522000741-main.pdf` | Experimental study on the spray characteristics of a pintle injector element | spray, TMR/LMR, block factor |
| `1-s2.0-S0094576524001103-main.pdf` | Impact of chemical modeling on the numerical analysis of a LOx/GCH4 rocket engine pintle injector | CFD, chemistry, LOX/GCH4 |
| `19700005097.pdf` | Extend the design criteria for the coaxial pintle injector to higher chamber pressures and to gaseous fuel | NASA/TRW pintle design criteria |
| `19710019929 (1).pdf` | Design of Liquid Propellant Rocket Engines, NASA SP-125 | chamber/cycle design |
| `19710025474.pdf` | Liquid rocket engine turbopump inducers | pump inlet/NPSH |
| `19720026079.pdf` | Liquid propellant rocket combustion instability | stability |
| `19730011150.pdf` | Apollo experience report: Descent propulsion system | Apollo pintle heritage |
| `19730022965.pdf` | Liquid rocket engine fluid-cooled combustion chambers | cooling/chamber |
| `19740020848.pdf` | Liquid rocket engine centrifugal flow turbopumps | pump output/capacity |
| `19750012398.pdf` | Turbopump systems for liquid rocket engines | pump system |
| `19760023196.pdf` | Liquid rocket engine injectors, NASA SP-8089 | injector baseline |
| `19770009165.pdf` | Liquid rocket engine nozzles, NASA SP-8120 | nozzle coupling |
| `19780023221.pdf` | Liquid rocket engine axial-flow turbopumps | pump system |
| `19940018570.pdf` | Inherent stability of central element coaxial liquid-liquid injectors | stability |
| `20150002584.pdf` | Designing Liquid Rocket Engine Injectors for Performance, Stability, and Cost | injector design practice |
| `20150016316.pdf` | Advancing the State-of-the-Practice for Liquid Rocket Engine Injector Design | injector design practice |
| `20150021468.pdf` | The Effect of Resistance on Rocket Injector Acoustics | injector acoustics |
| `A_Design_Tool_for_Liquid_Rocket_Engine_Injectors.pdf` | A design tool for liquid rocket engine injectors | design tool architecture |
| `Characterization-of-pintle-engine-performance-for-GO2-kerosene-propellants.pdf` | Characterization of pintle-engine performance for GO2/kerosene propellants | performance, heat flux, throttle |
| `Ma8uFpj30pN7K%2BP3wB2syoyllyK6k68jQ0r8DxEM8jQ%3D.pdf` | title not fully resolved from PDF metadata | likely combustion/chamber support |
| `Tesi_dottorato_Cavalieri.pdf` | PhD in Aeronautics and Space Engineering | spray/combustion support |
| `aerospace-09-00494-v2.pdf` | Effect of Local Momentum Ratio on Spray Windward Distribution of a Gas-Liquid Pintle Injector Element | LMR, gas-liquid spray |
| `ahn2014.pdf` | Effects of injector recess on heat flux in a combustion chamber with cooling channels | recess, heat flux |
| `applsci-15-02696.pdf` | Experimental Investigation on a Throttleable Pintle-Centrifugal Injector | throttle, pintle-centrifugal |
| `austin2002.pdf` | Characterization of Pintle Engine Performance for Nontoxic Hypergolic Bipropellants | engine test/performance |
| `austin2005.pdf` | Characterization of Pintle Engine Performance for Nontoxic Hypergolic Bipropellants | engine test/performance |
| `dressler2000.pdf` | TRW pintle engine heritage and performance characteristics | heritage/performance |
| `erkal2019.pdf` | AIAA paper; title unresolved from metadata | student/design-test support |
| `freeberg2019.pdf` | Spray Cone Formation from Pintle-Type Injector Systems in Liquid Rocket Engines | spray cone |
| `gavitt2001.pdf` | Testing of the 650Klbf LOX/LH2 Low Cost Pintle Engine | hot-fire/scale-up |
| `gromski2010.pdf` | Northrop Grumman TR202 LOX/LH2 Deep Throttling Engine Project Status | deep throttling |
| `kang2022.pdf` | Design of pintle injector using Kerosene-LOx and solving pintle tip thermal damage | design/thermal |
| `melcher2014.pdf` | Combustion Stability Characteristics of the Project Morpheus LOX/LCH4 Main Engine | LOX/methane stability |
| `mueller2000.pdf` | TRW 40 klbf LOX/RP-1 low cost pintle engine test results | hot-fire/performance |
| `nardi2015.pdf` | Experiments with Pintle Injector Design and Development | design/development |
| `ninish2018.pdf` | Spray characteristics of liquid-liquid Pintle injector | spray/atomization |
| `radhakrishnan2018 (1).pdf` | Lagrangian approach to axisymmetric spray simulation of pintle injector | spray simulation |
| `radhakrishnan2018.pdf` | Effect of injection conditions on mixing performance of pintle injector | mixing |
| `radhakrishnan2020.pdf` | title unresolved from metadata; duplicate pair present | spray/atomization |
| `radhakrishnan2021.pdf` | Detailed modelling on spray atomisation and combustion of LOX/GCH4 in variable area pintle injector | CFD/atomization/combustion |
| `s11630-015-0753-7.pdf` | Effects of Momentum Ratio and Weber Number on Spray Half Angles of Liquid Controlled Pintle Injector | TMR/We/spray angle |
| `s11630-016-0838-y.pdf` | Verification on Spray Simulation of a Pintle Injector for Liquid Rocket Engine | spray simulation |
| `s41598-024-64027-2.pdf` | Transient spray combustion characteristics in a gas-liquid pintle rocket engine under acoustic excitation | transient/acoustics |
| `s42405-022-00489-w.pdf` | Geometric Effects of Liquid Rocket Engine Pintle Injectors in Supercritical Combustion | supercritical geometry |
| `sakaki2015.pdf` | Optical Measurements of Ethanol/Liquid Oxygen Rocket Engine Combustor with Planar Pintle Injector | optical diagnostics |
| `sakaki2016.pdf` | Performance Evaluation of Rocket Engine Combustors using Ethanol/Liquid Oxygen Pintle Injector | performance |
| `sakaki2017.pdf` | Combustion Characteristics of Ethanol/LOX Rocket-Engine Combustor with Planar Pintle Injector | combustion |
| `sakaki2018.pdf` | Longitudinal combustion instability of a pintle injector for a liquid rocket engine combustor | instability |
| `son2016.pdf` | Design Procedure of a Movable Pintle Injector for Liquid Rocket Engines | movable pintle design |
| `son2017.pdf` | Design Procedure of a Movable Pintle Injector for Liquid Rocket Engines | movable pintle design |
| `son2017 (2).pdf` | Numerical study on combustion characteristics of a fuel-centered pintle injector for methane rocket engines | methane CFD |
| `song2021.pdf` | Atomization of gelled kerosene by multi-hole pintle injector for rocket engines | gelled propellant |
| `thermal-performance-study-of-mmhnto-rocket-thrust-chamber-a223.pdf` | Thermal performance study of MMH/NTO rocket thrust chamber | thermal/chamber |
| `tokudome2010.pdf` | A High-Speed Response LOX/LH2 Full Expander Cycle Engine with Deep Throttling Capability | cycle/throttle |

## Add or verify outside the local folder

These came from NTRS searches, Crossref title lookups, and reference trails in
the local pintle papers. Some may already exist elsewhere in `propulsion_texts`
under non-obvious filenames; verify before downloading.

| Source | Link | Why it matters |
|---|---|---|
| Elverum, *Combustion apparatus having a coaxial-pintle reactant injector*, 1980 | https://ntrs.nasa.gov/citations/20080004184 | Pintle patent/mechanism details |
| Gilroy and Sackheim, *The Lunar Module Descent Engine - A Historical Perspective*, AIAA 1989-2385 | https://doi.org/10.2514/6.1989-2385 | Apollo LMDE design history and throttle context |
| Elverum et al., *The descent engine for the lunar module*, AIAA 1967-521 | https://doi.org/10.2514/6.1967-521 | Original DPS engine paper |
| Gordon, *Summary of Deep Throttling Rocket Engines with Emphasis on Apollo LMDE*, AIAA 2006-5220 | https://doi.org/10.2514/6.2006-5220 | Throttle architecture and operating envelope |
| Gromski et al., *TR202 LOX/LH2 Deep Throttling Engine Technology Project Status*, AIAA 2010-6725 | https://ntrs.nasa.gov/citations/20100033110 | Deep throttle, LOX/LH2, feed/cycle context |
| Kim et al., *Design and development testing of the TR108*, AIAA 2005-3566 | https://doi.org/10.2514/6.2005-3566 | Pump-fed peroxide/hydrocarbon pintle heritage |
| Calvignac et al., *Design and testing of non-toxic RCS thrusters for Second Generation Reusable Launch Vehicle*, AIAA 2003-4922 | https://ntrs.nasa.gov/citations/20030067582 | Pintle/RCS test methods and nontoxic bipropellant context |
| Bedard et al., *Student design/build/test of a throttleable LOX-LCH4 thrust chamber*, AIAA 2012-3883 | https://doi.org/10.2514/6.2012-3883 | End-to-end throttleable chamber design/test example |
| Fang and Shen, *Study on atomization and combustion characteristics of LOX methane pintle injectors*, 2017 | https://doi.org/10.1016/j.actaastro.2017.03.025 | LOX/methane atomization/combustion |
| Zhou and Shen, *Influence of momentum ratio control mode on spray and combustion characteristics of a LOX/LCH4 pintle injector*, 2022 | https://doi.org/10.1631/jzus.A2100402 | TMR control-mode effect |
| Zhou, Shen, Jin, *Numerical study on the morphology of a liquid-liquid pintle injector element primary breakup spray*, 2020 | https://doi.org/10.1631/jzus.A1900624 | Primary breakup modeling |
| Kang et al., *A feasibility study of using pintle injector as sole-throttling device for shallow throttling condition*, 2020 | https://doi.org/10.1016/j.actaastro.2019.11.005 | Fixed/movable geometry throttle feasibility |
| Lee et al., *Spray characteristics of a pintle injector based on annular orifice area*, 2020 | https://doi.org/10.1016/j.actaastro.2019.11.008 | Annular area ratio and spray behavior |
| Lee et al., *Effects of skip distance on the spray characteristics of a pintle injector*, 2021 | https://doi.org/10.1016/j.actaastro.2020.09.043 | Skip distance / impingement distance |
| Son, Lee, Koo, *Characteristics of anchoring locations and angles for GOX/GCH4 flames of an annular pintle injector*, 2020 | https://doi.org/10.1016/j.actaastro.2020.08.036 | Flame anchoring and gas-gas/gas-liquid combustor behavior |
| Cheng, Li, Xu, Kang, *On the prediction of spray angle of liquid-liquid pintle injectors*, 2017 | https://doi.org/10.1016/j.actaastro.2017.05.037 | Spray angle correlation |
| Cheng, Li, Chen, *Flow characteristics of a pintle injector element*, 2019 | https://doi.org/10.1016/j.actaastro.2018.10.020 | Local flow and injection metrics |
| Chen, Li, Cheng, *Experimental research on the spray characteristics of pintle injector*, 2019 | https://doi.org/10.1016/j.actaastro.2019.06.032 | Experimental spray database |
| Zhao et al., *Review of atomization and mixing characteristics of pintle injectors*, 2022 | https://doi.org/10.1016/j.actaastro.2022.08.042 | Review and correlation selection |
| Zhang et al., *Experimental and numerical investigations on flow field characteristics of pintle injector*, 2020 | https://doi.org/10.1016/j.ast.2020.105924 | Flow field and CFD validation |
| Zhou et al., *Experimental and numerical investigations on the spray characteristics of liquid-gas pintle injector*, 2022 | https://doi.org/10.1016/j.ast.2022.107354 | Liquid-gas branch |
| Ma et al., *Generative Adversarial Networks with Physical Evaluators for Spray Simulation of Pintle Injector*, 2020 | https://arxiv.org/abs/2101.01217 | Optional surrogate-model reference, not a first-principles design source |

## Parameter links to encode in the CLI

- Engine target parameters should drive injector sizing through thrust,
  chamber pressure, ambient pressure, expansion ratio, mixture ratio, and
  propellant states. These set throat area, total mass flow, fuel/oxidizer mass
  split, injector pressure drops, and pump/manifold pressure requirements.
- Add explicit user questions for target injector pressure-drop fractions,
  allowable pump outlet pressures, inlet temperatures, propellant phases,
  radial stream choice, desired throttle range, minimum manufacturing feature,
  chamber radius/length or L-star policy, allowable wall wetting/film-cooling
  intent, and validation mode (screening vs cold-flow-calibrated).
- Replace the current generic vector spray angle with selectable correlations:
  SP-8089 screening, Cheng et al. spray-angle prediction, Zhou and Shen
  TMR/LMR block-factor relation, and Son/Lee skip-distance/annular-area
  corrections where input geometry is available.
- Treat pump output as a hard feasibility boundary:
  `P_pump_out >= Pc + dP_injector + manifold_losses + line_losses + regen_losses`
  for each stream, with NPSH/cavitation gates from the turbopump monographs and
  vapor-pressure margin from fluid properties.
- Tie chamber/nozzle geometry to injector gates: throat area and c-star set
  mass flow; chamber radius sets spray-wall interception; chamber length and
  L-star set breakup/vaporization residence time; nozzle throat heat flux and
  chamber wall cooling are affected by spray asymmetry, recess, film cooling,
  and flame anchoring.
- Separate liquid-liquid, gas-liquid, gas-gas, supercritical, gelled, and
  flashing/two-phase branches. The literature correlations are not
  interchangeable.

## Immediate implementation implications

- `raosim/injector.py` already has the right integration boundary:
  operating point -> mass-flow split -> dP -> areas -> TMR -> gates.
- The most urgent upgrade is replacing the current screening spray angle and
  SMD gates with literature-selectable correlations and clear validity domains.
- The second upgrade is adding geometry knobs from the literature:
  block factor, local momentum ratio, skip distance, annular area ratio,
  recess, radial-hole/slot layout, pintle-tip shape, and film-cooling fraction.
- The third upgrade is feed-system closure:
  pump/tank outlet pressure and flow capacity should be checked against the
  full chain of line, regen, manifold, and injector losses, not just
  `Pc + injector dP`.
