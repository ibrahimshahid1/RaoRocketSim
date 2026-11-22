# MDO remediation: literature traceability

This record distinguishes what the cited propulsion literature actually says
from repository calculations and software-policy decisions. Runtime code does
not parse this document or the transcription corpus. The original PDFs remain
the authority; Markdown under `propulsion_texts/propulsion_texts_for_agents`
is the search/index layer.

## Source ledger

| ID | Source | Location checked in the original PDF | Identifier |
|---|---|---|---|
| SP125 | Huzel, D. K. & Huang, D. H., *Design of Liquid Propellant Rocket Engines* (1967), NASA SP-125 | §2.1, source-PDF p. 40 / printed p. 31; injector design, source-PDF p. 137 / printed p. 128; coaxial-shell chamber design, source-PDF p. 118 / printed p. 109; propellant-tank design §8.1.2.3, source-PDF p. 348 / printed p. 339 | NASA report `19710019929` |
| SP8087 | *Liquid Rocket Engine Fluid-Cooled Combustion Chambers* (April 1972), NASA SP-8087 | coolant-temperature/coking guidance and channel-wall structural discussion; NTRS scan indexed as 1973 | NASA report `19730022965` |
| CYCLES | Parsley, R. C. & Zhang, B., “Thermodynamic Power Cycles for Pump-Fed Liquid Rocket Engines” (2004) | §I, source-PDF pp. 2–3 (chapter pp. 622–623) | DOI `10.2514/5.9781600866760.0621.0648` |
| HTD | Nasuti, F. & Pizzarelli, M., “Pseudo-boiling and heat transfer deterioration while heating supercritical liquid rocket engine propellants” (2021) | fluid behavior, source-PDF pp. 2–3; deterioration criterion, Eq. (9), source-PDF p. 6 | DOI `10.1016/j.supflu.2020.105066` |
| RAO58 | Rao, G. V. R., “Exhaust Nozzle Contour for Optimum Thrust” (1958) | source-PDF p. 1 / journal p. 377 and governing equations/figures that follow | DOI `10.2514/8.7324` |

Local originals:

- `propulsion_texts/19710019929.pdf`
- `propulsion_texts/19730022965.pdf`
- `propulsion_texts/fuel_pump_design/thermodynamic-power-cycles-for-pumpfed-liquid-rocket-engines-2004.pdf`
- `propulsion_texts/nasuti2021.pdf`
- `propulsion_texts/rao1958.pdf`

## Claim and decision matrix

| Remediation topic | Source-backed statement | Repository calculation / policy decision |
|---|---|---|
| O/F contract | SP125 §2.1 lists mixture ratio among the engine requirements that must be stated. | An explicit `DesignLayout`, a positive finite effective-O/F resolver, and an always-physical 11-value contract vector are software contracts. SP125 does not prescribe array shapes or sentinel handling. |
| Requirements ownership | SP125 §2.1 treats thrust and condition, performance, duration, mixture ratio, burnout weight, and envelope as defining requirements. | Requirement-owned fields cannot be overwritten by an analysis-config mapping, and callers cannot remove required solver rows while retaining a “requirements met” verdict. |
| Injector pressure-drop screen | SP125 injector design (source-PDF p. 137 / printed p. 128) labels 15–20% of chamber/nozzle stagnation pressure a rule-of-thumb design range. SP-194 separately explains qualitatively that increased injector pressure drop can suppress chug. | The repository’s 0.20 lower screen is preliminary sizing policy, not a proof of stability. It must not be described as a universal SP-194 numeric law. |
| Pressure search windows | Parsley & Zhang report about 10 MPa for expander-cycle potential, a 10–15 MPa gas-generator total-performance optimum, and onset of staged-combustion hardware limitation around 20–25 MPa, all with architecture-specific mechanisms. | Those values are recommended numerical search guidance, not universal hard validity domains. The electric-pump 1.5–6 MPa window is a repository default and is user-overridable. Hard admissibility comes from validated property/model domains and live component constraints. |
| Property tables | No paper citation makes a path string evidence of thermochemical validity. | Schema/interpolator versions, propellant identities, mode, units/O/F convention, finite monotone axes, shapes, physical sign checks, meaningful O/F dependence, and a content SHA-256 are required before a sampled table grants coverage. A constant fallback’s interpolation axes are not physical bounds. |
| HTD coverage | Nasuti & Pizzarelli Eq. (9) depends on heat flux, mass flux, friction factor, and the bulk real-fluid `(beta/cp)` term; the paper gives a threshold `K = 0.187`. It also shows that hydrogen and light hydrocarbons traverse strongly temperature-dependent regimes, with behavior dependent on reduced pressure. | Methane/hydrogen direct MDO is not called physics-feasible without validated real-fluid coolant surfaces. Treating hydrogen as requiring a real-fluid coverage check is a conservative repository decision, not a claim that every hydrogen channel deteriorates. Screening may proceed only with an explicit incomplete-physics opt-in, and its physics verdict remains unknown. |
| Chamber metal volume/mass | SP8087 distinguishes liner/channel-wall and outer reinforcement load paths. SP125’s coaxial-shell section discusses coolant-pressure hoop loading of the outer shell. SP125 Eq. (8-32), however, is in **propellant-tank design** and gives a cylindrical tank-section shell weight. | Liner, land, and closeout volumes are Pappus/shell-geometry calculations. Eq. (8-32) may corroborate the geometric surface-times-thickness form; it is not a thrust-chamber mass correlation or a chamber-thickness model. |
| Mass objective | The literature establishes that liner, lands/channels, and closeout are physical hardware regions. | `min_dry_mass_partial` includes the smooth electric package plus liner, channel lands, and closeout. It remains explicitly partial because injector hardware, manifolds, valves, lines, gimbal, and mounts are absent. |
| Constraint/feasibility contract | Individual physical margins retain the sources documented beside their implementing equations. | One ordered manifest owns names, scales, applicability, availability, optimizer role, and reporting order. Unavailable governing physics reduces to `unknown`, never `pass`; solver status and finiteness are unconditional final gates. |
| Hardware ledger | Literature-based shell/loading relations do not establish that every CAD subsystem was built. | Completeness is relative to a versioned named scope contract. Missing scopes and invalid geometry produce `None` plus a stable reason, never zero or an authoritative upper-bound mass. Mass and exported CAD share the same resolved geometry/body identity where a build-once path exists. |
| Rao contour fidelity | Rao 1958 formulates an optimum-thrust nozzle as a variational problem whose solution uses a control surface and the method of characteristics, with nozzle length, ambient pressure, and near-throat flow among the governing conditions. | The cheap fixed-topology Rao/TOP interpolation remains the inner-MDO provider. The variational/MOC solver is a post-solve higher-fidelity check until its convergence domain and differentiable sensitivities are demonstrated. This sequencing is a numerical-architecture decision, not a claim made by Rao. |

## Interpretation rule

“Literature-sourced” means the cited document directly supports the stated
physical relation, range, or applicability. “Repository calculation” means a
derivation from resolved geometry/inputs. “Policy decision” means a validation,
availability, optimization, or versioning contract. A source citation must not
be used to promote a calculation or policy decision into an experimental or
flight-qualified model.
