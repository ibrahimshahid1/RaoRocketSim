"""Machine-readable provenance and validity metadata for design models.

The registry deliberately distinguishes a published equation from a repository
heuristic.  A citation is not, by itself, evidence that a particular use of an
equation is validated for the current geometry, propellant, scale, or operating
state.  Consumers can therefore expose the same status in JSON reports and
refuse to promote screening assumptions into hardware authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProvenance:
    """One physical model, correlation, or design-policy assumption."""

    model_id: str
    subsystem: str
    quantity: str
    relation: str
    source: str
    validity: str
    status: str
    verification: str
    notes: str = ""
    local_source: str | None = None
    equation_ref: str = ""
    validation_level: str = "software_verified"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


_MODELS = (
    ModelProvenance(
        "injector.incompressible_orifice", "injector", "mass flux",
        "G = Cd*sqrt(2*rho*dP)",
        "Sutton & Biblarz, Rocket Propulsion Elements; NASA SP-8089",
        "single-phase, incompressible liquid; calibrated Cd required",
        "published_equation", "unit identity and mass-flow closure tests",
        local_source="propulsion_texts/19760023196.pdf",
        equation_ref="NASA SP-8089 injector pressure-drop sizing practice",
    ),
    ModelProvenance(
        "injector.compressible_orifice", "injector", "gas mass flux",
        "isentropic subsonic/choked orifice mass flux",
        "Anderson, Modern Compressible Flow; Sutton & Biblarz",
        "single-phase gas with known stagnation state and gamma/R; ideal-gas screen",
        "published_equation_screening_use", "choke-boundary and continuity tests",
        local_source="propulsion_texts/Fluid Mechanics, 7th Ed. (Mcgraw-Hill Series in Mechanical Engineering).pdf",
        equation_ref="compressible-flow/orifice relations",
    ),
    ModelProvenance(
        "injector.auto_pintle_diameter", "injector", "pintle diameter",
        "Dp = 0.30*Dc when Dp is omitted", "repository packaging policy",
        "no demonstrated performance-validity range",
        "repository_heuristic", "reported in result provenance; user override supported",
        "Must not be presented as a NASA or textbook sizing law.",
        validation_level="assumption_only",
    ),
    ModelProvenance(
        "injector.hinze_stable_drop", "injector", "stable drop diameter",
        "d = We_crit*sigma/(rho_g*u_rel^2), We_crit=13",
        "Hinze, AIChE Journal 1 (1955)",
        "dilute liquid drops in turbulent/aerodynamic breakup; not gas or transcritical flow",
        "published_correlation_screening_use",
        "equation identity, phase gate, pressure-validity tests",
        local_source="propulsion_texts/pintle_injector/PINTLE_LITERATURE_CATALOG.md",
        equation_ref="cataloged Hinze/Reitz screen; primary Hinze paper absent locally",
        validation_level="screening_not_physically_validated",
    ),
    ModelProvenance(
        "injector.primary_breakup_15dh", "injector", "primary breakup length",
        "Lb = 15*Dh",
        "repository midpoint of an order-of-magnitude Reitz-Bracco prompt-breakup range",
        "unvalidated for pintle annular sheets and rectangular slots",
        "repository_heuristic", "equation identity and geometry-applicability tests",
        "Replace with geometry-specific LISA/WAVE, slot-sheet, or round-jet models.",
        validation_level="assumption_only",
    ),
    ModelProvenance(
        "injector.d2_evaporation", "injector", "drop evaporation",
        "d^2(t) = d0^2 - K*t",
        "classical d-squared law; Priem-Heidmann vaporization-limited context",
        "isolated subcritical droplets; K must match propellant and gas state",
        "published_equation_user_calibration_required",
        "closed-form 99-percent time and remaining-volume tests",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="evaporation-model context and Eq. 16",
        validation_level="screening_not_physically_validated",
    ),
    ModelProvenance(
        "injector.throttle_area_schedule", "injector", "effective metering area",
        "solve A(f) at commanded dP/Pc for each throttle point",
        "orifice-law consequence; repository schedule",
        "commanded effective areas only; no actuator kinematics or Cd(stroke)",
        "derived_schedule_not_hardware_model",
        "area-ratio and pressure-drop closure tests",
        local_source="propulsion_texts/19760023196.pdf",
        equation_ref="orifice pressure-drop basis; actuator schedule is repository-derived",
        validation_level="software_verified_not_hardware_validated",
    ),
    ModelProvenance(
        "injector.son2017_movable_continuous_gap", "injector",
        "movable-pintle radial metering area",
        "R_f=D_post/2-t_post; A_tip=pi/sin(theta_pt)*"
        "[R_f^2-(R_f-L_open*sin(theta_pt)*cos(theta_pt))^2] "
        "(theta_pt->0: A_tip=2*pi*R_f*L_open); "
        "A_cg=pi/4*(D_cg^2-D_pr^2); A_eff=min(A_tip,A_cg)",
        "Son, Hong, and Koo, Design Procedure of a Movable Pintle "
        "Injector, Journal of Propulsion and Power 33(4), 2017",
        "single-phase incompressible cold-flow geometry below the "
        "A_tip=A_cg transition; configuration-specific Cd(opening,Re) "
        "calibration with an exact geometry fingerprint, position metrology, leakage, actuator load/stress "
        "inputs, and independently measured/VOF sheet thickness are required; "
        "mechanical opening is not liquid-sheet thickness",
        "published_geometry_with_calibrated_hydraulic_and_actuator_inputs",
        "exact Eq.-1/center-gap identities and Son Table-2 transition "
        "benchmarks for D_post=8 mm, t_post=0.5 mm, D_cg=4.55 mm, "
        "D_pr=3 mm: L_transition=0.417946, 0.454243, and 0.568310 mm "
        "at theta_pt=0, 20, and 40 deg (published 0.418/0.454/0.568 mm)",
        "Son Eq. 3 defines the discharge-coefficient measurement but does "
        "not provide a universal Cd-versus-stroke map. Static force and stem "
        "screens are repository ledgers, not actuator qualification; no swept "
        "moving CAD assembly or hot-fire validation is claimed.",
        local_source="propulsion_texts/pintle_injector/son2017.pdf",
        equation_ref="Eqs. 1-3 and Table 2, printed pages 859-861",
        validation_level=(
            "software_verified_geometry_and_literature_transition_only_"
            "not_hydraulic_actuator_sheet_or_hardware_validated"
        ),
    ),
    ModelProvenance(
        "injector.spray_cstar_fixed_point", "injector", "cycle mass-flow coupling",
        "eta_cstar=eta_vaporization*eta_mixing*eta_combustion; mdot=Pc*At/(eta_cstar*cstar)",
        "algebraic cycle closure with explicit repository coupling policy",
        "opt-in only; eta_mixing and eta_combustion must be independently supplied; direct-fuel regen re-solves coolant/wall/feed duty, while independent coolant/bypass topologies are excluded",
        "derived_repository_policy_not_combustion_model", "fixed-point/final-state closure, direct-fuel regen, pump-duty, stale-state, and invalid-state tests",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="vaporization-to-reacting-spray staged handoff and c-star efficiency context",
        validation_level="software_verified_not_hot_fire_calibrated",
    ),
    ModelProvenance(
        "spray.prescribed_one_way_carrier", "spray", "carrier state at parcels",
        "(u_g,rho_g,mu_g,T_g,p_g,k,epsilon)=bilinear_xr(prescribed field); no numerical extrapolation",
        "repository one-way carrier-interface policy; Euler-Lagrange source-term closure in Cavalieri thesis",
        "steady prescribed uniform or axisymmetric rectilinear field; one_way only; carrier mass/momentum feedback and carrier/droplet energy coupling are not applied",
        "repository_interface_policy",
        "manufactured-field interpolation, axis regularity, domain-policy, and immutable-state tests",
        "Parcel ledgers report equal/opposite carrier source demand, but a prescribed field cannot establish global two-way momentum or energy closure.",
        local_source="propulsion_texts/pintle_injector/Tesi_dottorato_Cavalieri.pdf",
        equation_ref="Thesis Eqs. 3.5-3.11 show the carrier source terms and droplet mass/momentum/energy equations omitted by one-way prescription",
        validation_level="software_verified_one_way_interface_not_cfd_or_physically_validated",
    ),
    ModelProvenance(
        "spray.schiller_naumann_drag", "spray", "spherical parcel acceleration",
        "Re_p=rho_g*|u_g-u_p|*d/mu_g; Cd=24/Re_p*(1+0.15*Re_p^0.687) to Re_p=1000, then Cd=0.44; a_d=3*Cd*rho_g*|u_rel|*u_rel/(4*rho_l*d)",
        "Schiller-Naumann spherical-particle drag; locally reproduced in Cavalieri thesis Eq. 3.12",
        "isolated spherical continuum particles/drops; one_way carrier; deformation, dense-spray interaction, history, lift, added mass, and carrier reaction feedback excluded",
        "published_correlation_one_way_use",
        "Stokes limit, zero-slip, high-Re branch, sign, and vector-acceleration tests",
        "Equation identity does not validate the spherical-drag approximation for deforming, evaporating, or dense pintle sprays.",
        local_source="propulsion_texts/pintle_injector/Tesi_dottorato_Cavalieri.pdf",
        equation_ref="PDF page 57, thesis Eq. 3.12 and parcel momentum Eq. 3.9",
        validation_level="software_verified_not_spray_drag_physically_validated",
    ),
    ModelProvenance(
        "spray.seeded_discrete_random_walk", "spray", "turbulent parcel dispersion",
        "tau_e=C_L*k/epsilon; u'_i~Normal(0,2*k/3); renew fluctuation at eddy expiry with explicit local RNG seed",
        "Fluent-style discrete random walk used by Radhakrishnan, Lee, and Koo (2021), with repository deterministic-RNG policy",
        "requires prescribed RANS k and epsilon and explicit C_L; isotropic Gaussian eddies, one_way coupling, and no crossing-trajectory or two-way turbulence modulation",
        "published_method_with_repository_policy",
        "same-seed reproducibility, eddy-lifetime, laminar no-RNG-consumption, and invalid-turbulence-state tests",
        "The paper states stochastic tracking and an eddy-lifetime time-scale constant of 0.15 but publishes no seed or carrier turbulence field; tau_e and isotropic variance are an explicit closure, not experimental truth.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Section 2.2, PDF page 11 / printed page 10: discrete random walk and eddy-life time-scale constant 0.15",
        validation_level="software_reproducible_closure_not_dispersion_physically_validated",
    ),
    ModelProvenance(
        "spray.wave_kh_radhakrishnan_2018", "spray", "WAVE/KH breakup scales",
        "2018 printed variant: Lambda/r=9.02*(1+0.45*Oh^0.5)*(1+0.4*Ta^0.7)/(1+0.87*We_g^1.67)^0.6; tau=3.726*B1*r/(Lambda*Omega)",
        "Radhakrishnan et al., Lagrangian Approach to Axisymmetric Spray Simulation (2018)",
        "secondary aerodynamic breakup of the paper's ambient water-air pintle cases; explicit liquid/carrier properties and B0/B1 required; one_way and not transferable to reacting/transcritical flow without validation",
        "published_equation_variant_literature_reproduction_use",
        "equation-identity, radius/diameter, coefficient-selection, breakup-relaxation, and conservation tests",
        "The exact paper variant (0.4/0.87) and the Reitz/OpenFOAM-compatible variant (0.4/0.865) are separate named selections.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2018 (1).pdf",
        equation_ref="Equations 2-7, PDF pages 8-9 / printed pages 450-451",
        validation_level="software_verified_literature_reproduction_not_physically_validated",
    ),
    ModelProvenance(
        "spray.wave_kh_radhakrishnan_2021", "spray", "WAVE/KH breakup scales",
        "2021 variant: Lambda/h=9.02*(1+0.45*Oh^0.5)*(1+0.45*Ta^0.7)/(1+0.87*We_g^1.67)^0.6; tau=3.726*B1*h/(Lambda*Omega)",
        "Radhakrishnan, Lee, and Koo, Combustion Theory and Modelling (2021)",
        "author modelling-2 secondary-breakup formulation with h=half the Table-7 full sheet thickness; explicit B0/B1 and properties; one_way; Tables 7/8 are author CFD, not experiment",
        "published_equation_variant_literature_reproduction_use",
        "coefficient-selection, VOF half-thickness calibration, equation-identity, time-march, and mass/momentum identity tests",
        "The coefficient version is explicit so 2018, 2021, and OpenFOAM/Reitz variants cannot be silently blended.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Equations 1-8, PDF pages 9-11 / printed pages 8-10",
        validation_level="software_verified_literature_reproduction_not_physically_validated",
    ),
    ModelProvenance(
        "spray.rayleigh_taylor_openfoam", "spray", "optional RT breakup event",
        "OpenFOAM ReitzKHRT fastest-growing RT frequency/wavelength plus c_tau timer and c_RT diameter update",
        "OpenFOAM Foundation ReitzKHRT model, version-10 source implementation",
        "optional and disabled by default; requires caller-supplied effective acceleration; absent from the Radhakrishnan modelling-2 validation path; one_way parcel update only",
        "external_open_source_model_optional_unvalidated",
        "zero-forcing, timer, event-selection, diameter, and represented mass/momentum identity tests",
        "The external source version is named but not pinned in the local literature corpus; this branch has no repository physical-validation dataset.",
        equation_ref="OpenFOAM Foundation v10 ReitzKHRT.C RT frequency, wave number, timer, and diameter logic",
        validation_level="software_identity_only_external_source_not_pinned_or_physically_validated",
    ),
    ModelProvenance(
        "spray.spalding_evaporation_eq16", "spray", "parcel evaporation rate",
        "dm/dt=-n*(Sh*D/d)*(pi*d^2)*rho_g*ln(1+Bm); Bm=(Y_s-Y_inf)/(1-Y_s); Sh=2 or 2+0.6*Re^0.5*Sc^(1/3)",
        "Radhakrishnan, Lee, and Koo (2021) Eq. 16 with explicit stagnant-sphere or Ranz-Marshall Sherwood closure",
        "caller supplies diffusivity, Bm/vapor fractions, densities, viscosity, slip, and named Sh closure; one_way vapor source demand only; no droplet-temperature/energy equation, phase equilibrium, real-fluid, or carrier-energy feedback",
        "published_equation_with_explicit_user_closure",
        "Eq.-16 sign/area identity, Sh/Re/Sc limits, frozen-step d-squared integration, and local mass/momentum source-demand tests",
        "The Sherwood and diffusivity/Bm closures are separate inputs; citing Eq. 16 alone does not determine them or validate LOX evaporation.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Eq. 16, PDF page 13 / printed page 12; convection-diffusion evaporation context",
        validation_level="software_verified_not_evaporation_or_energy_physically_validated",
    ),
    ModelProvenance(
        "spray.weighted_smd_rosin_rammler", "spray", "droplet-cloud statistics",
        "d32=sum(N_i*d_i^3)/sum(N_i*d_i^2); RR mass survival=exp[-(d/dbar)^n]; mass percentiles weight N_i*d_i^3",
        "Radhakrishnan et al. 2018 Eq. 8 and Radhakrishnan, Lee, and Koo 2021 Eq. 9/Rosin-Rammler discussion; repository multiplicity-aware policy",
        "requires positive physical droplet multiplicities and a representative sampled cloud; RR fit is a statistical summary, not an atomization or combustion model",
        "published_statistic_with_repository_policy",
        "analytic weighted-d32 identities, known-distribution RR recovery, percentile, convention, and seeded-sampling tests",
        "The 2021 expression is a decreasing mass-survival fraction, not an increasing CDF; no raw Table-8 parcel samples or per-case RR fits were published.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Equation 9 and paragraph following Table 7, PDF pages 11 and 22 / printed pages 10 and 21",
        validation_level="software_verified_statistic_not_spray_physically_validated",
    ),
    ModelProvenance(
        "spray.benchmark_radhakrishnan2018_water_air", "spray", "cold-flow validation fixture",
        "SHA-pinned 2018 geometry/BC/VOF/WAVE rows with experimental spray-half-angle and SMD targets kept origin-separated",
        "repository evidence policy applied to Radhakrishnan et al. 2018 water-air measurements",
        "ambient water-air pintle only; angle/SMD are experimental targets, while sheet thickness, velocity, and B0/B1 are author VOF/derived values; missing carrier/property/parcel data block strict end-to-end reproduction",
        "repository_evidence_policy_with_published_experimental_target",
        "PDF SHA, schema/unit/origin, exact-row, duplicate-id, and readiness-blocker tests",
        "The nominal 0.2-mm 2021 Table-5 restatement is a distinct publication revision and is never averaged into this fixture.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2018 (1).pdf",
        equation_ref="Tables 1-3, Figure 12, and adjacent uncertainty paragraph",
        validation_level="experimental_target_fixture_not_end_to_end_model_validation",
    ),
    ModelProvenance(
        "spray.benchmark_radhakrishnan2021_water_air", "spray", "cold-flow validation fixture revision",
        "SHA-pinned 2021 Table-5 water-air experiment/author-simulation comparison stored separately from the 2018 rows",
        "repository evidence policy applied to Radhakrishnan, Lee, and Koo 2021 Table 5",
        "single ambient water-air case; experimental and author-simulation columns remain origin-separated; incomplete carrier field, property, parcel, and RNG data block strict end-to-end reproduction",
        "repository_evidence_policy_with_published_experimental_target",
        "PDF SHA, revision-separation, exact-value, schema/unit/origin, and readiness-blocker tests",
        "The 2021 sheet thickness and B0/B1 differ from the nominally similar 2018 case; this is a separate fixture revision.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Table 5 and preceding validation prose, PDF pages 15-16 / printed pages 14-15",
        validation_level="experimental_target_fixture_distinct_revision_not_end_to_end_model_validation",
    ),
    ModelProvenance(
        "spray.benchmark_radhakrishnan2021_tables_7_8", "spray", "literature reproduction fixture",
        "SHA-pinned Table-3 inputs joined to Table-7 VOF outputs and Table-8 WAVE/Lagrangian SMD outputs with column-level origins",
        "repository evidence policy applied to Radhakrishnan, Lee, and Koo 2021 author CFD",
        "literature_reproduction_only; Tables 7 and 8 are author VOF/Lagrangian CFD, not experimental LOX/GCH4 measurements; missing carrier/property/parcel data block strict end-to-end validation",
        "repository_evidence_policy_for_published_author_cfd_not_experiment",
        "PDF SHA, schema/unit/origin, exact-table, anti-experimental-relabeling, and readiness-blocker tests",
        "Agreement can be reported as reproduction of published author outputs only and must never be promoted as physical validation.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Tables 3, 7, and 8, PDF pages 8, 22, and 25 / printed pages 7, 21, and 24",
        validation_level="literature_reproduction_only_not_experimental_validation",
    ),
    ModelProvenance(
        "spray.primary_geometry_dispatch", "spray", "liquid parcel source geometry",
        "mdot*Delta_t = sum(N_i*rho_l*pi*d_i^3/6); deterministic azimuthal placement and geometry-specific source direction",
        "Radhakrishnan et al. 2018/2021 radial-sheet setup plus explicit repository geometry-dispatch policy",
        "the literature-calibrated primary path is limited to the movable radial sheet; axial annuli, slots, and holes are geometric secondary-breakup blobs only",
        "published_geometry_mapping_with_repository_applicability_policy",
        "source mass/momentum identity, antipodal symmetry, geometry rejection, and primary-path gate tests",
        "Primary-path eligibility is not cycle-coupling eligibility; current repository injector forms remain blocked from the paper-specific primary path.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Figure 2 and Eqs. 1-8 radial-sheet-to-Lagrangian handoff",
        validation_level="software_verified_geometry_dispatch_not_primary_breakup_physically_validated",
    ),
    ModelProvenance(
        "spray.deterministic_parcel_march", "spray", "parcel trajectories and reservoirs",
        "du_p/dt=a_drag+g; dx_p/dt=u_p with WAVE, DRW, Eq.16 evaporation, exact represented-mass reservoirs, and segment boundary events",
        "Euler-Lagrange equations routed through Cavalieri thesis and Radhakrishnan 2018/2021; repository operator-split march",
        "prescribed one-way carrier, fixed parcel temperature, timestep-converged dilute spherical parcels; no two-way mass/momentum/energy feedback",
        "published_equations_with_repository_numerical_policy",
        "manufactured ballistic/drag/wall/outlet/evaporation/breakup, determinism, sampling, and conservation tests",
        "A closed parcel ledger proves software accounting only; global carrier momentum and energy remain open and are failed gates.",
        local_source="propulsion_texts/pintle_injector/Tesi_dottorato_Cavalieri.pdf",
        equation_ref="Thesis Eqs. 3.5-3.12 and Radhakrishnan 2021 Eq. 16",
        validation_level="software_verified_one_way_march_not_cfd_or_physically_validated",
    ),
    ModelProvenance(
        "spray.typed_cycle_handoff", "spray", "parcel-to-cycle evidence contract",
        "eta_vap=sum(m_vaporized,liquid)/sum(m_injected,liquid), with gas-carrier eta undefined and eligibility derived from all required gates",
        "repository physical-integrity and provenance policy",
        "reporting/interface only; requires exact stream continuity, carrier fingerprint, conservation, energy, convergence, target benchmark, phase, and non-regen scope",
        "repository_verification_policy",
        "gas/liquid representation, stale-field fingerprint, role/flow identity, source-hash, serialization, and fail-closed gate tests",
        "The handoff is currently always cycle-ineligible because phase/critical evidence, two-way carrier energy/momentum, and strict target benchmarks are absent.",
        local_source="propulsion_texts/pintle_injector/PINTLE_LITERATURE_CATALOG.md",
        equation_ref="repository evidence routing for Lagrangian spray models",
        validation_level="software_verified_fail_closed_interface_not_cycle_validated",
    ),
    ModelProvenance(
        "spray.openfoam13_external_gap_vof", "spray", "external VOF case export",
        "mdot_wedge=mdot_360*theta/(2*pi); U_r=mdot_360/(rho*2*pi*r_tip*L_open); incompressibleVoF(alpha.water,U,p_rgh)",
        "Radhakrishnan et al. 2018 Section 2.3 plus OpenFOAM Foundation v13 tag 20260624 incompressibleVoF/nozzleFlow2D and damBreakLaminar tutorials",
        "water-only, isothermal incompressible, subcritical external-gap wedge; mechanical opening is prescribed and the internal center-gap turn/post/tip surfaces are not meshed",
        "published_workflow_reduced_external_case_template",
        "fixture SHA, SI mapping, wedge mass-flux, dictionary-set, patch, hash, deterministic-writer, and fail-closed gate tests",
        "The 3.03 g/s annular air stream belongs to the later Lagrangian carrier stage. Static export is not an OpenFOAM run, paper reproduction, or physical validation.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2018 (1).pdf",
        equation_ref="Section 2.3 and Figures 1-4, PDF pages 4-6 / printed pages 446-448; OpenFOAM v13 patch 20260624 dictionaries",
        validation_level="software_template_static_verified_not_openfoam_run_or_physically_validated",
    ),
    ModelProvenance(
        "spray.vof_to_lagrangian_handoff", "spray", "VOF sheet-to-parcel evidence contract",
        "h_full,U_sheet,mdot_liq,J_liq and carrier(x,r) are accepted only after mass/momentum, stationarity, mesh, timestep, domain and averaging gates pass",
        "Radhakrishnan et al. 2018/2021 VOF-to-Lagrangian workflow with repository provenance and conservation policy",
        "typed interface only; requires fingerprinted extraction, full-thickness convention, alpha-isocontour definition, converged VOF fields and carrier coverage; does not itself run or validate CFD",
        "published_workflow_with_repository_fail_closed_interface_policy",
        "immutability, fingerprint, full-thickness, variation, carrier-domain, four-refinement, mass/momentum and conversion-block tests",
        "A caller cannot promote author-VOF outputs or an unconverged field to parcel authority merely by supplying finite sheet values.",
        local_source="propulsion_texts/pintle_injector/radhakrishnan2021.pdf",
        equation_ref="Modelling-2 VOF sheet to WAVE/Lagrangian sequence, Figures 3-4 and Tables 7-8",
        validation_level="software_verified_fail_closed_interface_not_cfd_or_physically_validated",
    ),
    ModelProvenance(
        "chamber.lstar_volume", "chamber", "chamber volume",
        "Vc = Lstar*At",
        "Huzel & Huang; Sutton & Biblarz",
        "empirical propellant/injector design input; not predicted combustion closure",
        "published_design_parameter", "exact revolved-volume closure tests",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="NASA SP-125 chamber characteristic-length practice",
    ),
    ModelProvenance(
        "chamber.shoulder_fill_fraction", "chamber", "shoulder fillet",
        "Rs = 0.8*Rs,max",
        "repository geometry policy; docs/shoulder_radius_design_basis.md",
        "geometric feasibility only; no demonstrated performance optimum",
        "repository_heuristic", "tangency and feasibility tests",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="SP-125 qualitative contraction factors; 0.8 is repository policy",
        validation_level="assumption_only",
    ),
    ModelProvenance(
        "pump.auto_efficiency_flow_head", "pump", "hydraulic efficiency",
        "piecewise eta(Q) with high-head deductions",
        "repository conservative screening policy",
        "preliminary duty estimate only; replace with measured pump map",
        "repository_heuristic", "bounds and monotonic-duty regression tests",
        validation_level="assumption_only",
    ),
    ModelProvenance(
        "pump.meanline_coefficients", "pump", "impeller dimensions",
        "psi=gH/U2^2; phi2=Cm2/U2; Q=b2*(pi*D2-Z*t/sin(beta2))*Cm2",
        "NASA SP-8109 centrifugal-pump meanline practice",
        "best-efficiency meanline screening; requires slip/loss validation and pump map",
        "published_equations_screening_use", "velocity-triangle identity tests",
        local_source="propulsion_texts/fuel_pump_design/19740020848.pdf",
        equation_ref="NASA SP-8109 centrifugal-pump meanline design",
        validation_level="software_verified_not_hardware_validated",
    ),
    ModelProvenance(
        "pump.annular_eye_free_area", "pump", "shaft/hub/eye and beta1",
        "Q=pi*(R1^2-Rh^2)*(1-B1)*phi1*omega*R1; B=Z*t/(2*pi*r*sin(beta))",
        "NASA SP-8109 inlet/discharge free-area and velocity-triangle practice",
        "one-dimensional blockage/zero-incidence meanline; blade-to-blade CFD, stress, cavitation and pump-map tests required",
        "published_equations_screening_use", "continuity, coefficient, blockage, shaft-fit, beta1 and CAD-identity tests",
        local_source="propulsion_texts/fuel_pump_design/19740020848.pdf",
        equation_ref="NASA SP-8109 impeller inlet/discharge velocity and free-area criteria",
        validation_level="software_verified_not_hardware_validated",
    ),
    ModelProvenance(
        "pump.split_casing_joint", "pump", "separable volute pressure boundary",
        "Fsep=p*Aprojected; Fclamp=1.5*Fsep; sigma_b=Fclamp/(N*At)",
        "standard pressure-joint mechanics with NASA SP-8109 volute/casing practice",
        "bounded machining/assembly topology and preliminary clamp screen only; gasket, flange FEA, threads, dowels, fatigue and proof test excluded",
        "published_equations_screening_use", "two-solid validity, volume closure, tool access, bolt/flow clearance and STEP round-trip tests",
        local_source="propulsion_texts/fuel_pump_design/19740020848.pdf",
        equation_ref="NASA SP-8109 volute/casing practice plus standard pressure-separation balance",
        validation_level="software_verified_not_joint_qualified",
    ),
    ModelProvenance(
        "performance.isentropic_area_mach", "performance", "Mach/area state",
        "A/Astar=f(M,gamma); p/p0 and T/T0 are isentropic functions",
        "standard compressible-flow relations",
        "calorically perfect, constant-gamma, one-dimensional isentropic flow",
        "published_equations", "inverse/forward identity and reference-value tests",
        local_source="propulsion_texts/Fluid Mechanics, 7th Ed. (Mcgraw-Hill Series in Mechanical Engineering).pdf",
        equation_ref="compressible-flow/nozzle chapters",
    ),
    ModelProvenance(
        "performance.frozen_variable_cp_quasi1d", "performance",
        "thermally-perfect frozen-composition nozzle state and thrust",
        "h0-h(T)=u^2/2; p/p0=exp[int(T0->T,cp/T dT)/R]; "
        "u*=sqrt(gamma(T*)*R*T*); A/A*=G*/G; "
        "c*=p0/G*; Cf=G*ue/p0+epsilon*(pe-pa)/p0",
        "Anderson, Modern Compressible Flow, 3rd ed., Sections 17.3-17.6; "
        "NASA SP-125 frozen-flow practice",
        "fixed ideal-gas composition and R with bounded piecewise-linear cp(T); "
        "adiabatic, inviscid, isentropic quasi-one-dimensional flow; initial "
        "integrated-design use is Bezier-only and excludes MOC/Rao characteristics, "
        "equilibrium chemistry, boundary-layer/thermal/Bartz and Hall-Cd authority",
        "published_equations_with_repository_numerical_policy",
        "exact cp and cp/T segment-integral tests, sonic/energy/entropy/area/mass "
        "closure, constant-cp collapse, branch/inverse-pressure tests, strict "
        "provenance-schema and fingerprint tests",
        "Closure residuals verify software identities only. Property-grid "
        "refinement, a pinned external thermochemistry fixture, profile-aware "
        "thermal/viscous adapters, CFD and test evidence remain release gates.",
        local_source="propulsion_texts/5f36b7c4ded79bb3e90754d0f81682f7a68014be.pdf",
        equation_ref="PDF pp. 669-685, Sections 17.3-17.6: high-temperature "
        "nozzle flow, frozen composition/mixture cp, and frozen sound speed",
        validation_level="software_verified_conservation_only_not_property_or_hardware_validated",
    ),
    ModelProvenance(
        "performance.ideal_thrust_coefficient", "performance", "thrust coefficient",
        "Cf=momentum term+epsilon*(pe-pa)/Pc",
        "standard rocket-nozzle momentum balance",
        "steady attached quasi-one-dimensional flow; no separation loss",
        "published_equation", "closed-form identity and engine tests",
        local_source="propulsion_texts/19770009165.pdf",
        equation_ref="NASA SP-8120 performance and expansion design",
        validation_level="software_verified_not_cfd_validated",
    ),
    ModelProvenance(
        "throat.hall_discharge_coefficient", "throat", "inviscid discharge coefficient",
        "Cd~=1-(gamma+1)/(96*(Ru/Rt)^2)",
        "Hall transonic leading term as used by NASA SP-8120",
        "leading-order inviscid screen over the documented curvature range",
        "published_equation_screening_use", "forward/inverse regression tests",
        local_source="propulsion_texts/19770009165.pdf",
        equation_ref="NASA SP-8120 section 2.1.1.1",
    ),
    ModelProvenance(
        "nozzle.rao_top_bezier", "nozzle", "bell contour",
        "Rao angle charts plus equivalent-cone length and quadratic Bezier",
        "Rao TOP chart approximation and NASA SP-8120 design practice",
        "axisymmetric preliminary contour inside the tabulated epsilon/length grid",
        "published_chart_with_repository_interpolant", "chart and external-geometry regressions",
        local_source="propulsion_texts/19770009165.pdf",
        equation_ref="Rao/TOP chart figures and bell-contour guidance",
        validation_level="benchmark_screened_not_hardware_validated",
    ),
    ModelProvenance(
        "nozzle.moc", "nozzle", "characteristic flow field",
        "axisymmetric compatibility and MOC unit processes",
        "NASA/JHU MOC source port and classical MOC",
        "steady inviscid isentropic constant-gamma supersonic region",
        "source_port_experimental", "NASA M3.5 perfect-nozzle fixture regressions",
        local_source="propulsion_texts/20030067852.pdf",
        equation_ref="NASA/JHU nozzle design-code report",
        validation_level="reference_matched_subset",
    ),
    ModelProvenance(
        "nozzle.rao_variational_bvp", "nozzle", "optimum-thrust control surface",
        "Rao stationarity plus characteristic compatibility and mass/length closure",
        "G. V. R. Rao, Exhaust Nozzle Contour for Optimum Thrust (1958)",
        "axisymmetric inviscid isentropic constant-gamma smooth-flow region",
        "published_equations_experimental_solver",
        "residual/topology suite and strict Rao 1958 Nozzle B Table II thrust benchmark",
        local_source="propulsion_texts/rao1958.pdf",
        equation_ref="Eqs. 1-19 and Figs. 1-4",
        validation_level="strict_literature_benchmark_passed_not_cfd_validated",
    ),
    ModelProvenance(
        "thermal.bartz", "thermal", "gas-side heat-transfer coefficient",
        "Bartz correlation with throat-diameter and property correction sigma",
        "Bartz rocket-nozzle heat-transfer correlation",
        "turbulent thrust chambers inside the correlation/calibration envelope",
        "published_correlation_screening_use", "exponent transcription and magnitude tests",
        local_source="propulsion_texts/technical-notes-1957.pdf",
        equation_ref="Bartz 1957 heat-transfer correlation",
        validation_level="software_verified_not_cht_validated",
    ),
    ModelProvenance(
        "thermal.sieder_tate", "thermal", "coolant film coefficient",
        "Nu=0.027*Re^0.8*Pr^(1/3)*(mu_b/mu_w)^0.14",
        "Sieder-Tate correlation; regenerative-cooling practice",
        "fully turbulent internal flow; property and entrance restrictions apply",
        "published_correlation_screening_use", "equation and trend tests",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="NASA SP-125 regenerative-cooling analysis context",
        validation_level="software_verified_not_cht_validated",
    ),
    ModelProvenance(
        "thermal.rectangular_laminar_nusselt", "thermal", "laminar film coefficient",
        "Shah-London all-walls-uniform-heat-flux rectangular-duct polynomial",
        "Shah and London rectangular-duct solution",
        "fully developed laminar flow with all walls uniformly heated",
        "published_equation_external_source", "limiting-aspect-ratio tests",
        local_source="propulsion_texts/Fluid Mechanics, 7th Ed. (Mcgraw-Hill Series in Mechanical Engineering).pdf",
        equation_ref="noncircular-duct section and Ref. 34 routing to Shah & London",
        validation_level="software_verified_local_secondary_source",
    ),
    ModelProvenance(
        "thermal.darcy_pressure_loss", "thermal", "coolant pressure loss",
        "dP=f*(L/Dh)*rho*v^2/2 plus local losses",
        "Darcy-Weisbach with laminar/Blasius/Swamee-Jain friction factors",
        "single-phase internal flow; roughness and local-loss inputs required",
        "published_equations_screening_use", "equation and limiting-regime tests",
        local_source="propulsion_texts/Fluid Mechanics, 7th Ed. (Mcgraw-Hill Series in Mechanical Engineering).pdf",
        equation_ref="internal-flow pressure-loss chapters",
    ),
    ModelProvenance(
        "structure.sp125_liner_stress", "structure", "liner combined stress",
        "pressure, thermal, and constraint terms from SP-125 screening equations",
        "NASA SP-125 liquid-rocket thrust-chamber design",
        "preliminary thin-wall/liner screen; detailed joints and 3-D loads excluded",
        "published_equations_screening_use", "independent equation reconstruction tests",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="SP-125 Eqs. 4-29, 4-31, and 4-32",
        validation_level="software_verified_not_fea_validated",
    ),
    ModelProvenance(
        "structure.strain_life", "structure", "fatigue life",
        "Coffin-Manson/Basquin total-strain relation",
        "classical strain-life fatigue; SP-125 chamber-life context",
        "requires process/temperature/cycle-specific coefficients and creep/TMF treatment",
        "published_equation_user_data_required", "synthetic-coefficient equation tests",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="SP-125 cyclic-life discussion",
        validation_level="screening_not_material_qualified",
    ),
    ModelProvenance(
        "materials.narloy_grcop", "materials", "temperature-dependent properties",
        "interpolated representative conductivity/strength/fatigue properties",
        "NASA NARloy-Z and GRCop reports",
        "screening values; heat/lot/process-specific design allowables required",
        "published_data_screening_catalog", "positivity and trend tests",
        local_source="propulsion_texts/materials_science/19740017910.pdf",
        equation_ref="NARloy-Z property data; GRCop sources separately cataloged",
        validation_level="screening_not_material_qualified",
    ),
    ModelProvenance(
        "pump.inducer_npsh", "pump", "inducer and suction margin",
        "continuity, incidence/solidity, suction-specific-speed and NPSH screens",
        "NASA SP-8052 liquid-rocket inducer design",
        "meanline preliminary design; cavitation testing required",
        "published_equations_screening_use", "geometry identities and chart-point tests",
        local_source="propulsion_texts/fuel_pump_design/19710025474.pdf",
        equation_ref="NASA SP-8052 inducer design criteria",
        validation_level="software_verified_not_cavitation_validated",
    ),
    ModelProvenance(
        "pump.electric_drive", "pump", "motor/inverter/battery sizing",
        "hydraulic power to shaft/electrical power to energy/current/mass",
        "energy balance with technology-specific efficiency and density inputs",
        "screening assumptions; measured maps and pulse thermal data required",
        "derived_balance_with_repository_defaults", "power/energy identity and Lee reference-point tests",
        local_source="propulsion_texts/fuel_pump_design/s42405-020-00325-z.pdf",
        equation_ref="electric-pump-fed engine package study",
        validation_level="software_verified_not_hardware_validated",
    ),
    ModelProvenance(
        "interface.pressure_bolt_plate", "interface", "injector/chamber joint",
        "pressure force, thin-wall hoop, clamped-plate bending, bolt screens",
        "standard pressure-vessel/plate/fastener relations with SP-125 context",
        "preliminary screen; gasket, preload scatter, threads and flange FEA excluded",
        "published_equations_screening_use", "force/stress identity and geometry tests",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="NASA SP-125 structural/joint design context",
        validation_level="software_verified_not_joint_qualified",
    ),
    ModelProvenance(
        "separation.empirical_onset", "separation", "separation pressure",
        "Kalt-Badal/Schmucker empirical pressure-ratio criteria",
        "empirical rocket-nozzle separation literature",
        "onset screen only; side loads and separated thrust require CFD/test",
        "published_correlations_screening_use", "formula/provenance and trend tests",
        local_source="propulsion_texts/hagemann1998.pdf",
        equation_ref="overexpanded-nozzle separation review",
        validation_level="screening_not_cfd_validated",
    ),
    ModelProvenance(
        "cad.kernel_topology", "cad", "solid validity",
        "OpenCascade valid-solid/re-import/bounding-box/volume checks",
        "geometric-kernel verification policy",
        "proves exchange topology only, not manufacturability or pressure integrity",
        "repository_verification_policy", "STEP/STL round-trip tests",
        local_source="propulsion_texts/CAD_04.pdf",
        equation_ref="CAD exchange/solid-modeling background",
        validation_level="topology_only",
    ),
    ModelProvenance(
        "thermochemistry.rocketcea_snapshot", "thermochemistry", "chamber products",
        "RocketCEA chamber gamma, molecular weight, temperature, and cstar snapshot",
        "NASA CEA through the optional RocketCEA implementation",
        "frozen chamber-property snapshot only; variable-composition expansion excluded",
        "external_solver_interface", "adapter/plumbing tests; real CEA regression still required",
        local_source="propulsion_texts/19710019929.pdf",
        equation_ref="SP-125 equilibrium-performance context; RocketCEA is external authority",
        validation_level="external_solver_not_locally_regression_validated",
    ),
    ModelProvenance(
        "flow.boundary_layer_displacement", "flow", "effective nozzle area loss",
        "turbulent flat-plate displacement-thickness screening correlation",
        "classical turbulent boundary-layer scaling",
        "attached smooth-wall flow; pressure gradient, curvature, chemistry and roughness simplified",
        "published_correlation_screening_use", "trend and bounded-area-loss tests",
        local_source="propulsion_texts/Fluid Mechanics, 7th Ed. (Mcgraw-Hill Series in Mechanical Engineering).pdf",
        equation_ref="turbulent boundary-layer chapter",
        validation_level="screening_not_cfd_validated",
    ),
    ModelProvenance(
        "thermal.fin_efficiency", "thermal", "regen rib heat spreading",
        "eta_f=tanh(mH)/(mH), m=sqrt(2h/(k*t_land))",
        "straight-fin heat-transfer solution applied to channel lands",
        "one-dimensional fin with idealized base/tip and uniform properties",
        "published_equation_screening_use", "limiting-behavior tests",
        local_source="propulsion_texts/atefi2019.pdf",
        equation_ref="regen channel-fin treatment",
        validation_level="software_verified_not_cht_validated",
    ),
    ModelProvenance(
        "thermal.radiation", "thermal", "gas/wall radiative heat flux",
        "grey-gas/wall emissivity screening contribution",
        "rocket-chamber radiation literature",
        "band/soot/species dependence simplified; property calibration required",
        "published_model_screening_use", "sign, bounds and energy-accounting tests",
        local_source="propulsion_texts/leccese2018.pdf",
        equation_ref="rocket thrust-chamber radiative heat-transfer treatment",
        validation_level="screening_not_spectral_validated",
    ),
    ModelProvenance(
        "thermal.chf", "thermal", "critical heat flux margin",
        "conservative Zuber-class boiling reference with optional property backend",
        "classical pool-boiling CHF used as a conservative reference",
        "not a validated forced-flow cryogenic channel CHF correlation",
        "published_reference_screen", "finite/bounds and margin tests",
        local_source="docs/thermofluid_literature_provenance.md",
        equation_ref="repository provenance note and declared forced-flow limitation",
        validation_level="conservative_reference_not_design_correlation",
    ),
    ModelProvenance(
        "injector.momentum_resultant_angle", "injector", "spray direction",
        "theta=atan2(M_radial*cos(delta), M_axial+M_radial*sin(delta))",
        "first-order momentum-vector construction; NASA SP-8089 design variables",
        "screening direction only; cold-flow distribution and interaction physics required",
        "derived_physics_screen", "vector-limit and trend tests",
        local_source="propulsion_texts/19760023196.pdf",
        equation_ref="NASA SP-8089 momentum/geometry design practice",
        validation_level="screening_not_cold_flow_validated",
    ),
    ModelProvenance(
        "injector.n_tau_stability", "injector", "combustion response band",
        "low-order n-tau sensitive-frequency screen",
        "classical time-lag combustion-stability screening",
        "not a chamber transfer-function or stability-margin prediction",
        "published_concept_screening_use", "frequency/time-scale trend tests",
        local_source="propulsion_texts/19760023196.pdf",
        equation_ref="NASA SP-8089 combustion-stability design/testing guidance",
        validation_level="screening_not_stability_validated",
    ),
    ModelProvenance(
        "pump.synthetic_map", "pump", "off-design head/efficiency curve",
        "repository-generated nondimensional curve about the meanline duty point",
        "repository screening policy",
        "not a measured map and not valid for cavitation or control design",
        "repository_heuristic", "shape/bounds regression tests",
        validation_level="assumption_only",
    ),
    ModelProvenance(
        "altitude.attached_thrust", "altitude", "off-design thrust",
        "quasi-one-dimensional attached-flow Cf with explicit separation applicability",
        "standard nozzle momentum balance plus empirical separation onset",
        "attached flow only; separated-flow thrust must be withheld or separately modeled",
        "published_equation_with_applicability_gate", "altitude/separation regression tests",
        local_source="propulsion_texts/hagemann1998.pdf",
        equation_ref="overexpanded nozzle flow and separation regimes",
        validation_level="screening_not_separated_flow_validated",
    ),
    ModelProvenance(
        "trajectory.vertical_point_mass", "trajectory", "altitude/velocity history",
        "m*dv/dt=T-D-mg with mass depletion and vertical kinematics",
        "Newtonian point-mass flight mechanics",
        "one-dimensional vertical screen; no guidance, wind, staging dynamics or 6-DOF",
        "published_equations_screening_use", "constant-acceleration and integrated-dv tests",
        local_source="propulsion_texts/Fluid Mechanics, 7th Ed. (Mcgraw-Hill Series in Mechanical Engineering).pdf",
        equation_ref="continuum drag background; point-mass mechanics is elementary",
        validation_level="software_verified_mission_screen",
    ),
    ModelProvenance(
        "regen.boolean_brep", "cad", "regenerative passage geometry",
        "revolved envelopes plus patterned passage subtraction and optional manifolds/ports",
        "repository CAD construction policy",
        "topology/connectivity only; process, min-wall, maldistribution and proof require evidence",
        "repository_verification_policy", "STEP re-import and passage-connectivity tests",
        local_source="propulsion_texts/gradl2018.pdf",
        equation_ref="additively manufactured channel-wall design context",
        validation_level="topology_only",
    ),
)

MODEL_REGISTRY: dict[str, ModelProvenance] = {
    item.model_id: item for item in _MODELS
}


def get_model_provenance(model_id: str) -> ModelProvenance:
    """Return one registry entry, raising ``KeyError`` for an unknown model."""

    return MODEL_REGISTRY[model_id]


def model_provenance_dict(*, subsystem: str | None = None) -> dict[str, dict]:
    """Serialize the registry, optionally selecting one subsystem."""

    return {
        model_id: entry.to_dict()
        for model_id, entry in MODEL_REGISTRY.items()
        if subsystem is None or entry.subsystem == subsystem
    }


def audit_model_registry(repo_root: str | Path) -> dict[str, object]:
    """Audit metadata completeness and every declared local source path."""

    root = Path(repo_root)
    missing_sources: list[str] = []
    published_without_local_source: list[str] = []
    incomplete: list[str] = []
    unlabeled_heuristics: list[str] = []
    for model_id, entry in MODEL_REGISTRY.items():
        if not all((
            entry.quantity,
            entry.relation,
            entry.source,
            entry.validity,
            entry.status,
            entry.verification,
            entry.validation_level,
        )):
            incomplete.append(model_id)
        if entry.local_source and not (root / entry.local_source).exists():
            missing_sources.append(model_id)
        if "published" in entry.status and not entry.local_source:
            published_without_local_source.append(model_id)
        if "repository" in entry.source.lower() and not any(
            label in entry.status for label in ("heuristic", "policy", "schedule")
        ):
            unlabeled_heuristics.append(model_id)
    return {
        "entry_count": len(MODEL_REGISTRY),
        "subsystems": sorted({entry.subsystem for entry in MODEL_REGISTRY.values()}),
        "missing_local_sources": sorted(missing_sources),
        "published_without_local_source": sorted(published_without_local_source),
        "incomplete_entries": sorted(incomplete),
        "unlabeled_repository_heuristics": sorted(unlabeled_heuristics),
        "passed": not (
            missing_sources
            or published_without_local_source
            or incomplete
            or unlabeled_heuristics
        ),
    }
