"""Canonical LREKit physical quantities, symbols, and units.

This is a *normalisation* table, not scientific authority.  It records how the
same physical quantity is written across the propulsion literature and across
LREKit source code so retrieval can find it, and it records dimensionality so
two quantities are never merged merely because they share a glyph.

Every entry carries ``provenance="SYSTEM_INFERS"`` semantics: the canonical
name and the alias list are this system's terminology decision.  A quantity's
*value*, *validity range*, or *correlation* is never stored here - those live
in equations/claims with document-level provenance.

Ambiguity is preserved.  ``resolve_symbol("k")`` returns several candidates;
callers must disambiguate with subsystem/document context or report
``AMBIGUOUS_SYMBOL``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Quantity:
    qid: str
    canonical: str
    dimension: str           # e.g. "M L T^-2"; "1" for dimensionless
    si_unit: str
    symbols: tuple = ()      # symbols used in literature
    code_symbols: tuple = () # names used in LREKit source
    aliases: tuple = ()
    subsystems: tuple = ()
    note: str = ""


def _q(qid, canonical, dimension, si_unit, symbols=(), code_symbols=(), aliases=(), subsystems=(), note=""):
    return Quantity(qid, canonical, dimension, si_unit, tuple(symbols), tuple(code_symbols),
                    tuple(aliases), tuple(subsystems), note)


QUANTITIES: tuple[Quantity, ...] = (
    # ---------------- performance ----------------
    _q("QTY_THRUST", "thrust", "M L T^-2", "N", ("F", "F_t", "T"), ("thrust", "thrust_n", "F"),
       ("engine thrust",), ("performance",),
       note="Symbol T also denotes temperature; never merge on the glyph alone."),
    _q("QTY_ISP", "specific impulse", "T", "s", ("Isp", "I_sp", "I_s"), ("isp", "isp_s", "I_sp"),
       ("specific impulse (mass basis)",), ("performance",),
       note="Reported in seconds here (I = F/(mdot*g0)); some sources report N.s/kg."),
    _q("QTY_MDOT", "total mass flow rate", "M T^-1", "kg/s", ("ṁ", "m_dot", "w"), ("mdot", "m_dot", "mdot_total"),
       ("propellant flow rate",), ("performance", "feed_system")),
    _q("QTY_MDOT_OX", "oxidizer mass flow rate", "M T^-1", "kg/s", ("ṁ_o", "w_o"), ("mdot_ox", "mdot_o"),
       (), ("performance", "injector")),
    _q("QTY_MDOT_FUEL", "fuel mass flow rate", "M T^-1", "kg/s", ("ṁ_f", "w_f"), ("mdot_fuel", "mdot_f"),
       (), ("performance", "injector", "cooling")),
    _q("QTY_MR", "mixture ratio", "1", "-", ("O/F", "r", "MR"), ("of_ratio", "mixture_ratio", "OF"),
       ("oxidizer-to-fuel ratio",), ("performance", "thermochemistry")),
    _q("QTY_CSTAR", "characteristic velocity", "L T^-1", "m/s", ("c*", "c_star"), ("cstar", "c_star"),
       (), ("performance", "thermochemistry")),
    _q("QTY_CF", "thrust coefficient", "1", "-", ("C_F", "CF"), ("cf", "thrust_coefficient"),
       (), ("performance", "nozzle_contour"),
       note="Distinct from discharge coefficient C_d and from skin-friction coefficient C_f."),
    _q("QTY_ETA_CSTAR", "c-star efficiency", "1", "-", ("η_c*",), ("eta_cstar", "eta_c_star"),
       ("combustion efficiency",), ("performance", "injector")),
    _q("QTY_ETA_CF", "thrust-coefficient efficiency", "1", "-", ("η_CF",), ("eta_cf",),
       ("nozzle efficiency",), ("performance", "nozzle_contour")),

    # ---------------- pressures / temperatures ----------------
    _q("QTY_PC", "chamber pressure", "M L^-1 T^-2", "Pa", ("Pc", "p_c", "P_c", "p_0"), ("pc", "p_c", "chamber_pressure"),
       ("combustion chamber stagnation pressure",), ("performance", "chamber"),
       note="Several sources use p_c for static chamber pressure and p_0 for stagnation; check the source."),
    _q("QTY_PA", "ambient pressure", "M L^-1 T^-2", "Pa", ("p_a", "P_amb", "p_∞"), ("pa", "p_ambient", "p_amb"),
       (), ("performance", "altitude" if False else "performance")),
    _q("QTY_PE", "nozzle exit pressure", "M L^-1 T^-2", "Pa", ("p_e", "P_e"), ("pe", "p_exit", "p_e"),
       (), ("nozzle_contour", "performance")),
    _q("QTY_P0", "stagnation pressure", "M L^-1 T^-2", "Pa", ("p_0", "P_t", "p_t"), ("p0", "p_stag"),
       ("total pressure",), ("gas_dynamics",)),
    _q("QTY_TC", "chamber temperature", "Θ", "K", ("T_c", "T_0"), ("tc", "t_c", "chamber_temperature"),
       ("combustion temperature",), ("thermochemistry", "chamber")),
    _q("QTY_TWG", "gas-side wall temperature", "Θ", "K", ("T_wg", "T_w,g"), ("t_wg", "twg", "wall_temp_gas"),
       (), ("cooling", "heat_transfer", "structures")),
    _q("QTY_TWC", "coolant-side wall temperature", "Θ", "K", ("T_wc", "T_w,c"), ("t_wc", "twc"),
       (), ("cooling", "heat_transfer")),
    _q("QTY_TCOOL", "bulk coolant temperature", "Θ", "K", ("T_b", "T_co"), ("t_coolant", "t_bulk"),
       (), ("cooling",)),
    _q("QTY_TAW", "adiabatic wall temperature", "Θ", "K", ("T_aw",), ("t_aw", "taw"),
       ("recovery temperature",), ("heat_transfer",)),

    # ---------------- heat transfer ----------------
    _q("QTY_QFLUX", "heat flux", "M T^-3", "W/m^2", ("q", "q''", "q̇"), ("q", "heat_flux", "q_flux"),
       (), ("cooling", "heat_transfer")),
    _q("QTY_HG", "gas-side heat-transfer coefficient", "M T^-3 Θ^-1", "W/m^2/K", ("h_g",), ("h_g", "hg"),
       (), ("heat_transfer",)),
    _q("QTY_HC", "coolant-side heat-transfer coefficient", "M T^-3 Θ^-1", "W/m^2/K", ("h_c", "h_co"), ("h_c", "hc"),
       (), ("cooling",)),
    _q("QTY_NU", "Nusselt number", "1", "-", ("Nu",), ("nu", "nusselt"), (), ("heat_transfer",)),
    _q("QTY_PR", "Prandtl number", "1", "-", ("Pr",), ("pr", "prandtl"), (), ("heat_transfer",)),
    _q("QTY_RE", "Reynolds number", "1", "-", ("Re",), ("re", "reynolds"), (), ("heat_transfer", "gas_dynamics")),
    _q("QTY_CHF", "critical heat flux", "M T^-3", "W/m^2", ("q_CHF",), ("chf", "q_chf"),
       ("burnout heat flux",), ("cooling",)),
    _q("QTY_FIN_EFF", "fin efficiency", "1", "-", ("η_f",), ("fin_efficiency", "eta_fin"), (), ("cooling",)),

    # ---------------- geometry ----------------
    _q("QTY_EPS", "expansion ratio", "1", "-", ("ε", "A_e/A_t", "AR"), ("eps", "epsilon", "area_ratio", "expansion_ratio"),
       ("area ratio",), ("nozzle_contour",)),
    _q("QTY_CR", "contraction ratio", "1", "-", ("ε_c", "A_c/A_t"), ("contraction_ratio", "cr"),
       (), ("chamber",)),
    _q("QTY_AT", "throat area", "L^2", "m^2", ("A_t", "A*"), ("a_t", "at", "throat_area"), (), ("throat",)),
    _q("QTY_RT", "throat radius", "L", "m", ("r_t", "R_t"), ("r_t", "rt", "throat_radius"), (), ("throat",)),
    _q("QTY_AE", "nozzle exit area", "L^2", "m^2", ("A_e",), ("a_e", "exit_area"), (), ("nozzle_contour",)),
    _q("QTY_RE_EXIT", "nozzle exit radius", "L", "m", ("r_e", "R_e"), ("r_e", "exit_radius"), (), ("nozzle_contour",),
       note="Do not confuse the symbol R_e (exit radius) with Re (Reynolds number)."),
    _q("QTY_LSTAR", "characteristic chamber length", "L", "m", ("L*",), ("l_star", "lstar"),
       ("L-star",), ("chamber",)),
    _q("QTY_TW", "wall thickness", "L", "m", ("t_w", "δ"), ("t_wall", "wall_thickness"), (), ("structures", "cooling")),
    _q("QTY_THETA_N", "initial wall angle at nozzle attachment", "1", "rad", ("θ_n", "θ_i"), ("theta_n", "theta_i"),
       (), ("nozzle_contour",)),
    _q("QTY_THETA_E", "nozzle exit wall angle", "1", "rad", ("θ_e",), ("theta_e",), (), ("nozzle_contour",)),
    _q("QTY_THETA_B", "flow angle at the control-surface attachment point B", "1", "rad", ("θ_B",), ("theta_b",),
       (), ("nozzle_contour",), note="Rao variational construction; see the Rao control-surface literature."),

    # ---------------- gas dynamics ----------------
    _q("QTY_MACH", "Mach number", "1", "-", ("M", "Ma"), ("mach", "m_number"), (), ("gas_dynamics",),
       note="Symbol M also denotes molar mass and moment; disambiguate by dimension."),
    _q("QTY_GAMMA", "specific-heat ratio", "1", "-", ("γ", "k"), ("gamma", "g"), (), ("gas_dynamics", "thermochemistry"),
       note="Sources using k for gamma conflict with k for thermal conductivity. Never merge on the glyph."),
    _q("QTY_SOUND", "speed of sound", "L T^-1", "m/s", ("a", "c"), ("a_sound", "sound_speed"), (), ("gas_dynamics",)),
    _q("QTY_RGAS", "specific gas constant", "L^2 T^-2 Θ^-1", "J/kg/K", ("R", "R_s"), ("r_gas", "rgas"),
       (), ("thermochemistry",), note="Distinct from the universal gas constant R_u and from radius R."),
    _q("QTY_MU", "dynamic viscosity", "M L^-1 T^-1", "Pa.s", ("μ",), ("mu", "viscosity"), (), ("heat_transfer",)),
    _q("QTY_RHO", "density", "M L^-3", "kg/m^3", ("ρ",), ("rho", "density"), (), ("gas_dynamics", "cooling")),
    _q("QTY_KCOND", "thermal conductivity", "M L T^-3 Θ^-1", "W/m/K", ("k", "λ"), ("k_cond", "conductivity", "k_w"),
       (), ("heat_transfer", "materials"),
       note="Symbol k collides with the specific-heat ratio in older propulsion texts."),
    _q("QTY_CP", "specific heat at constant pressure", "L^2 T^-2 Θ^-1", "J/kg/K", ("c_p",), ("cp", "c_p"),
       (), ("thermochemistry", "cooling")),

    # ---------------- hydraulics / injector ----------------
    _q("QTY_CD", "discharge coefficient", "1", "-", ("C_d", "Cd"), ("cd", "c_d", "discharge_coefficient"),
       (), ("injector", "feed_system"),
       note="Geometry- and Reynolds-dependent; a literature value for one orifice geometry is not transferable."),
    _q("QTY_DP", "pressure drop", "M L^-1 T^-2", "Pa", ("Δp", "ΔP", "dP"), ("dp", "delta_p", "pressure_drop"),
       (), ("injector", "cooling", "feed_system")),
    _q("QTY_TMR", "total momentum ratio", "1", "-", ("TMR",), ("tmr", "momentum_ratio"),
       ("momentum ratio",), ("injector",)),
    _q("QTY_SKIP", "pintle skip distance", "L", "m", ("L_s",), ("skip_distance",), (), ("injector",)),
    _q("QTY_BF", "pintle blockage factor", "1", "-", ("BF",), ("blockage_factor", "bf"), (), ("injector",)),
    _q("QTY_SMD", "Sauter mean diameter", "L", "m", ("SMD", "D_32"), ("smd", "d32"), (), ("spray",)),

    # ---------------- pumps / feed ----------------
    _q("QTY_HEAD", "pump head rise", "L", "m", ("H",), ("head", "head_m"), (), ("pump",)),
    _q("QTY_DP_PUMP", "pump pressure rise", "M L^-1 T^-2", "Pa", ("Δp_pump",), ("delta_p_pump", "pump_dp"),
       (), ("pump", "feed_system")),
    _q("QTY_PHI", "flow coefficient", "1", "-", ("φ",), ("phi", "flow_coefficient"), (), ("pump",)),
    _q("QTY_PSI", "head coefficient", "1", "-", ("ψ",), ("psi", "head_coefficient"), (), ("pump",)),
    _q("QTY_NS", "specific speed", "1", "-", ("N_s", "Ns", "ω_s"), ("ns", "specific_speed"), (), ("pump",),
       note="Dimensional in US practice (rpm.gpm^0.5/ft^0.75) and dimensionless in SI practice; the two differ by a fixed factor. Record which convention the source uses."),
    _q("QTY_NPSH", "net positive suction head", "L", "m", ("NPSH", "NPSHr", "NPSHa"), ("npsh", "npsh_r", "npsh_a"),
       (), ("pump",), note="Required (NPSHr) and available (NPSHa) are different quantities."),
    _q("QTY_SIGMA_CAV", "cavitation number", "1", "-", ("σ",), ("cavitation_number", "sigma_cav"), (), ("pump",),
       note="Symbol sigma also denotes stress."),
    _q("QTY_OMEGA", "rotational speed", "T^-1", "rad/s", ("ω", "N"), ("omega", "rpm", "shaft_speed"), (), ("pump",),
       note="N is rpm in pump practice and a count elsewhere; record units explicitly."),
    _q("QTY_UTIP", "impeller tip speed", "L T^-1", "m/s", ("U_2", "u_t"), ("u_tip", "tip_speed"), (), ("pump",)),
    _q("QTY_ETA_PUMP", "pump efficiency", "1", "-", ("η_p",), ("eta_pump",), (), ("pump",)),
    _q("QTY_PSHAFT", "shaft power", "M L^2 T^-3", "W", ("P_s",), ("shaft_power",), (), ("pump", "electric_drive")),
    _q("QTY_PMOTOR", "motor power", "M L^2 T^-3", "W", ("P_m",), ("motor_power",), (), ("electric_drive",)),
    _q("QTY_EBATT", "battery energy", "M L^2 T^-2", "J", ("E_b",), ("battery_energy",), (), ("electric_drive",)),

    # ---------------- structures / materials ----------------
    _q("QTY_SIGMA", "stress", "M L^-1 T^-2", "Pa", ("σ",), ("sigma", "stress"), (), ("structures",)),
    _q("QTY_STRAIN", "strain", "1", "-", ("ε",), ("strain",), (), ("structures",),
       note="Symbol epsilon also denotes expansion ratio and emissivity."),
    _q("QTY_FOS", "factor of safety", "1", "-", ("FS", "FoS"), ("fos", "factor_of_safety", "safety_factor"),
       (), ("structures",)),
    _q("QTY_BUCKLE", "buckling margin", "1", "-", ("MS_b",), ("buckling_margin",), (), ("structures",)),
    _q("QTY_NFATIGUE", "cycles to failure", "1", "cycles", ("N_f",), ("n_cycles", "fatigue_life"), (), ("structures",)),

    # ---------------- mass ----------------
    _q("QTY_MASS_COMP", "component mass", "M", "kg", ("m",), ("mass_kg", "component_mass"), (), ("mass",)),
    _q("QTY_MASS_DRY", "engine dry mass", "M", "kg", ("m_dry",), ("dry_mass", "engine_dry_mass"), (), ("mass",)),
)

BY_ID = {q.qid: q for q in QUANTITIES}


def _symbol_index() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for q in QUANTITIES:
        for s in list(q.symbols) + list(q.code_symbols) + list(q.aliases) + [q.canonical]:
            idx.setdefault(s.strip().lower(), []).append(q.qid)
    return idx


SYMBOL_INDEX = _symbol_index()

# Glyphs that are provably overloaded in this corpus.  Callers MUST supply
# document/subsystem context before treating a match as an identification.
KNOWN_AMBIGUOUS = {
    "k": ["QTY_GAMMA", "QTY_KCOND"],
    "r": ["QTY_RGAS", "QTY_MR"],
    "t": ["QTY_THRUST", "QTY_TC"],
    "m": ["QTY_MACH", "QTY_MASS_COMP"],
    "n": ["QTY_OMEGA", "QTY_NFATIGUE"],
    "a": ["QTY_SOUND", "QTY_AT"],
    "c": ["QTY_SOUND", "QTY_CSTAR"],
    "ε": ["QTY_EPS", "QTY_STRAIN"],
    "σ": ["QTY_SIGMA", "QTY_SIGMA_CAV"],
    "φ": ["QTY_PHI"],
}


def resolve_symbol(symbol: str, subsystem: str | None = None) -> dict:
    """Resolve a symbol to candidate quantities.

    Returns ``{"state": ..., "candidates": [...]}``.  ``state`` is
    ``"RESOLVED"``, ``"AMBIGUOUS_SYMBOL"`` or ``"UNRESOLVED_ENTITY"``.  This
    function never silently picks one candidate.
    """
    key = symbol.strip().lower()
    cands = list(dict.fromkeys(SYMBOL_INDEX.get(key, [])))
    if key in KNOWN_AMBIGUOUS:
        cands = list(dict.fromkeys(cands + KNOWN_AMBIGUOUS[key]))
    if subsystem:
        narrowed = [c for c in cands if subsystem in BY_ID[c].subsystems]
        if narrowed:
            cands = narrowed
    if not cands:
        return {"state": "UNRESOLVED_ENTITY", "symbol": symbol, "candidates": []}
    state = "RESOLVED" if len(cands) == 1 else "AMBIGUOUS_SYMBOL"
    return {
        "state": state,
        "symbol": symbol,
        "candidates": [
            {"qid": c, "canonical": BY_ID[c].canonical, "si_unit": BY_ID[c].si_unit,
             "dimension": BY_ID[c].dimension, "note": BY_ID[c].note}
            for c in cands
        ],
    }


def to_records() -> list[dict]:
    return [
        {
            "qid": q.qid, "type": "PhysicalQuantity", "canonical": q.canonical,
            "dimension": q.dimension, "si_unit": q.si_unit,
            "symbols": list(q.symbols), "code_symbols": list(q.code_symbols),
            "aliases": list(q.aliases), "subsystems": list(q.subsystems),
            "note": q.note, "provenance_kind": "SYSTEM_INFERS",
            "provenance_note": "Canonical naming and alias grouping are a knowledge-system terminology decision, not a source claim.",
        }
        for q in QUANTITIES
    ]
