"""
propellants.py – Propellant combustion-product database.

Each entry stores the thermodynamic properties of the *exhaust gas*
(combustion products) at the nominal O/F ratio, not the raw propellants.

Fields
------
gamma    : ratio of specific heats of combustion products  (γ)
Mw       : mean molecular weight of exhaust  [kg/mol]
Tc       : adiabatic flame / chamber stagnation temperature  [K]
eta_cstar: characteristic-velocity (combustion) efficiency  η_c*
eta_CF   : thrust-coefficient (nozzle) efficiency  η_CF
eta_Isp  : overall specific-impulse efficiency  η_Isp = η_c*·η_CF
OF       : nominal oxidiser-to-fuel mass ratio
source   : provenance string for the thermochemical constants

Efficiency convention
---------------------
Combustion efficiency and nozzle efficiency act on physically distinct
quantities and must not be lumped together (see Sutton & Biblarz,
*Rocket Propulsion Elements*, §3 / §5):

    η_c*  lowers the delivered c*  → mass flow Pc·A_t/c* rises (more
          propellant is needed to hold the same chamber pressure when the
          combustion is incomplete);
    η_CF  lowers the delivered thrust coefficient (divergence, friction,
          two-phase, kinetic losses);
    η_Isp = η_c*·η_CF  is the net specific-impulse multiplier.

Back-compatibility: callers that supply only ``eta_Isp`` (the historical
single lumped multiplier) get ``eta_CF = eta_Isp`` and ``eta_cstar = 1.0``,
which reproduces the previous "all loss on Cf, ideal mass flow" behaviour
bit-for-bit.  Supply ``eta_cstar``/``eta_CF`` explicitly for the honest
split, in which case ``eta_Isp`` is overridden with their product.

Derived quantities (computed at import time)
--------------------------------------------
R_gas   : specific gas constant  R_universal / Mw   [J/(kg·K)]
c_star  : ideal characteristic velocity  [m/s]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from raosim.gas_dynamics import characteristic_velocity

R_UNIVERSAL = 8314.46  # J/(kmol·K)  ← note: we store Mw in kg/mol, so R = R_u/Mw


@dataclass
class Propellant:
    name: str
    gamma: float
    Mw: float        # kg/mol  (e.g. 0.022 for ~22 g/mol)
    Tc: float        # K
    eta_Isp: float = 1.0
    OF: float = 0.0
    # Explicit split.  ``None`` means "derive from eta_Isp" (back-compat:
    # eta_CF = eta_Isp, eta_cstar = 1.0).  When both are given, eta_Isp is
    # recomputed as their product.
    eta_cstar: float | None = None
    eta_CF: float | None = None
    source: str | None = None


    R_gas: float = field(init=False)
    c_star: float = field(init=False)

    def __post_init__(self):
        # Resolve the efficiency split before anything reads it.  The four
        # cases keep every legacy `eta_Isp=`-only caller bit-identical while
        # exposing an honest η_c*·η_CF decomposition when requested.
        if self.eta_cstar is None and self.eta_CF is None:
            self.eta_cstar = 1.0
            self.eta_CF = float(self.eta_Isp)
        elif self.eta_cstar is None:
            self.eta_cstar = 1.0
            self.eta_CF = float(self.eta_CF)
            self.eta_Isp = self.eta_cstar * self.eta_CF
        elif self.eta_CF is None:
            self.eta_cstar = float(self.eta_cstar)
            self.eta_CF = 1.0
            self.eta_Isp = self.eta_cstar * self.eta_CF
        else:
            self.eta_cstar = float(self.eta_cstar)
            self.eta_CF = float(self.eta_CF)
            self.eta_Isp = self.eta_cstar * self.eta_CF

        Mw_kg_per_kmol = self.Mw * 1000.0
        self.R_gas = R_UNIVERSAL / Mw_kg_per_kmol
        self.c_star = characteristic_velocity(self.gamma, self.R_gas, self.Tc)

    @property
    def c_star_effective(self) -> float:
        """Delivered characteristic velocity, c*·η_c*  [m/s]."""
        return self.c_star * self.eta_cstar




PROPELLANT_DB: dict[str, Propellant] = {}


def _register(p: Propellant):
    PROPELLANT_DB[p.name.lower()] = p


# ---------------------------------------------------------------------------
# Constant-gamma combustion-product table.
#
# These are SCREENING-grade constants for preliminary sizing; for design use
# the CEA path (raosim/cea.py).  Chamber-product gamma / Mw / Tc are taken at
# (or near) the listed O/F from:
#   - Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed., Table 5-5
#     ("Theoretical chamber performance ... at Pc ≈ 1000 psia, optimum
#     expansion, shifting equilibrium");
#   - Huzel & Huang, *Modern Engineering for Design of Liquid-Propellant
#     Rocket Engines* (NASA SP-125), combustion-product tables.
# Efficiencies follow Sutton §3.2 / §5: combustion (c*) efficiency and nozzle
# (thrust-coefficient) efficiency are reported separately, never lumped, with
# typical liquid-engine ranges eta_c* ≈ 0.94-0.99 and eta_CF ≈ 0.97-0.99.
# ---------------------------------------------------------------------------

_register(Propellant(
    name="N2O/Ethanol",
    gamma=1.22,
    Mw=0.0264,    # ~26.4 g/mol
    Tc=2950.0,    # K  (O/F ~5.5)
    eta_cstar=0.94,
    eta_CF=0.98,
    OF=5.5,
    source="Approximate constants (N2O/ethanol not in SP-125/Sutton "
           "tables); CEA strongly recommended for design.",
))

_register(Propellant(
    name="LOX/RP-1",
    gamma=1.24,
    Mw=0.0219,    # 21.9 g/mol  (Sutton Table 5-5, O/F 2.24)
    Tc=3571.0,    # K           (Sutton Table 5-5)
    eta_cstar=0.975,
    eta_CF=0.985,
    OF=2.27,
    source="Sutton & Biblarz RPE 9th ed. Table 5-5 (LOX/RP-1, O/F 2.24, "
           "Pc 1000 psia); eta split per RPE §3.2.",
))

_register(Propellant(
    name="LOX/LCH4",
    gamma=1.20,
    Mw=0.0203,    # ~20.3 g/mol
    Tc=3533.0,    # K  (O/F ~3.5)
    eta_cstar=0.975,
    eta_CF=0.985,
    OF=3.5,
    source="LOX/CH4 chamber products (Gordon-McBride CEA / SP-125 class), "
           "O/F 3.5, Pc ~7 MPa; eta split per Sutton RPE §3.2.",
))

_register(Propellant(
    name="LOX/LH2",
    gamma=1.26,
    Mw=0.0089,    # 8.9 g/mol  (Sutton Table 5-5, O/F 4.83; H2-rich exhaust)
    Tc=2999.0,    # K          (Sutton Table 5-5)
    eta_cstar=0.99,
    eta_CF=0.99,
    OF=4.83,
    source="Sutton & Biblarz RPE 9th ed. Table 5-5 (LOX/LH2, O/F 4.83, "
           "Pc 1000 psia); eta split per RPE §3.2.",
))

_register(Propellant(
    name="N2O4/MMH",
    gamma=1.23,
    Mw=0.0215,    # 21.5 g/mol  (Sutton Table 5-5, O/F 2.17)
    Tc=3122.0,    # K           (Sutton Table 5-5)
    eta_cstar=0.97,
    eta_CF=0.985,
    OF=2.17,
    source="Sutton & Biblarz RPE 9th ed. Table 5-5 (N2O4/MMH, O/F 2.17, "
           "Pc 1000 psia); eta split per RPE §3.2.",
))


def get_propellant(name: str) -> Propellant:
    """Lookup by case-insensitive name.  Raises KeyError if not found."""
    key = name.lower().replace(" ", "").replace("_", "")

    for k, v in PROPELLANT_DB.items():
        if k.replace(" ", "").replace("/", "").replace("-", "") == key.replace("/", "").replace("-", ""):
            return v
    raise KeyError(
        f"Unknown propellant '{name}'.  Available: {list(PROPELLANT_DB.keys())}"
    )


def custom_propellant(gamma: float, Mw: float, Tc: float,
                      eta_Isp: float = 0.95, OF: float = 0.0,
                      eta_cstar: float | None = None,
                      eta_CF: float | None = None,
                      source: str | None = None) -> Propellant:
    """Create a custom propellant from user-supplied thermodynamic data.

    Supply ``eta_cstar``/``eta_CF`` for the honest combustion-vs-nozzle split;
    omit them to keep the legacy single ``eta_Isp`` multiplier (loss on Cf).
    """
    return Propellant(name="Custom", gamma=gamma, Mw=Mw, Tc=Tc,
                      eta_Isp=eta_Isp, OF=OF, eta_cstar=eta_cstar,
                      eta_CF=eta_CF, source=source)


def list_propellants() -> list[str]:
    return [v.name for v in PROPELLANT_DB.values()]
