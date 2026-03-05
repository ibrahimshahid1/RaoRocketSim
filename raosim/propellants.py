"""
propellants.py – Propellant combustion-product database.

Each entry stores the thermodynamic properties of the *exhaust gas*
(combustion products) at the nominal O/F ratio, not the raw propellants.

Fields
------
gamma   : ratio of specific heats of combustion products  (γ)
Mw      : mean molecular weight of exhaust  [kg/mol]
Tc      : adiabatic flame / chamber stagnation temperature  [K]
eta_Isp : empirical Isp efficiency multiplier (accounts for nozzle losses,
          mixture-ratio deviation, cooling, etc.)
OF      : nominal oxidiser-to-fuel mass ratio

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
    eta_Isp: float
    OF: float


    R_gas: float = field(init=False)
    c_star: float = field(init=False)

    def __post_init__(self):
        Mw_kg_per_kmol = self.Mw * 1000.0
        self.R_gas = R_UNIVERSAL / Mw_kg_per_kmol
        self.c_star = characteristic_velocity(self.gamma, self.R_gas, self.Tc)




PROPELLANT_DB: dict[str, Propellant] = {}


def _register(p: Propellant):
    PROPELLANT_DB[p.name.lower()] = p


_register(Propellant(
    name="N2O/Ethanol",
    gamma=1.22,
    Mw=0.0260,    # ~26 g/mol
    Tc=2800.0,    # K
    eta_Isp=0.92,
    OF=5.5,
))

_register(Propellant(
    name="LOX/RP-1",
    gamma=1.23,
    Mw=0.0235,    # ~23.5 g/mol
    Tc=3400.0,
    eta_Isp=0.96,
    OF=2.6,
))

_register(Propellant(
    name="LOX/LCH4",
    gamma=1.24,
    Mw=0.0220,    # ~22 g/mol
    Tc=3500.0,
    eta_Isp=0.96,
    OF=3.5,
))

_register(Propellant(
    name="LOX/LH2",
    gamma=1.20,
    Mw=0.0100,    # ~10 g/mol  (H₂O-rich exhaust)
    Tc=3250.0,
    eta_Isp=0.98,
    OF=6.0,
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
                      eta_Isp: float = 0.95, OF: float = 0.0) -> Propellant:
    """Create a custom propellant from user-supplied thermodynamic data."""
    return Propellant(name="Custom", gamma=gamma, Mw=Mw, Tc=Tc,
                      eta_Isp=eta_Isp, OF=OF)


def list_propellants() -> list[str]:
    return [v.name for v in PROPELLANT_DB.values()]
