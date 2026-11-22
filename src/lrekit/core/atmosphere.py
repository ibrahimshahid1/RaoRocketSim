"""
atmosphere.py – Simplified ISA (International Standard Atmosphere) model.

Provides temperature, pressure, and density as a function of geometric
altitude up to ~86 km.  Uses the standard 7-layer piecewise-linear
temperature model for the first three key layers.
"""

from __future__ import annotations
import math

# ISA sea-level constants
T0_ISA = 288.15    # K
P0_ISA = 101325.0  # Pa
RHO0_ISA = 1.225   # kg/m³
R_AIR = 287.05     # J/(kg·K)
g0 = 9.80665       # m/s²


def isa(h: float) -> tuple[float, float, float]:
    """
    Return (T [K], p [Pa], ρ [kg/m³]) at geometric altitude h [m]
    using a simplified 3-segment ISA.

    Layers
    ------
    0–11 000 m  : troposphere   (lapse −6.5 K/km)
    11–20 000 m : tropopause    (isothermal 216.65 K)
    20–47 000 m : stratosphere  (lapse +1.0 K/km)
    >47 000 m   : thin-air exponential tail (rough approximation)
    """
    if h < 0:
        h = 0.0

    if h <= 11000:
        T = T0_ISA - 0.0065 * h
        p = P0_ISA * (T / T0_ISA) ** (g0 / (0.0065 * R_AIR))
    elif h <= 20000:
        # base values at 11 km
        T11 = 216.65
        p11 = P0_ISA * (T11 / T0_ISA) ** (g0 / (0.0065 * R_AIR))
        T = T11
        p = p11 * math.exp(-g0 * (h - 11000) / (R_AIR * T))
    elif h <= 47000:
        # base values at 20 km
        T11 = 216.65
        p11 = P0_ISA * (T11 / T0_ISA) ** (g0 / (0.0065 * R_AIR))
        p20 = p11 * math.exp(-g0 * 9000 / (R_AIR * 216.65))
        T20 = 216.65
        lapse = 0.001   # K/m
        T = T20 + lapse * (h - 20000)
        p = p20 * (T / T20) ** (-g0 / (lapse * R_AIR))
    else:
        # exponential decay (very rough above 47 km)
        T = 270.65       # approximate
        T11 = 216.65
        p11 = P0_ISA * (T11 / T0_ISA) ** (g0 / (0.0065 * R_AIR))
        p20 = p11 * math.exp(-g0 * 9000 / (R_AIR * 216.65))
        T20 = 216.65
        p47 = p20 * ((T20 + 0.001 * 27000) / T20) ** (-g0 / (0.001 * R_AIR))
        p = p47 * math.exp(-g0 * (h - 47000) / (R_AIR * T))

    rho = p / (R_AIR * T)
    return T, p, rho


def density(h: float) -> float:
    """Air density [kg/m³] at altitude h [m]."""
    return isa(h)[2]


def pressure(h: float) -> float:
    """Atmospheric pressure [Pa] at altitude h [m]."""
    return isa(h)[1]


def temperature(h: float) -> float:
    """Air temperature [K] at altitude h [m]."""
    return isa(h)[0]
