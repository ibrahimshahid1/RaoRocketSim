"""
gas_dynamics.py – Quasi-1-D isentropic compressible-flow relations.

All functions assume calorically perfect ideal gas with constant γ.
References: Anderson, *Modern Compressible Flow*; the provided Rao-nozzle
formulation document (isentropic relations, area-Mach, Prandtl-Meyer).
"""

from __future__ import annotations
import math



def isentropic_temperature_ratio(M: float, gamma: float) -> float:
    """T / T₀  =  (1 + (γ-1)/2 · M²)⁻¹"""
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-1.0)


def isentropic_pressure_ratio(M: float, gamma: float) -> float:
    """p / p₀  =  (T/T₀)^(γ/(γ-1))"""
    return isentropic_temperature_ratio(M, gamma) ** (gamma / (gamma - 1.0))


def isentropic_density_ratio(M: float, gamma: float) -> float:
    """ρ / ρ₀  =  (T/T₀)^(1/(γ-1))"""
    return isentropic_temperature_ratio(M, gamma) ** (1.0 / (gamma - 1.0))




def area_mach_relation(M: float, gamma: float) -> float:
    """
    A/A* for a given Mach number and γ.

        A/A* = (1/M) · [ (2/(γ+1)) · (1 + (γ-1)/2 · M²) ]^((γ+1)/(2(γ-1)))
    """
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return (1.0 / M) * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * M * M)) ** (gp1 / (2.0 * gm1))


def mach_from_area_ratio(area_ratio: float, gamma: float,
                         supersonic: bool = True) -> float:
    """
    Invert the area-Mach relation for the supersonic (default) or subsonic
    branch using Newton-Raphson iteration.

    Parameters
    ----------
    area_ratio : A/A*  (must be >= 1)
    gamma      : ratio of specific heats
    supersonic : if True return the supersonic root, else the subsonic root

    Returns
    -------
    Mach number
    """
    if area_ratio < 1.0:
        raise ValueError("area_ratio must be >= 1.0")

    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    exp = gp1 / (2.0 * gm1)


    def _f(M):
        return area_mach_relation(M, gamma) - area_ratio


    def _df(M):
        bracket = 1.0 + 0.5 * gm1 * M * M
        A_over_Astar = area_mach_relation(M, gamma)
        return A_over_Astar * (-1.0 / M + exp * gm1 * M / bracket)

    if supersonic:
        M = 1.0 + 0.5 * math.log(area_ratio)   # rough guess
        if M < 1.01:
            M = 1.01
    else:
        M = 0.5

    for _ in range(100):
        fval = _f(M)
        dfval = _df(M)
        if abs(dfval) < 1e-30:
            break
        dM = -fval / dfval
        M += dM
        if M < 1e-6:
            M = 1e-6
        if abs(dM) < 1e-12:
            break

    return M




def exit_mach_from_pressure_ratio(Pc: float, Pe: float,
                                  gamma: float) -> float:
    """
    Compute exit Mach number from chamber pressure Pc and exit pressure Pe,
    assuming isentropic expansion.

        Pe/Pc = (1 + (γ-1)/2 · Me²)^(-γ/(γ-1))

    Solved for Me (supersonic root).
    """
    ratio = Pe / Pc  # < 1
    gm1 = gamma - 1.0
    # Me² = (2/(γ-1)) · [ (Pe/Pc)^(-(γ-1)/γ)  - 1 ]
    Me_sq = (2.0 / gm1) * (ratio ** (-gm1 / gamma) - 1.0)
    if Me_sq < 0:
        raise ValueError("Pressure ratio yields subsonic exit (Pe > Pc critical)")
    return math.sqrt(Me_sq)


def expansion_ratio_from_pressure(Pc: float, Pa: float,
                                  gamma: float) -> tuple[float, float]:
    """
    Return (ε, Me) for matched-expansion (Pe = Pa) given Pc and Pa.
    """
    Me = exit_mach_from_pressure_ratio(Pc, Pa, gamma)
    eps = area_mach_relation(Me, gamma)
    return eps, Me




def thrust_coefficient(Me: float, gamma: float,
                       Pe_over_Pc: float, Pa_over_Pc: float,
                       epsilon: float) -> float:
    """
    Ideal 1-D thrust coefficient.

        Cf = sqrt{ 2γ²/(γ-1) · (2/(γ+1))^((γ+1)/(γ-1)) · [1 - (Pe/Pc)^((γ-1)/γ)] }
             + (Pe/Pc - Pa/Pc) · ε
    """
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    momentum = math.sqrt(
        (2.0 * gamma * gamma / gm1)
        * (2.0 / gp1) ** (gp1 / gm1)
        * (1.0 - Pe_over_Pc ** (gm1 / gamma))
    )
    pressure = (Pe_over_Pc - Pa_over_Pc) * epsilon
    return momentum + pressure




def prandtl_meyer(M: float, gamma: float) -> float:
    """
    Prandtl-Meyer function ν(M) in **radians**.

        ν(M) = sqrt((γ+1)/(γ-1)) · arctan(sqrt((γ-1)/(γ+1)·(M²-1)))
               - arctan(sqrt(M²-1))
    """
    if M < 1.0:
        raise ValueError("Prandtl-Meyer function is undefined for M < 1")
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    q = math.sqrt(gp1 / gm1)
    msq = M * M - 1.0
    return q * math.atan(math.sqrt(gm1 / gp1 * msq)) - math.atan(math.sqrt(msq))


def mach_angle(M: float) -> float:
    """Mach angle μ = arcsin(1/M) in radians. Requires M ≥ 1."""
    if M < 1.0:
        raise ValueError("Mach angle undefined for M < 1")
    return math.asin(1.0 / M)


def mach_from_prandtl_meyer(nu: float, gamma: float,
                             tol: float = 1e-10,
                             max_iter: int = 50) -> float:
    """
    Invert the Prandtl-Meyer function: given ν [radians], find M.

    Uses Newton-Raphson with the analytical derivative:

        dν/dM = √(M²−1) / (M · (1 + (γ−1)/2 · M²))

    Parameters
    ----------
    nu    : Prandtl-Meyer angle [radians], must be ≥ 0
    gamma : ratio of specific heats

    Returns
    -------
    Mach number M ≥ 1
    """
    if nu < 0.0:
        raise ValueError("ν must be ≥ 0")
    if nu == 0.0:
        return 1.0

    gm1 = gamma - 1.0
    M = 1.0 + nu  # initial guess (linearization near M=1)
    if M < 1.01:
        M = 1.01

    M_max = 100.0

    for _ in range(max_iter):
        if M > M_max:
            M = M_max
        nu_current = prandtl_meyer(M, gamma)
        msq_m1 = M * M - 1.0
        if msq_m1 < 1e-30:
            M += 0.01
            continue
        denom = M * (1.0 + 0.5 * gm1 * M * M)
        if denom < 1e-30 or not math.isfinite(denom):
            break
        dnu_dM = math.sqrt(msq_m1) / denom
        if dnu_dM < 1e-30 or not math.isfinite(dnu_dM):
            break
        dM = (nu - nu_current) / dnu_dM
        if not math.isfinite(dM):
            break
        M += dM
        if M < 1.0:
            M = 1.0 + 1e-6
        if abs(dM) < tol:
            break

    return min(M, M_max)



def characteristic_velocity(gamma: float, R_gas: float, Tc: float) -> float:
    """
    Ideal characteristic velocity c* [m/s].

        c* = sqrt(γ · R · Tc) / { γ · sqrt( [2/(γ+1)]^((γ+1)/(γ-1)) ) }

    Equivalently:

        c* = (1/γ) · sqrt( γ·R·Tc · [(γ+1)/2]^((γ+1)/(γ-1)) )
    """
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return math.sqrt(gamma * R_gas * Tc) / (
        gamma * math.sqrt((2.0 / gp1) ** (gp1 / gm1))
    )
