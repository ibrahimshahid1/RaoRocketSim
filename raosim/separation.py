"""
separation.py – Flow separation prediction for overexpanded nozzles.

Implements four empirical separation criteria for full flow separation (FSS),
with forms and constants taken from the criteria survey in Östlund (2002)
(`propulsion_texts/fulltext01.pdf`, PDF p. 51–52; md mirror
`propulsion_texts_for_agents/markdown/fulltext01.md`) and the design practice
of NASA SP-8120:

  • Summerfield (1940s test base, conical nozzles) — Östlund Eq. 28:
        p_sep / Pa ≈ 0.4
  • Schilling (1962, conical + truncated-ideal test base) — Östlund Eq. 29:
        p_sep / Pa = k1 · (Pc/Pa)^k2
        k1 = 0.582, k2 = −0.195   (contoured nozzles)
        k1 = 0.541, k2 = −0.136   (conical nozzles)
  • Kalt–Badal (1965) — Schilling's form refit, Östlund p. 52:
        p_sep / Pa = (2/3) · (Pc/Pa)^(−0.2)
  • Schmucker (1970s, fully turbulent BL) — Östlund Eq. 30, "still widely
    used"; local-Mach form, evaluated at each wall station:
        p_sep / Pa = (1.88 · M_i − 1)^(−0.64)

Design margin: NASA SP-8120 (`19770009165.pdf`) recommends reducing the
expansion ratio when the exit wall pressure is within 20 percent of the
separation pressure; ``check_separation`` therefore reports a design-margin
verdict at ``Pe ≥ design_margin · p_sep`` (default 1.2) alongside the raw
separated/attached prediction.

History note (2026-07-22): before this date the module carried the Schmucker
and Kalt–Badal *labels* on interchanged/altered forms ("Schmucker" =
(Pa/Pc)^0.8 / M, "Kalt–Badal" = 1/(1.88·M − 1)).  Both were off their
literature namesakes (~1.75× at Me = 3); see
docs/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md §A.2.1.  The JAX mirror
(`raosim/jax/thermal.py:schmucker_separation_margin`) is kept in lock-step —
change both together (parity test: tests/test_jax_thermal_design_opt.py).

References
----------
- J. Östlund, "Flow Processes in Rocket Engine Nozzles with Focus on Flow
  Separation and Side-Loads", KTH thesis (2002).  [in propulsion_texts]
- NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976).  [in propulsion_texts]
- R. Stark, "Flow Separation in Rocket Nozzles – An Overview" (2009).
  [secondary; not in the local corpus]
"""

from __future__ import annotations
import math
import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
)

#: SP-8120 design rule: exit pressure should clear the separation pressure by
#: at least 20 % (reduce ε when Pe is within 20 % of p_sep).
SP8120_DESIGN_MARGIN = 1.2


def summerfield_separation_pressure(Pa: float) -> float:
    """
    Summerfield criterion (Östlund 2002, Eq. 28; conical-nozzle test base):

        p_sep ≈ 0.4 · Pa

    Returns the wall pressure [Pa] at which separation is expected.
    """
    return 0.4 * Pa


def schilling_separation_ratio(Pc_over_Pa: float, *, contoured: bool = True) -> float:
    """
    Schilling (1962) criterion (Östlund 2002, Eq. 29).  Returns p_sep / Pa:

        p_sep/Pa = k1 · (Pc/Pa)^k2

    with (k1, k2) = (0.582, −0.195) for contoured nozzles and
    (0.541, −0.136) for conical nozzles.  NASA adopted a similar correlation
    for truncated contoured nozzles in the mid-1970s (Östlund p. 52).
    """
    if Pc_over_Pa <= 1.0:
        return float("inf")  # not an overexpanded supersonic nozzle regime
    k1, k2 = (0.582, -0.195) if contoured else (0.541, -0.136)
    return k1 * Pc_over_Pa ** k2


def kalt_badal_separation_ratio(Pc_over_Pa: float) -> float:
    """
    Kalt–Badal (1965) criterion (Östlund 2002, p. 52): Schilling's form with
    k1 = 2/3, k2 = −0.2.  Returns p_sep / Pa:

        p_sep/Pa = (2/3) · (Pc/Pa)^(−0.2)

    Pressure-ratio rule — independent of local Mach number.

    .. note:: Signature changed 2026-07-22 (was ``(Me, gamma)`` carrying a
       mislabeled Mach-form; see module docstring).
    """
    if Pc_over_Pa <= 1.0:
        return float("inf")
    return (2.0 / 3.0) * Pc_over_Pa ** (-0.2)


def schmucker_separation_ratio(Me: float, Pa_over_Pc: float) -> float:
    """
    Schmucker criterion (Östlund 2002, Eq. 30; fully turbulent BL):

        p_sep/Pa = (1.88 · Me − 1)^(−0.64)

    evaluated with the *local* inviscid wall Mach number.  Returns p_sep / Pc
    (i.e. the ambient-referenced ratio rescaled by ``Pa_over_Pc`` so callers
    can compare directly against p_wall/Pc).

    Validity: supersonic, fully turbulent boundary layers (test base
    Me ≳ 1.5); the expression is finite for Me > 0.532.
    """
    denom = 1.88 * Me - 1.0
    if denom <= 0.0:
        return float("inf")  # criterion inapplicable (deep subsonic)
    return denom ** (-0.64) * Pa_over_Pc


_METHODS = ("summerfield", "kalt_badal", "schmucker", "schilling")


def check_separation(
    contour: dict,
    Pc: float,
    Pa: float,
    gamma: float,
    method: str = 'schmucker',
    *,
    frozen_expansion=None,
    design_margin: float = SP8120_DESIGN_MARGIN,
) -> dict:
    """
    Check whether the nozzle will experience flow separation at the given
    ambient pressure.

    Parameters
    ----------
    contour : dict from ``bell_nozzle_contour``
    Pc      : chamber pressure  [Pa]
    Pa      : ambient pressure  [Pa]
    gamma   : ratio of specific heats
    method  : 'summerfield', 'kalt_badal', 'schmucker', or 'schilling'
    design_margin : SP-8120-style design factor; the design verdict requires
        ``Pe ≥ design_margin · p_sep`` (default 1.2 = the SP-8120 "within
        20 percent" rule).  Does not affect the physical separated/attached
        prediction.

    Returns
    -------
    dict with:
        'separated'     : bool   (physical prediction)
        'method'        : str
        'p_sep'         : separation pressure  [Pa]
        'x_sep'         : axial location of separation  [m]  (None if no sep)
        'y_sep'         : radial location  [m]  (None if no sep)
        'margin'        : Pe/p_sep  (>1 means no separation)
        'design_margin_required' : the SP-8120-style factor used
        'design_margin_ok'       : margin ≥ design_margin and not separated
        'exit_pressure' : Pe  [Pa]
    """
    x = np.asarray(contour['x'], dtype=float)
    y = np.asarray(contour['y'], dtype=float)
    Rt = contour['Rt']
    At = np.pi * Rt**2
    epsilon = contour['epsilon']


    if frozen_expansion is not None:
        if not math.isclose(
            float(frozen_expansion.chamber_pressure_pa),
            float(Pc), rel_tol=1.0e-10, abs_tol=0.0,
        ):
            raise ValueError("frozen expansion chamber pressure does not match Pc")
        if not math.isclose(
            float(frozen_expansion.expansion_ratio),
            float(epsilon), rel_tol=1.0e-10, abs_tol=1.0e-12,
        ):
            raise ValueError("frozen expansion ratio does not match contour epsilon")
        Me = float(frozen_expansion.exit.mach)
        Pe = float(frozen_expansion.exit.pressure_pa)
    else:
        Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
        Pe = Pc * isentropic_pressure_ratio(Me, gamma)


    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {_METHODS}.")

    def criterion_pressure(M_local: float) -> float:
        if Pa <= 0.0:
            return 0.0  # vacuum: no ambient back-pressure, no separation
        if method == 'summerfield':
            return summerfield_separation_pressure(Pa)
        if method == 'kalt_badal':
            return kalt_badal_separation_ratio(Pc / Pa) * Pa
        if method == 'schilling':
            return schilling_separation_ratio(Pc / Pa) * Pa
        return schmucker_separation_ratio(M_local, Pa / Pc) * Pc

    # Schmucker is a local-Mach criterion and must be evaluated at each
    # candidate wall station; the pressure-ratio criteria (Summerfield,
    # Schilling, Kalt-Badal) yield a station-independent threshold.  Using Me
    # once and then marching against a constant threshold would mix an exit
    # condition with upstream wall states for the local-Mach form.
    p_sep_exit = criterion_pressure(Me)
    separated = False
    x_sep = None
    y_sep = None
    onset_threshold = None
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    for i in range(throat_idx, len(x)):
        A_local = np.pi * y[i]**2
        ar = max(A_local / At, 1.0)
        if frozen_expansion is not None:
            station = frozen_expansion.station(ar, supersonic=True)
            M_local = station.mach
            p_local = station.pressure_pa
        else:
            try:
                M_local = mach_from_area_ratio(ar, gamma, supersonic=True)
            except Exception:
                continue
            p_local = Pc * isentropic_pressure_ratio(M_local, gamma)
        threshold = criterion_pressure(M_local)
        if p_local <= threshold:
            separated = True
            x_sep = float(x[i])
            y_sep = float(y[i])
            onset_threshold = float(threshold)
            break

    p_sep = onset_threshold if onset_threshold is not None else p_sep_exit
    margin = Pe / p_sep_exit if p_sep_exit > 0 else float('inf')

    return {
        'separated': separated,
        'method': method,
        'p_sep': p_sep,
        'exit_criterion_pressure': p_sep_exit,
        'onset_criterion_pressure': onset_threshold,
        'criterion_evaluated_locally': True,
        'x_sep': x_sep,
        'y_sep': y_sep,
        'margin': margin,
        'design_margin_required': float(design_margin),
        'design_margin_ok': bool((not separated) and margin >= float(design_margin)),
        'exit_pressure': Pe,
        'expansion_model': (
            'frozen_variable_cp_q1d'
            if frozen_expansion is not None else 'constant_gamma'
        ),
    }


def separation_summary(result: dict) -> str:
    """Format a human-readable separation check summary."""
    lines = []
    lines.append(f"  Separation check ({result['method']}):")
    lines.append(f"    Exit pressure Pe = {result['exit_pressure']:.0f} Pa")
    lines.append(f"    Separation pressure p_sep = {result['p_sep']:.0f} Pa")
    lines.append(f"    Margin Pe/p_sep = {result['margin']:.3f}")
    if result['separated']:
        lines.append(f"    ⚠  SEPARATION PREDICTED at x = {result['x_sep']*1000:.1f} mm")
    else:
        lines.append(f"    ✓  No separation expected")
        if not result.get('design_margin_ok', True):
            lines.append(
                "    ⚠  SP-8120 design margin NOT met "
                f"(Pe/p_sep {result['margin']:.2f} < "
                f"{result['design_margin_required']:.2f}): consider reducing ε"
            )
    return "\n".join(lines)
