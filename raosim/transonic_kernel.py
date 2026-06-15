"""
Kliegel-Levine 1969 transonic kernel for the MOC starting line.

Kliegel, J.R. and Levine, J.N., *Transonic Flow in Small Throat Radius of
Curvature Nozzles*, AIAA J. 7(7) 1375-1378, 1969.  Open access:
<https://arc.aiaa.org/doi/pdf/10.2514/3.5355>.  Also reproduced in
Zucrow & Hoffman, *Gas Dynamics* Vol. 2 Ch. 16, and Östlund 2002 §3.

The closed-form 3rd-order toroidal-coordinate expansion is valid down
to Rc/Rt ~ 0.5, well below the leading-order Hall expansion's ~2.0
floor.  This module's coefficients are a direct port of NASA/JHU
``MOC_GridCalc_BDE.cpp::KLThroat`` (lines 3103-3178) with two typo
corrections in the AXI ``v[3]`` polynomial (NASA line 3133) and the
analogous typo in the TWOD ``v[3]`` polynomial (NASA line 3157).

Typo corrections
================
NASA AXI line 3133 reads::

    z*((556*G*G + 1737*G + 3069)*y*y*y*y*y/1728
       * (388*G*G + 1161*G + 1181)*y*y/576
       + (304*G*G + 831*G + 1242)*y/864) + ...

There are three problems with the middle sub-term:

* The ``*`` between the y^5 term and the y^2 term must be ``+``: with
  ``*``, the dimensional power becomes y^7 which is inconsistent with
  a third-order radial-expansion coefficient and breaks the series
  recurrence.
* The constant ``1181`` is a typo for ``1881`` (cross-check Kliegel-
  Levine 1969 Table 1; the same coefficient ``1881`` appears in the
  ``u[3]`` polynomial at NASA line 3127 as ``(388*G*G + 1161*G + 1881)``,
  confirming that v[3] should match).
* The exponent ``y*y`` (= y^2) is a typo for ``y*y*y`` (= y^3): the
  2D branch's analogous term at line 3157 uses ``y*y*y``, and the
  y^5 → y^3 → y power progression matches the descending parity
  expected of a third-order coefficient.

NASA AXI line 3129 reads::

    z*((52*G*G + 51*G + 327)*y*y*y*y/34
       - (52*G*G + 75*G + 279)*y*y/192
       + (92*G*G + 180*G + 639)/1152)

The divisor ``/34`` on the y^4 z-coupling term in ``u[3]`` is off by
~11× — every other divisor in the same polynomial is /24, /144, /192,
/1152, /82944 (powers of 2 × 3).  The correct divisor is ``/384``
(2 × /192, the natural recurrence factor for the y^4 term immediately
above the /192 y^2 term).  Cross-check Kliegel-Levine 1969 Table 1.
This implementation uses the corrected ``/384.0``.

The 2D analogue (NASA line 3157) has the same ``*`` → ``+`` typo in
the middle of the z-bracket; the constant ``1665`` is correct.

This module uses the corrected forms.  Do **not** "fix" the comments
back to literal NASA source — the original lines are mathematically
wrong; see ``test_transonic_kernel.py`` for the validation gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


GEOM_AXI = "AXI"
GEOM_TWOD = "TWOD"


@dataclass(frozen=True)
class TransonicState:
    """Local transonic state returned by :func:`kliegel_levine`."""

    u_over_astar: float
    v_over_astar: float
    M: float
    theta: float
    p_over_p0: float


def _kl_axi_coeffs(y: float, z: float, G: float) -> tuple[float, float, float, float, float, float]:
    """Polynomial coefficients ``u1..u3, v1..v3`` for the axisymmetric KL series.

    Port of ``MOC_GridCalc_BDE.cpp::KLThroat`` AXI branch (lines 3122-3135),
    with the ``v[3]`` line-3133 typo corrected as documented at the top
    of this module.
    """
    y2 = y * y
    y3 = y2 * y
    y4 = y3 * y
    y5 = y4 * y
    y6 = y5 * y
    y7 = y6 * y
    z2 = z * z
    z3 = z2 * z

    u1 = 0.5 * y2 - 0.25 + z
    v1 = y3 / 4.0 - y / 4.0 + y * z
    u2 = (
        (2 * G + 9) * y4 / 24.0
        - (4 * G + 15) * y2 / 24.0
        + (10 * G + 57) / 288.0
        + z * (y2 - 5.0 / 8.0)
        - (2 * G - 3) * z2 / 6.0
    )
    v2 = (
        (G + 3) * y5 / 9.0
        - (20 * G + 63) * y3 / 96.0
        + (28 * G + 93) * y / 288.0
        + z * ((2 * G + 9) * y3 / 6.0 - (4 * G + 15) * y / 12.0)
        + y * z2
    )
    u3 = (
        (556 * G * G + 1737 * G + 3069) * y6 / 10368.0
        - (388 * G * G + 1161 * G + 1881) * y4 / 2304.0
        + (304 * G * G + 831 * G + 1242) * y2 / 1728.0
        - (2708 * G * G + 7839 * G + 14211) / 82944.0
        + z * (
            (52 * G * G + 51 * G + 327) * y4 / 384.0
            - (52 * G * G + 75 * G + 279) * y2 / 192.0
            + (92 * G * G + 180 * G + 639) / 1152.0
        )
        + z2 * (-(7 * G - 3) * y2 / 8.0 + (13 * G - 27) / 48.0)
        + (4 * G * G - 57 * G + 27) * z3 / 144.0
    )
    # CORRECTED v3 (see module docstring): NASA line 3133 has `*` between
    # the y^5 and y^3 sub-terms (should be `+`), an exponent `y*y` that
    # should be `y*y*y`, and a constant `1181` that should be `1881`.
    v3 = (
        (6836 * G * G + 23031 * G + 30627) * y7 / 82944.0
        - (3380 * G * G + 11391 * G + 15291) * y5 / 13824.0
        + (3424 * G * G + 11271 * G + 15228) * y3 / 13824.0
        - (7100 * G * G + 22311 * G + 30249) * y / 82944.0
        + z * (
            (556 * G * G + 1737 * G + 3069) * y5 / 1728.0
            + (388 * G * G + 1161 * G + 1881) * y3 / 576.0
            + (304 * G * G + 831 * G + 1242) * y / 864.0
        )
        + z2 * (
            (52 * G * G + 51 * G + 327) * y3 / 192.0
            - (52 * G * G + 75 * G + 279) * y / 192.0
        )
        - z3 * (7 * G - 3) * y / 12.0
    )
    return u1, u2, u3, v1, v2, v3


def _kl_twod_coeffs(y: float, z: float, G: float) -> tuple[float, float, float, float, float, float]:
    """Polynomial coefficients for the 2D planar KL series.

    Port of ``MOC_GridCalc_BDE.cpp::KLThroat`` TWOD branch (lines 3146-3159).
    The line-3157 ``*`` typo inside the ``v[3]`` z-bracket is corrected to
    ``+`` analogously to the AXI v[3] correction at the top of this module.

    NASA line 3151 reads ``(782*G*G + 5523 + 2*G*2887)/272160``: this is
    interpreted as ``(782*G*G + 5774*G + 5523)/272160`` (the loose
    ``2*G*2887`` is NASA shorthand for ``5774*G``).
    """
    y2 = y * y
    y3 = y2 * y
    y4 = y3 * y
    y5 = y4 * y
    y6 = y5 * y
    y7 = y6 * y
    z2 = z * z
    z3 = z2 * z

    u1 = 0.5 * y2 - 1.0 / 6.0 + z
    v1 = y3 / 6.0 - y / 6.0 + y * z
    u2 = (
        (y + 6) * y4 / 18.0
        - (2 * G + 9) * y2 / 18.0
        + (G + 30) / 270.0
        + z * (y2 - 0.5)
        - (2 * G - 3) * z2 / 6.0
    )
    v2 = (
        (22 * G + 75) * y5 / 360.0
        - (5 * G + 21) * y3 / 54.0
        + (34 * G + 195) * y / 1080.0
        + z / 9.0 * ((2 * G + 12) * y3 - (2 * G + 9) * y)
        + y * z2
    )
    u3 = (
        (362 * G * G + 1449 * G + 3177) * y6 / 12960.0
        - (194 * G * G + 837 * G + 1665) * y4 / 2592.0
        + (854 * G * G + 3687 * G + 6759) * y2 / 12960.0
        - (782 * G * G + 5774 * G + 5523) / 272160.0
        + z * (
            (26 * G * G + 27 * G + 237) * y4 / 288.0
            - (26 * G * G + 51 * G + 189) * y2 / 144.0
            + (134 * G * G + 429 * G + 1743) / 4320.0
        )
        + z2 * (-5 * G * y2 / 4.0 + (7 * G - 18) / 36.0)
        + z3 * (2 * G * G - 33 * G + 9) / 72.0
    )
    # CORRECTED v3: NASA line 3157 has `*` between the y^5 and y^3
    # sub-terms in the z-bracket; the math requires `+`.
    v3 = (
        (6574 * G * G + 26481 * G + 40059) * y7 / 181440.0
        - (2254 * G * G + 10113 * G + 16479) * y5 / 25920.0
        + (5026 * G * G + 25551 * G + 46377) * y3 / 77760.0
        - (7570 * G * G + 45927 * G + 98757) * y / 544320.0
        + z * (
            (362 * G * G + 1449 * G + 3177) * y5 / 2160.0
            + (194 * G * G + 837 * G + 1665) * y3 / 648.0
            + (854 * G * G + 3687 * G + 6759) * y / 6480.0
        )
        + z2 * (
            (26 * G * G + 27 * G + 237) * y3 / 144.0
            - (26 * G * G + 51 * G + 189) / 144.0
        )
        + z3 * (-5 * G * y / 6.0)
    )
    return u1, u2, u3, v1, v2, v3


def kliegel_levine(
    r_over_Rt: float,
    x_over_Rt: float,
    gamma: float,
    Rc_over_Rt: float,
    geom: str = GEOM_AXI,
) -> TransonicState:
    """Third-order Kliegel-Levine transonic state at ``(r, x)`` near the throat.

    Parameters
    ----------
    r_over_Rt : float
        Radial coordinate normalised by the throat radius.
    x_over_Rt : float
        Axial coordinate normalised by the throat radius.  ``x = 0`` is
        the throat plane; positive ``x`` is downstream.
    gamma : float
        Ratio of specific heats.
    Rc_over_Rt : float
        Throat curvature ratio (``Rd / Rt`` in this codebase's nomenclature).
    geom : str
        ``"AXI"`` for axisymmetric flow, ``"TWOD"`` for planar.

    Returns
    -------
    TransonicState
        Local velocity components (u/a*, v/a* in NASA's normalised units;
        per NASA's KLThroat the magnitude ``sqrt(U^2 + V^2)`` is taken
        directly as the Mach number, matching the source-code definition
        at line 3168), the corresponding Mach number, the flow angle, and
        ``p/p0`` via the local isentropic relation.

    References
    ----------
    * Kliegel, J.R. and Levine, J.N., *Transonic Flow in Small Throat
      Radius of Curvature Nozzles*, AIAA J. 7(7) 1375-1378, 1969.
    * Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Ch. 16 (coefficient table).
    * Östlund 2002, KTH thesis §3 (cross-check).

    Port reference
    --------------
    ``MOC_Grid_BDE/MOC_GridCalc_BDE.cpp::KLThroat`` lines 3103-3178
    (NASA/JHU Three-Dimensional-Nozzle-Design-Code, 2003).
    """
    if Rc_over_Rt <= 0.0:
        raise ValueError("Rc_over_Rt must be positive")
    G = float(gamma)
    y = float(r_over_Rt)

    if geom == GEOM_AXI:
        # NASA Eq. 12 toroidal coordinate (axisymmetric).
        z = float(x_over_Rt) * math.sqrt(2.0 * Rc_over_Rt / (G + 1.0))
        RSP = float(Rc_over_Rt) + 1.0
        u1, u2, u3, v1, v2, v3 = _kl_axi_coeffs(y, z, G)
        U = 1.0 + u1 / RSP + (u1 + u2) / (RSP ** 2) + (u1 + 2 * u2 + u3) / (RSP ** 3)
        V = math.sqrt((G + 1.0) / (2.0 * RSP)) * (
            v1 / RSP
            + (1.5 * v1 + v2) / (RSP ** 2)
            + (15.0 / 8.0 * v1 + 2.5 * v2 + v3) / (RSP ** 3)
        )
    elif geom == GEOM_TWOD:
        z = float(x_over_Rt) * math.sqrt(Rc_over_Rt / (G + 1.0))
        RS = float(Rc_over_Rt)
        u1, u2, u3, v1, v2, v3 = _kl_twod_coeffs(y, z, G)
        U = 1.0 + u1 / RS + u2 / (RS ** 2) + u3 / (RS ** 3)
        V = math.sqrt((G + 1.0) / RS) * (
            v1 / RS + v2 / (RS ** 2) + v3 / (RS ** 3)
        )
    else:
        raise ValueError(f"geom must be {GEOM_AXI!r} or {GEOM_TWOD!r}, got {geom!r}")

    if abs(V) < 1e-5:
        V = 0.0
    theta = math.atan2(V, U)
    if abs(theta) < 1e-5:
        theta = 0.0
    M = math.hypot(U, V)

    from raosim.gas_dynamics import isentropic_pressure_ratio

    if M > 1.0:
        p_over_p0 = isentropic_pressure_ratio(M, G)
    else:
        p_over_p0 = isentropic_pressure_ratio(max(M, 1.0 + 1e-9), G)

    return TransonicState(U, V, M, theta, p_over_p0)
