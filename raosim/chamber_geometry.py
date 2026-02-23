"""
chamber_geometry.py – Combustion chamber and convergent section geometry.

Generates the full upstream geometry that connects to the nozzle:
  1. Cylindrical chamber section (constant radius = Rc)
  2. Conical or curved convergent section → merges at the upstream fillet

Parameterized by:
  - L*  : characteristic length  (V_chamber / A_throat)  [m]
  - CR  : contraction ratio  (Ac / At)
  - convergent half-angle
"""

from __future__ import annotations
import math
import numpy as np


def chamber_contour(
    Rt: float,
    L_star: float = 1.0,
    contraction_ratio: float = 2.5,
    convergent_half_angle_deg: float = 30.0,
    n_pts_chamber: int = 50,
    n_pts_convergent: int = 80,
) -> dict:
    """
    Generate the combustion chamber + convergent section contour.

    Parameters
    ----------
    Rt                      : throat radius  [m]
    L_star                  : characteristic length V_chamber/At  [m]  (default 1.0)
    contraction_ratio       : Ac/At  (default 2.5)
    convergent_half_angle_deg : half-angle of convergent cone  [°]  (default 30)
    n_pts_chamber           : points in the cylindrical section
    n_pts_convergent        : points in the convergent section

    Returns
    -------
    dict with:
        'x', 'y'           : full upstream contour  (chamber → convergent)
        'x_chamber', 'y_chamber' : cylindrical section
        'x_conv', 'y_conv'       : convergent section
        'Rc'                : chamber radius  [m]
        'Lc'                : cylindrical chamber length  [m]
        'L_conv'            : convergent section length  [m]
        'V_chamber'         : chamber volume  [m³]
    """
    At = math.pi * Rt**2
    Rc = Rt * math.sqrt(contraction_ratio)   # Ac = π·Rc² = CR · At
    Ac = math.pi * Rc**2

    # Volume = L* · At
    V_chamber = L_star * At

    # Convergent section: cone from Rc down to the nozzle inlet
    # The inlet of the nozzle upstream fillet is at radius ≈ Rt (at x=0)
    # The convergent cone goes from Rc to approximately Rt + 1.5·Rt·(1 - cos(conv_angle))
    # For simplicity, the convergent section goes from Rc down to the upstream
    # fillet entry radius.  With a 1.5Rt fillet, the fillet starts at angle
    # -(π/2 + conv_angle), so the tangent point radius is:
    Ru = 1.5 * Rt
    conv_angle_rad = math.radians(convergent_half_angle_deg)

    # The fillet entry point (top of the upstream arc):
    # Arc center at (0, Rt + Ru)
    # Arc ends at angle -(π/2 + conv_angle)
    # Entry point: x_entry = Ru·cos(-(π/2 + conv_angle)) = -Ru·sin(conv_angle)
    # y_entry = (Rt + Ru) + Ru·sin(-(π/2 + conv_angle)) = (Rt + Ru) - Ru·cos(conv_angle)
    x_fillet_entry = -Ru * math.sin(conv_angle_rad)
    y_fillet_entry = (Rt + Ru) - Ru * math.cos(conv_angle_rad)

    # Convergent cone: from Rc to y_fillet_entry, at the given half-angle
    # Length of convergent cone:
    L_conv = (Rc - y_fillet_entry) / math.tan(conv_angle_rad)

    # x coordinates: convergent starts at x_fillet_entry - L_conv
    x_conv_start = x_fillet_entry - L_conv

    # Cylindrical chamber volume:
    # Total V_chamber = V_cylinder + V_cone_frustum
    # V_cone_frustum = (π/3)·L_conv·(Rc² + Rc·y_fillet_entry + y_fillet_entry²)
    V_frustum = (math.pi / 3.0) * L_conv * (
        Rc**2 + Rc * y_fillet_entry + y_fillet_entry**2
    )
    V_cylinder_needed = V_chamber - V_frustum
    if V_cylinder_needed < 0:
        # Chamber is all convergent section — L* too small or CR too large
        V_cylinder_needed = 0.001 * V_chamber  # minimal cylinder length
    Lc = V_cylinder_needed / Ac  # Cylinder length

    # ── Build the contour ────────────────────────────────────────

    # Cylindrical section: from left end to convergent section start
    x_cyl_start = x_conv_start - Lc
    x_chamber = np.linspace(x_cyl_start, x_conv_start, n_pts_chamber)
    y_chamber = np.full_like(x_chamber, Rc)

    # Convergent cone: from x_conv_start to x_fillet_entry
    x_conv = np.linspace(x_conv_start, x_fillet_entry, n_pts_convergent)
    y_conv = Rc - (x_conv - x_conv_start) * math.tan(conv_angle_rad)

    # Concatenate
    x_full = np.concatenate([x_chamber, x_conv[1:]])  # skip duplicate point
    y_full = np.concatenate([y_chamber, y_conv[1:]])

    return {
        'x': x_full,
        'y': y_full,
        'x_chamber': x_chamber,
        'y_chamber': y_chamber,
        'x_conv': x_conv,
        'y_conv': y_conv,
        'Rc': Rc,
        'Lc': Lc,
        'L_conv': L_conv,
        'V_chamber': V_chamber,
        'L_star': L_star,
        'contraction_ratio': contraction_ratio,
    }


def full_engine_contour(
    chamber: dict,
    nozzle: dict,
) -> dict:
    """
    Stitch the chamber + convergent contour with the nozzle contour to
    produce one continuous engine profile.

    Parameters
    ----------
    chamber : dict from ``chamber_contour``
    nozzle  : dict from ``bell_nozzle_contour``

    Returns
    -------
    dict with 'x', 'y' for the complete engine
    """
    x_full = np.concatenate([chamber['x'], nozzle['x']])
    y_full = np.concatenate([chamber['y'], nozzle['y']])
    return {
        'x': x_full,
        'y': y_full,
        'chamber': chamber,
        'nozzle': nozzle,
    }
