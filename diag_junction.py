"""Diagnose the chamber/nozzle geometry gate failure (position + slope)."""
import math
import numpy as np
import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig
from raosim.throat_geometry import ThroatGeometrySpec
from raosim.chamber_geometry import (
    chamber_contour, full_engine_contour, thrust_chamber_geometry_checks,
    _join_angle_deg,
)

# user's interview params
Rt, eps, lpct, gamma, pa = 0.07, 25.0, 80.0, 1.22, 0.01
ru_f, rd_f, conv = 1.5, 0.382, 45.0

cfg = RaoSolverConfig(
    Rt=Rt, epsilon=eps, gamma=gamma, pa_over_p0=pa, length_pct=lpct,
    max_nfev=0,                      # seed only -> fast
    solver_backend="numpy", wall_method="bde",
    throat_upstream_radius_factor=ru_f,
    throat_downstream_radius_factor=rd_f,
    kernel_d_fraction_max=0.7,
)
sol = rv.solve_rao_bvp(cfg)
spec = ThroatGeometrySpec(upstream_radius_ratio=ru_f,
                          downstream_radius_ratio=rd_f,
                          convergent_half_angle_deg=conv)
nozzle = sol.to_contour_dict(Rt=Rt, epsilon=eps, length_pct=lpct,
                             pa_over_p0=pa, Ru_factor=ru_f, Rd_factor=rd_f,
                             convergent_half_angle_deg=conv)
nozzle["throat_geometry"] = spec.to_dict()
nozzle["throat_location"] = spec.throat_location

chamber = chamber_contour(Rt, L_star=1.75, contraction_ratio=3.0,
                          throat_geometry=spec, shoulder_radius_factor=0.5,
                          minimum_cylindrical_length=0.45)
contour = full_engine_contour(chamber, nozzle)
checks = contour["geometry_checks"]

print("theta_N(report) =", math.degrees(sol.theta_N))
print("reliability     =", sol.reliability.value)

Rd = rd_f * Rt
wx = np.asarray(nozzle["x_throat"]); wy = np.asarray(nozzle["y_throat"])
bx = np.asarray(nozzle["x_bell"]);   by = np.asarray(nozzle["y_bell"])
arc_end = math.asin(min(max(float(bx[0]) / Rd, 0.0), 1.0))
print("\n--- throat arc <-> bell (MOC wall) junction ---")
print(f"Rd = {Rd:.6f} m")
print(f"arc_end (matched to wall x0) = {math.degrees(arc_end):.4f} deg")
print(f"throat-arc end  (Nx,Ny) = ({wx[-1]:.8f}, {wy[-1]:.8f})")
print(f"wall start      (x0,r0) = ({bx[0]:.8f}, {by[0]:.8f})")
print(f"  dx = {wx[-1]-bx[0]:+.3e}   dr = {wy[-1]-by[0]:+.3e}  (radial gap)")
arc_slope = math.degrees(arc_end)  # local wall angle of circular arc at arc_end
wall_slope = math.degrees(math.atan2(by[1]-by[0], bx[1]-bx[0]))
print(f"arc tangent angle at end = {arc_slope:.4f} deg")
print(f"wall initial slope angle = {wall_slope:.4f} deg")
print(f"  slope jump = {abs(arc_slope-wall_slope):.4f} deg")

print("\n--- individual join angles (deg) ---")
print("chamber cyl->shoulder :",
      _join_angle_deg(chamber["x_chamber"], chamber["y_chamber"],
                      chamber["x_shoulder"], chamber["y_shoulder"]))
print("shoulder->convergent  :",
      _join_angle_deg(chamber["x_shoulder"], chamber["y_shoulder"],
                      chamber["x_conv"], chamber["y_conv"]))
print("convergent->upst arc  :",
      _join_angle_deg(chamber["x_conv"], chamber["y_conv"],
                      chamber["x_upstream_arc"], chamber["y_upstream_arc"]))
print("throat arc->bell      :",
      _join_angle_deg(wx, wy, bx, by))

print("\n--- gate summary ---")
for k in ("position_continuity", "seam_position_gap", "throat_bell_position_gap",
          "slope_continuity", "maximum_join_angle_deg"):
    print(f"  {k} = {checks[k]}")
