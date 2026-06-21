"""Verify the throat-bell tangent-reconciliation fix clears all geometry gates."""
import math
import numpy as np
import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig
from raosim.throat_geometry import ThroatGeometrySpec
from raosim.chamber_geometry import (
    chamber_contour, full_engine_contour,
    failed_thrust_chamber_geometry_checks, HARD_THRUST_CHAMBER_GEOMETRY_CHECKS,
)

Rt, eps, lpct, gamma, pa = 0.07, 25.0, 80.0, 1.22, 0.01
ru_f, rd_f, conv = 1.5, 0.382, 45.0

cfg = RaoSolverConfig(
    Rt=Rt, epsilon=eps, gamma=gamma, pa_over_p0=pa, length_pct=lpct,
    max_nfev=0, solver_backend="numpy", wall_method="bde",
    throat_upstream_radius_factor=ru_f, throat_downstream_radius_factor=rd_f,
    kernel_d_fraction_max=0.7,
)
sol = rv.solve_rao_bvp(cfg)
spec = ThroatGeometrySpec(upstream_radius_ratio=ru_f, downstream_radius_ratio=rd_f,
                          convergent_half_angle_deg=conv)
nozzle = sol.to_contour_dict(Rt=Rt, epsilon=eps, length_pct=lpct, pa_over_p0=pa,
                             Ru_factor=ru_f, Rd_factor=rd_f,
                             convergent_half_angle_deg=conv)
nozzle["throat_geometry"] = spec.to_dict()
nozzle["throat_location"] = spec.throat_location

chamber = chamber_contour(Rt, L_star=1.75, contraction_ratio=3.0,
                          throat_geometry=spec, shoulder_radius_factor=0.5,
                          minimum_cylindrical_length=0.45)
contour = full_engine_contour(chamber, nozzle)
checks = contour["geometry_checks"]
failed = failed_thrust_chamber_geometry_checks(checks)

print("HARD GATES:")
for name in HARD_THRUST_CHAMBER_GEOMETRY_CHECKS:
    print(f"  {'PASS' if checks.get(name) else 'FAIL'}  {name}")
print("\nkey metrics:")
print(f"  position_continuity      = {checks['position_continuity']}")
print(f"  seam_position_gap        = {checks['seam_position_gap']:.3e} m")
print(f"  throat_bell_position_gap = {checks['throat_bell_position_gap']:.3e} m")
print(f"  slope_continuity         = {checks['slope_continuity']}")
print(f"  maximum_join_angle_deg   = {checks['maximum_join_angle_deg']:.4f} deg")
print(f"  axial monotonic          = {checks['axial_coordinates_monotonic']}")

rec = nozzle["throat_bell_reconciliation"]
Re_des = rec["exit_radius_design"]; Re_built = rec["exit_radius_built"]
eps_built = (Re_built / Rt) ** 2
print("\nthroat bridge (tangent, exit-preserving):")
print(f"  bridge kind           = {rec['throat_bridge']}")
print(f"  wall theta0           = {rec['wall_theta0_deg']:.4f} deg")
print(f"  exit radius design    = {Re_des:.6f} m  (eps={eps:.3f})")
print(f"  exit radius built     = {Re_built:.6f} m  (eps={eps_built:.3f})")
print(f"  eps drift             = {100*(eps_built-eps)/eps:+.4f} %  (expect ~0)")

print("\nRESULT:", "ALL HARD GATES PASS -> would export" if not failed
      else f"STILL FAILING: {failed}")
