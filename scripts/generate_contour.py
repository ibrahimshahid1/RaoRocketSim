"""Parametric Rao nozzle contour generator (converged BVP + BDE wall).

Runs the current solver configuration (characteristic formulation, JAX
backend, constraint-weight ladder, full D-state continuity) for the
design point you specify, builds the wall via NASA's BDE-region march,
and writes:

    <out>/contour.csv      wall polyline (x, r) in metres
    <out>/contour_3d.stl   binary STL surface of revolution
    <out>/contour.png      2-D profile + residual summary
    <out>/contour_3d.png   3-D rendering (matplotlib)

Usage examples
--------------
    PYTHONPATH=. python scripts/generate_contour.py
    PYTHONPATH=. python scripts/generate_contour.py \
        --rt 0.025 --epsilon 8 --length-pct 85 --gamma 1.23 --out builds/my_nozzle

Reliability: the solve must pass the 2e-3 residual gate or the script
exits nonzero (use --allow-unconverged to override).  Outputs are
preliminary design geometry — NOT hardware-qualified (no CFD, thermal,
structural, or hot-fire validation; see RaoSolution.warnings).
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig


def solve(args):
    rv.PHYSICS_WEIGHT = 1.0
    cfg = RaoSolverConfig(
        Rt=args.rt, epsilon=args.epsilon, gamma=args.gamma,
        pa_over_p0=args.pa_over_p0, length_pct=args.length_pct,
        n_control=args.n_control, n_kernel=args.n_kernel,
        max_nfev=4000, residual_tol=2e-3,
        evaluate_moc=True, wall_method="bde", couple_wall=False,
        kernel_d_fraction_max=0.7,
        solver_backend="jax", thetaN_guess_deg=args.theta_b_guess,
        formulation="characteristic", pin_d_theta=True, pin_d_mach=True,
        jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
    )
    return rv.solve_rao_bvp(cfg)


def revolve(wall: np.ndarray, n_theta: int = 96):
    """Surface of revolution: (n_pts, n_theta, 3) vertices."""
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    x = wall[:, 0][:, None] * np.ones_like(th)[None, :]
    y = wall[:, 1][:, None] * np.cos(th)[None, :]
    z = wall[:, 1][:, None] * np.sin(th)[None, :]
    return np.stack([x, y, z], axis=-1)


def write_stl(path: Path, verts: np.ndarray):
    """Binary STL from the revolve grid (two triangles per quad)."""
    n_i, n_j, _ = verts.shape
    tris = []
    for i in range(n_i - 1):
        for j in range(n_j):
            j2 = (j + 1) % n_j
            a, b = verts[i, j], verts[i, j2]
            c, d = verts[i + 1, j], verts[i + 1, j2]
            tris.append((a, b, c))
            tris.append((b, d, c))
    with open(path, "wb") as f:
        f.write(b"RaoRocketSim BDE contour".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else np.array([0.0, 0.0, 1.0])
            f.write(struct.pack("<3f", *n))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rt", type=float, default=0.020, help="throat radius [m]")
    ap.add_argument("--epsilon", type=float, default=10.0, help="area ratio Ae/At")
    ap.add_argument("--length-pct", type=float, default=80.0,
                    help="bell length as %% of 15-deg cone")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--pa-over-p0", type=float, default=0.01)
    ap.add_argument("--theta-b-guess", type=float, default=22.0,
                    help="initial expansion angle seed [deg]")
    ap.add_argument("--n-control", type=int, default=24)
    ap.add_argument("--n-kernel", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path("builds/contour_run"))
    ap.add_argument("--allow-unconverged", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Solving Rao BVP: Rt={args.rt} m, eps={args.epsilon}, "
          f"L%={args.length_pct}, gamma={args.gamma} ...", flush=True)
    sol = solve(args)
    r = sol.residuals
    gate = r.max_scaled <= 2e-3
    print(f"max_scaled = {r.max_scaled:.3e} (gate 2e-3) "
          f"mass = {r.mass_residual_rel:+.1e}  length = {r.length_residual_rel:+.1e}")
    print(f"kdf = {sol.control_surface.kernel_d_fraction:.3f}  "
          f"reliability = {sol.reliability.value}")
    if not gate and not args.allow_unconverged:
        print("NOT CONVERGED to the residual gate; rerun with "
              "--allow-unconverged to export anyway.", file=sys.stderr)
        sys.exit(2)

    wall = sol.wall_raw
    L = rv._target_length(args.rt, args.epsilon, args.length_pct)
    Re = math.sqrt(args.epsilon) * args.rt
    print(f"wall: {wall.shape[0]} points; exit ({wall[-1,0]:.5f}, "
          f"{wall[-1,1]:.5f}) vs target ({L:.5f}, {Re:.5f})")

    np.savetxt(args.out / "contour.csv", wall, delimiter=",",
               header="x_m,r_m", comments="")

    verts = revolve(wall)
    write_stl(args.out / "contour_3d.stl", verts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(wall[:, 0] * 1e3, wall[:, 1] * 1e3, "b-", lw=1.5, label="bell wall (BDE)")
    ax.plot(wall[:, 0] * 1e3, -wall[:, 1] * 1e3, "b-", lw=1.5)
    ax.plot([L * 1e3], [Re * 1e3], "r*", ms=12, label="commanded exit E")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("r [mm]")
    ax.set_title(
        f"Rao TOP contour  Rt={args.rt*1e3:.1f} mm  eps={args.epsilon:g}  "
        f"L={args.length_pct:g}%  gamma={args.gamma:g}\n"
        f"max_scaled={r.max_scaled:.2e}  mass={r.mass_residual_rel:+.1e}  "
        f"len={r.length_residual_rel:+.1e}  kdf={sol.control_surface.kernel_d_fraction:.2f}")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "contour.png", dpi=160)

    fig3 = plt.figure(figsize=(7, 5.6))
    ax3 = fig3.add_subplot(111, projection="3d")
    v = np.concatenate([verts, verts[:, :1, :]], axis=1)  # close the seam
    ax3.plot_surface(v[..., 0] * 1e3, v[..., 1] * 1e3, v[..., 2] * 1e3,
                     cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_xlabel("x [mm]"); ax3.set_ylabel("y [mm]"); ax3.set_zlabel("z [mm]")
    ax3.set_title("Surface of revolution (exported to contour_3d.stl)")
    try:
        ax3.set_box_aspect((np.ptp(v[..., 0]), np.ptp(v[..., 1]), np.ptp(v[..., 2])))
    except Exception:
        pass
    fig3.tight_layout()
    fig3.savefig(args.out / "contour_3d.png", dpi=160)

    print(f"wrote: {args.out}/contour.csv, contour_3d.stl, contour.png, contour_3d.png")
    print("NOTE: preliminary design geometry; not hardware-qualified.")


if __name__ == "__main__":
    main()
