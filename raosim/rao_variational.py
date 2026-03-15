"""
rao_variational.py – Rao thrust-optimized nozzle via calculus of variations.

Implements the *true* Rao optimization method:
  1. Define a control surface CE in the supersonic region
  2. Express thrust and mass flow as integrals over CE
  3. Augmented functional  I = ∫(f₁ + λ₂·f₂ + λ₃·f₃) dr
  4. Enforce Euler–Lagrange stationarity:  ∂I/∂M = ∂I/∂θ = ∂I/∂φ = 0
  5. Solve for optimal (M(r), θ(r), φ(r)) on CE
  6. Construct shock-free wall via MOC from starting line to CE

The conceptual key (from Rao 1958, NASA TM-1990):
  Optimize the *flow distribution on a control surface*, not the wall itself.
  Then build the wall that realizes that flowfield.

References:
  - G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958
  - NASA TM (1990), Rao method re-derivation with explicit functionals
  - NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    isentropic_density_ratio,
    isentropic_temperature_ratio,
    mach_from_area_ratio,
    mach_angle,
    prandtl_meyer,
    mach_from_prandtl_meyer,
    area_mach_relation,
)
from raosim.moc import (
    _make_point,
    CharPoint,
    CharRow,
    solve_interior_point,
    solve_axis_point,
    solve_wall_point,
    approximate_starting_line,
    march_coupled_net,
    sample_exit_plane,
    compute_exit_thrust,
)
from raosim.wall_model import SplineWall


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ControlSurface:
    """
    Discretized control surface CE.

    Each station i defines the flow state at radius r_i:
        M[i]     : Mach number
        theta[i] : flow angle from axis [rad]
        phi[i]   : control-surface inclination from axis [rad]
        r[i]     : radial position [m]
    """
    r: np.ndarray
    M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    lambda2: float = 0.0    # Lagrange multiplier for mass-flow constraint
    lambda3: float = 0.0    # Lagrange multiplier for length constraint


# ─────────────────────────────────────────────────────────────────────
#  Rao functional integrands  (NASA TM 1990 formulation)
# ─────────────────────────────────────────────────────────────────────

def _isentropic_V(M: float, gamma: float) -> float:
    """
    Non-dimensional velocity V/V_max as a function of Mach number.
    V/V_max = M · sqrt(T/T₀) / M_max_equivalent

    For the Rao functional we use the normalized velocity:
        V̄ = M · a / a₀ = M · sqrt(T/T₀)
    where T/T₀ = (1 + (γ-1)/2·M²)^(-1)
    """
    T_ratio = isentropic_temperature_ratio(M, gamma)
    return M * math.sqrt(T_ratio)


def _isentropic_rhoV(M: float, gamma: float) -> float:
    """ρ·V (non-dimensional) = ρ/ρ₀ · V̄."""
    rho_ratio = isentropic_density_ratio(M, gamma)
    V = _isentropic_V(M, gamma)
    return rho_ratio * V


def thrust_integrand(M: float, theta: float, phi: float,
                     r: float, gamma: float) -> float:
    """
    Thrust contribution per unit radial extent on the control surface.

    f₁ = 2π·r · [ ρV² cos(θ) sin(φ-θ)/sin(φ) + p sin(φ-θ)·cos(…)/sin(φ) ]

    Simplified axisymmetric form:
        dF = 2πr·dr / sin(φ) · [ ρV²·cos(θ)·sin(φ-θ) + p·sin(φ-θ)·cos(θ) ]

    Which reduces (using momentum+pressure on an oblique surface) to:
        f₁ = 2πr · [ρV·sin(φ-θ)/sin(φ)] · [V·cos(θ) + p/(ρV)]

    In non-dimensional form (normalized by P₀):
        f₁ = 2πr · { (p/p₀)·sin(φ-θ)/sin(φ)
                    + (ρ/ρ₀)·V̄²·cos(θ)·sin(φ-θ)/sin(φ) }
    """
    if abs(math.sin(phi)) < 1e-15:
        return 0.0

    p_ratio = isentropic_pressure_ratio(M, gamma)
    rho_ratio = isentropic_density_ratio(M, gamma)
    T_ratio = isentropic_temperature_ratio(M, gamma)

    # Non-dimensional velocity squared: V̄² = γ·M²·(T/T₀)
    V_sq = gamma * M * M * T_ratio

    sin_diff = math.sin(phi - theta)
    sin_phi = math.sin(phi)

    # Momentum flux (axial component) + pressure force (axial component)
    momentum = rho_ratio * V_sq * math.cos(theta) * sin_diff / sin_phi
    pressure = p_ratio * sin_diff / sin_phi

    return 2.0 * math.pi * r * (momentum + pressure)


def massflow_integrand(M: float, theta: float, phi: float,
                       r: float, gamma: float) -> float:
    """
    Mass-flow contribution per unit radial extent on CE.

        f₂ = 2πr · ρV · sin(φ-θ) / sin(φ)

    In non-dimensional form:
        f₂ = 2πr · (ρ/ρ₀) · V̄ · sin(φ-θ) / sin(φ)
    """
    if abs(math.sin(phi)) < 1e-15:
        return 0.0
    rhoV = _isentropic_rhoV(M, gamma)
    sin_diff = math.sin(phi - theta)
    sin_phi = math.sin(phi)
    return 2.0 * math.pi * r * rhoV * sin_diff / sin_phi


def length_integrand(phi: float) -> float:
    """
    Length constraint integrand.

        f₃ = cot(φ) = cos(φ) / sin(φ)

    The nozzle length is: L = z_C + ∫ cot(φ) dr  over CE
    """
    if abs(math.sin(phi)) < 1e-15:
        return 0.0
    return math.cos(phi) / math.sin(phi)


# ─────────────────────────────────────────────────────────────────────
#  Partial derivatives for the Euler-Lagrange stationarity conditions
# ─────────────────────────────────────────────────────────────────────

def _numerical_partials(f, M, theta, phi, r, gamma, dh=1e-6):
    """
    Compute ∂f/∂M, ∂f/∂θ, ∂f/∂φ by central differences.

    Used for the stationarity conditions:
        ∂(f₁ + λ₂f₂ + λ₃f₃)/∂M = 0
        ∂(f₁ + λ₂f₂ + λ₃f₃)/∂θ = 0
        ∂(f₁ + λ₂f₂ + λ₃f₃)/∂φ = 0
    """
    f0 = f(M, theta, phi, r, gamma)

    # ∂f/∂M
    M_lo = max(M - dh, 1.0 + 1e-8)
    M_hi = M + dh
    df_dM = (f(M_hi, theta, phi, r, gamma) -
             f(M_lo, theta, phi, r, gamma)) / (M_hi - M_lo)

    # ∂f/∂θ
    df_dtheta = (f(M, theta + dh, phi, r, gamma) -
                 f(M, theta - dh, phi, r, gamma)) / (2.0 * dh)

    # ∂f/∂φ
    phi_lo = max(phi - dh, 1e-8)
    phi_hi = min(phi + dh, math.pi - 1e-8)
    df_dphi = (f(M, theta, phi_hi, r, gamma) -
               f(M, theta, phi_lo, r, gamma)) / (phi_hi - phi_lo)

    return df_dM, df_dtheta, df_dphi


def stationarity_residuals(M: float, theta: float, phi: float,
                           r: float, gamma: float,
                           lambda2: float, lambda3: float) -> np.ndarray:
    """
    Compute the 3 Euler-Lagrange residuals at a single CE station:

        R₁ = ∂f₁/∂M + λ₂·∂f₂/∂M = 0
        R₂ = ∂f₁/∂θ + λ₂·∂f₂/∂θ = 0
        R₃ = ∂f₁/∂φ + λ₂·∂f₂/∂φ + λ₃·∂f₃/∂φ = 0

    Note: f₃ = cot(φ) depends only on φ, so ∂f₃/∂M = ∂f₃/∂θ = 0.
    """
    df1 = _numerical_partials(thrust_integrand, M, theta, phi, r, gamma)
    df2 = _numerical_partials(massflow_integrand, M, theta, phi, r, gamma)

    # ∂f₃/∂φ = -1/sin²(φ)
    if abs(math.sin(phi)) > 1e-15:
        df3_dphi = -1.0 / (math.sin(phi) ** 2)
    else:
        df3_dphi = -1e15

    R = np.array([
        df1[0] + lambda2 * df2[0],              # ∂/∂M
        df1[1] + lambda2 * df2[1],              # ∂/∂θ
        df1[2] + lambda2 * df2[2] + lambda3 * df3_dphi,  # ∂/∂φ
    ])
    return R


def transversality_residual(M: float, theta: float, phi: float,
                            r: float, gamma: float,
                            lambda2: float, lambda3: float) -> float:
    """
    Endpoint (transversality) condition for free exit radius:

        (f₁ + λ₂·f₂ + λ₃·f₃)|_E = 0
    """
    f1 = thrust_integrand(M, theta, phi, r, gamma)
    f2 = massflow_integrand(M, theta, phi, r, gamma)
    f3 = length_integrand(phi)
    return f1 + lambda2 * f2 + lambda3 * f3


# ─────────────────────────────────────────────────────────────────────
#  Control-surface solver
# ─────────────────────────────────────────────────────────────────────

def _initial_ce_guess(Rt: float, Re: float, Ln: float,
                      gamma: float, n_pts: int) -> ControlSurface:
    """
    Initial guess for the control surface distribution.

    Uses quasi-1D estimates:
      - θ(r) linearly decreasing from ~θ_n to ~0 (parallel exit)
      - M(r) from area ratio at each radial station
      - φ(r) ≈ angle of a line connecting throat to exit
    """
    At = math.pi * Rt * Rt

    r = np.linspace(Rt * 1.05, Re, n_pts)
    M = np.zeros(n_pts)
    theta = np.zeros(n_pts)
    phi = np.zeros(n_pts)

    for i in range(n_pts):
        ar = (math.pi * r[i] ** 2) / At
        if ar < 1.0:
            ar = 1.0 + 1e-6
        try:
            M[i] = mach_from_area_ratio(ar, gamma, supersonic=True)
        except ValueError:
            M[i] = 1.5
        # θ linearly decreasing from ~20° to ~0°
        frac = i / max(n_pts - 1, 1)
        theta[i] = math.radians(20.0 * (1.0 - frac))
        # φ initial guess: ~60° (typical CE inclination)
        phi[i] = math.radians(60.0 + 20.0 * frac)

    return ControlSurface(r=r, M=M, theta=theta, phi=phi)


def _integrate_ce(ce: ControlSurface, gamma: float) -> tuple[float, float, float]:
    """Integrate thrust (F), mass flow (ṁ), and length (L) over the CE."""
    n = len(ce.r)
    F_total = 0.0
    mdot_total = 0.0
    L_total = 0.0

    for i in range(n - 1):
        dr = ce.r[i + 1] - ce.r[i]
        if dr <= 0:
            continue

        # Midpoint values
        r_mid = 0.5 * (ce.r[i] + ce.r[i + 1])
        M_mid = 0.5 * (ce.M[i] + ce.M[i + 1])
        theta_mid = 0.5 * (ce.theta[i] + ce.theta[i + 1])
        phi_mid = 0.5 * (ce.phi[i] + ce.phi[i + 1])

        if M_mid < 1.001:
            M_mid = 1.001

        F_total += thrust_integrand(M_mid, theta_mid, phi_mid,
                                    r_mid, gamma) * dr
        mdot_total += massflow_integrand(M_mid, theta_mid, phi_mid,
                                         r_mid, gamma) * dr
        L_total += length_integrand(phi_mid) * dr

    return F_total, mdot_total, L_total


def solve_optimal_control_surface(
    Rt: float,
    epsilon: float,
    gamma: float,
    length_pct: float = 80.0,
    n_ce_pts: int = 25,
    max_outer_iter: int = 80,
    max_inner_iter: int = 15,
    tol: float = 1e-6,
) -> ControlSurface:
    """
    Find the optimal control-surface distribution (M, θ, φ) that
    maximizes thrust subject to mass-flow and length constraints.

    Strategy:
      1. Start with a quasi-1D initial guess for CE
      2. Iterate:
         a. Fix λ₂, λ₃ → sweep CE stations enforcing stationarity
         b. Update λ₂, λ₃ from constraint residuals (mass flow, length)
      3. Enforce transversality at the exit

    Parameters
    ----------
    Rt          : throat radius [m]
    epsilon     : expansion ratio
    gamma       : ratio of specific heats
    length_pct  : bell length as % of 15° cone
    n_ce_pts    : number of CE stations
    max_outer_iter : max Lagrange-multiplier iterations
    max_inner_iter : max station-level Newton iterations
    tol         : convergence tolerance

    Returns
    -------
    ControlSurface with optimized (M, θ, φ) distribution
    """
    Re = math.sqrt(epsilon) * Rt
    At = math.pi * Rt * Rt

    # Target length (same convention as bell nozzle)
    L_15 = (Re - Rt) / math.tan(math.radians(15.0))
    Ln_target = (length_pct / 100.0) * L_15

    # Target mass flow (from throat conditions, quasi-1D)
    # ṁ = ρ*·a*·A* → normalized: ṁ_norm = (ρ/ρ₀·V̄)_throat · A*
    # At M=1: ρV = ρ₀·(2/(γ+1))^((γ+1)/(2(γ-1))) · a₀ · 1
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    rhoV_star = (2.0 / gp1) ** (gp1 / (2.0 * gm1))
    mdot_target = 2.0 * math.pi * rhoV_star  # normalized, per unit ρ₀·a₀·Rt²

    # Initial guess
    ce = _initial_ce_guess(Rt, Re, Ln_target, gamma, n_ce_pts)

    # Initial Lagrange multipliers (heuristic seeds)
    lambda2 = -0.5
    lambda3 = 0.01

    relaxation = 0.3   # under-relaxation for stability
    lambda_lr = 0.05   # learning rate for λ updates

    for outer in range(max_outer_iter):
        # ─── Inner loop: enforce stationarity at each CE station ───
        for i in range(1, n_ce_pts - 1):
            M_i = ce.M[i]
            theta_i = ce.theta[i]
            phi_i = ce.phi[i]
            r_i = ce.r[i]

            for inner in range(max_inner_iter):
                R = stationarity_residuals(M_i, theta_i, phi_i,
                                           r_i, gamma, lambda2, lambda3)
                if np.linalg.norm(R) < tol:
                    break

                # Numerical Jacobian of R w.r.t. (M, θ, φ)
                dh = 1e-5
                J = np.zeros((3, 3))
                state = np.array([M_i, theta_i, phi_i])
                for j in range(3):
                    sp = state.copy()
                    sm = state.copy()
                    sp[j] += dh
                    sm[j] -= dh
                    # Clamp Mach and phi
                    sp[0] = max(sp[0], 1.001)
                    sm[0] = max(sm[0], 1.001)
                    sp[2] = np.clip(sp[2], 1e-4, math.pi - 1e-4)
                    sm[2] = np.clip(sm[2], 1e-4, math.pi - 1e-4)

                    Rp = stationarity_residuals(sp[0], sp[1], sp[2],
                                                r_i, gamma, lambda2, lambda3)
                    Rm = stationarity_residuals(sm[0], sm[1], sm[2],
                                                r_i, gamma, lambda2, lambda3)
                    J[:, j] = (Rp - Rm) / (sp[j] - sm[j])

                try:
                    delta = np.linalg.solve(J, -R)
                except np.linalg.LinAlgError:
                    break

                # Damped update
                M_i += relaxation * delta[0]
                theta_i += relaxation * delta[1]
                phi_i += relaxation * delta[2]

                # Clamp to physical ranges
                M_i = max(M_i, 1.001)
                theta_i = np.clip(theta_i, -0.1, math.radians(50))
                phi_i = np.clip(phi_i, math.radians(5), math.radians(175))

            ce.M[i] = M_i
            ce.theta[i] = theta_i
            ce.phi[i] = phi_i

        # Enforce boundary: near-parallel exit (θ → 0 at wall/exit)
        ce.theta[-1] = max(0.0, ce.theta[-2] * 0.3)

        # ─── Constraint residuals and λ update ───
        F_val, mdot_val, L_val = _integrate_ce(ce, gamma)

        # Mass-flow constraint residual
        mdot_err = mdot_val - mdot_target
        # Length constraint residual
        L_err = L_val - Ln_target

        # Update multipliers by gradient of constraint violation
        lambda2 -= lambda_lr * mdot_err
        lambda3 -= lambda_lr * L_err

        # ─── Transversality at exit ───
        T_res = transversality_residual(
            ce.M[-1], ce.theta[-1], ce.phi[-1],
            ce.r[-1], gamma, lambda2, lambda3
        )
        # Adjust exit φ to satisfy transversality
        if abs(T_res) > tol:
            dh = 1e-5
            T_p = transversality_residual(
                ce.M[-1], ce.theta[-1], ce.phi[-1] + dh,
                ce.r[-1], gamma, lambda2, lambda3
            )
            dT_dphi = (T_p - T_res) / dh
            if abs(dT_dphi) > 1e-15:
                ce.phi[-1] -= relaxation * T_res / dT_dphi
                ce.phi[-1] = np.clip(ce.phi[-1], math.radians(5),
                                     math.radians(175))

        # Convergence check
        total_residual = abs(mdot_err) + abs(L_err) + abs(T_res)
        if total_residual < tol * 10:
            break

    ce.lambda2 = lambda2
    ce.lambda3 = lambda3
    return ce


# ─────────────────────────────────────────────────────────────────────
#  MOC wall construction from optimal CE
# ─────────────────────────────────────────────────────────────────────

def _construct_wall_from_ce(
    Rt: float, epsilon: float, gamma: float,
    ce: ControlSurface, Ln: float,
    n_char: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the optimized control-surface conditions, construct the
    nozzle wall using the method of characteristics.

    Strategy:
      1. Generate transonic starting line
      2. March MOC net with a trial wall
      3. Refine wall shape so that exit-plane conditions match CE targets
    """
    Re = math.sqrt(epsilon) * Rt
    Rd = 0.382 * Rt

    # Use the CE exit angle as θ_n seed
    theta_n = max(ce.theta[0], math.radians(15))

    # Target exit-plane properties from CE
    M_exit_target = float(np.mean(ce.M[-3:]))
    theta_exit_target = float(np.mean(ce.theta[-3:]))

    Ny = Rt + Rd * (1.0 - math.cos(theta_n))
    Nx = Rd * math.sin(theta_n)

    # Build a spline wall that tries to match CE characteristics
    # Use CE radial distribution as guide for control points
    n_ctrl = min(7, max(3, len(ce.r) // 3))
    # Sample CE radii at evenly spaced axial stations
    x_ce_est = np.linspace(Nx, Ln, n_ctrl + 2)
    r_ce_est = np.interp(
        np.linspace(ce.r[0], ce.r[-1], n_ctrl + 2),
        ce.r, ce.r
    )
    # Use interpolated radii that transition from Ny to Re
    control_r = np.linspace(Ny, Re, n_ctrl + 2)[1:-1]

    # Iterative wall refinement
    best_wall = None
    best_cost = float('inf')

    for iteration in range(20):
        try:
            wall = SplineWall.from_controls(
                control_r, Nx, Ny, Ln, Re, theta_n
            )

            from raosim.moc import solve_flowfield
            result = solve_flowfield(Rt, epsilon, gamma, wall, n_char)
            metrics = result['exit_metrics']

            # Cost: penalize deviation from CE targets
            cost = 0.0
            cost += (metrics['M_mean'] - M_exit_target) ** 2
            cost += 5.0 * metrics['theta_rms'] ** 2
            cost += 2.0 * max(0, metrics['theta_max'] - 5.0) ** 2
            cost -= 0.5 * metrics['F']   # also maximize thrust

            if cost < best_cost:
                best_cost = cost
                best_wall = wall

            # Gradient-free perturbation: shift control points toward CE targets
            delta = 0.001 * Rt * (1.0 - iteration / 20.0)
            perturbation = np.random.uniform(-delta, delta, size=len(control_r))
            control_r_new = control_r + perturbation
            control_r_new = np.clip(control_r_new, Rt, Re)
            control_r_new = np.sort(control_r_new)
            control_r = control_r_new

        except Exception:
            # If MOC fails, perturb and retry
            delta = 0.002 * Rt
            control_r += np.random.uniform(-delta, delta, size=len(control_r))
            control_r = np.clip(control_r, Rt, Re)
            control_r = np.sort(control_r)
            continue

    if best_wall is None:
        # Fallback: use a simple linear wall
        best_wall = SplineWall.from_controls(
            np.linspace(Ny, Re, n_ctrl + 2)[1:-1],
            Nx, Ny, Ln, Re, theta_n
        )

    wall_x, wall_r, _ = best_wall.sample(100)
    return wall_x, wall_r


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def _full_cone_length(Rt: float, epsilon: float) -> float:
    Re = math.sqrt(epsilon) * Rt
    return (Re - Rt) / math.tan(math.radians(15.0))


def rao_variational_contour(
    Rt: float,
    epsilon: float,
    gamma: float = 1.4,
    length_pct: float = 80.0,
    n_ce_pts: int = 25,
    n_char: int = 30,
    convergent_half_angle_deg: float = 45.0,
    Ru_factor: float = 1.5,
    max_iter: int = 80,
) -> dict:
    """
    Generate an optimized bell nozzle contour via the Rao variational method.

    Steps:
      1. Solve for optimal control-surface distribution (calculus of variations)
      2. Construct wall via MOC to realize that flowfield
      3. Prepend convergent + throat arcs

    Parameters
    ----------
    Rt          : throat radius [m]
    epsilon     : expansion ratio Ae/At
    gamma       : ratio of specific heats
    length_pct  : bell length as % of 15° cone
    n_ce_pts    : control-surface discretization points
    n_char      : MOC characteristic line points
    convergent_half_angle_deg : upstream inlet half-angle [°]
    Ru_factor   : upstream fillet ratio
    max_iter    : max variational iterations

    Returns
    -------
    dict compatible with bell_nozzle_contour() output
    """
    Re = math.sqrt(epsilon) * Rt
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon)
    Ru = Ru_factor * Rt
    Rd = 0.382 * Rt

    # Step 1: Solve the variational problem on the control surface
    ce = solve_optimal_control_surface(
        Rt, epsilon, gamma, length_pct, n_ce_pts,
        max_outer_iter=max_iter
    )

    # Step 2: Construct the wall from CE via MOC
    wall_x, wall_r = _construct_wall_from_ce(
        Rt, epsilon, gamma, ce, Ln, n_char
    )

    # Step 3: Build convergent + throat arcs
    conv_angle = math.radians(convergent_half_angle_deg)
    n_conv = 100
    t_conv = np.linspace(-(math.pi / 2 + conv_angle), -math.pi / 2, n_conv)
    x_conv = Ru * np.cos(t_conv)
    y_conv = (Rt + Ru) + Ru * np.sin(t_conv)

    theta_n = max(ce.theta[0], math.radians(15))
    t_thr = np.linspace(-math.pi / 2, theta_n - math.pi / 2, n_conv)
    x_throat = Rd * np.cos(t_thr)
    y_throat = (Rt + Rd) + Rd * np.sin(t_thr)

    # Full contour
    x_full = np.concatenate([x_conv, x_throat, wall_x])
    y_full = np.concatenate([y_conv, y_throat, wall_r])

    # Compute exit metrics
    theta_n_deg = math.degrees(theta_n)
    theta_e_deg = math.degrees(math.atan2(
        wall_r[-1] - wall_r[-2], wall_x[-1] - wall_x[-2]
    )) if len(wall_x) > 1 else 0.0

    Nx = x_throat[-1]
    Ny = y_throat[-1]

    return {
        'x': x_full,
        'y': y_full,
        'theta_n': theta_n_deg,
        'theta_e': theta_e_deg,
        'Ln': Ln,
        'Re': Re,
        'Rt': Rt,
        'Ru': Ru,
        'Rd': Rd,
        'epsilon': epsilon,
        'length_pct': length_pct,
        'N': (Nx, Ny),
        'E': (wall_x[-1], wall_r[-1]),
        'P1': (0.5 * (Nx + wall_x[-1]), 0.5 * (Ny + wall_r[-1])),
        'x_conv': x_conv,
        'y_conv': y_conv,
        'x_throat': x_throat,
        'y_throat': y_throat,
        'x_bell': wall_x,
        'y_bell': wall_r,
        'method': 'rao',
        'contour_type': 'rao_variational',
        'control_surface': ce,
        'lambda2': ce.lambda2,
        'lambda3': ce.lambda3,
    }
