#!/usr/bin/env python3
"""Gimbal-TVC actuator sizing study over LREKit-generated engine contours.

Consumes the ``contour.csv`` files written by ``python -m raosim`` (one per
thrust point), builds the 316L pressure-vessel shell, and evaluates the
per-axis gimbal torque budget and the actuator feasibility screen.

The three screens applied to every candidate actuator are, in order of how
often they are the binding one in practice:

  1. TORQUE     tau_avail = N * eta * tau_motor  >=  tau_required
  2. SPEED      omega_motor = N * omega_gimbal   <=  omega_motor_max
  3. INERTIA    N^2 * J_rotor                    <=  k * J_load
                (k = 10 open-loop stepper, 30 closed-loop stepper, 100 servo;
                 Oriental Motor / PMD published load-to-rotor inertia limits)

Screens 1 and 2 pull the gear ratio N in opposite directions, so they define a
feasible band in N.  Screen 3 caps the band from above independently.  Where the
band is empty the actuator cannot do the job at any reduction -- which is the
correct way to answer "can I use two NEMA 17s?", because the naive comparison
of motor holding torque against required gimbal torque ignores that gearing is
free to fix that.

Literature basis for the requirement side
-----------------------------------------
* Bandwidth 4 Hz with 20-25 deg phase lag at 1 Hz -- the SSME TVC requirement
  envelope, Cowan & Weir, "Design and Test of Electromechanical Actuators for
  Thrust Vector Control", NASA MSFC, NTRS 19940025147.
* Gimbal deflection +/-5 to +/-10 deg is the conventional design band.
* Reflected rotor inertia dominates TVC actuator sizing: "the amount of inertia
  created by the rest of the system is essentially negligible when compared to
  the inertia created by the cyclic action of each rotor inertia" -- ibid.
* Gimbal bearing friction torque is small relative to inertia and flex-line
  terms for rolling-element gimbals -- Neugebauer et al., "Bearing Development
  for a Rocket Engine Gimbal" (VINCI), 38th AMS / ESMATS 2006.
* 316L rho = 8000 kg/m^3, S_y = 290 MPa -- raosim/materials.py catalog entry.

Usage
-----
    python scripts/tvc_sizing_study.py --contours /tmp/tvc --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ------------------------------------------------------------------ constants

RHO_316L = 8000.0      # kg/m^3   raosim/materials.py "Stainless 316L"
SY_316L = 290.0e6      # Pa       raosim/materials.py "Stainless 316L"
G0 = 9.80665

# Hot-wall knockdown on yield.  316L retains roughly half of room-temperature
# yield in the 700-900 K band; combined with a 2.0 burst factor of safety this
# is the allowable used for the hoop-stress wall law below.
HOT_KNOCKDOWN = 0.50
FOS_PRESSURE = 2.0
T_WALL_MIN = 0.0010    # m, manufacturing floor (machining / L-PBF minimum)


# --------------------------------------------------------------- mass properties

@dataclass
class Geometry:
    thrust_N: float
    Pc_Pa: float
    r_throat: float
    r_exit: float
    r_chamber: float
    x_injector: float
    x_throat: float
    x_exit: float
    length: float
    t_wall: float
    shell_volume: float
    shell_mass: float
    x_cg: float
    pivot_x: float
    d_cg_pivot: float
    I_pivot: float
    I_cg: float


def load_contour(path: Path) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    with path.open() as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            pts.append((float(row[0]), float(row[1])))
    return pts


def hoop_thickness(Pc: float, r_chamber: float) -> float:
    """Thin-wall hoop-stress thickness with hot derate and FOS.

    sigma_hoop = Pc * r / t  ->  t = FOS * Pc * r / (knockdown * S_y)
    """
    sigma_allow = HOT_KNOCKDOWN * SY_316L
    return max(T_WALL_MIN, FOS_PRESSURE * Pc * r_chamber / sigma_allow)


def build_geometry(case_dir: Path, thrust_N: float, Pc: float,
                   pivot_mode: str = "throat") -> Geometry:
    pts = load_contour(case_dir / "contour.csv")
    r_throat = min(r for _, r in pts)
    x_throat = min(pts, key=lambda p: p[1])[0]
    x_inj, r_cham = pts[0]
    x_exit, r_exit = pts[-1]

    t = hoop_thickness(Pc, r_cham)

    m_tot = 0.0
    mx = 0.0
    elems = []
    for (x0, r0), (x1, r1) in zip(pts[:-1], pts[1:]):
        dx = x1 - x0
        if dx <= 0:
            continue
        r_i = 0.5 * (r0 + r1)
        r_o = r_i + t
        dm = RHO_316L * math.pi * (r_o**2 - r_i**2) * dx
        xc = 0.5 * (x0 + x1)
        m_tot += dm
        mx += dm * xc
        elems.append((dm, xc, r_i, r_o, dx))
    x_cg = mx / m_tot

    pivot_x = {"throat": x_throat, "injector": x_inj,
               "cg": x_cg, "exit": x_exit}[pivot_mode]

    def moi(about: float) -> float:
        return sum(
            dm * (r_o**2 + r_i**2) / 4.0 + dm * dx**2 / 12.0 + dm * (xc - about) ** 2
            for dm, xc, r_i, r_o, dx in elems
        )

    return Geometry(
        thrust_N=thrust_N, Pc_Pa=Pc,
        r_throat=r_throat, r_exit=r_exit, r_chamber=r_cham,
        x_injector=x_inj, x_throat=x_throat, x_exit=x_exit,
        length=x_exit - x_inj, t_wall=t,
        shell_volume=m_tot / RHO_316L, shell_mass=m_tot, x_cg=x_cg,
        pivot_x=pivot_x, d_cg_pivot=x_cg - pivot_x,
        I_pivot=moi(pivot_x), I_cg=moi(x_cg),
    )


# ------------------------------------------------------------------- torque budget

@dataclass
class Requirement:
    theta_max_deg: float = 5.0
    bandwidth_Hz: float = 4.0       # SSME TVC envelope, NTRS 19940025147
    assembly_factor: float = 2.2    # shell -> gimbaled assembly (wet)
    mu_bearing: float = 0.003       # rolling-element gimbal bearing
    misalign_deg: float = 0.5       # thrust-line misalignment about the pivot
    vehicle_g: float = 4.0          # axial acceleration for the unbalance term
    k_flex_frac: float = 0.25       # flex-line spring torque as frac of inertia
    margin: float = 2.0             # design margin on the sized actuator


@dataclass
class Budget:
    thrust_N: float
    gimbaled_mass: float
    I_pivot: float
    alpha: float
    omega_gimbal: float
    tau_inertia: float
    tau_flex: float
    tau_friction: float
    tau_misalign: float
    tau_unbalance: float
    tau_total: float
    tau_sized: float
    power_peak_W: float


def torque_budget(g: Geometry, req: Requirement) -> Budget:
    theta = math.radians(req.theta_max_deg)
    wc = 2.0 * math.pi * req.bandwidth_Hz
    alpha = theta * wc**2
    omega = theta * wc

    m = g.shell_mass * req.assembly_factor
    I = g.I_pivot * req.assembly_factor

    tau_J = I * alpha
    tau_k = req.k_flex_frac * tau_J
    tau_f = req.mu_bearing * g.thrust_N * (1.6 * g.r_throat)
    tau_m = g.thrust_N * abs(g.x_exit - g.pivot_x) * math.sin(math.radians(req.misalign_deg))
    tau_u = m * req.vehicle_g * G0 * abs(g.d_cg_pivot) * math.sin(theta)

    total = tau_J + tau_k + tau_f + tau_m + tau_u
    sized = total * req.margin
    # Peak mechanical power of the sinusoid: inertial term peaks a quarter
    # period away from the rate peak, so use the conservative product.
    power = sized * omega
    return Budget(
        thrust_N=g.thrust_N, gimbaled_mass=m, I_pivot=I, alpha=alpha,
        omega_gimbal=omega, tau_inertia=tau_J, tau_flex=tau_k,
        tau_friction=tau_f, tau_misalign=tau_m, tau_unbalance=tau_u,
        tau_total=total, tau_sized=sized, power_peak_W=power,
    )


# ------------------------------------------------------------------ actuator model

@dataclass
class Actuator:
    name: str
    kind: str                 # "stepper" | "servo" | "rc-servo" | "hydraulic"
    tau_cont_Nm: float        # usable continuous/dynamic shaft torque
    n_max_rpm: float          # usable speed before torque collapse
    J_rotor_kgm2: float
    inertia_ratio_limit: float  # allowable J_load / J_rotor
    price_usd: float
    note: str = ""

    @property
    def omega_max(self) -> float:
        return self.n_max_rpm * 2.0 * math.pi / 60.0

    @property
    def power_W(self) -> float:
        return self.tau_cont_Nm * self.omega_max


# Catalogue.  Stepper entries use *dynamic* torque at the stated speed, which is
# well below the catalogue holding torque -- holding torque is a standstill
# number and is not available while slewing.
ACTUATORS = [
    Actuator("NEMA 17 (17HS15-1504S, 45 Ncm)", "stepper",
             tau_cont_Nm=0.20, n_max_rpm=600, J_rotor_kgm2=5.4e-6,
             inertia_ratio_limit=10.0, price_usd=9.13,
             note="0.45 Nm holding; ~0.20 Nm at 600 rpm. Open loop."),
    Actuator("NEMA 17 long (17HS26-2304S, 79 Ncm)", "stepper",
             tau_cont_Nm=0.35, n_max_rpm=600, J_rotor_kgm2=1.4e-5,
             inertia_ratio_limit=10.0, price_usd=23.12,
             note="0.79 Nm holding, 42x42x67 mm, 2.3 A."),
    Actuator("NEMA 17 closed-loop + encoder", "stepper",
             tau_cont_Nm=0.35, n_max_rpm=1000, J_rotor_kgm2=1.4e-5,
             inertia_ratio_limit=30.0, price_usd=60.0,
             note="Encoder removes step loss; inertia limit 30x."),
    Actuator("NEMA 23 (1.9 Nm)", "stepper",
             tau_cont_Nm=0.90, n_max_rpm=800, J_rotor_kgm2=2.8e-5,
             inertia_ratio_limit=10.0, price_usd=35.0,
             note="1.9 Nm holding; ~0.9 Nm at 800 rpm."),
    Actuator("RC servo 50 kg-cm coreless (8.4 V)", "rc-servo",
             tau_cont_Nm=4.9, n_max_rpm=91, J_rotor_kgm2=0.0,
             inertia_ratio_limit=1e9, price_usd=45.0,
             note="0.11 s/135 deg -> 1227 deg/s no-load. Internal gearbox; "
                  "rotor inertia already reflected by the maker."),
    Actuator("Brushless RC servo M50BHW (14 V)", "rc-servo",
             tau_cont_Nm=4.9, n_max_rpm=111, J_rotor_kgm2=0.0,
             inertia_ratio_limit=1e9, price_usd=120.0,
             note="50 kg-cm at 0.09 s/60 deg -> 667 deg/s."),
    Actuator("BLDC 400 W + 20:1 planetary", "servo",
             tau_cont_Nm=1.27, n_max_rpm=3000, J_rotor_kgm2=1.5e-5,
             inertia_ratio_limit=100.0, price_usd=400.0,
             note="Motor-shaft torque before the 20:1 stage; closed loop."),
    Actuator("BLDC 1 kW servo + roller screw", "servo",
             tau_cont_Nm=3.18, n_max_rpm=3000, J_rotor_kgm2=6.0e-5,
             inertia_ratio_limit=100.0, price_usd=1200.0,
             note="Architecture of the NASA MSFC EMTVC actuator, scaled down."),
]


@dataclass
class Screen:
    actuator: str
    n_units: int
    N_min_torque: float       # smallest ratio that meets torque
    N_max_speed: float        # largest ratio that still meets rate
    N_max_inertia: float      # largest ratio the inertia limit allows
    N_upper: float
    feasible: bool
    binding: str
    N_pick: float | None
    margin_torque: float | None
    power_ratio: float        # required peak power / actuator power
    detail: str = ""


def screen_actuator(a: Actuator, b: Budget, n_units: int = 1,
                    eta: float = 0.85) -> Screen:
    """Feasible band in reduction ratio N (motor rad per gimbal rad)."""
    tau_motor = a.tau_cont_Nm * n_units
    J_rotor = a.J_rotor_kgm2 * n_units

    # 1. torque:  N * eta * tau_motor >= tau_sized
    N_torque = b.tau_sized / (eta * tau_motor)

    # 2. speed:   N * omega_gimbal <= omega_max
    N_speed = a.omega_max / b.omega_gimbal

    # 3. inertia: N^2 * J_rotor <= limit * J_load
    if J_rotor > 0.0:
        N_inertia = math.sqrt(a.inertia_ratio_limit * b.I_pivot / J_rotor)
    else:
        N_inertia = float("inf")

    N_upper = min(N_speed, N_inertia)
    feasible = N_torque <= N_upper

    if N_speed <= N_inertia:
        binding = "speed"
    else:
        binding = "reflected inertia"
    if feasible:
        # geometric mid-band pick
        N_pick = math.sqrt(N_torque * N_upper)
        margin = (N_pick * eta * tau_motor) / b.tau_sized
    else:
        N_pick, margin = None, None

    p_req = b.power_peak_W
    p_avail = a.tau_cont_Nm * a.omega_max * n_units * eta
    return Screen(
        actuator=a.name, n_units=n_units,
        N_min_torque=N_torque, N_max_speed=N_speed, N_max_inertia=N_inertia,
        N_upper=N_upper, feasible=feasible, binding=binding,
        N_pick=N_pick, margin_torque=margin,
        power_ratio=p_req / p_avail if p_avail > 0 else float("inf"),
        detail=a.note,
    )


# ------------------------------------------------------------------ print envelope

@dataclass
class PrintPlan:
    thrust_N: float
    scale: float
    length_mm: float
    exit_dia_mm: float
    chamber_dia_mm: float
    fits_bed_upright: bool
    fits_bed_diagonal: bool
    n_parts: int
    note: str


BED = 256.0     # mm, Bambu Lab P1S build volume 256 x 256 x 256
BED_USABLE = 240.0   # keep off the plate edges and the Z end-stop


def print_plan(g: Geometry, scale: float = 1.0) -> PrintPlan:
    """Envelope check against a P1S.

    Diagonal placement is reported but never recommended: a tall slender
    revolve laid on the build diagonal needs full-height support on one side,
    which destroys the surface the demonstrator exists to show.  Anything
    longer than the usable Z height is split at a bolted flange instead --
    which is also where a real engine has a joint.
    """
    L = g.length * 1000.0 * scale
    d_exit = 2.0 * (g.r_exit + g.t_wall) * 1000.0 * scale
    d_cham = 2.0 * (g.r_chamber + g.t_wall) * 1000.0 * scale
    d_max = max(d_exit, d_cham)
    diag = BED * math.sqrt(3.0)

    upright = L <= BED_USABLE and d_max <= BED_USABLE
    diagonal = (not upright) and L <= diag and d_max <= BED * 0.6

    if upright:
        n, note = 1, "single part, printed vertically nozzle-down"
    elif d_max > BED_USABLE:
        n = int(math.ceil(L / BED_USABLE))
        note = (f"exceeds bed in diameter ({d_max:.0f} mm) -- print at "
                f"{BED_USABLE/d_max:.2f}x scale, or sector-split")
    else:
        n = int(math.ceil(L / BED_USABLE))
        note = (f"split into {n} at a bolted flange"
                + (" (diagonal placement possible but support-heavy)"
                   if diagonal else ""))
    return PrintPlan(g.thrust_N, scale, L, d_exit, d_cham, upright, diagonal, n, note)


# ------------------------------------------------------------------------- driver

CASES = [
    # thrust N, Pc Pa
    (100, 3.0e6), (250, 3.0e6), (500, 3.0e6),
    (1000, 3.0e6), (2000, 3.0e6), (5000, 3.0e6), (10000, 3.0e6),
    (15000, 3.0e6), (25000, 3.0e6), (50000, 3.0e6), (100000, 3.0e6),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contours", default="/tmp/tvc",
                    help="directory holding F<thrust>/contour.csv")
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--bandwidth", type=float, default=4.0)
    args = ap.parse_args()

    root = Path(args.contours)
    req = Requirement(theta_max_deg=args.theta, bandwidth_Hz=args.bandwidth)

    out = {"requirement": asdict(req),
           "constants": {"rho_316L": RHO_316L, "Sy_316L": SY_316L,
                         "hot_knockdown": HOT_KNOCKDOWN,
                         "fos_pressure": FOS_PRESSURE, "t_min": T_WALL_MIN},
           "actuators": [asdict(a) for a in ACTUATORS],
           "cases": []}

    print(f"{'F[N]':>8} {'t[mm]':>6} {'Rt[mm]':>7} {'Dex[mm]':>8} {'L[mm]':>7} "
          f"{'m_sh[kg]':>9} {'m_gim':>7} {'I[kgm2]':>9} {'tau_s[Nm]':>10} {'P[W]':>7}")
    for F, Pc in CASES:
        d = root / f"F{F}"
        if not (d / "contour.csv").exists():
            continue
        g = build_geometry(d, F, Pc)
        b = torque_budget(g, req)
        screens = {}
        for a in ACTUATORS:
            for n in (1, 2):
                screens[f"{a.name} x{n}"] = asdict(screen_actuator(a, b, n))
        out["cases"].append({
            "geometry": asdict(g), "budget": asdict(b),
            "screens": screens, "print": asdict(print_plan(g)),
        })
        print(f"{F:>8} {g.t_wall*1e3:>6.2f} {g.r_throat*1e3:>7.2f} "
              f"{2*g.r_exit*1e3:>8.1f} {g.length*1e3:>7.1f} {g.shell_mass:>9.3f} "
              f"{b.gimbaled_mass:>7.2f} {b.I_pivot:>9.4f} {b.tau_sized:>10.2f} "
              f"{b.power_peak_W:>7.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
