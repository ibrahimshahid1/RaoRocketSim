#!/usr/bin/env python3
"""
rao_bell_toolbox.py – interactive early-design tool
• Solves ε from Pc/Pa (or vice-versa)
• Optimises θₙ, θₑ for axial thrust (Rao 1961)
• Generates and plots a reduced-length Rao bell
   (60 %, 80 % or 90 % of full length)
PyPI deps: numpy, matplotlib, requests
"""

import math, requests, numpy as np, matplotlib.pyplot as plt
from bisect import bisect_left
from datetime import datetime, timezone
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d import Axes3D   # noqa – needed for proj='3d'
from pathlib import Path
# ────────────────────────────────────────────
g0 = 9.80665
PROPS = {
    "n2o/ethanol": {"gamma": 1.22, "eta_Isp": 0.92, "default_OF": 5.5},
    "lox/kerosene": {"gamma": 1.23, "eta_Isp": 0.96, "default_OF": 2.6},
    "lox/lch4":    {"gamma": 1.24, "eta_Isp": 0.96, "default_OF": 3.5},
    "lox/lh2":     {"gamma": 1.20, "eta_Isp": 0.98, "default_OF": 6.0},
}
R_EARTH = 6371000.0              # m

# ───────────────────────────── standard atmosphere (simplified ISA)
def rho_atm(h):
    """Return air density [kg/m³] up to 86 km using a 3-segment ISA fit."""
    if h < 11000:                       # troposphere
        T = 288.15 - 0.0065*h
        p = 101325*(T/288.15)**5.256
    elif h < 20000:                     # lower stratosphere
        T = 216.65
        p = 22632*math.exp(-(h-11000)/(29.3*T))
    else:                               # very thin air
        T = 216.65 + 0.001*(h-20000)
        p = 5474*math.exp(-(h-20000)/(29.3*T))
    return p / (287.05*T)
# ───────────────────────────── helper math
def area(r): return math.pi * r * r


def area_mach_relation(M, g):
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M ** 2)) ** ((g + 1) / (2 * (g - 1)))

def mach_from_area(ar, g):
    lo, hi = 1.01, 50.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if area_mach_relation(mid, g) > ar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def eps_from_pressures(Pc, Pa, g):
    lo, hi = 1.01, 50.0
    def PePc(M): return (1 + 0.5 * (g - 1) * M ** 2) ** (-g / (g - 1))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if PePc(mid) > Pa / Pc:
            lo = mid
        else:
            hi = mid
    Me = 0.5 * (lo + hi)
    eps = area_mach_relation(Me, g)
    return eps, Me

def cf_ideal(Me, g, Pc, Pa, eps):
    PePc = (1 + 0.5 * (g - 1) * Me ** 2) ** (-g / (g - 1))
    return math.sqrt(
        (2 * g ** 2) / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) *
        (1 - PePc ** ((g - 1) / g))
    ) + (PePc - Pa / Pc) * eps

def cosine_loss(tn, te):     # Rao axial-loss approximation
    return math.cos(math.radians(tn + 2 * te) / 6)

def optimise_angles(eps, g, Pc, Pa):
    best = (0, 0, 0)
    Me = mach_from_area(eps, g)
    cf_i = cf_ideal(Me, g, Pc, Pa, eps)
    for tn in np.arange(15, 46, 0.25):
        for te in np.arange(4, 19, 0.25):
            cf_ax = cf_i * cosine_loss(tn, te)
            if cf_ax > best[0]:
                best = (cf_ax, tn, te)
    return best  # (Cf_ax, θn, θe)

def full_length(Rt, eps):                # 100 % Rao bell
    return (math.sqrt(eps) - 1) * Rt / math.tan(math.radians(15))

# ───────────────────────────── bell-contour generator
def bell_nozzle(Rt, eps, theta_n, theta_e, l_percent=80, n=120):
    """
    Return (Ln, θn, θe), (x, y) arrays for one side (upper half) of the bell.
    Symmetric reflections are trivial.
    """
    # —— 1. throat fillet (1.5 Rt, centre at 0, 2.5 Rt) ——
    ea = np.linspace(math.radians(-135), -math.pi/2, n)
    x1 = 1.5 * Rt * np.cos(ea)
    y1 = 1.5 * Rt * np.sin(ea) + 2.5 * Rt

    # —— 2. throat-to-bell fillet (0.382 Rt, centre at 0, 1.382 Rt) ——
    eb = np.linspace(-math.pi/2, math.radians(theta_n) - math.pi/2, n)
    x2 = 0.382 * Rt * np.cos(eb)
    y2 = 0.382 * Rt * np.sin(eb) + 1.382 * Rt

    # —— 3. quadratic Bézier bell section ——
    #   N = end of small fillet
    Nx, Ny = x2[-1], y2[-1]
    #   E = exit plane point
    Ln   = l_percent/100 * full_length(Rt, eps)
    Ex   = Ln
    Ey   = math.sqrt(eps) * Rt
    #   Q = intersection of the two straight-line tangents through N & E
    m1   = math.tan(math.radians(theta_n))
    m2   = math.tan(math.radians(theta_e))
    C1   = Ny - m1*Nx
    C2   = Ey - m2*Ex
    Qx   = (C2 - C1)/(m1 - m2)
    Qy   = (m1*C2 - m2*C1)/(m1 - m2)

    t = np.linspace(0, 1, n)
    x3 = (1-t)**2 * Nx + 2*(1-t)*t*Qx + t**2 * Ex
    y3 = (1-t)**2 * Ny + 2*(1-t)*t*Qy + t**2 * Ey

    # concatenate the three bits
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    return (Ln, theta_n, theta_e), (x, y)

# ───────────────────────────── plotting helpers
def ring(r, h, a=0, n_theta=32, n_height=4):
    θ, v = np.meshgrid(np.linspace(0, 2*np.pi, n_theta), np.linspace(a, a+h, n_height))
    return r*np.cos(θ), r*np.sin(θ), v

def set_axes_equal_3d(ax):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ori  = np.mean(lims, axis=1)
    rad  = 0.5 * np.max(np.abs(lims[:,1] - lims[:,0]))
    ax.set_xlim3d(ori[0]-rad, ori[0]+rad)
    ax.set_ylim3d(ori[1]-rad, ori[1]+rad)
    ax.set_zlim3d(ori[2]-rad, ori[2]+rad)

def plot_nozzle(Rt, eps, angles, contour):
    Ln, θn, θe = angles
    x, y  = contour
    plt.figure(figsize=(12,5))

    # --- 2-D view
    ax = plt.subplot(121)
    ax.set_aspect('equal')
    ax.plot(x,  y,  'b', lw=2.2)
    ax.plot(x, -y,  'b', lw=2.2)
    ax.axhline(0, color='k', lw=.4, ls='--')
    ax.axvline(0, color='k', lw=.4, ls='--')
    ax.grid(True, which='both', ls=':')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.title(f"Rao bell – ε={eps:.1f}, Ln={Ln*1000:.0f} mm, θn={θn:.1f}°, θe={θe:.1f}°")

    # annotate a few key radii
    ax.annotate(f"Rt={Rt*1000:.1f} mm", xy=(0,0), xytext=(0.05,0.05),
                textcoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=.7))

    # --- rudimentary 3-D pseudo-revolve
    ax3 = plt.subplot(122, projection='3d')
    for xi, yi in zip(x, y):
        X,Y,Z = ring(yi, 0.005, xi)          # 5 mm slices
        ax3.plot_surface(X,Y,Z, color='lightsteelblue', linewidth=0, antialiased=False)
    set_axes_equal_3d(ax3)
    ax3.view_init(elev=18, azim=-145)
    ax3.set_title("extruded half-theta view")
    ax3.set_axis_off()

    plt.tight_layout()
    plt.show()

# ───────────────────────────── local pressure helper
def get_local_pressure(fallback_hPa=1013):
    try:
        loc = requests.get("https://ipinfo.io/json", timeout=2).json()["loc"]
        lat, lon = map(float, loc.split(","))
        url = "https://api.open-meteo.com/v1/forecast"
        w = requests.get(url, params={"latitude":lat, "longitude":lon,
                                      "hourly":"pressure_msl"}, timeout=3).json()
        return np.mean(w["hourly"]["pressure_msl"]) * 100  # Pa
    except Exception:
        return fallback_hPa * 100
    

def simulate_vertical_ascent(m0, mp, thrust, Isp, Cd, A_ref, dt=0.01):
   
    m_dot  = thrust / (g0*Isp)
    t_burn = mp / m_dot
    m      = m0
    v      = 0.0     # velocity [m/s]
    h      = 0.0     # altitude [m]
    loss_g = 0.0
    loss_d = 0.0
    t      = 0.0

    while t < t_burn:
        rho = rho_atm(h)
        D   = 0.5*rho*v*v*Cd*A_ref
        a   = (thrust - D) / m - g0*(R_EARTH/(R_EARTH+h))**2
        loss_g += g0*dt
        loss_d += D/m*dt
        v += a*dt
        h += v*dt
        m -= m_dot*dt
        t += dt

    # coast to apogee after burnout
    while v > 0:
        rho = rho_atm(h)
        D   = 0.5*rho*v*v*Cd*A_ref
        a   = -(g0*(R_EARTH/(R_EARTH+h))**2) - D/m
        loss_g += g0*dt
        loss_d += D/m*dt
        v += a*dt
        h += v*dt
        t += dt

    ideal_dv = g0*Isp*math.log(m0/(m0-mp))
    dv_actual = math.sqrt(2*g0*h)                # energy-equivalent
    return dict(m_dot=m_dot, burn=t_burn, ideal_dv=ideal_dv,
                dv_actual=dv_actual, loss_g=loss_g, loss_d=loss_d,
                apogee=h)
   # ───────────────── 1-D single-stage point-mass integrator ─────────────
def simulate_stage(m0, mp, thrust, Isp, Cd, Aref, h0=0.0, v0=0.0, dt=0.02):
    m_dot  = thrust / (g0*Isp)
    t_burn = mp / m_dot
    m      = m0
    h, v   = h0, v0
    loss_g = loss_d = 0.0
    t      = 0.0
    while t < t_burn:
        rho = rho_atm(h)
        D   = 0.5*rho*v*v*Cd*Aref
        a   = (thrust - D)/m - g0*(R_EARTH/(R_EARTH+h))**2
        loss_g += g0*dt
        loss_d += D/m*dt
        v += a*dt
        h += v*dt
        m -= m_dot*dt
        t += dt
    while v > 0:                       # coast to peak for hand-off
        rho = rho_atm(h)
        D   = 0.5*rho*v*v*Cd*Aref
        a   = -(g0*(R_EARTH/(R_EARTH+h))**2) - D/m
        loss_g += g0*dt
        loss_d += D/m*dt
        v += a*dt
        h += v*dt
        t += dt
    ideal_dv  = g0*Isp*math.log(m0/(m0-mp))
    dv_actual = math.sqrt(max(0, 2*g0*(h-h0)))
    return dict(burn=t_burn, h=h, v=v, m=m,
                ideal_dv=ideal_dv, dv_actual=dv_actual,
                loss_g=loss_g, loss_d=loss_d)

# ───────────────── Ask user for one stage’s mass & aero data ───────────
def get_stage_inputs(num):
    print(f"\n── Stage {num} parameters ─────────────────────────────")
    engines = int(input(f"Engines on stage {num} (blank=1): ") or 1)
    mp      = float(input("Propellant mass [kg]          : ") or 0)
    mdry    = float(input("Dry mass [kg]                 : ") or 0)
    Cd      = float(input("Ballistic Cd (blank=0.45)     : ") or 0.45)
    dia     = float(input("Stage diameter [m] (blank=0.25): ") or 0.25)
    Aref    = math.pi*(dia/2)**2
    return engines, mp, mdry, Cd, Aref
             


# ───────────────────────────── CLI entry-point
def main():
    print("┌── Rao Bell Toolbox  –", datetime.now(timezone.utc).strftime('%F %T'), "UTC")
    print("│  leave blank for default                                 └─\n")

    # ── USER INPUTS (unchanged nozzle wizard) ──────────────────
    prop = input(f"Propellant {list(PROPS)} [n2o/ethanol] : ") or "n2o/ethanol"
    pdat = PROPS.get(prop.lower(), PROPS["n2o/ethanol"])

    γ  = float(input(f"γ (blank={pdat['gamma']}) : ") or pdat['gamma'])
    Of = float(input(f"O/F ratio (blank={pdat['default_OF']}) : ") or pdat['default_OF'])
    Pc_bar = float(input("Chamber pressure Pc [bar] (blank=45): ") or 45)
    Pc = Pc_bar*1e5
    Pa = float(input("Ambient pressure Pa [kPa] (0→auto): ") or 0)*1e3
    if Pa == 0:
        Pa = get_local_pressure()
        print(f"   ↳ local weather: {Pa/1000:.1f} kPa")

    Rt = float(input("Throat radius Rt [m] (blank=0.020): ") or 0.020)

    re_in = input("Exit radius Re [m] (blank=auto): ")
    if re_in.strip() == "":
        eps, Me = eps_from_pressures(Pc, Pa, γ)
        Re = math.sqrt(eps)*Rt
    else:
        Re  = float(re_in)
        eps = area(Re)/area(Rt)
        Me  = mach_from_area(eps, γ)

    print(f"   solved ε = {eps:.2f}  (Me={Me:.2f})")

    Cf_ax, θn, θe = optimise_angles(eps, γ, Pc, Pa)
    print(f"   optimum θn={θn:.1f}°, θe={θe:.1f}°, Cf(ax)={Cf_ax:.3f}")

    l_pct = int(input("Bell length [% of full, 60/80/90] (blank=80): ") or 80)

    # ── BUILD CONTOUR ─────────────────────────────────────────
    angle_tuple, (x, y) = bell_nozzle(Rt, eps, θn, θe, l_pct, n=201)
    Ln, *_ = angle_tuple
    print(f"   Rao {l_pct}% bell length = {Ln*1000:.1f} mm")

    # ── BASIC ENGINE PERFORMANCE ————————————————
    At= area(Rt)
    engine_thrust = Cf_ax * Pc * At
    Isp_sl        = Cf_ax * (math.sqrt(8314/22*3000)/g0) * pdat['eta_Isp']
    m_dot         = engine_thrust / (g0*Isp_sl)
    print(f"   Thrust ≈ {engine_thrust/1000:.1f} kN,  ṁ ≈ {m_dot:.2f} kg/s")


    # ── TRAJECTORY : get user data for TWO stages ───────────────────────────
    print("\nEnter vehicle data for both stages (leave blank to accept defaults).")
    s1 = get_stage_inputs(1)
    s2 = get_stage_inputs(2)

# scale thrust by engines
    thrust_s1 = engine_thrust * s1[0]
    thrust_s2 = engine_thrust * s2[0]

# Stage-1 carries Stage-2 dry-plus-prop mass as payload
    m0_s1 = s1[1] + s1[2] + s2[1] + s2[2]

    # ...existing code...

    res1  = simulate_stage(m0_s1, s1[1],
                       thrust_s1, Isp_sl,
                       s1[3], s1[4])

    # Stage-2 starts at Stage-1 burnout altitude & velocity
    m0_s2 = s2[1] + s2[2]
    # crude +15 s Isp vacuum bonus for upper bell
    res2  = simulate_stage(m0_s2, s2[1],
                       thrust_s2, Isp_sl + 15,
                       s2[3], s2[4],
                       h0=res1['h'], v0=res1['v'])

    # ── PRINT RESULTS ───────────────────────────────────────────────────────
    def show(n, d):
        print(f"\n Stage {n}")
        print(f"   Burn time ........... {d['burn']:.2f} s")
        print(f"   Ideal Δv ............ {d['ideal_dv']/1000:.2f} km/s")
        print(f"   Actual Δv ........... {d['dv_actual']/1000:.2f} km/s")
        print(f"   Gravity loss ........ {d['loss_g']:.0f} m/s")
        print(f"   Drag loss ........... {d['loss_d']:.0f} m/s")
        print(f"   Burn-out altitude ... {d['h']/1000:.1f} km")

    show(1, res1)
    show(2, res2)

    print("\n── Stack summary ──────────────────────────────────────")
    print(f"   Apogee .............. {res2['h']/1000:.1f} km")
    print(f"   Stack Δv (ideal) .... {(res1['ideal_dv']+res2['ideal_dv'])/1000:.2f} km/s")
    print(f"   Stack Δv (actual) ... {(res1['dv_actual']+res2['dv_actual'])/1000:.2f} km/s\n")

    # ── CSV export & plot exactly as before ──────────────────
    n_csv = int(input("Number of CSV points (blank=301): ") or 301)
    idx   = np.linspace(0, len(x)-1, n_csv).astype(int)
    xv, yv = x[idx], y[idx]
    csv_name = input("CSV file name (blank=rao_nozzle_profile.csv): ") or "rao_nozzle_profile.csv"
    csv_path = Path(csv_name).expanduser().resolve()
    with open(csv_path, "w") as f:
        f.write("x_m,y_m\n")
        for xi, yi in zip(xv, yv):
            f.write(f"{xi:.6e},{yi:.6e}\n")
    print(f"   → wrote {n_csv} points to '{csv_path}'")

    plot_nozzle(Rt, eps, angle_tuple, (x, y))
    

# ...existing code...
if __name__ == "__main__":
    main()



