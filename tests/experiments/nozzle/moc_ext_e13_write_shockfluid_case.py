"""E13 (prototype pending user decision D1): write-only OpenFOAM Foundation v13
foamRun/shockFluid three-mesh wedge case set for the accepted Nozzle-B contour.

Never runs OpenFOAM or any other process; writes text files and runs static
checks only.  Not library code.

.venv-jax/bin/python tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import time

import numpy as np

ROOT = str(_repository_root())
PLAN = os.path.join(ROOT, "builds", "moc_extensions_plan_2026-11-21")
OUT_DEFAULT = os.path.join(PLAN, "opus_b", "e13")
T19_PKL_DIR = os.path.join(ROOT, "builds", "moc_audit_2026-11-19", "sol_b", "t19_refinement")
T19_EVID_DIR = os.path.join(ROOT, "docs", "subsystems", "S06-rao-moc-bvp-contour", "audit_2026-11-19",
                            "evidence", "validation", "t19_refinement")

STATUS_LABEL = "prototype pending user decision D1"
TEMPLATE_ID = "raosim.moc_ext.e13.shockfluid_nozzle_wedge"
TEMPLATE_VERSION = 0
MANIFEST_NAME = "raosim_shockfluid_manifest.json"
OF = dict(distribution="OpenFOAM Foundation", major_version=13, tag="version-13",
          commit="65eda17ad4caa916bfd0606039db92509670f9f7", runner="foamRun", module="shockFluid",
          source_url="https://github.com/OpenFOAM/OpenFOAM-13/tree/version-13", bib=["openfoam13", "openfoam13src"])
R_UNIVERSAL = 8314.46261815324

DEFAULTS = dict(
    wedge_total_deg=5.0, turn_limit_deg=0.125, convergent_half_angle_deg=45.0, contraction_ratio=2.5,
    coarse_dx_rt=0.01, coarse_nr=32, size_ratio_limit=1.25,
    p0_pa=1.0e6, T0_k=3000.0, mol_weight=28.96, p_init_over_p0=0.02, n_flow_through=10.0,
    rt_m=1.0, decompose_subdomains=(4, 8, 16), source="refined",
)
LEVELS = (("L1_coarse", 1, None), ("L2_medium", 2, None), ("L3_fine", 4, None), ("L1_coarse_wedge0p2", 1, 0.2))
V13_REFS_DIR = os.path.join(OUT_DEFAULT, "v13_refs")
PATCHES = ("inlet", "outlet", "wall", "wedgeFront", "wedgeBack")
INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"

# Evidence classes for each dictionary entry: V13_TUTORIAL = read in
# tutorials/shockFluid/LadenburgJet60psi at tag version-13 (scout4 W26, scout5 W3);
# V13_HEADER = read in a v13 source header (scout5 W2/W4); V13_SOURCE = read in a
# v13 .C file; V13_TREE = class exists in the v13 tree listing but its usage was
# not read; UNVERIFIED = not confirmed from v13 source or documentation.
ENTRY_EVIDENCE = [
    ("system/controlDict", "solver shockFluid", "V13_TUTORIAL", "scout5 W3"),
    ("system/controlDict", "startFrom/startTime/stopAt/endTime/deltaT/writeControl/writeInterval/cycleWrite/"
     "writeFormat/writePrecision/writeCompression/timeFormat/timePrecision/adjustTimeStep/maxCo/maxDeltaT",
     "V13_TUTORIAL", "scout5 W3 (LadenburgJet60psi controlDict)"),
    ("system/controlDict", "numeric values of endTime/writeInterval/deltaT", "UNVERIFIED",
     "prototype choice from 1-D flow-through estimate; stationarity must be checked by the user"),
    ("system/fvSchemes", "fluxScheme Kurganov", "V13_SOURCE", "shockFluid.C lookupOrDefault fluxScheme (scout5 W2)"),
    ("system/fvSchemes", "ddt Euler; grad Gauss linear; div none; laplacian Gauss linear corrected; "
     "interpolation linear + reconstruct(rho) vanLeer, reconstruct(U) vanLeerV, reconstruct(T) vanLeer; "
     "snGrad corrected", "V13_TUTORIAL", "scout5 W2/W3"),
    ("system/fvSolution", "solvers (rho|rhoU|rhoE).* diagonal; U.* smoothSolver GaussSeidel; e.* $U; PIMPLE {}",
     "V13_TUTORIAL", "scout5 W2/W3 (L_system_fvSolution)"),
    ("system/blockMeshDict", "convertToMeters; vertices; blocks hex (...) (nx nr 1) simpleGrading (1 1 1); "
     "collapsed axis vertices; boundary patch/wedge; defaultPatch type empty", "V13_TUTORIAL",
     "scout4 downloads/openfoam13/system_blockMeshDict"),
    ("system/blockMeshDict", "patch face vertex order: same pattern as the tutorial (inlet face normal into the "
     "block, all other patch faces outward)", "V13_TUTORIAL", "computed from both vertex lists"),
    ("system/blockMeshDict", "boundary type wall", "V13_TREE",
     "src/OpenFOAM/meshes/polyMesh/polyPatches/derived/wall/wallPolyPatch.H in tree listing; usage not read"),
    ("system/blockMeshDict", "blockMesh places interior points of a straight-edged uniformly graded block by "
     "trilinear interpolation (assumed by the static cell-volume check)", "UNVERIFIED", "blockMesh source not read"),
    ("constant/physicalProperties", "hePsiThermo, pureMixture, sutherland, hConst, perfectGas, specie, "
     "sensibleInternalEnergy; molWeight; Cp; hf; As; Ts; Pr", "V13_TUTORIAL", "scout5 W3"),
    ("constant/physicalProperties", "inviscid path selected iff max(mu) == 0", "V13_SOURCE",
     "shockFluid.C lines 148-150 (scout5 W2, verify_scout5 W2)"),
    ("constant/physicalProperties", "sutherland with As 0 gives mu identically 0", "UNVERIFIED",
     "sutherlandTransport source not read at the tag; runtime inviscid path not exercised (scout5 open question)"),
    ("constant/physicalProperties", "OpenFOAM universal gas constant equals 8314.46261815324 J/(kmol K)",
     "UNVERIFIED", "used only to derive Cp from gamma; |dR/R| <= 1e-5 changes gamma by < 3e-6"),
    ("constant/momentumTransport", "simulationType laminar", "V13_TUTORIAL", "scout5 W3; not constructed when inviscid"),
    ("0/p", "totalPressure: rho none; psi psi; gamma; p0; value (isentropic 'supersonic' law for psi with gamma > 1)",
     "V13_HEADER", "totalPressureFvPatchScalarField.H (scout5 W4); mode mapping from header table, .C selection "
     "logic not read"),
    ("0/p", "waveTransmissive: field p; psi psi; gamma; value; fieldInf/lInf omitted (optional)", "V13_HEADER",
     "waveTransmissiveFvPatchField.H + advectiveFvPatchField.H (scout5 W4); 'field' entry as in tutorial"),
    ("0/T", "totalTemperature: gamma; T0; value", "V13_HEADER", "totalTemperatureFvPatchScalarField.H (scout5 W4)"),
    ("0/U", "inletOutlet: inletValue; value", "V13_TUTORIAL", "LadenburgJet60psi 0/U outlet"),
    ("0/U", "zeroGradient on a totalPressure inflow patch", "V13_TUTORIAL", "LadenburgJet60psi 0/U freestream"),
    ("0/U", "slip (no entries) on the wall", "V13_TREE",
     "src/finiteVolume/fields/fvPatchFields/derived/slip/slipFvPatchField.H in tree listing; header not read"),
    ("0/*", "wedge (no entries) on paired wedge patches; empty on defaultFaces", "V13_HEADER",
     "wedgeFvPatchField.H / wedgePolyPatch.H (scout5 W4); defaultFaces empty as in tutorial"),
    ("Allrun", "RunFunctions/runApplication, blockMesh, checkMesh -allTopology -allGeometry, foamRun",
     "UNVERIFIED", "repository convention raosim/openfoam/case.py; tutorial Allrun not read"),
    ("Allrun", "WM_PROJECT_VERSION environment check", "UNVERIFIED", "environment convention, not read at the tag"),
    ("system/decomposeParDict", "numberOfSubdomains; method scotch", "UNVERIFIED",
     "decomposePar exists in the v13 tree; dictionary syntax not read"),
    ("system/functions", "file read without a controlDict reference; #includeFunc <template>(name=..., key=value, ...) "
     "argument syntax", "V13_TUTORIAL", "tutorials/shockFluid/biconic25-55Run35 and incompressibleFluid/pitzDailySteady "
     "system/functions fetched at tag version-13 (e13/v13_refs)"),
    ("system/functions", "patchFlowRate(patch): sum of phi, a mass flow because shockFluid phi = interpolate(rho U).Sf",
     "V13_ETC_TEMPLATE", "etc/caseDicts/functions/surfaceFieldValue/patchFlowRate at tag; shockFluid.C phi_ constructor"),
    ("system/functions", "forcesCompressible(patches, CofR, pitchAxis)", "V13_ETC_TEMPLATE",
     "etc/caseDicts/functions/forces/forcesCompressible and forces.cfg at tag"),
    ("system/functions", "forces pressure reference (pRef default) for the absolute wall pressure force", "UNVERIFIED",
     "forces source not read; vacuum CF needs the absolute-pressure wall integral"),
    ("system/functions", "graphUniform(start, end, nPoints, fields, axis); patchSurface(patch, fields, interpolate)",
     "V13_ETC_TEMPLATE", "etc/caseDicts/functions/graphs/graphUniform(.cfg), surface/patchSurface(.cfg) at tag"),
    ("system/functions", "MachNo without arguments", "V13_TUTORIAL", "tutorials/fluid/aerofoilNACA0012Steady/system/functions"),
    ("run_controls", "shockFluid Courant number Co = 0.5 max(sum(amaxSf)/V) dt", "V13_SOURCE", "shockFluid.C lines 50-55"),
    ("run_controls", "wedge-patch faces contribute c|Sf| to fvc::surfaceSum(amaxSf) (upper time-step-count estimate)",
     "UNVERIFIED", "fvc::surfaceSum and wedge amaxSf boundary values not read"),
    ("run_slurm.sbatch", "Slurm template and parallel commands (decomposePar, foamRun -parallel, reconstructPar)",
     "UNVERIFIED", "site-specific template; not read at the tag"),
]

BC_REQUIRED = {"totalPressure": {"p0"}, "totalTemperature": {"gamma", "T0"}, "waveTransmissive": {"gamma"},
               "inletOutlet": {"inletValue"}, "fixedValue": {"value"}, "zeroGradient": set(), "slip": set(),
               "wedge": set(), "empty": set()}
CONSTRAINT_TYPES = {"wedge", "empty"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def git_head_noproc():
    gd = os.path.join(ROOT, ".git")
    try:
        head = open(os.path.join(gd, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        p = os.path.join(gd, ref)
        if os.path.exists(p):
            return open(p).read().strip()
        for line in open(os.path.join(gd, "packed-refs")):
            if line.strip().endswith(ref):
                return line.split()[0]
    except OSError as exc:
        return f"unavailable: {exc}"
    return "unavailable"


def fmt(v):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("non-finite scalar")
    if v == 0.0:
        return "0"
    return format(v, ".15g")


def foam_header(obj, cls, location=None):
    loc = "" if location is None else f'    location    "{location}";\n'
    return ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "  =========                 |\n"
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
            "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
            "    \\\\  /    A nd           | Version:  13\n"
            "     \\\\/     M anipulation  |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n{\n    format      ascii;\n"
            f"    class       {cls};\n{loc}    object      {obj};\n}}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")


FOOTER = "\n// ************************************************************************* //\n"


# --------------------------------------------------------------------------- geometry
def load_source(kind):
    report_path = os.path.join(T19_EVID_DIR, "report.json")
    report = json.load(open(report_path))
    pkl = os.path.join(T19_PKL_DIR, f"{kind}_solution.pkl")
    npz = os.path.join(T19_EVID_DIR, f"{kind}_field.npz")
    pkl_sha = sha256_file(pkl)
    npz_sha = sha256_file(npz)
    if pkl_sha != report["artifact_sha256"][f"{kind}_solution.pkl"]:
        raise RuntimeError("solution pickle sha256 does not match report.json")
    if npz_sha != report["artifact_sha256"][f"{kind}_field.npz"]:
        raise RuntimeError("field npz sha256 does not match report.json")
    sys.path.insert(0, ROOT)
    with open(pkl, "rb") as f:
        sol = pickle.load(f)
    topo = sol.topology_solved
    cfg = report[kind]["config"]
    return dict(
        wall_raw=np.asarray(sol.wall_raw, dtype=float), theta_B=float(topo.theta_B),
        B=(float(topo.B.x), float(topo.B.r)), E=(float(topo.E.x), float(topo.E.r)),
        E_state=dict(M=float(topo.E.M), theta_rad=float(topo.E.theta)),
        B_state=dict(M=float(topo.B.M), theta_rad=float(topo.B.theta)),
        D=(float(topo.D.x), float(topo.D.r)), thrust_coefficient=float(sol.thrust_coefficient),
        Rt=float(cfg["Rt"]), Rd=float(cfg["throat_downstream_radius_factor"]),
        Ru=float(cfg["throat_upstream_radius_factor"]), gamma=float(cfg["gamma"]), epsilon=float(cfg["epsilon"]),
        config=cfg, kind=kind,
        files={os.path.relpath(pkl, ROOT): pkl_sha, os.path.relpath(npz, ROOT): npz_sha,
               os.path.relpath(report_path, ROOT): sha256_file(report_path)},
    )


def chord_turn_deg(x, r):
    ang = np.arctan2(np.diff(r), np.diff(x))
    return np.degrees(np.abs(np.diff(ang)))


def build_wall(src, p):
    """Wall polyline stations (x, r, section) inlet to exit, one coarse axial cell per block.

    Segment length h(s) = min(coarse_dx, 0.9 turn_limit / |curvature|), graded so that
    |dh/ds| <= size_ratio_limit - 1 - 0.05, then equidistributed separately upstream and
    downstream of the throat station x = 0."""
    from scipy.interpolate import PchipInterpolator

    Rt, Rd, Ru = src["Rt"], src["Rd"] * src["Rt"], src["Ru"] * src["Rt"]
    dlt = math.radians(p["turn_limit_deg"])
    a = math.radians(p["convergent_half_angle_deg"])
    Rc = Rt * math.sqrt(p["contraction_ratio"])
    xa, ra = -Ru * math.sin(a), Rt + Ru * (1.0 - math.cos(a))
    if Rc <= ra:
        raise ValueError("contraction ratio too small for the convergent arc")
    x_in = xa - (Rc - ra) / math.tan(a)
    tb = src["theta_B"]
    w = src["wall_raw"]
    xu, iu = np.unique(w[:, 0], return_index=True)
    pch = PchipInterpolator(xu, w[iu, 1], extrapolate=False)
    xB, xE = Rd * math.sin(tb), src["E"][0]
    ds = 2.0e-5 * Rt

    def cone(t):
        return x_in + t * (xa - x_in), Rc + t * (ra - Rc)

    def up(ph):
        return Ru * np.sin(ph), Rt + Ru * (1.0 - np.cos(ph))

    def down(ph):
        return Rd * np.sin(ph), Rt + Rd * (1.0 - np.cos(ph))

    def rao(x):
        return x, pch(x)

    sections = [("cone", cone, 0.0, 1.0, math.hypot(xa - x_in, ra - Rc), 0.0),
                ("upstream_arc", up, -a, 0.0, Ru * a, 1.0 / Ru),
                ("downstream_arc", down, 0.0, tb, Rd * tb, 1.0 / Rd),
                ("rao_BE", rao, xB, xE, None, None)]
    S, PAR, SEC, KAP, XX, RR = [], [], [], [], [], []
    s0 = 0.0
    for si, (name, fn, t0, t1, length, kap) in enumerate(sections):
        if name == "rao_BE":
            xs = np.linspace(t0, t1, int(math.ceil((t1 - t0) / (0.5 * ds))) + 1)
            rs = pch(xs)
            seg = np.hypot(np.diff(xs), np.diff(rs))
            s_loc = np.concatenate([[0.0], np.cumsum(seg)])
            phi = np.arctan(pch(xs, 1))
            kp = np.abs(np.gradient(phi, s_loc))
            par = xs
        else:
            n = int(math.ceil(length / ds)) + 1
            par = np.linspace(t0, t1, n)
            xs, rs = fn(par)
            xs, rs = np.asarray(xs, float), np.asarray(rs, float)
            s_loc = np.linspace(0.0, length, n)
            kp = np.full(n, kap)
        if si > 0:
            par, xs, rs, s_loc, kp = par[1:], xs[1:], rs[1:], s_loc[1:], kp[1:]
        S.append(s0 + s_loc)
        PAR.append(par)
        SEC.append(np.full(par.size, si))
        KAP.append(kp)
        XX.append(xs)
        RR.append(rs)
        s0 = S[-1][-1]
    S, PAR, SEC, KAP, XX, RR = (np.concatenate(v) for v in (S, PAR, SEC, KAP, XX, RR))
    h = np.minimum(p["coarse_dx_rt"] * Rt, 0.9 * dlt / np.maximum(KAP, 1e-12))
    g = p["size_ratio_limit"] - 1.0 - 0.05
    dS = np.diff(S)
    for i in range(1, h.size):
        h[i] = min(h[i], h[i - 1] + g * dS[i - 1])
    for i in range(h.size - 2, -1, -1):
        h[i] = min(h[i], h[i + 1] + g * dS[i])
    i_throat = int(np.nonzero((SEC == 2))[0][0]) - 1
    if abs(XX[i_throat]) > 1e-12 or abs(RR[i_throat] - Rt) > 1e-12:
        raise RuntimeError("throat sample not found")
    stations_s = []
    for lo, hi in ((0, i_throat), (i_throat, S.size - 1)):
        seg_s = S[lo:hi + 1]
        inv = 1.0 / h[lo:hi + 1]
        xi = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(seg_s))])
        n_seg = int(math.ceil(xi[-1] - 1e-9))
        targets = np.linspace(0.0, xi[-1], n_seg + 1)
        st = np.interp(targets, xi, seg_s)
        st[0], st[-1] = seg_s[0], seg_s[-1]
        stations_s.append(st if lo == 0 else st[1:])
    st = np.concatenate(stations_s)
    stations = []
    for sv in st:
        j0 = int(np.clip(np.searchsorted(S, sv, side="right") - 1, 0, S.size - 2))
        j1 = j0 + 1
        f = (sv - S[j0]) / (S[j1] - S[j0])
        si = int(SEC[j1])
        p0 = PAR[j0] if SEC[j0] == si else sections[si][2]
        name, fn = sections[si][0], sections[si][1]
        x, r = fn(p0 + f * (PAR[j1] - p0))
        stations.append((float(x), float(r), name))
    stations[-1] = (float(xE), float(pch(xE)), "rao_BE")
    X = np.array([s[0] for s in stations])
    Rr = np.array([s[1] for s in stations])
    sec = [s[2] for s in stations]
    if np.any(np.diff(X) <= 0):
        raise RuntimeError("non-monotone stations")
    t = chord_turn_deg(X, Rr)
    L = np.hypot(np.diff(X), np.diff(Rr))
    ratio = np.maximum(L[1:] / L[:-1], L[:-1] / L[1:])
    arc_x = Rd * np.sin(np.linspace(0, tb, 2001))
    arc_r = Rt + Rd * (1 - np.cos(np.linspace(0, tb, 2001)))
    i_B = int(np.argmin(np.abs(X - xB)))
    diag = dict(
        n_stations=len(stations), n_blocks=len(stations) - 1,
        section_counts={k: sec.count(k) for k in ("cone", "upstream_arc", "downstream_arc", "rao_BE")},
        x_inlet=float(X[0]), r_inlet=float(Rr[0]), x_exit=float(X[-1]), r_exit=float(Rr[-1]),
        throat_is_station=bool(np.any((np.abs(X) < 1e-12) & (np.abs(Rr - Rt) < 1e-12))),
        max_chord_turn_deg=float(t.max()), turn_limit_deg=p["turn_limit_deg"],
        max_chord_turn_near_B_deg=float(t[max(i_B - 3, 0):i_B + 2].max()),
        segment_length_min_max_rt=[float(L.min() / Rt), float(L.max() / Rt)],
        max_adjacent_segment_ratio=float(ratio.max()), segment_ratio_limit=p["size_ratio_limit"],
        pchip_slope_at_B_minus_tan_thetaB=float(pch(xB, 1) - math.tan(tb)),
        pchip_vs_analytic_arc_max_abs=float(np.nanmax(np.abs(pch(arc_x) - arc_r))),
        exit_radius_vs_target=float(Rr[-1] - src["E"][1]),
        area_ratio_exit=float((Rr[-1] / Rt) ** 2),
        contraction_area_ratio=float((Rr[0] / Rt) ** 2),
        convergent_construction=("straight cone at convergent half-angle tangent to the upstream arc Ru, "
                                 "from inlet radius sqrt(contraction_ratio) Rt"),
        divergent_construction=("analytic downstream arc Rd from 0 to theta_B, then PCHIP through the "
                                "solved raw wall nodes (repository export basis) from B to E"),
        station_rule=("segment length h(s) = min(coarse_dx, 0.9 turn_limit/|curvature|), Lipschitz-graded with "
                      "slope size_ratio_limit - 1.05, equidistributed on each side of the throat; one coarse "
                      "axial cell per block"),
    )
    diag["polyline_turn_ok"] = bool(diag["max_chord_turn_deg"] <= p["turn_limit_deg"] + 1e-9)
    diag["segment_ratio_ok"] = bool(diag["max_adjacent_segment_ratio"] <= p["size_ratio_limit"] + 1e-9)
    return X * p["rt_m"], Rr * p["rt_m"], sec, diag


# --------------------------------------------------------------------------- rendering
def vertices_of(X, R, half):
    V = []
    for x, r in zip(X, R):
        V.append((x, 0.0, 0.0))
        V.append((x, r * math.cos(half), -r * math.sin(half)))
        V.append((x, r * math.cos(half), r * math.sin(half)))
    return V


def blocks_of(N):
    out = []
    for k in range(N):
        a0, a1 = 3 * k, 3 * (k + 1)
        out.append((a0, a1, a1 + 1, a0 + 1, a0, a1, a1 + 2, a0 + 2))
    return out


def faces_of(N):
    wall = [(3 * k + 1, 3 * k + 2, 3 * (k + 1) + 2, 3 * (k + 1) + 1) for k in range(N)]
    front = [(3 * k, 3 * k + 1, 3 * (k + 1) + 1, 3 * (k + 1)) for k in range(N)]
    back = [(3 * k, 3 * (k + 1), 3 * (k + 1) + 2, 3 * k + 2) for k in range(N)]
    inlet = [(0, 1, 2, 0)]
    outlet = [(3 * N, 3 * N + 1, 3 * N + 2, 3 * N)]
    return dict(inlet=inlet, outlet=outlet, wall=wall, wedgeFront=front, wedgeBack=back)


def render_block_mesh(V, B, nx, nr, F):
    vl = "\n".join(f"    ({fmt(x)} {fmt(y)} {fmt(z)})" for x, y, z in V)
    bl = "\n".join(f"    hex ({' '.join(str(i) for i in b)}) ({n} {nr} 1) simpleGrading (1 1 1)"
                   for b, n in zip(B, nx))
    types = dict(inlet="patch", outlet="patch", wall="wall", wedgeFront="wedge", wedgeBack="wedge")

    def patch(name):
        fl = "\n".join(f"            ({' '.join(str(i) for i in f)})" for f in F[name])
        return f"    {name}\n    {{\n        type {types[name]};\n        faces\n        (\n{fl}\n        );\n    }}\n"
    bnd = "\n".join(patch(nm) for nm in PATCHES)
    return (foam_header("blockMeshDict", "dictionary") + "convertToMeters 1;\n\nvertices\n(\n" + vl + "\n);\n\n"
            "blocks\n(\n" + bl + "\n);\n\ndefaultPatch\n{\n    type empty;\n}\n\nboundary\n(\n" + bnd + ");\n" + FOOTER)


def render_control(end_time, write_interval, delta_t):
    return foam_header("controlDict", "dictionary", "system") + f"""solver          shockFluid;

startFrom       latestTime;

startTime       0;

stopAt          endTime;

endTime         {fmt(end_time)};

deltaT          {fmt(delta_t)};

writeControl    adjustableRunTime;

writeInterval   {fmt(write_interval)};

cycleWrite      0;

writeFormat     ascii;

writePrecision  15;

writeCompression off;

timeFormat      general;

timePrecision   8;

adjustTimeStep  yes;

maxCo           0.5;

maxDeltaT       1;
""" + FOOTER


def render_schemes():
    return foam_header("fvSchemes", "dictionary", "system") + """fluxScheme      Kurganov;

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;

    reconstruct(rho) vanLeer;
    reconstruct(U)  vanLeerV;
    reconstruct(T)  vanLeer;
}

snGradSchemes
{
    default         corrected;
}
""" + FOOTER


def render_solution():
    return foam_header("fvSolution", "dictionary", "system") + """solvers
{
    "(rho|rhoU|rhoE).*"
    {
        solver          diagonal;
    }

    "U.*"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-10;
        relTol          0;
    }

    "e.*"
    {
        $U;
        tolerance       1e-10;
        relTol          0;
    }
}

PIMPLE
{}
""" + FOOTER


def render_physical(cp, W):
    return foam_header("physicalProperties", "dictionary", "constant") + f"""thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie
    {{
        molWeight       {fmt(W)};
    }}
    thermodynamics
    {{
        Cp              {fmt(cp)};
        hf              0;
    }}
    transport
    {{
        As              0;
        Ts              110.4;
        Pr              1;
    }}
}}
""" + FOOTER


def render_momentum():
    return foam_header("momentumTransport", "dictionary", "constant") + "simulationType  laminar;\n" + FOOTER


def render_field(name, cls, dims, internal, bcs):
    body = []
    for patch in PATCHES + ("defaultFaces",):
        entries = bcs[patch]
        lines = "".join(f"        {k:<16}{v};\n" for k, v in entries)
        body.append(f"    {patch}\n    {{\n{lines}    }}\n")
    return (foam_header(name, cls, "0") + f"dimensions      {dims};\n\ninternalField   {internal};\n\n"
            "boundaryField\n{\n" + "\n".join(body) + "}\n" + FOOTER)


def field_bcs(p):
    g = fmt(p["gamma"])
    p0, T0, pi = fmt(p["p0_pa"]), fmt(p["T0_k"]), fmt(p["p_init_over_p0"] * p["p0_pa"])
    wedge = [("type", "wedge")]
    empty = [("type", "empty")]
    P = dict(inlet=[("type", "totalPressure"), ("rho", "none"), ("psi", "psi"), ("gamma", g),
                    ("p0", f"uniform {p0}"), ("value", f"uniform {p0}")],
             outlet=[("type", "waveTransmissive"), ("field", "p"), ("psi", "psi"), ("gamma", g),
                     ("value", f"uniform {pi}")],
             wall=[("type", "zeroGradient")], wedgeFront=wedge, wedgeBack=wedge, defaultFaces=empty)
    T = dict(inlet=[("type", "totalTemperature"), ("gamma", g), ("T0", f"uniform {T0}"), ("value", f"uniform {T0}")],
             outlet=[("type", "zeroGradient")], wall=[("type", "zeroGradient")], wedgeFront=wedge, wedgeBack=wedge,
             defaultFaces=empty)
    U = dict(inlet=[("type", "zeroGradient")],
             outlet=[("type", "inletOutlet"), ("inletValue", "uniform (0 0 0)"), ("value", "uniform (0 0 0)")],
             wall=[("type", "slip")], wedgeFront=wedge, wedgeBack=wedge, defaultFaces=empty)
    return P, T, U


def render_allrun():
    return """#!/bin/sh
cd "${0%/*}" || exit 1

if [ "${WM_PROJECT_VERSION:-}" != "13" ]; then
    echo "This case targets OpenFOAM Foundation v13 (tag version-13)." >&2
    exit 2
fi

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

runApplication blockMesh
runApplication checkMesh -allTopology -allGeometry
runApplication foamRun
"""


def render_allclean():
    return """#!/bin/sh
cd "${0%/*}" || exit 1

. "$WM_PROJECT_DIR/bin/tools/CleanFunctions"

cleanCase
"""


def render_decompose(n):
    return foam_header("decomposeParDict", "dictionary", "system") + f"""numberOfSubdomains {n};

method          scotch;
""" + FOOTER


def render_slurm(case_id, n):
    return f"""#!/bin/bash
#SBATCH --job-name={case_id}
#SBATCH --ntasks={n}
#SBATCH --time=24:00:00
# UNVERIFIED site-specific template: load or source OpenFOAM Foundation v13 below before use.
# module load openfoam/13
set -eu
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$0")}}"
if [ "${{WM_PROJECT_VERSION:-}}" != "13" ]; then
    echo "OpenFOAM Foundation v13 environment not loaded" >&2
    exit 2
fi
blockMesh > log.blockMesh 2>&1
checkMesh -allTopology -allGeometry > log.checkMesh 2>&1
decomposePar > log.decomposePar 2>&1
srun foamRun -parallel > log.foamRun 2>&1
reconstructPar > log.reconstructPar 2>&1
"""


def sample_points(X, R, half):
    xs = X[-1] - 0.1 * (X[-1] - X[-2])
    rw = float(np.interp(xs, X, R)) * math.cos(half)
    eps = 1e-4 * R.min()
    return dict(exit_start=(xs, eps, 0.0), exit_end=(xs, 0.999 * rw, 0.0),
                axis_start=(X[0] + 1e-3 * (X[-1] - X[0]), eps, 0.0), axis_end=(X[-1] - 1e-3 * (X[-1] - X[0]), eps, 0.0))


def render_functions(X, R, half):
    sp = sample_points(X, R, half)

    def pt(v):
        return "(" + " ".join(fmt(c) for c in v) + ")"
    return foam_header("functions", "dictionary", "system") + f"""#includeFunc patchFlowRate(name=inletMassFlow, patch=inlet)

#includeFunc patchFlowRate(name=outletMassFlow, patch=outlet)

#includeFunc forcesCompressible(name=wallForces, patches=(wall), CofR=(0 0 0), pitchAxis=(0 1 0))

#includeFunc MachNo

#includeFunc graphUniform
(
    name=exitPlaneLine,
    start={pt(sp["exit_start"])},
    end={pt(sp["exit_end"])},
    nPoints=401,
    fields=(p T U rho),
    axis=y
)

#includeFunc graphUniform
(
    name=axisLine,
    start={pt(sp["axis_start"])},
    end={pt(sp["axis_end"])},
    nPoints=2001,
    fields=(p T U rho),
    axis=x
)

#includeFunc patchSurface(name=wallSurface, patch=wall, fields=(p T U rho), interpolate=false)

#includeFunc patchSurface(name=outletSurface, patch=outlet, fields=(p T U rho), interpolate=false)
""" + FOOTER


def timestep_estimate(X, R, nx, nr, half, p, Rt, end_time):
    g = p["gamma"]
    Rgas = R_UNIVERSAL / p["mol_weight"]
    sa = math.sin(2 * half)
    best = {"meridional": (0.0, None), "with_wedge_faces": (0.0, None)}
    for k in range(len(X) - 1):
        xm, rm = 0.5 * (X[k] + X[k + 1]), 0.5 * (R[k] + R[k + 1])
        M = mach_from_area_ratio(max((rm / Rt) ** 2, 1.0 + 1e-12), g, supersonic=xm > 0.0)
        T = p["T0_k"] / (1.0 + 0.5 * (g - 1.0) * M * M)
        c = math.sqrt(g * Rgas * T)
        u = M * c
        dx = (X[k + 1] - X[k]) / nx[k]
        dr = rm / nr
        for j in (0, 1, nr - 1):
            r1, r2 = j * dr, (j + 1) * dr
            V = dx * 0.5 * (r2 * r2 - r1 * r1) * sa
            s_r = dx * 2.0 * math.sin(half) * (r1 + r2)
            mer = (abs(u) + c) / dx + 0.5 * c * s_r / V
            wed = mer + c * dx * dr / V
            for key, val in (("meridional", mer), ("with_wedge_faces", wed)):
                if val > best[key][0]:
                    best[key] = (val, dict(block=k, radial_index=j, x=xm, M=M))
    out = {}
    for key, (val, where) in best.items():
        dt = 0.5 / val
        out[key] = dict(dt_min_s=dt, n_steps_to_end_time=end_time / dt, limiting_cell=where)
    out["method"] = ("Co = 0.5 sum(amaxSf)/V dt with amaxSf = (|u_n| + c)|Sf| from the 1-D isentropic state at each "
                     "block (u along x); maxCo 0.5; 'with_wedge_faces' adds c|Sf| on the two wedge faces")
    return out


# --------------------------------------------------------------------------- static checks
TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[{}()\[\];]|[^\s{}()\[\];"]+(?:\([^\s{}();"]*\))?')


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def balance_ok(text):
    t = strip_comments(text)
    pairs = {"}": "{", ")": "(", "]": "["}
    stack = []
    for ch in re.sub(r'"(?:[^"\\]|\\.)*"', "", t):
        if ch in "{([":
            stack.append(ch)
        elif ch in "})]":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def parse_foam(text):
    toks = TOKEN_RE.findall(strip_comments(text))
    pos = [0]

    def value_until_semicolon():
        vals = []
        while toks[pos[0]] != ";":
            vals.append(item())
        pos[0] += 1
        return vals

    def item():
        t = toks[pos[0]]
        if t == "(":
            pos[0] += 1
            lst = []
            while toks[pos[0]] != ")":
                lst.append(item())
            pos[0] += 1
            return lst
        if t == "[":
            pos[0] += 1
            lst = []
            while toks[pos[0]] != "]":
                lst.append(toks[pos[0]])
                pos[0] += 1
            pos[0] += 1
            return ("dims", lst)
        if t == "{":
            pos[0] += 1
            d = parse_dict("}")
            pos[0] += 1
            return d
        if t in ")]};":
            raise SyntaxError(f"unexpected {t!r} at token {pos[0]}")
        pos[0] += 1
        return t

    def parse_dict(end):
        d = {}
        while pos[0] < len(toks) and toks[pos[0]] != end:
            key = toks[pos[0]].strip('"')
            pos[0] += 1
            if key.startswith("#"):
                name = toks[pos[0]]
                pos[0] += 1
                args = item() if pos[0] < len(toks) and toks[pos[0]] == "(" else []
                d.setdefault(key, []).append([name, args])
                continue
            if key.startswith("$") and toks[pos[0]] == ";":
                pos[0] += 1
                d[key] = "macro"
                continue
            if toks[pos[0]] == "{":
                pos[0] += 1
                d[key] = parse_dict("}")
                if toks[pos[0]] != "}":
                    raise SyntaxError("unterminated dictionary")
                pos[0] += 1
            else:
                d[key] = value_until_semicolon()
        return d

    out = parse_dict(None)
    if pos[0] != len(toks):
        raise SyntaxError("trailing tokens")
    return out


def polyhedron_volume(P):
    faces = [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7)]
    ref = P.mean(axis=-2)
    vol = np.zeros(P.shape[:-2])
    for f in faces:
        Q = P[..., f, :]
        c = Q.mean(axis=-2)
        for i in range(4):
            a, b = Q[..., i, :], Q[..., (i + 1) % 4, :]
            vol += np.einsum("...i,...i->...", a - ref, np.cross(b - ref, c - ref)) / 6.0
    return vol


def cell_points(Vb, nx, nr):
    """Trilinear subdivision of one block (8x3) into nx*nr*1 cells -> (nx, nr, 8, 3)."""
    u = np.linspace(0, 1, nx + 1)
    v = np.linspace(0, 1, nr + 1)
    w = np.array([0.0, 1.0])
    U, Vv, Ww = np.meshgrid(u, v, w, indexing="ij")
    c = [(1 - U) * (1 - Vv) * (1 - Ww), U * (1 - Vv) * (1 - Ww), U * Vv * (1 - Ww), (1 - U) * Vv * (1 - Ww),
         (1 - U) * (1 - Vv) * Ww, U * (1 - Vv) * Ww, U * Vv * Ww, (1 - U) * Vv * Ww]
    G = sum(ci[..., None] * Vb[i] for i, ci in enumerate(c))
    idx = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    return np.stack([G[di:di + nx, dj:dj + nr, dk] for di, dj, dk in idx], axis=2)


def static_checks(files, V, Bk, F, nx, nr, half, X, R):
    res = {}
    parsed = {}
    bal = {}
    for path, txt in files.items():
        if path.endswith(".json") or path in ("Allrun", "Allclean", "run_slurm.sbatch"):
            continue
        bal[path] = balance_ok(txt)
        try:
            parsed[path] = parse_foam(txt)
        except (SyntaxError, IndexError) as exc:
            parsed[path] = None
            bal[path] = False
            res.setdefault("parse_errors", {})[path] = str(exc)
    res["braces_balanced_and_parsed"] = all(bal.values())
    hdr = {}
    for path, d in parsed.items():
        ok = bool(d) and isinstance(d.get("FoamFile"), dict) and d["FoamFile"].get("object") == [os.path.basename(path)] \
            and d["FoamFile"].get("format") == ["ascii"] and "class" in d["FoamFile"]
        hdr[path] = ok
    res["foamfile_headers_ok"] = all(hdr.values())
    bm = parsed["system/blockMeshDict"]
    blist = bm["boundary"][0]
    names = [blist[i] for i in range(0, len(blist), 2)]
    btypes = {blist[i]: blist[i + 1]["type"][0] for i in range(0, len(blist), 2)}
    mesh_patches = set(names) | ({"defaultFaces"} if "defaultPatch" in bm else set())
    btypes["defaultFaces"] = bm["defaultPatch"]["type"][0]
    field_patch_sets, bc_ok, type_match = {}, {}, {}
    for fld in ("0/p", "0/T", "0/U"):
        bf = parsed[fld]["boundaryField"]
        field_patch_sets[fld] = sorted(bf)
        for patch, ent in bf.items():
            t = ent["type"][0]
            req = BC_REQUIRED.get(t)
            bc_ok[f"{fld}:{patch}"] = req is not None and req.issubset(ent.keys())
            mt = btypes.get(patch)
            type_match[f"{fld}:{patch}"] = (t == mt) if (t in CONSTRAINT_TYPES or mt in CONSTRAINT_TYPES) else True
    res["patch_sets_identical"] = all(set(s) == mesh_patches for s in field_patch_sets.values())
    fn = parsed.get("system/functions") or {}
    calls = fn.get("#includeFunc", [])
    known = {"patchFlowRate", "forcesCompressible", "MachNo", "graphUniform", "patchSurface"}
    flat = []
    for name, args in calls:
        toks_ = []

        def walk(a):
            if isinstance(a, list):
                for b in a:
                    walk(b)
            else:
                toks_.append(str(a))
        walk(args)
        flat.append((name, " ".join(toks_)))
    refs = re.findall(r"patch(?:es)?=\(?([A-Za-z]+)", " ".join(t for _, t in flat))
    res["function_templates_known"] = bool(calls) and all(nm in known for nm, _ in calls)
    res["function_patch_references_exist"] = bool(refs) and all(r_ in mesh_patches for r_ in refs)
    sp = sample_points(X, R, half)
    inside = []
    for key_, (px, py, pz) in sp.items():
        rw = float(np.interp(px, X, R)) * math.cos(half)
        inside.append(X[0] < px < X[-1] and 0.0 < py < rw and pz == 0.0)
    res["sample_points_inside_domain"] = bool(all(inside))
    res["patch_names"] = sorted(mesh_patches)
    res["bc_required_entries_present"] = all(bc_ok.values())
    res["constraint_types_consistent"] = all(type_match.values())
    res["blockMesh_patch_types"] = btypes
    Vn = np.array(V)
    nverts = len(bm["vertices"][0])
    res["vertex_count_matches"] = nverts == len(V)
    blocks_tok = bm["blocks"][0]
    nb = sum(1 for t in blocks_tok if t == "hex")
    res["block_count_matches"] = nb == len(Bk)
    idx_ok = all(0 <= i < len(V) for b in Bk for i in b)
    collapsed_ok = all(b[0] == b[4] and b[1] == b[5] and abs(Vn[b[0], 1]) + abs(Vn[b[0], 2]) == 0.0
                       and abs(Vn[b[1], 1]) + abs(Vn[b[1], 2]) == 0.0 for b in Bk)
    res["block_vertex_indices_valid"] = bool(idx_ok)
    res["collapsed_vertices_only_on_axis"] = bool(collapsed_ok)
    wall_m = Vn[1::3]
    wall_p = Vn[2::3]
    ang_m = np.degrees(np.arctan2(wall_m[:, 2], wall_m[:, 1]))
    ang_p = np.degrees(np.arctan2(wall_p[:, 2], wall_p[:, 1]))
    total = ang_p - ang_m
    res["wedge_total_angle_deg_min_max"] = [float(total.min()), float(total.max())]
    res["wedge_symmetric_about_xy_plane"] = bool(np.max(np.abs(ang_p + ang_m)) < 1e-10)
    res["wedge_angle_ok"] = bool(np.max(np.abs(total - 2 * math.degrees(half))) < 1e-10)
    n_front = np.array([0.0, math.sin(half), math.cos(half)])
    n_back = np.array([0.0, -math.sin(half), math.cos(half)])
    dev_front = max(abs(float(Vn[i] @ n_front)) for f in F["wedgeFront"] for i in f)
    dev_back = max(abs(float(Vn[i] @ n_back)) for f in F["wedgeBack"] for i in f)
    res["wedge_faces_planar_max_dev_m"] = max(dev_front, dev_back)
    res["wedge_faces_planar"] = bool(max(dev_front, dev_back) < 1e-12 * max(1.0, float(np.max(np.abs(Vn)))))
    cents = Vn[np.array(Bk)].mean(axis=1)
    orient = {}
    for name, faces in F.items():
        outward = 0
        for k, f in enumerate(faces):
            Q = Vn[list(f)]
            nv = np.zeros(3)
            for i in range(4):
                a_, b_ = Q[i], Q[(i + 1) % 4]
                nv += np.array([(a_[1] - b_[1]) * (a_[2] + b_[2]), (a_[2] - b_[2]) * (a_[0] + b_[0]),
                                (a_[0] - b_[0]) * (a_[1] + b_[1])])
            kb = 0 if name == "inlet" else (len(Bk) - 1 if name == "outlet" else k)
            outward += int(float(nv @ (Q.mean(axis=0) - cents[kb])) > 0)
        orient[name] = [outward, len(faces)]
    res["patch_face_outward_counts"] = orient
    res["face_orientation_matches_v13_tutorial_pattern"] = (orient["inlet"][0] == 0 and all(
        orient[nm][0] == orient[nm][1] for nm in ("outlet", "wall", "wedgeFront", "wedgeBack")))
    P = Vn[np.array(Bk)]
    bv = polyhedron_volume(P)
    L = np.diff(X)
    analytic = 0.5 * math.sin(2 * half) * L * (R[:-1] ** 2 + R[:-1] * R[1:] + R[1:] ** 2) / 3.0
    res["block_volumes_positive"] = bool(np.all(bv > 0))
    res["block_volume_min_m3"] = float(bv.min())
    res["block_volume_vs_analytic_max_rel"] = float(np.max(np.abs(bv / analytic - 1.0)))
    cell_v = []
    cell_dx, cell_dr = [], []
    for k, b in enumerate(Bk):
        cp_ = cell_points(Vn[list(b)], int(nx[k]), int(nr))
        vol = polyhedron_volume(cp_)
        cell_v.append(vol.ravel())
        cell_dx.append(np.full(int(nx[k]), L[k] / nx[k]))
        cell_dr.append(np.array([R[k], R[k + 1]]).max() / nr)
    cv = np.concatenate(cell_v)
    res["n_cells"] = int(cv.size)
    res["cell_volumes_positive"] = bool(np.all(cv > 0))
    res["cell_volume_min_m3"] = float(cv.min())
    res["cell_volume_sum_vs_block_sum_rel"] = float(abs(cv.sum() / bv.sum() - 1.0))
    dx_all = np.concatenate(cell_dx)
    res["axial_cell_size_min_max_m"] = [float(dx_all.min()), float(dx_all.max())]
    res["max_adjacent_axial_size_ratio"] = float(np.max(np.maximum(dx_all[1:] / dx_all[:-1], dx_all[:-1] / dx_all[1:])))
    res["radial_cell_size_max_m"] = float(max(cell_dr))
    res["max_cell_aspect_dr_over_dx"] = float(max(max(R[k], R[k + 1]) / nr / (L[k] / nx[k]) for k in range(len(Bk))))
    res["all_pass"] = all(res[k] for k in ("braces_balanced_and_parsed", "foamfile_headers_ok", "patch_sets_identical",
                                            "function_templates_known", "function_patch_references_exist",
                                            "sample_points_inside_domain",
                                            "bc_required_entries_present", "constraint_types_consistent",
                                            "vertex_count_matches", "block_count_matches",
                                            "block_vertex_indices_valid", "collapsed_vertices_only_on_axis",
                                            "wedge_angle_ok", "wedge_symmetric_about_xy_plane", "wedge_faces_planar",
                                            "block_volumes_positive", "cell_volumes_positive"))
    return res


# --------------------------------------------------------------------------- isentropic helpers
def mach_from_area_ratio(eps, g, supersonic=True):
    def f(M):
        return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0))) - eps
    lo, hi = (1.0 + 1e-12, 50.0) if supersonic else (1e-9, 1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(lo) > 0) == (f(mid) > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- package
def build_package(src, p, level_name, factor, X, R, sec, geo_diag, nx_coarse):
    half = 0.5 * math.radians(p["wedge_total_deg"])
    nx = nx_coarse * factor
    nr = p["coarse_nr"] * factor
    V = vertices_of(X, R, half)
    Bk = blocks_of(len(X) - 1)
    F = faces_of(len(X) - 1)
    g = p["gamma"]
    Rgas = R_UNIVERSAL / p["mol_weight"]
    cp = g * Rgas / (g - 1.0)
    a0 = math.sqrt(g * Rgas * p["T0_k"])
    Me = mach_from_area_ratio(geo_diag["area_ratio_exit"], g)
    Ue = Me * a0 / math.sqrt(1.0 + 0.5 * (g - 1.0) * Me * Me)
    t_ft = (X[-1] - X[0]) / Ue
    end_time = p["n_flow_through"] * t_ft
    files = {
        "system/blockMeshDict": render_block_mesh(V, Bk, nx, nr, F),
        "system/controlDict": render_control(end_time, t_ft, 1e-9),
        "system/fvSchemes": render_schemes(),
        "system/fvSolution": render_solution(),
        "system/decomposeParDict": render_decompose(p["decompose_subdomains"][int(round(math.log2(factor)))]),
        "constant/physicalProperties": render_physical(cp, p["mol_weight"]),
        "constant/momentumTransport": render_momentum(),
        "system/functions": render_functions(X, R, half),
    }
    P_, T_, U_ = field_bcs(p)
    files["0/p"] = render_field("p", "volScalarField", "[1 -1 -2 0 0 0 0]",
                                f"uniform {fmt(p['p_init_over_p0'] * p['p0_pa'])}", P_)
    files["0/T"] = render_field("T", "volScalarField", "[0 0 0 1 0 0 0]", f"uniform {fmt(p['T0_k'])}", T_)
    files["0/U"] = render_field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", U_)
    case_id = f"nozzleB_shockfluid_{level_name}"
    files["Allrun"] = render_allrun()
    files["Allclean"] = render_allclean()
    files["run_slurm.sbatch"] = render_slurm(case_id, p["decompose_subdomains"][int(round(math.log2(factor)))])
    checks = static_checks(files, V, Bk, F, nx, nr, half, X, R)
    art = {k: sha256_bytes(v.encode()) for k, v in sorted(files.items())}
    tstep = timestep_estimate(X, R, nx, nr, half, p, src["Rt"] * p["rt_m"], end_time)
    refs_sha = {}
    if os.path.isdir(V13_REFS_DIR):
        refs_sha = {os.path.relpath(os.path.join(V13_REFS_DIR, f), ROOT): sha256_file(os.path.join(V13_REFS_DIR, f))
                    for f in sorted(os.listdir(V13_REFS_DIR))}
    a_star_wedge = 0.5 * (src["Rt"] * p["rt_m"]) ** 2 * math.sin(2 * half)
    manifest = dict(
        schema_version=1,
        status_label=STATUS_LABEL,
        template=dict(id=TEMPLATE_ID, version=TEMPLATE_VERSION, case_id=case_id, mesh_level=level_name,
                      refinement_factor_vs_coarse=factor,
                      profile="nozzleB_t19_accepted_contour_inviscid_constant_gamma_wedge"),
        solver=dict(OF, runtime_version_verified=False),
        solver_executed=False, runtime_version_verified=False,
        source_contour=dict(kind=src["kind"], files_sha256=src["files"], theta_B_rad=src["theta_B"],
                            B_over_rt=src["B"], E_over_rt=src["E"], D_over_rt=src["D"],
                            moc_thrust_coefficient_vacuum=src["thrust_coefficient"],
                            moc_config={k: src["config"][k] for k in ("Rt", "epsilon", "gamma", "pa_over_p0",
                                                                      "length_pct", "throat_upstream_radius_factor",
                                                                      "throat_downstream_radius_factor", "n_control",
                                                                      "n_kernel", "kernel_dtheta_limit_deg",
                                                                      "starting_line_method")}),
        geometry=dict(geo_diag, rt_m=p["rt_m"], wedge_total_deg=p["wedge_total_deg"],
                      vertex_convention="(x, r cos(a/2), -/+ r sin(a/2)); wall vertices on the true radius",
                      wedge_throat_area_m2=a_star_wedge,
                      full_revolution_factor_for_flat_wedge=2.0 * math.pi / math.sin(2 * half),
                      convergent_parameters=dict(convergent_half_angle_deg=p["convergent_half_angle_deg"],
                                                 contraction_ratio=p["contraction_ratio"],
                                                 basis="lrekit.components.chamber.geometry defaults (repository convention, "
                                                       "not literature); throat curvature Ru from the MOC config",
                                                 moc_difference=("the CFD includes this convergent section while the "
                                                                 "MOC uses a Kliegel-Levine start line with Ru only; "
                                                                 "this difference is part of what CFD1 tests"))),
        mesh_and_numerics=dict(n_blocks=len(Bk), n_vertices=len(V), radial_cells=nr,
                               axial_cells=int(nx.sum()), n_cells=checks["n_cells"],
                               coarse_dx_target_rt=p["coarse_dx_rt"], size_ratio_limit=p["size_ratio_limit"],
                               turn_limit_deg=p["turn_limit_deg"],
                               flux_scheme="Kurganov", reconstruction="vanLeer/vanLeerV", maxCo=0.5,
                               refinement_ratio_between_levels=2.0,
                               refinement=("uniform, ratio 2 between levels in both directions: axial cells per block "
                                           "1/2/4, radial cells 32/64/128 (blocks and wall polyline identical on all "
                                           "levels); celik2008 Step 2 calls r > 1.3 desirable"),
                               role=("GCI level" if p["wedge_total_deg"] == DEFAULTS["wedge_total_deg"]
                                     else "wedge-angle sensitivity (coarse level only, report-only)"),
                               throat_cell_aspect_ratio_dr_over_dx=checks["max_cell_aspect_dr_over_dx"],
                               time_step_estimate=tstep),
        thermo=dict(gamma=g, mol_weight=p["mol_weight"], R_specific=Rgas, Cp=cp, transport="sutherland As 0 (mu = 0)",
                    p0_pa=p["p0_pa"], T0_k=p["T0_k"], note="constant-gamma perfect gas; dimensional scale arbitrary "
                    "for inviscid nondimensional comparison"),
        initial_conditions=dict(p_over_p0=p["p_init_over_p0"], T_over_T0=1.0, U="0",
                                note="uniform quiescent start; only the converged steady state is compared"),
        run_controls=dict(one_d_exit_mach=Me, one_d_exit_velocity_m_s=Ue, flow_through_time_s=t_ft,
                          end_time_s=end_time, write_interval_s=t_ft, n_flow_through=p["n_flow_through"],
                          convergence_monitors=dict(
                              status="PROPOSED thresholds",
                              mass_imbalance="|mdot_in - mdot_out|/mdot_out from inletMassFlow/outletMassFlow, "
                                             "PROPOSED < 1e-4 averaged over the last flow-through time",
                              cf_change="relative change of CF_vac over the last flow-through time, PROPOSED < 1e-5")),
        function_objects=dict(file="system/functions",
                              entries=["inletMassFlow, outletMassFlow: patchFlowRate (mass flow, wedge)",
                                       "wallForces: forcesCompressible on wall (pressure force for CF via wall "
                                       "integral; see UNVERIFIED pRef)",
                                       "MachNo", "exitPlaneLine, axisLine: graphUniform sample sets (p T U rho)",
                                       "wallSurface, outletSurface: patchSurface cell data (p T U rho)"],
                              v13_reference_files_sha256=refs_sha),
        boundary_conditions=dict(inlet="p totalPressure (psi, gamma) p0; T totalTemperature T0; U zeroGradient",
                                 outlet="p waveTransmissive (gamma; no fieldInf/lInf); T zeroGradient; U inletOutlet",
                                 wall="p zeroGradient; T zeroGradient; U slip",
                                 wedgeFront_wedgeBack="wedge", defaultFaces="empty (collapsed axis faces)"),
        dictionary_entry_evidence=[dict(file=f, entry=e, status=s, evidence=r) for f, e, s, r in ENTRY_EVIDENCE],
        unverified_entries=[dict(file=f, entry=e, status=s) for f, e, s, r in ENTRY_EVIDENCE
                            if s in ("UNVERIFIED", "V13_TREE")],
        static_checks=checks,
        gates=dict(static_dictionary_set_complete=True, static_checks_passed=bool(checks["all_pass"]),
                   source_contour_sha256_verified=True, exact_openfoam_runtime_verified=False,
                   block_mesh_executed=False, check_mesh_passed=False, solver_executed=False,
                   inviscid_path_confirmed_at_runtime=False, steady_state_verified=False,
                   mass_conservation_verified_from_results=False, mesh_convergence_verified=False,
                   moc_comparison_performed=False, physically_validated=False, hardware_qualified=False),
        prohibited_claims=["Any statement about the flow before the user runs the cases.",
                           "Agreement between this Euler case and the MOC is physical validation.",
                           "This constant-gamma inviscid case represents hot-gas, viscous or separated flow."],
        references=["openfoam13", "openfoam13src", "greenshields2010", "nparcspatial2021", "celik2008",
                    "haimes1998", "vandenberghe2024"],
        artifact_sha256=art,
    )
    fp_payload = {k: manifest[k] for k in ("schema_version", "template", "solver", "source_contour", "geometry",
                                           "mesh_and_numerics", "thermo", "initial_conditions",
                                           "boundary_conditions", "artifact_sha256")}
    manifest["case_fingerprint_sha256"] = sha256_bytes(canonical(fp_payload))
    files[MANIFEST_NAME] = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    return files, manifest, dict(V=V, Bk=Bk, nx=nx, nr=nr)


def write_case(files, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    for rel, txt in files.items():
        path = os.path.join(dest, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        if rel in ("Allrun", "Allclean", "run_slurm.sbatch"):
            os.chmod(path, 0o755)


# --------------------------------------------------------------------------- plans
def comparison_plan(src, manifests):
    return dict(
        schema="raosim.moc_ext.e13.comparison_plan.v0", status_label=STATUS_LABEL,
        purpose=("Compare the user-run OpenFOAM v13 shockFluid Euler solutions on three meshes with the accepted "
                 "Nozzle-B MOC solution (t19 refined, 48/64, dtheta 0.03125 deg)."),
        evidence_class="benchmark agreement (code-to-code, same inviscid constant-gamma model); not physical validation",
        moc_reference=dict(files_sha256=src["files"], thrust_coefficient_vacuum=src["thrust_coefficient"],
                           thrust_definition=("lrekit.components.nozzle.solvers.nasa.surface_thrust_coefficient on C-D-E: "
                                              "F/(p0 pi Rt^2), momentum + pressure, pa = 0"),
                           E_state=src["E_state"], B_state=src["B_state"],
                           field_npz="refined_field.npz: xy_over_rt, mach, theta_rad (1850 nodes); ce_* (48 nodes)",
                           moc_numerical_uncertainty_estimate=(
                               "use the t19 baseline-vs-refined deltas (report.json refinement.metrics) as U_MOC: "
                               "Mach 8.0e-4, theta 0.0229 deg, p/p0 7.5e-5, CF relative 4.0e-5")),
        cases=[dict(case_id=m["template"]["case_id"], fingerprint=m["case_fingerprint_sha256"],
                    n_cells=m["mesh_and_numerics"]["n_cells"]) for m in manifests],
        prerequisites_per_run=[
            "CFD1 step 0: checkMesh output and a 10-step dry-run log on L1 returned by the user",
            "blockMesh and checkMesh -allTopology -allGeometry pass (log retained, sha256 recorded)",
            "log confirms the inviscid path (no momentumTransport model constructed) or record otherwise",
            "convergence monitors from system/functions (inletMassFlow, outletMassFlow, wallForces) meet the PROPOSED "
            "thresholds in acceptance_metrics.numerical_quality_prerequisites",
            "solver version string and binary/tag fingerprint recorded"],
        integrated_functionals=[
            dict(id="Cd", definition="mdot_outlet / mdot_1D_ideal(A_star_wedge, p0, T0, gamma)", gci=True,
                 moc_counterpart="kernel/BD mass flow relative to 1-D ideal (report-only; KL start line embeds Cd)"),
            dict(id="CF_vac", definition=("integral over outlet patch of (rho u_x^2 + p) dA / (p0 A_star_wedge) from "
                                          "outletSurface; cross-check: inlet momentum flux plus wallForces pressure "
                                          "integral"),
                 gci=True, moc_counterpart="moc_reference.thrust_coefficient_vacuum (same control-volume momentum "
                                           "flux: no wall between C-D-E and the exit plane)"),
            dict(id="CF_over_Cd", definition="CF_vac / Cd (thrust per unit actual mass flow)", gci=True,
                 moc_counterpart="CF_MOC / (MOC mass flow ratio)"),
            dict(id="wall_axial_pressure_force", definition="integral over divergent wall of p n_x dA / (p0 A_star_wedge)",
                 gci=True, moc_counterpart="wall integral of the MOC wall pressure"),
            dict(id="exit_mass_averaged_mach", definition="mass-weighted M on the outlet patch", gci=True,
                 moc_counterpart="mass-weighted M on C-D-E or exit plane via field interpolation")],
        pointwise_comparisons=[
            dict(id="wall_p_over_p0", where="divergent wall, sampled at polyline segment midpoints (avoid kink "
                 "vertices: 0.125 deg chord turning)", against="MOC wall_raw nodes, linear interpolation in x",
                 gci="GCI only in smooth regions (no Richardson where M_n >= 1); otherwise undetermined"),
            dict(id="axis_mach", where="r = 0 (cell centres of the axis-adjacent prism row, extrapolated)",
                 against="MOC field nodes on/near axis"),
            dict(id="field_M_theta_p", where="MOC refined_field.npz node coordinates, CFD cell-centre linear "
                 "interpolation; exclude nodes within 0.2 Rt of the sonic line and within one CFD cell of walls/axis",
                 against="MOC mach, theta_rad, isentropic p/p0",
                 metrics="per-point agreement rule; report fraction of points satisfying it plus max, RMS, p99"),
            dict(id="exit_profiles", where="x = x_E: M(r), theta(r), p/p0(r)", against="MOC field interpolated to x_E")],
        shock_detection=dict(criterion="normal Mach number M_n = M . grad(p)/|grad(p)| >= 1 with a |grad p| "
                                       "threshold; positive M_n compression (haimes1998, NTRS 19980009319 PDF 19)",
                             use="flag any compression shock in the divergent section; Nozzle-B MOC reports "
                                 "shock_free = True", threshold="|grad p| Rt/p0 > 0.05 (PROPOSED)"),
        gci_procedure=dict(
            source=["nparcspatial2021", "celik2008 Eqs. (1)-(7), PDF 5-7 of the ASME-hosted policy file"],
            grids="three systematically refined meshes, refinement ratio r = 2 in x and r (blocks fixed)",
            representative_h="h = [(1/N) sum dA_i]^(1/2) for the 2-D meridional mesh (celik2008 Eq. 2); r21 = r32 = 2 "
                             "by construction",
            observed_order="p = ln[(f3 - f2)/(f2 - f1)]/ln r (constant r); if (f3-f2)/(f2-f1) < 0 the convergence is "
                           "oscillatory: report and do not extrapolate",
            extrapolation="f_ext = (r^p f1 - f2)/(r^p - 1)",
            gci_fine="GCI21 = 1.25 |(f1 - f2)/f1| / (r^p - 1)",
            asymptotic_check="GCI32 / (r^p GCI21) reported; near 1 indicates the asymptotic range",
            scope="integrated functionals only; Richardson extrapolation is invalid at discontinuities "
                  "(nparcspatial2021)"),
        acceptance_metrics=dict(
            status=("agreement rule PRE-REGISTERED by the planner (opus_a answers_round1 A11); numerical-quality "
                    "thresholds are PROPOSED"),
            agreement_rule="|f_MOC - f_CFD| <= U_comb = sqrt(U_CFD^2 + U_MOC^2), evaluated per quantity",
            f_CFD="fine-mesh value f1 (L3_fine); the Richardson value f_ext is reported alongside, not used in the rule",
            U_CFD=("GCI21_fine * |f1| (celik2008 Eq. 7), only for integrated functionals and for samples in smooth "
                   "regions; Richardson extrapolation is invalid at discontinuities (nparcspatial2021); where the "
                   "observed order is undefined (oscillatory or non-monotone convergence) U_CFD is unavailable and the "
                   "comparison is reported as undetermined"),
            U_MOC=("t19 baseline-vs-refined delta of the same quantity (report.json refinement.metrics for global "
                   "outputs; pointwise from the two t19 field arrays interpolated to the sample coordinates)"),
            numerical_quality_prerequisites=dict(
                status="PROPOSED thresholds",
                items=["prerequisites_per_run satisfied for L1, L2 and L3",
                       "mass imbalance |mdot_in - mdot_out|/mdot_out < 1e-4 over the last flow-through time",
                       "relative change of CF_vac over the last flow-through time < 1e-5"]),
            wedge_sensitivity="L1_coarse_wedge0p2 vs L1_coarse: differences of all integrated functionals reported, "
                              "report-only, no acceptance",
            failure_handling="failed or undetermined comparisons stay recorded; no threshold relaxation"),
        convergent_section_difference=("the CFD includes the convergent section (45 deg cone, contraction ratio 2.5, "
                                       "repository convention) while the MOC uses a Kliegel-Levine start line with "
                                       "Ru only; this difference is part of what CFD1 tests"),
        can_establish=["Code-to-code benchmark agreement of the MOC wall pressure, field and CF with an independent "
                       "Euler discretisation on the same faceted geometry and constant gamma",
                       "Discharge coefficient of the 2-D throat for Ru = 1.5 Rt in the Euler model (indirect check of "
                       "the KL start line, MOC-08/09)", "Absence or presence of a captured compression shock in "
                       "Nozzle-B at the resolved mesh scale", "Discretisation uncertainty (GCI) of the CFD functionals"],
        cannot_establish=["Physical validation (inviscid model on both sides)",
                          "Viscous, thermal, separation, or hot-gas variable-gamma effects",
                          "The KL coefficients themselves", "Accuracy at the sonic line or axis singularity beyond "
                          "mesh resolution", "Shock position more precisely than the captured shock width",
                          "Anything before the user runs the cases"],
        known_systematic_effects=["polyline wall with <= 0.125 deg chord turning: local wall-pressure sawtooth near "
                                  "vertices in supersonic flow", "flat-faced wedge: wedge-area factor sin(alpha)/alpha "
                                  "is common to all stations, so normalise by A_star_wedge from the mesh",
                                  "convergent geometry (45 deg cone, contraction 2.5) is a repository default, not part "
                                  "of the MOC solution"],
    )


def handoff_schema():
    return dict(
        schema="raosim.moc_ext.shockfluid_nozzle_handoff.v0", status_label=STATUS_LABEL,
        mirrors="raosim/openfoam/handoff.py (VOFArtifactProvenance, VOFConvergenceStudy, VOFHandoffGate field names)",
        provenance=dict(case_id="str", case_fingerprint_sha256="sha256 from the case manifest",
                        input_fingerprint_sha256="sha256 of source contour pickle", solver_name="foamRun/shockFluid",
                        solver_version="str (from log header)", solver_version_fingerprint_sha256="sha256",
                        extraction_code_fingerprint_sha256="sha256 of the post-processing script"),
        run_evidence=dict(log_sha256={"log.blockMesh": "sha256", "log.checkMesh": "sha256", "log.foamRun": "sha256"},
                          final_time_s="float", n_flow_through_completed="float",
                          inviscid_path_confirmed="bool", check_mesh_failed_checks="int"),
        functionals_per_mesh={"<case_id>": {"Cd": "float", "CF_vac": "float", "CF_over_Cd": "float",
                                            "wall_axial_pressure_force": "float", "exit_mass_averaged_mach": "float",
                                            "monitor_change_last_flow_through": "float",
                                            "mass_balance_rel": "float"}},
        sampled_data_files={"<file>": "sha256; CSV with SI units and x/Rt, r/Rt columns"},
        convergence_studies=[dict(kind="mesh", baseline_run_fingerprint_sha256="sha256",
                                  refined_run_fingerprint_sha256="sha256", monitored_metric="CF_vac",
                                  relative_change="float", acceptance_tolerance="float in (0,1]",
                                  refinement_ratio=2.0),
                             dict(kind="iterative", monitored_metric="CF_vac", relative_change="float",
                                  acceptance_tolerance="float")],
        gates=[dict(name="str", passed="bool", detail="str", evidence_fingerprint_sha256="sha256", required="bool")],
    )


# --------------------------------------------------------------------------- figure
def figure(out, X, R, sec, nx1, nr1, src, half):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "text.color": INK, "axes.linewidth": 0.7, "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.4), gridspec_kw=dict(height_ratios=[1.45, 1.0]))
    w = src["wall_raw"]
    xs_all = np.concatenate([np.linspace(X[k], X[k + 1], nx1[k] + 1)[:-1] for k in range(len(X) - 1)] + [X[-1:]])
    rs_all = np.interp(xs_all, X, R)
    for ax, xlim, ylim, stride, lw, equal in ((axs[0], (X[0] - 0.05, X[-1] + 0.05), (-0.05, R.max() + 0.1), 4, 0.25, True),
                                              (axs[1], (-0.3, 0.5), (0.0, 1.25), 1, 0.15, False)):
        for i in range(0, xs_all.size, stride):
            if xlim[0] <= xs_all[i] <= xlim[1]:
                ax.plot([xs_all[i], xs_all[i]], [0, rs_all[i]], color=MUTED, lw=lw, alpha=0.7)
        m = (xs_all >= xlim[0] - 0.05) & (xs_all <= xlim[1] + 0.05)
        for j in range(0, nr1 + 1, 4 if stride > 1 else 1):
            ax.plot(xs_all[m], rs_all[m] * j / nr1, color=MUTED, lw=lw, alpha=0.7)
        for k in range(len(X)):
            if k in (0, len(X) - 1) or sec[k] != sec[k - 1]:
                ax.plot([X[k], X[k]], [0, R[k]], color=INK, lw=0.9)
        ax.plot(X, R, color=INK, lw=1.0)
        ax.plot([X[0], X[-1]], [0, 0], color=INK, lw=1.0)
        ax.plot(w[:, 0], w[:, 1], "--", color=ACCENT, lw=1.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if equal:
            ax.set_aspect("equal")
        ax.set_xlabel(r"$x/R_t$ (-)")
        ax.set_ylabel(r"$r/R_t$ (-)")
    axs[0].plot([], [], color=INK, lw=1.0, label="block outline and section boundaries (wall on wedge-plane edges)")
    axs[0].plot([], [], "--", color=ACCENT, lw=1.0, label="MOC wall, t19 refined (wall_raw)")
    axs[0].plot([], [], color=MUTED, lw=0.6, label="coarse-mesh cell lines (every 4th)")
    axs[0].legend(loc="upper left", frameon=False, fontsize=7)
    axs[0].text(0.995, 0.03, "(a)", transform=axs[0].transAxes, ha="right", va="bottom", fontsize=9)
    axs[1].text(0.995, 0.03, "(b) throat region, all coarse cell lines, axes not to scale", transform=axs[1].transAxes,
                ha="right", va="bottom", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    fig.tight_layout()
    base = os.path.join(out, "e13_mesh_preview")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--source", choices=("refined", "baseline"), default=DEFAULTS["source"])
    ap.add_argument("--wedge-total-deg", type=float, default=DEFAULTS["wedge_total_deg"])
    ap.add_argument("--turn-limit-deg", type=float, default=DEFAULTS["turn_limit_deg"])
    ap.add_argument("--coarse-nr", type=int, default=DEFAULTS["coarse_nr"])
    ap.add_argument("--coarse-dx-rt", type=float, default=DEFAULTS["coarse_dx_rt"])
    args = ap.parse_args(argv)
    t0 = time.time()
    p = dict(DEFAULTS)
    p.update(source=args.source, wedge_total_deg=args.wedge_total_deg, turn_limit_deg=args.turn_limit_deg,
             coarse_nr=args.coarse_nr, coarse_dx_rt=args.coarse_dx_rt)
    src = load_source(p["source"])
    p["gamma"] = src["gamma"]
    X, R, sec, geo = build_wall(src, p)
    nx1 = np.ones(len(X) - 1, dtype=int)
    out = args.out
    os.makedirs(out, exist_ok=True)
    manifests, level_info = [], {}
    determinism = {}
    for name, factor, wedge in LEVELS:
        pc = dict(p, wedge_total_deg=wedge) if wedge is not None else p
        files, man, mesh = build_package(src, pc, name, factor, X, R, sec, geo, nx1)
        files2, man2, _ = build_package(src, pc, name, factor, X, R, sec, geo, nx1)
        determinism[name] = all(files[k] == files2[k] for k in files) and set(files) == set(files2)
        dest = os.path.join(out, "cases", man["template"]["case_id"])
        write_case(files, dest)
        disk = {rel: sha256_file(os.path.join(dest, *rel.split("/"))) for rel in files}
        mem = {rel: sha256_bytes(txt.encode()) for rel, txt in files.items()}
        determinism[name + "_disk_matches_memory"] = disk == mem
        manifests.append(man)
        level_info[name] = dict(case_dir=os.path.relpath(dest, ROOT), case_fingerprint_sha256=man["case_fingerprint_sha256"],
                                n_cells=man["mesh_and_numerics"]["n_cells"], radial_cells=mesh["nr"],
                                axial_cells=int(mesh["nx"].sum()), static_checks_all_pass=man["static_checks"]["all_pass"],
                                wedge_total_deg=pc["wedge_total_deg"], role=man["mesh_and_numerics"]["role"],
                                time_step_estimate=man["mesh_and_numerics"]["time_step_estimate"],
                                file_sha256=disk)
    cp_path = os.path.join(out, "comparison_plan.json")
    with open(cp_path, "w") as f:
        json.dump(comparison_plan(src, manifests), f, indent=1, sort_keys=False)
    hs_path = os.path.join(out, "handoff_schema.json")
    with open(hs_path, "w") as f:
        json.dump(handoff_schema(), f, indent=1)
    set_manifest = dict(schema="raosim.moc_ext.e13.case_set.v0", status_label=STATUS_LABEL, solver_executed=False,
                        runtime_version_verified=False, levels=level_info, refinement_ratio=2.0,
                        source_contour=src["files"], parameters={k: v for k, v in p.items()},
                        set_fingerprint_sha256=sha256_bytes(canonical({k: v["case_fingerprint_sha256"]
                                                                       for k, v in level_info.items()})))
    sm_path = os.path.join(out, "case_set_manifest.json")
    with open(sm_path, "w") as f:
        json.dump(set_manifest, f, indent=1, default=list)
    half = 0.5 * math.radians(p["wedge_total_deg"])
    base = figure(out, X / p["rt_m"], R / p["rt_m"], sec, nx1, p["coarse_nr"], src, half)
    script = os.path.abspath(__file__)
    all_static = all(m["static_checks"]["all_pass"] for m in manifests)
    deterministic = all(determinism.values())
    status = "PASS" if (all_static and deterministic and geo["polyline_turn_ok"] and geo["segment_ratio_ok"]) else "FAIL"
    loaded = sorted(os.path.relpath(m.__file__, ROOT) for n_, m in sys.modules.items()
                    if (n_ == "lrekit" or n_.startswith("lrekit.")) and getattr(m, "__file__", None))
    result = dict(
        id="E13", status=status, status_label=STATUS_LABEL, evidence_class="verification of generated files only",
        command=" ".join([".venv-jax/bin/python", os.path.relpath(script, ROOT)] + list(argv if argv is not None else sys.argv[1:])),
        git_head=git_head_noproc(), script=os.path.relpath(script, ROOT), script_sha256=sha256_file(script),
        raosim_sources_sha256={f: sha256_file(os.path.join(ROOT, f)) for f in loaded},
        raosim_import_note="lrekit modules are imported only by unpickling the t19 RaoSolution",
        inputs=src["files"], configuration=p,
        metrics=dict(levels=level_info, geometry=geo, determinism=determinism,
                     static_checks_coarse=manifests[0]["static_checks"],
                     n_unverified_entries=len(manifests[0]["unverified_entries"])),
        hypotheses=dict(case_generated_deterministically_and_passes_static_checks=("supported" if status == "PASS"
                                                                                  else "not supported")),
        figures={"e13_mesh_preview": {"pdf": os.path.relpath(base + ".pdf", ROOT), "pdf_sha256": sha256_file(base + ".pdf"),
                                      "png": os.path.relpath(base + ".png", ROOT), "png_sha256": sha256_file(base + ".png")}},
        outputs={os.path.relpath(x, ROOT): sha256_file(x) for x in (cp_path, hs_path, sm_path)},
        wall_time_s=time.time() - t0, processes_spawned=0,
        caption_facts=dict(
            configuration=("Meridional block outline and coarse-mesh cell lines of the write-only OpenFOAM v13 "
                           "shockFluid wedge case (5 deg, polyline wall with <= %.3g deg chord turning, %d blocks, "
                           "%d coarse cells) with the t19 refined MOC wall overlaid; panel (b) is not to scale." %
                           (p["turn_limit_deg"], len(X) - 1, manifests[0]["mesh_and_numerics"]["n_cells"])),
            evidence_class="verification of generated files only",
            not_established="Anything about the flow; that blockMesh, checkMesh or shockFluid accept the files."),
        not_established="Anything about the flow.",
        amendments=dict(
            source="opus_a/answers_round1.json A11 (round1_amendments E13-amend)",
            applied=["convergent section labelled repository convention; MOC difference stated in manifests and plan",
                     "5 deg primary wedge kept; coarse 0.2 deg sensitivity case L1_coarse_wedge0p2 added",
                     "uniform ratio-2 refinement, throat aspect ratio and time-step-count estimates stated in manifests",
                     "system/functions added: mass flow, wall forces, MachNo, exit-plane and axis sample sets, wall and "
                     "outlet surfaces; convergence monitors with PROPOSED thresholds",
                     "comparison_plan agreement rule replaced by |MOC - CFD| <= sqrt(U_CFD^2 + U_MOC^2)",
                     "test docstring states prototype pending decision D1"],
            previous_version={os.path.relpath(os.path.join(out, "manifest_v1.json"), ROOT):
                              sha256_file(os.path.join(out, "manifest_v1.json"))}
            if os.path.exists(os.path.join(out, "manifest_v1.json")) else {},
            v13_refs_provenance=("fetched once by opus_b3 with curl from raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/"
                                 "version-13/<path> into e13/v13_refs (read as text only); hashes in each manifest "
                                 "function_objects.v13_reference_files_sha256")),
    )
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else list(o))
    print(json.dumps(dict(status=status, levels={k: (v["n_cells"], v["static_checks_all_pass"]) for k, v in level_info.items()},
                          determinism=determinism, geometry=geo), indent=1, default=str))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
