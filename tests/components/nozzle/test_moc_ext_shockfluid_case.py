"""Static checks of the opt-in E13 shockFluid case writer (scripts/, not library code).

Prototype pending decision D1: if D1 is not option (a), the script and this test may be removed.
Uses a synthetic contour; never runs OpenFOAM or any other process.
"""

from lrekit.paths import repository_root as _repository_root
import importlib.util
import math
import os
import subprocess

import numpy as np
import pytest

ROOT = str(_repository_root())
_SPEC = importlib.util.spec_from_file_location(
    "moc_ext_e13_write_shockfluid_case",
    os.path.join(ROOT, "tests", "experiments", "nozzle", "moc_ext_e13_write_shockfluid_case.py"))
w = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(w)


def _synthetic_src():
    Rt, Rd, tb = 1.0, 0.382, math.radians(20.0)
    ph = np.linspace(0.0, tb, 400)
    arc = np.c_[Rd * np.sin(ph), Rt + Rd * (1.0 - np.cos(ph))]
    xB, rB = arc[-1]
    xE, rE = 2.0, 1.6
    t = np.linspace(0.0, 1.0, 30)[1:]
    dx = xE - xB
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    r = h00 * rB + h10 * dx * math.tan(tb) + h01 * rE + h11 * dx * math.tan(math.radians(8.0))
    wall = np.vstack([arc, np.c_[xB + t * dx, r]])
    cfg = dict(Rt=1.0, epsilon=rE**2, gamma=1.23, pa_over_p0=0.0, length_pct=80.0,
               throat_upstream_radius_factor=1.5, throat_downstream_radius_factor=Rd, n_control=8, n_kernel=8,
               kernel_dtheta_limit_deg=0.5, starting_line_method="kliegel_levine")
    return dict(wall_raw=wall, theta_B=tb, B=(xB, rB), E=(xE, rE), E_state=dict(M=2.5, theta_rad=0.14),
                B_state=dict(M=1.9, theta_rad=tb), D=(1.0, 1.0), thrust_coefficient=1.5, Rt=Rt, Rd=Rd, Ru=1.5,
                gamma=1.23, epsilon=rE**2, config=cfg, kind="synthetic", files={})


@pytest.fixture(scope="module")
def built():
    p = dict(w.DEFAULTS, coarse_dx_rt=0.05, turn_limit_deg=0.5, coarse_nr=4, gamma=1.23)
    src = _synthetic_src()
    X, R, sec, geo = w.build_wall(src, p)
    nx1 = np.ones(len(X) - 1, dtype=int)
    return p, src, X, R, sec, geo, nx1


def test_geometry_policies(built):
    p, src, X, R, sec, geo, nx1 = built
    assert geo["polyline_turn_ok"] and geo["segment_ratio_ok"] and geo["throat_is_station"]
    assert np.all(np.diff(X) > 0)
    assert X[-1] == pytest.approx(src["E"][0]) and R[-1] == pytest.approx(src["E"][1])
    assert geo["contraction_area_ratio"] == pytest.approx(p["contraction_ratio"])


@pytest.mark.parametrize("level,factor", [("L1_coarse", 1), ("L2_medium", 2)])
def test_package_static_checks_and_determinism(built, tmp_path, monkeypatch, level, factor):
    p, src, X, R, sec, geo, nx1 = built

    def _no_process(*a, **k):
        raise AssertionError("writer spawned a process")
    monkeypatch.setattr(subprocess, "Popen", _no_process)
    files, man, mesh = w.build_package(src, p, level, factor, X, R, sec, geo, nx1)
    files2, _, _ = w.build_package(src, p, level, factor, X, R, sec, geo, nx1)
    assert files == files2
    chk = man["static_checks"]
    assert chk["all_pass"], chk
    assert chk["face_orientation_matches_v13_tutorial_pattern"]
    assert chk["function_patch_references_exist"] and chk["sample_points_inside_domain"]
    assert "system/functions" in files
    assert chk["block_volume_vs_analytic_max_rel"] < 1e-12
    assert chk["n_cells"] == (len(X) - 1) * factor * p["coarse_nr"] * factor
    assert man["solver_executed"] is False and man["runtime_version_verified"] is False
    assert man["status_label"] == w.STATUS_LABEL
    dest = tmp_path / level
    w.write_case(files, str(dest))
    for rel, txt in files.items():
        assert (dest / rel).read_text(encoding="utf-8") == txt
        assert man["artifact_sha256"].get(rel, w.sha256_bytes(txt.encode())) == w.sha256_bytes(txt.encode())


def test_static_checks_detect_defects(built):
    p, src, X, R, sec, geo, nx1 = built
    files, man, mesh = w.build_package(src, p, "L1_coarse", 1, X, R, sec, geo, nx1)
    half = 0.5 * math.radians(p["wedge_total_deg"])
    F = w.faces_of(len(X) - 1)
    args = (mesh["V"], mesh["Bk"], F, mesh["nx"], mesh["nr"], half, X, R)
    bad = dict(files)
    bad["0/U"] = bad["0/U"].replace("    wall\n    {\n        type            slip;\n    }\n", "")
    assert not w.static_checks(bad, *args)["patch_sets_identical"]
    bad = dict(files)
    bad["0/T"] = bad["0/T"].replace("T0              uniform", "Tzero           uniform")
    assert not w.static_checks(bad, *args)["bc_required_entries_present"]
    bad = dict(files)
    bad["system/fvSchemes"] = bad["system/fvSchemes"].replace("default         Euler;\n}", "default         Euler;\n")
    assert not w.static_checks(bad, *args)["braces_balanced_and_parsed"]
    flipped = [(b[3], b[2], b[1], b[0], b[7], b[6], b[5], b[4]) for b in mesh["Bk"]]
    assert not w.static_checks(files, mesh["V"], flipped, F, mesh["nx"], mesh["nr"], half, X, R)["block_volumes_positive"]
    assert w.balance_ok("a { b (1 2); }") and not w.balance_ok("a { b (1 2; }")
