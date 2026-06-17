import math
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from raosim.gas_dynamics import prandtl_meyer
from raosim.nasa_moc import MOCNode
from raosim.rao_existence_scan import (
    ExistenceScanConfig,
    MODEL_FAN,
    MODEL_POSITION,
    MODEL_SMOOTH,
    SCAN_STATIONARITY,
    prandtl_meyer_fan_post_state,
    refine_from_scan_best,
    scan_existence,
)


def test_prandtl_meyer_fan_relation_is_explicit():
    gamma = 1.4
    pre = MOCNode(x=0.1, r=0.03, M=2.0, theta=math.radians(8.0), gamma=gamma)
    turn = math.radians(4.0)

    post = prandtl_meyer_fan_post_state(pre, turn, gamma)

    assert post.x == pre.x
    assert post.r == pre.r
    assert post.theta - pre.theta == pytest.approx(turn, abs=1e-12)
    assert (
        prandtl_meyer(post.M, gamma) - prandtl_meyer(pre.M, gamma)
    ) == pytest.approx(turn, abs=1e-10)
    assert post.M > pre.M


def test_tiny_existence_scan_returns_three_closure_fields():
    cfg = ExistenceScanConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        theta_b_values_deg=np.array([23.0, 25.0]),
        kdf_values=np.array([0.20, 0.35]),
        models=(MODEL_SMOOTH, MODEL_POSITION, MODEL_FAN),
        n_kernel=8,
        n_de_points=6,
        position_theta_count=3,
        position_mach_count=3,
        fan_turn_count=3,
    )

    result = scan_existence(cfg)

    assert set(result.closures) == {MODEL_SMOOTH, MODEL_POSITION, MODEL_FAN}
    for closure in result.closures.values():
        assert closure.residual_norm.shape == (2, 2)
        assert closure.radius_residual.shape == (2, 2)
        assert closure.length_residual.shape == (2, 2)
        assert closure.sigma_E_rad.shape == (2, 2)
        assert closure.mass_residual.shape == (2, 2)
        assert np.any(np.isfinite(closure.residual_norm))

    fan = result.closures[MODEL_FAN]
    assert np.nanmin(fan.d_mach_jump) >= -1e-10


def test_position_model_minimizes_over_post_state_grid():
    cfg = ExistenceScanConfig(
        Rt=0.020,
        epsilon=10.0,
        theta_b_values_deg=np.array([24.0]),
        kdf_values=np.array([0.30]),
        models=(MODEL_POSITION,),
        n_kernel=8,
        n_de_points=6,
        position_theta_span_deg=2.0,
        position_theta_count=3,
        position_mach_down=0.1,
        position_mach_up=0.2,
        position_mach_count=3,
    )

    closure = scan_existence(cfg).closures[MODEL_POSITION]

    assert closure.residual_norm.shape == (1, 1)
    assert np.isfinite(closure.residual_norm[0, 0])
    assert closure.success[0, 0]
    assert closure.d_mach_post[0, 0] >= 1.0


def test_stationarity_mode_uses_absolute_sigma_e():
    cfg = ExistenceScanConfig(
        Rt=0.020,
        epsilon=10.0,
        theta_b_values_deg=np.array([25.0]),
        kdf_values=np.array([0.15]),
        models=(MODEL_SMOOTH,),
        n_kernel=8,
        n_de_points=6,
        scan_mode=SCAN_STATIONARITY,
    )

    closure = scan_existence(cfg).closures[MODEL_SMOOTH]

    components = np.array([
        closure.radius_residual[0, 0],
        closure.length_residual[0, 0],
        closure.sigma_E_rad[0, 0],
    ])
    assert np.isfinite(closure.sigma_E_rad[0, 0])
    assert closure.residual_norm[0, 0] == pytest.approx(
        np.sqrt(np.mean(components ** 2))
    )


def test_root_refinement_closes_geometry_from_scan_seed():
    pytest.importorskip("scipy")
    cfg = ExistenceScanConfig(
        Rt=0.020,
        epsilon=10.0,
        theta_b_values_deg=np.array([25.0, 25.5]),
        kdf_values=np.array([0.14, 0.16]),
        models=(MODEL_SMOOTH,),
        n_kernel=8,
        n_de_points=8,
    )
    result = scan_existence(cfg)

    root = refine_from_scan_best(result, model=MODEL_SMOOTH, maxfev=40)

    assert root.residual_norm_geometry < 1e-3
    assert np.isfinite(root.sigma_E_rad)


def test_existence_scan_cli_runs_without_pythonpath(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/rao_existence_scan.py",
            "--theta-b-values-deg",
            "25.0",
            "--kdf-values",
            "0.15",
            "--models",
            "smooth",
            "--n-kernel",
            "8",
            "--n-de-points",
            "6",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "smooth_scan.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["config"]["n_kernel"] == 8
    assert summary["config"]["n_de_points"] == 6
    assert summary["config"]["scan_mode"] == "geometry"
