"""
wall_model.py – Spline-based wall geometry for MOC nozzle design.

Provides a SplineWall class that the MOC solver queries for:
  - r(x):            wall radius at axial position x
  - theta(x):        wall angle (atan(dr/dx)) at x
  - intersect_char(): where an incoming characteristic hits the wall
"""

from __future__ import annotations
import math
import numpy as np


class SplineWall:
    """
    Cubic Hermite spline wall with clamped endpoint slopes.

    The wall is defined by knot points (x_k, r_k) with computed
    tangent slopes, plus clamped slopes at the endpoints:
      - Start: slope = tan(theta_n)  (inflection)
      - End:   slope ≈ 0             (near-parallel exit)
    """

    def __init__(self, x_knots: np.ndarray, r_knots: np.ndarray,
                 slope_start: float, slope_end: float = 0.0):
        self.x_knots = np.asarray(x_knots, dtype=float)
        self.r_knots = np.asarray(r_knots, dtype=float)
        self.n = len(x_knots)

        self.slopes = np.zeros(self.n)
        self.slopes[0] = slope_start
        self.slopes[-1] = slope_end

        for i in range(1, self.n - 1):
            dx_l = self.x_knots[i] - self.x_knots[i - 1]
            dx_r = self.x_knots[i + 1] - self.x_knots[i]
            if dx_l > 1e-15 and dx_r > 1e-15:
                dr_l = (self.r_knots[i] - self.r_knots[i - 1]) / dx_l
                dr_r = (self.r_knots[i + 1] - self.r_knots[i]) / dx_r
                self.slopes[i] = 0.5 * (dr_l + dr_r)
            else:
                self.slopes[i] = 0.0

        self.x_start = float(x_knots[0])
        self.x_end = float(x_knots[-1])

    @classmethod
    def from_controls(cls, control_r: np.ndarray,
                      Nx: float, Ny: float,
                      Ex: float, Ey: float,
                      theta_n: float) -> SplineWall:
        n_ctrl = len(control_r)
        x_knots = np.linspace(Nx, Ex, n_ctrl + 2)
        r_knots = np.concatenate([[Ny], np.asarray(control_r), [Ey]])
        return cls(x_knots, r_knots,
                   slope_start=math.tan(theta_n), slope_end=0.0)

    def _find_segment(self, x: float) -> int:
        if x <= self.x_knots[0]:
            return 0
        if x >= self.x_knots[-1]:
            return self.n - 2
        idx = int(np.searchsorted(self.x_knots, x)) - 1
        return max(0, min(idx, self.n - 2))

    def _hermite_eval(self, seg: int, x: float) -> tuple[float, float]:
        x0 = self.x_knots[seg]
        x1 = self.x_knots[seg + 1]
        r0 = self.r_knots[seg]
        r1 = self.r_knots[seg + 1]
        m0 = self.slopes[seg]
        m1 = self.slopes[seg + 1]

        h = x1 - x0
        if h < 1e-15:
            return r0, m0

        t = (x - x0) / h
        t2 = t * t
        t3 = t2 * t

        r = (2*t3 - 3*t2 + 1)*r0 + (t3 - 2*t2 + t)*h*m0 + \
            (-2*t3 + 3*t2)*r1 + (t3 - t2)*h*m1

        dr_dt = (6*t2 - 6*t)*r0 + (3*t2 - 4*t + 1)*h*m0 + \
                (-6*t2 + 6*t)*r1 + (3*t2 - 2*t)*h*m1
        dr_dx = dr_dt / h

        return float(r), float(dr_dx)

    def r(self, x: float) -> float:
        seg = self._find_segment(x)
        r_val, _ = self._hermite_eval(seg, x)
        return r_val

    def dr_dx(self, x: float) -> float:
        seg = self._find_segment(x)
        _, slope = self._hermite_eval(seg, x)
        return slope

    def theta(self, x: float) -> float:
        return math.atan(self.dr_dx(x))

    def intersect_char(self, x0: float, r0: float,
                       char_slope: float,
                       tol: float = 1e-10, max_iter: int = 30) -> tuple[float, float]:
        """
        Find where a characteristic line from (x0, r0) with given
        slope (dr/dx along the characteristic) intersects this wall.

        Solves: r0 + char_slope*(x - x0) = wall.r(x) for x.
        Uses Newton's method.
        """
        x_guess = x0 + 0.01 * (self.x_end - self.x_start)
        if x_guess > self.x_end:
            x_guess = 0.5 * (x0 + self.x_end)

        for _ in range(max_iter):
            r_char = r0 + char_slope * (x_guess - x0)
            r_wall = self.r(x_guess)
            dr_wall = self.dr_dx(x_guess)

            f = r_char - r_wall
            df = char_slope - dr_wall

            if abs(df) < 1e-15:
                break
            dx = -f / df
            x_guess += dx

            x_guess = max(self.x_start, min(x_guess, self.x_end))

            if abs(dx) < tol:
                break

        x_hit = x_guess
        r_hit = self.r(x_hit)
        return x_hit, r_hit

    def sample(self, n: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(self.x_start, self.x_end, n)
        rs = np.array([self.r(x) for x in xs])
        thetas = np.array([self.theta(x) for x in xs])
        return xs, rs, thetas
