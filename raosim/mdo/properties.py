"""
raosim.mdo.properties — C¹ shape-preserving property surfaces (Phase 2).

Plan §9: CEA/CoolProp are never autodiffed.  Chamber thermochemistry is
sampled OFFLINE on a (Pc, O/F) grid (``scripts/sample_cea_surface.py`` on a
host with RocketCEA), stored as plain arrays, and evaluated here through a
C¹ monotone tensor-product Hermite interpolant:

* 1-D node slopes by the Fritsch–Carlson (1980) monotone limiter — the same
  scheme as SciPy's ``PchipInterpolator`` (parity is test-pinned in
  tests/test_mdo_properties.py against SciPy where available);
* 2-D bicubic Hermite patches from the limited partials with zero cross
  (twist) terms — C¹ across cell edges, shape-preserving along grid lines;
* evaluation is pure ``jnp`` (jit/grad-safe; the cell index is a
  ``stop_gradient`` integer — the interpolant is C¹ so the derivative is
  continuous across cells).

Domain policy (plan rule 7): the optimizer must be *constrained* to the
tabulated box.  Evaluation clamps queries to the box only to stay finite;
``domain_violation`` exposes the (positive-when-outside) distances so the NLP
layer can enforce membership as explicit constraints instead of silently
extrapolating.

Final designs are re-evaluated on the authoritative backends (Phase 11) —
these surfaces are a differentiable *stand-in*, not the truth source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

import raosim.jax  # noqa: F401  -- enables x64
import jax
import jax.numpy as jnp

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Fritsch–Carlson monotone slopes (NumPy, build-time)                          #
# --------------------------------------------------------------------------- #
def fritsch_carlson_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Monotonicity-limited node derivatives for cubic Hermite interpolation.

    Interior nodes: the weighted harmonic mean of adjacent secants when they
    share a sign, else zero (Fritsch & Carlson 1980).  End nodes: the
    one-sided three-point estimate with the standard shape-preserving clamps
    (identical to SciPy's PCHIP ``_edge_case``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("need at least two nodes")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")
    delta = np.diff(y) / h
    d = np.zeros_like(y)

    # interior
    for k in range(1, x.size - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or (
                np.sign(delta[k - 1]) != np.sign(delta[k])):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def edge(h0, h1, d0, d1):
        val = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(val) != np.sign(d0):
            val = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(val) > 3.0 * abs(d0):
            val = 3.0 * d0
        return val

    if x.size == 2:
        d[0] = d[1] = delta[0]
    else:
        d[0] = edge(h[0], h[1], delta[0], delta[1])
        d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


# --------------------------------------------------------------------------- #
# 1-D PCHIP evaluation (jnp)                                                   #
# --------------------------------------------------------------------------- #
def _hermite_basis(s: Array) -> tuple[Array, Array, Array, Array]:
    s2 = s * s
    s3 = s2 * s
    return (2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2)


@dataclass(frozen=True)
class Pchip1D:
    """C¹ monotone cubic over a static grid; jnp evaluation."""

    x: Array
    y: Array
    d: Array

    @classmethod
    def build(cls, x, y) -> "Pchip1D":
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        d = fritsch_carlson_slopes(x, y)
        return cls(jnp.asarray(x), jnp.asarray(y), jnp.asarray(d))

    def __call__(self, t: Array) -> Array:
        t = jnp.asarray(t, dtype=jnp.float64)
        tc = jnp.clip(t, self.x[0], self.x[-1])
        i = jnp.clip(jnp.searchsorted(self.x, tc, side="right") - 1,
                     0, self.x.size - 2)
        i = jax.lax.stop_gradient(i)
        h = self.x[i + 1] - self.x[i]
        s = (tc - self.x[i]) / h
        h00, h10, h01, h11 = _hermite_basis(s)
        return (h00 * self.y[i] + h01 * self.y[i + 1]
                + h * (h10 * self.d[i] + h11 * self.d[i + 1]))


# --------------------------------------------------------------------------- #
# 2-D tensor-product monotone-Hermite surface (jnp)                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PropertySurface2D:
    """z(x, y) over a static rectilinear grid, C¹, shape-preserving along
    grid lines.  ``name`` is carried for error messages / provenance."""

    xg: Array           # (m,) strictly increasing (e.g. Pc)
    yg: Array           # (n,) strictly increasing (e.g. O/F)
    Z: Array            # (m, n)
    Zx: Array           # (m, n) d/dx  (FC-limited per column)
    Zy: Array           # (m, n) d/dy  (FC-limited per row)
    name: str = "surface"

    @classmethod
    def build(cls, xg, yg, Z, *, name: str = "surface") -> "PropertySurface2D":
        xg = np.asarray(xg, float)
        yg = np.asarray(yg, float)
        Z = np.asarray(Z, float)
        if Z.shape != (xg.size, yg.size):
            raise ValueError(f"{name}: Z shape {Z.shape} != grid "
                             f"({xg.size}, {yg.size})")
        Zx = np.stack([fritsch_carlson_slopes(xg, Z[:, j])
                       for j in range(yg.size)], axis=1)
        Zy = np.stack([fritsch_carlson_slopes(yg, Z[i, :])
                       for i in range(xg.size)], axis=0)
        return cls(jnp.asarray(xg), jnp.asarray(yg), jnp.asarray(Z),
                   jnp.asarray(Zx), jnp.asarray(Zy), name)

    # -- evaluation ---------------------------------------------------------- #
    def __call__(self, x: Array, y: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        xc = jnp.clip(x, self.xg[0], self.xg[-1])
        yc = jnp.clip(y, self.yg[0], self.yg[-1])
        i = jnp.clip(jnp.searchsorted(self.xg, xc, side="right") - 1,
                     0, self.xg.size - 2)
        j = jnp.clip(jnp.searchsorted(self.yg, yc, side="right") - 1,
                     0, self.yg.size - 2)
        i = jax.lax.stop_gradient(i)
        j = jax.lax.stop_gradient(j)
        hx = self.xg[i + 1] - self.xg[i]
        hy = self.yg[j + 1] - self.yg[j]
        s = (xc - self.xg[i]) / hx
        u = (yc - self.yg[j]) / hy
        a00, a10, a01, a11 = _hermite_basis(s)   # value/deriv basis in x
        b00, b10, b01, b11 = _hermite_basis(u)   # value/deriv basis in y

        Z, Zx, Zy = self.Z, self.Zx, self.Zy
        f = (
            a00 * b00 * Z[i, j] + a01 * b00 * Z[i + 1, j]
            + a00 * b01 * Z[i, j + 1] + a01 * b01 * Z[i + 1, j + 1]
            + hx * (a10 * b00 * Zx[i, j] + a11 * b00 * Zx[i + 1, j]
                    + a10 * b01 * Zx[i, j + 1] + a11 * b01 * Zx[i + 1, j + 1])
            + hy * (a00 * b10 * Zy[i, j] + a01 * b10 * Zy[i + 1, j]
                    + a00 * b11 * Zy[i, j + 1] + a01 * b11 * Zy[i + 1, j + 1])
        )
        return f

    # -- domain policy (plan rule 7) ----------------------------------------- #
    def domain_violation(self, x: Array, y: Array) -> Array:
        """Positive components measure how far (x, y) sits outside the box.
        Feed these to the NLP as ``violation <= 0`` constraints."""
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return jnp.stack([
            self.xg[0] - x, x - self.xg[-1],
            self.yg[0] - y, y - self.yg[-1],
        ])


# --------------------------------------------------------------------------- #
# Chamber-property bundles                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChamberSurfaces:
    """gamma / Tc / R_gas over (Pc, O/F), plus derived ideal c*.

    ``c_star_ideal`` is *derived* from (gamma, Tc, R) at evaluation time so
    the c* convention stays pinned in exactly one place (plan rule 2)."""

    gamma: PropertySurface2D
    Tc: PropertySurface2D
    R_gas: PropertySurface2D
    provenance: str = "unspecified"

    def c_star_ideal(self, Pc: Array, OF: Array) -> Array:
        g = self.gamma(Pc, OF)
        Tc = self.Tc(Pc, OF)
        R = self.R_gas(Pc, OF)
        gp1 = g + 1.0
        gm1 = g - 1.0
        return jnp.sqrt(g * R * Tc) / (g * jnp.sqrt((2.0 / gp1) ** (gp1 / gm1)))

    def domain_violation(self, Pc: Array, OF: Array) -> Array:
        return self.gamma.domain_violation(Pc, OF)


def constant_chamber_surfaces(*, gamma: float, Tc: float, R_gas: float,
                              Pc_range=(1.0e6, 8.0e6), OF_range=(1.5, 3.5),
                              ) -> ChamberSurfaces:
    """Flat surfaces from constants — the skeleton fallback when no CEA table
    is available (sandbox/CI).  Marked in provenance; NOT for final results."""
    xg = np.linspace(Pc_range[0], Pc_range[1], 4)
    yg = np.linspace(OF_range[0], OF_range[1], 4)
    ones = np.ones((xg.size, yg.size))
    return ChamberSurfaces(
        gamma=PropertySurface2D.build(xg, yg, gamma * ones, name="gamma"),
        Tc=PropertySurface2D.build(xg, yg, Tc * ones, name="Tc"),
        R_gas=PropertySurface2D.build(xg, yg, R_gas * ones, name="R_gas"),
        provenance="constant_fallback(skeleton)",
    )


# --------------------------------------------------------------------------- #
# Offline sampling (host-only; requires RocketCEA)                             #
# --------------------------------------------------------------------------- #
def sample_cea_tables(Pc_grid, OF_grid, *, oxidizer: str, fuel: str,
                      ) -> dict[str, np.ndarray]:
    """Sample frozen-chamber properties on the grid via ``raosim.cea``.

    Host-only (RocketCEA + Fortran CEA backend); never called from jitted
    code.  Returns plain arrays for ``save_tables``/``build_chamber_surfaces``.
    Equilibrium sampling is deliberately NOT provided here yet — the
    constant-gamma solver cannot consume it (RQ4-B; evaluation report §D.3.2).
    """
    from raosim.cea import cea_propellant  # deferred: host dependency

    Pc_grid = np.asarray(Pc_grid, float)
    OF_grid = np.asarray(OF_grid, float)
    gamma = np.zeros((Pc_grid.size, OF_grid.size))
    Tc = np.zeros_like(gamma)
    R_gas = np.zeros_like(gamma)
    for i, Pc in enumerate(Pc_grid):
        for j, OF in enumerate(OF_grid):
            prop = cea_propellant(oxidizer=oxidizer, fuel=fuel, Pc=float(Pc),
                                  mixture_ratio=float(OF))
            gamma[i, j] = prop.gamma
            Tc[i, j] = prop.Tc
            R_gas[i, j] = 8.314462618 / prop.Mw
    return {"Pc_grid": Pc_grid, "OF_grid": OF_grid, "gamma": gamma,
            "Tc": Tc, "R_gas": R_gas}


def save_tables(path: str, tables: Mapping[str, np.ndarray], *,
                oxidizer: str, fuel: str) -> None:
    np.savez(path, oxidizer=np.str_(oxidizer), fuel=np.str_(fuel), **tables)


def load_chamber_surfaces(path: str) -> ChamberSurfaces:
    dat = np.load(path, allow_pickle=False)
    return ChamberSurfaces(
        gamma=PropertySurface2D.build(dat["Pc_grid"], dat["OF_grid"],
                                      dat["gamma"], name="gamma"),
        Tc=PropertySurface2D.build(dat["Pc_grid"], dat["OF_grid"],
                                   dat["Tc"], name="Tc"),
        R_gas=PropertySurface2D.build(dat["Pc_grid"], dat["OF_grid"],
                                      dat["R_gas"], name="R_gas"),
        provenance=f"cea_frozen_table:{path}",
    )
