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
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

import raosim.jax  # noqa: F401  -- enables x64
import jax
import jax.numpy as jnp

Array = jnp.ndarray

PROPERTY_TABLE_SCHEMA_VERSION = 1
INTERPOLATOR_VERSION = "tensor-pchip-hermite-v1"
OF_CONVENTION = "oxidizer_to_fuel_mass_ratio"
SAMPLED_BOUNDED = "sampled_bounded"
CONSTANT_UNBOUNDED = "constant_unbounded"

_TABLE_UNITS = {
    "Pc_grid": "Pa",
    "OF_grid": "kg_oxidizer/kg_fuel",
    "gamma": "1",
    "Tc": "K",
    "R_gas": "J/(kg*K)",
}


def _canonical_name(value: object) -> str:
    key = "".join(ch for ch in str(value).strip().lower() if ch.isalnum())
    aliases = {
        "o2": "lox", "oxygen": "lox", "liquidoxygen": "lox",
        "rp1": "rp1", "kerosene": "rp1",
        "ch4": "lch4", "methane": "lch4", "liquidmethane": "lch4",
        "h2": "lh2", "hydrogen": "lh2", "liquidhydrogen": "lh2",
        "etoh": "ethanol",
    }
    return aliases.get(key, key)


def _table_digest(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    """Content identity independent of file name or path provenance."""

    digest = hashlib.sha256()
    digest.update(b"raosim.chamber-property-table.v1\0")
    digest.update(json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    for name in ("Pc_grid", "OF_grid", "gamma", "Tc", "R_gas"):
        array = np.ascontiguousarray(np.asarray(arrays[name], dtype="<f8"))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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
    domain_policy: str = SAMPLED_BOUNDED
    oxidizer: str = "unspecified"
    fuel: str = "unspecified"
    thermochemistry_mode: str = "frozen"
    of_convention: str = OF_CONVENTION
    units_json: str = json.dumps(_TABLE_UNITS, sort_keys=True)
    schema_version: int = PROPERTY_TABLE_SCHEMA_VERSION
    interpolator_version: str = INTERPOLATOR_VERSION
    content_sha256: str = ""
    source_path: str = ""

    def c_star_ideal(self, Pc: Array, OF: Array) -> Array:
        g = self.gamma(Pc, OF)
        Tc = self.Tc(Pc, OF)
        R = self.R_gas(Pc, OF)
        gp1 = g + 1.0
        gm1 = g - 1.0
        return jnp.sqrt(g * R * Tc) / (g * jnp.sqrt((2.0 / gp1) ** (gp1 / gm1)))

    def domain_violation(self, Pc: Array, OF: Array) -> Array:
        return self.gamma.domain_violation(Pc, OF)

    def domain_margin(self, Pc: Array, OF: Array) -> Array:
        """Exact normalized distance to the governing sampled box.

        Constant fallback axes exist only because the interpolator needs a
        rectangular grid.  They are not a claimed physical validity domain and
        therefore return a finite inert value; the manifest separately marks
        the row not applicable.  Sampled surfaces use an exact min so a query on
        a table boundary evaluates to exactly zero rather than a biased smooth
        envelope violation.
        """

        if self.domain_policy == CONSTANT_UNBOUNDED:
            return jnp.asarray(1.0, dtype=jnp.float64)
        xg, yg = self.gamma.xg, self.gamma.yg
        xspan = xg[-1] - xg[0]
        yspan = yg[-1] - yg[0]
        margins = jnp.stack([
            (jnp.asarray(Pc) - xg[0]) / xspan,
            (xg[-1] - jnp.asarray(Pc)) / xspan,
            (jnp.asarray(OF) - yg[0]) / yspan,
            (yg[-1] - jnp.asarray(OF)) / yspan,
        ])
        return jnp.min(margins)

    def has_meaningful_of_dependence(self, *, rtol: float = 1.0e-10) -> bool:
        """Whether at least one chamber property varies across the O/F axis."""

        for surface in (self.gamma, self.Tc, self.R_gas):
            values = np.asarray(surface.Z, dtype=float)
            variation = np.max(np.ptp(values, axis=1))
            reference = max(float(np.max(np.abs(values))), 1.0)
            if variation > rtol * reference:
                return True
        return False

    def validate_for_of_optimization(self) -> None:
        if self.domain_policy != SAMPLED_BOUNDED:
            raise ValueError(
                "O/F optimization requires a bounded sampled chamber-property table"
            )
        if not self.has_meaningful_of_dependence():
            raise ValueError(
                "O/F optimization requires meaningful O/F dependence in at least "
                "one of gamma, Tc, or R_gas; the supplied table is flat"
            )


def constant_chamber_surfaces(*, gamma: float, Tc: float, R_gas: float,
                              Pc_range=(1.0e6, 8.0e6), OF_range=(1.5, 3.5),
                              ) -> ChamberSurfaces:
    """Flat surfaces from constants — the skeleton fallback when no CEA table
    is available (sandbox/CI).  Marked in provenance; NOT for final results."""
    xg = np.linspace(Pc_range[0], Pc_range[1], 4)
    yg = np.linspace(OF_range[0], OF_range[1], 4)
    ones = np.ones((xg.size, yg.size))
    arrays = {
        "Pc_grid": xg, "OF_grid": yg, "gamma": gamma * ones,
        "Tc": Tc * ones, "R_gas": R_gas * ones,
    }
    metadata = {
        "schema_version": PROPERTY_TABLE_SCHEMA_VERSION,
        "interpolator_version": INTERPOLATOR_VERSION,
        "oxidizer": "unspecified",
        "fuel": "unspecified",
        "thermochemistry_mode": "constant",
        "of_convention": OF_CONVENTION,
        "units_json": json.dumps(_TABLE_UNITS, sort_keys=True),
    }
    return ChamberSurfaces(
        gamma=PropertySurface2D.build(xg, yg, gamma * ones, name="gamma"),
        Tc=PropertySurface2D.build(xg, yg, Tc * ones, name="Tc"),
        R_gas=PropertySurface2D.build(xg, yg, R_gas * ones, name="R_gas"),
        provenance="constant_fallback(skeleton)",
        domain_policy=CONSTANT_UNBOUNDED,
        thermochemistry_mode="constant",
        content_sha256=_table_digest(arrays, metadata),
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
    """Write a self-identifying, content-hashed frozen CEA property table."""

    arrays = {
        name: np.asarray(tables[name], dtype=float)
        for name in ("Pc_grid", "OF_grid", "gamma", "Tc", "R_gas")
    }
    metadata = {
        "schema_version": PROPERTY_TABLE_SCHEMA_VERSION,
        "interpolator_version": INTERPOLATOR_VERSION,
        "oxidizer": str(oxidizer),
        "fuel": str(fuel),
        "thermochemistry_mode": "frozen",
        "of_convention": OF_CONVENTION,
        "units_json": json.dumps(_TABLE_UNITS, sort_keys=True),
    }
    identity = _table_digest(arrays, metadata)
    np.savez(
        path,
        **arrays,
        **{name: np.asarray(value) for name, value in metadata.items()},
        content_sha256=np.asarray(identity),
    )


def _scalar_text(dat: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(dat[name])
    if value.shape != ():
        raise ValueError(f"property table metadata {name!r} must be scalar")
    return str(value.item())


def _validate_table_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    Pc = np.asarray(arrays["Pc_grid"], dtype=float)
    OF = np.asarray(arrays["OF_grid"], dtype=float)
    if Pc.ndim != 1 or OF.ndim != 1 or Pc.size < 2 or OF.size < 2:
        raise ValueError("Pc_grid and OF_grid must be 1-D with at least two nodes")
    if not np.all(np.isfinite(Pc)) or not np.all(np.diff(Pc) > 0.0):
        raise ValueError("Pc_grid must be finite and strictly increasing")
    if not np.all(np.isfinite(OF)) or not np.all(np.diff(OF) > 0.0):
        raise ValueError("OF_grid must be finite and strictly increasing")
    expected = (Pc.size, OF.size)
    for name in ("gamma", "Tc", "R_gas"):
        values = np.asarray(arrays[name], dtype=float)
        if values.shape != expected:
            raise ValueError(
                f"{name}: shape {values.shape} does not match grid {expected}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(np.asarray(arrays["gamma"], dtype=float) <= 1.0):
        raise ValueError("gamma must be > 1 throughout a perfect-gas CEA surface")
    if np.any(np.asarray(arrays["Tc"], dtype=float) <= 0.0):
        raise ValueError("Tc must be positive throughout the property table")
    if np.any(np.asarray(arrays["R_gas"], dtype=float) <= 0.0):
        raise ValueError("R_gas must be positive throughout the property table")


def _arrays_have_of_dependence(
    arrays: Mapping[str, np.ndarray], *, rtol: float = 1.0e-10
) -> bool:
    for name in ("gamma", "Tc", "R_gas"):
        values = np.asarray(arrays[name], dtype=float)
        variation = np.max(np.ptp(values, axis=1))
        reference = max(float(np.max(np.abs(values))), 1.0)
        if variation > rtol * reference:
            return True
    return False


def load_chamber_surfaces(
    path: str,
    *,
    expected_propellant: str | None = None,
    require_of_dependence: bool = False,
) -> ChamberSurfaces:
    """Load and validate a sampled CEA surface before granting model coverage.

    Validation covers schema/interpolator versions, units and O/F convention,
    oxidizer/fuel identity, finite monotone axes, array shapes and physical sign
    conditions, and the content SHA-256.  The file path is retained solely as
    provenance; :attr:`ChamberSurfaces.content_sha256` is the identity.
    """

    table_path = Path(path).expanduser()
    if not table_path.is_file():
        raise ValueError(f"chamber property table does not exist: {table_path}")
    required = {
        "Pc_grid", "OF_grid", "gamma", "Tc", "R_gas", "oxidizer", "fuel",
        "schema_version", "interpolator_version", "thermochemistry_mode",
        "of_convention", "units_json", "content_sha256",
    }
    try:
        with np.load(table_path, allow_pickle=False) as dat:
            missing = sorted(required - set(dat.files))
            if missing:
                raise ValueError(
                    "property table is missing required metadata/arrays: "
                    + ", ".join(missing)
                )
            arrays = {
                name: np.asarray(dat[name], dtype=float)
                for name in ("Pc_grid", "OF_grid", "gamma", "Tc", "R_gas")
            }
            metadata = {
                "schema_version": int(np.asarray(dat["schema_version"]).item()),
                "interpolator_version": _scalar_text(dat, "interpolator_version"),
                "oxidizer": _scalar_text(dat, "oxidizer"),
                "fuel": _scalar_text(dat, "fuel"),
                "thermochemistry_mode": _scalar_text(dat, "thermochemistry_mode"),
                "of_convention": _scalar_text(dat, "of_convention"),
                "units_json": _scalar_text(dat, "units_json"),
            }
            claimed_digest = _scalar_text(dat, "content_sha256")
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("property table"):
            raise
        raise ValueError(f"could not read chamber property table {table_path}: {exc}") from exc

    if metadata["schema_version"] != PROPERTY_TABLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported property table schema_version={metadata['schema_version']}; "
            f"expected {PROPERTY_TABLE_SCHEMA_VERSION}"
        )
    if metadata["interpolator_version"] != INTERPOLATOR_VERSION:
        raise ValueError(
            "property table interpolator_version does not match this evaluator"
        )
    if metadata["thermochemistry_mode"] != "frozen":
        raise ValueError("only frozen-chamber CEA property tables are supported")
    if metadata["of_convention"] != OF_CONVENTION:
        raise ValueError(
            f"property table O/F convention must be {OF_CONVENTION!r}"
        )
    try:
        units = json.loads(metadata["units_json"])
    except json.JSONDecodeError as exc:
        raise ValueError("property table units_json is invalid") from exc
    if units != _TABLE_UNITS:
        raise ValueError(
            f"property table units do not match the required SI contract: {units!r}"
        )
    _validate_table_arrays(arrays)
    if require_of_dependence and not _arrays_have_of_dependence(arrays):
        raise ValueError(
            "O/F optimization requires meaningful O/F dependence in at least "
            "one of gamma, Tc, or R_gas; the supplied table is flat"
        )
    actual_digest = _table_digest(arrays, metadata)
    if claimed_digest != actual_digest:
        raise ValueError(
            "property table content SHA-256 mismatch; the arrays or metadata "
            "were changed after the table was written"
        )
    if expected_propellant is not None:
        try:
            expected_oxidizer, expected_fuel = str(expected_propellant).split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"expected propellant identity must be OXIDIZER/FUEL, got "
                f"{expected_propellant!r}"
            ) from exc
        actual_pair = (
            _canonical_name(metadata["oxidizer"]), _canonical_name(metadata["fuel"])
        )
        expected_pair = (
            _canonical_name(expected_oxidizer), _canonical_name(expected_fuel)
        )
        if actual_pair != expected_pair:
            raise ValueError(
                "property table propellant identity mismatch: table stores "
                f"{metadata['oxidizer']}/{metadata['fuel']}, mission requires "
                f"{expected_propellant}"
            )

    surfaces = ChamberSurfaces(
        gamma=PropertySurface2D.build(arrays["Pc_grid"], arrays["OF_grid"],
                                      arrays["gamma"], name="gamma"),
        Tc=PropertySurface2D.build(arrays["Pc_grid"], arrays["OF_grid"],
                                   arrays["Tc"], name="Tc"),
        R_gas=PropertySurface2D.build(arrays["Pc_grid"], arrays["OF_grid"],
                                      arrays["R_gas"], name="R_gas"),
        provenance=f"cea_frozen_table:{table_path}",
        domain_policy=SAMPLED_BOUNDED,
        oxidizer=metadata["oxidizer"],
        fuel=metadata["fuel"],
        thermochemistry_mode=metadata["thermochemistry_mode"],
        of_convention=metadata["of_convention"],
        units_json=metadata["units_json"],
        schema_version=metadata["schema_version"],
        interpolator_version=metadata["interpolator_version"],
        content_sha256=actual_digest,
        source_path=str(table_path),
    )
    return surfaces
