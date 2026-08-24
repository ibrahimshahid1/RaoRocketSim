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

from dataclasses import dataclass, field
import hashlib
from importlib import resources
from io import BytesIO
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

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

CEA_PROPERTY_FIELDS = ("gamma", "Tc", "R_gas")

CEA_CATALOG_SENTINEL = "catalog"
CEA_CATALOG_MANIFEST_SCHEMA_VERSION = 1
CEA_CATALOG_MANIFEST_RESOURCE = "data/cea_tables/catalog_v1.json"


@dataclass(frozen=True)
class CEAFieldValidationTolerance:
    """Acceptance limits for one held-out CEA property field.

    Gradient absolute tolerances are axis-specific because ``d/dPc`` and
    ``d/dOF`` have different dimensions and numerical scales.  A comparison
    passes when ``abs(error) <= atol + rtol * abs(reference)``.  Keeping every
    number explicit prevents a single permissive percentage from hiding a bad
    near-zero derivative.
    """

    value_rtol: float
    value_atol: float
    dPc_rtol: float
    dPc_atol: float
    dOF_rtol: float
    dOF_atol: float

    def __post_init__(self) -> None:
        for name in (
            "value_rtol", "value_atol", "dPc_rtol", "dPc_atol",
            "dOF_rtol", "dOF_atol",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.value_atol == self.value_rtol == 0.0:
            raise ValueError("value validation needs a nonzero tolerance")
        if self.dPc_atol == self.dPc_rtol == 0.0:
            raise ValueError("d/dPc validation needs a nonzero tolerance")
        if self.dOF_atol == self.dOF_rtol == 0.0:
            raise ValueError("d/dOF validation needs a nonzero tolerance")

    def as_dict(self) -> dict[str, float]:
        return {
            "value_rtol": float(self.value_rtol),
            "value_atol": float(self.value_atol),
            "dPc_rtol": float(self.dPc_rtol),
            "dPc_atol": float(self.dPc_atol),
            "dOF_rtol": float(self.dOF_rtol),
            "dOF_atol": float(self.dOF_atol),
        }


# Engineering acceptance limits for the Phase-2 interpolation gate.  These do
# not describe RocketCEA uncertainty; they bound error introduced by replacing
# live RocketCEA with the differentiable surface.  Gradient relative limits are
# deliberately looser than value limits, while the axis-specific absolute
# limits keep extrema/flat directions well posed.  A production table that
# misses these limits is refined; the gate is never weakened implicitly.
DEFAULT_CEA_VALIDATION_TOLERANCES: Mapping[
    str, CEAFieldValidationTolerance
] = {
    "gamma": CEAFieldValidationTolerance(
        value_rtol=5.0e-3,
        value_atol=5.0e-4,
        dPc_rtol=2.5e-1,
        dPc_atol=2.0e-10,       # 1/Pa
        dOF_rtol=2.5e-1,
        dOF_atol=2.0e-3,        # 1/(O/F)
    ),
    "Tc": CEAFieldValidationTolerance(
        value_rtol=5.0e-3,
        value_atol=5.0,         # K
        dPc_rtol=2.5e-1,
        dPc_atol=2.0e-6,        # K/Pa
        dOF_rtol=2.5e-1,
        dOF_atol=10.0,          # K/(O/F)
    ),
    "R_gas": CEAFieldValidationTolerance(
        value_rtol=5.0e-3,
        value_atol=5.0e-1,      # J/(kg K)
        dPc_rtol=2.5e-1,
        dPc_atol=2.0e-7,        # J/(kg K Pa)
        dOF_rtol=2.5e-1,
        dOF_atol=5.0e-1,        # J/(kg K)/(O/F)
    ),
}


class CEAPointSampler(Protocol):
    """Host-only point sampler used by table generation and validation.

    ``oxidizer`` and ``fuel`` are RocketCEA card identifiers (not necessarily
    the human-readable catalog spellings retained in table metadata).
    """

    def __call__(
        self, *, Pc: float, OF: float, oxidizer: str, fuel: str
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class CEAMetricValidationSummary:
    """Worst-case result for one field/quantity over all held-out points."""

    checked: int
    failures: int
    max_abs_error: float
    max_relative_error: float
    max_tolerance_ratio: float
    worst_point: tuple[float, float]
    surrogate_at_worst: float
    reference_at_worst: float

    @property
    def passed(self) -> bool:
        return self.failures == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checked": int(self.checked),
            "failures": int(self.failures),
            "max_abs_error": float(self.max_abs_error),
            "max_relative_error": float(self.max_relative_error),
            "max_tolerance_ratio": float(self.max_tolerance_ratio),
            "worst_point": {
                "Pc_Pa": float(self.worst_point[0]),
                "OF": float(self.worst_point[1]),
            },
            "surrogate_at_worst": float(self.surrogate_at_worst),
            "reference_at_worst": float(self.reference_at_worst),
        }


@dataclass(frozen=True)
class CEAFieldValidationSummary:
    value: CEAMetricValidationSummary
    dPc: CEAMetricValidationSummary
    dOF: CEAMetricValidationSummary

    @property
    def passed(self) -> bool:
        return self.value.passed and self.dPc.passed and self.dOF.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "value": self.value.as_dict(),
            "dPc": self.dPc.as_dict(),
            "dOF": self.dOF.as_dict(),
        }


@dataclass(frozen=True)
class CEAHoldoutValidationReport:
    """Auditable result of the Phase-2 held-out RocketCEA gate."""

    passed: bool
    table_path: str
    table_content_sha256: str
    table_schema_version: int
    interpolator_version: str
    oxidizer: str
    fuel: str
    rocketcea_oxidizer: str
    rocketcea_fuel: str
    backend_notes: tuple[str, ...]
    backend: str
    holdout_points: tuple[tuple[float, float], ...]
    finite_difference_fraction: float
    selection_seed: int | None
    tolerances: Mapping[str, CEAFieldValidationTolerance]
    fields: Mapping[str, CEAFieldValidationSummary]

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": "phase_2_heldout_cea_values_and_gradients",
            "passed": bool(self.passed),
            "table": {
                "path": self.table_path,
                "content_sha256": self.table_content_sha256,
                "schema_version": int(self.table_schema_version),
                "interpolator_version": self.interpolator_version,
            },
            "propellant": {
                "oxidizer": self.oxidizer,
                "fuel": self.fuel,
                "rocketcea_oxidizer": self.rocketcea_oxidizer,
                "rocketcea_fuel": self.rocketcea_fuel,
            },
            "backend": self.backend,
            "backend_notes": list(self.backend_notes),
            "acceptance_rule": (
                "abs(surrogate-reference) <= atol + rtol*abs(reference)"
            ),
            "reference_gradient_method": (
                "second_order_centered_backend_difference_within_fit_cell"
            ),
            "quantity_units": {
                "gamma": {"value": "1", "dPc": "1/Pa", "dOF": "1/(O/F)"},
                "Tc": {"value": "K", "dPc": "K/Pa", "dOF": "K/(O/F)"},
                "R_gas": {
                    "value": "J/(kg*K)",
                    "dPc": "J/(kg*K*Pa)",
                    "dOF": "J/(kg*K)/(O/F)",
                },
            },
            "holdout": {
                "count": len(self.holdout_points),
                "selection_seed": self.selection_seed,
                "finite_difference_fraction": self.finite_difference_fraction,
                "points": [
                    {"Pc_Pa": float(pc), "OF": float(of)}
                    for pc, of in self.holdout_points
                ],
            },
            "tolerances": {
                name: tolerance.as_dict()
                for name, tolerance in self.tolerances.items()
            },
            "fields": {
                name: summary.as_dict()
                for name, summary in self.fields.items()
            },
        }


@dataclass(frozen=True)
class CatalogCEATableResource:
    """One hash-verified CEA table resolved from installed package data.

    The logical package name is retained as provenance.  Raw bytes are private
    so the loader works for both ordinary wheel installs and non-filesystem
    ``importlib.resources`` providers without rewriting a mission's stable
    ``cea_table_path='catalog'`` intent to an installation-specific path.
    """

    propellant: str
    resource_name: str
    package_path: str
    catalog_version: str
    table_content_sha256: str
    file_sha256: str
    manifest_entry: Mapping[str, object]
    _payload: bytes = field(repr=False, compare=False)


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
_ROCKETCEA_CATALOG_CARD_PAIRS = {
    # Keep the catalog/table identity human-readable while passing identifiers
    # that RocketCEA 1.2.x actually registers.  In particular, ``LCH4`` is not
    # a stock fuel card; ``CH4`` is.  RP-1 currently aliases successfully, but
    # the registered ``RP1`` spelling is less version-dependent.
    ("lox", "rp1"): ("LOX", "RP1"),
    ("lox", "lch4"): ("LOX", "CH4"),
    ("lox", "lh2"): ("LOX", "LH2"),
    ("n2o4", "mmh"): ("N2O4", "MMH"),
    ("n2o", "ethanol"): ("N2O", "Ethanol"),
}


def rocketcea_card_names(oxidizer: str, fuel: str) -> tuple[str, str]:
    """Resolve catalog spellings to RocketCEA's stock card identifiers.

    Unknown pairs pass through unchanged so custom RocketCEA cards remain
    usable.  The resolved names are only backend inputs: :func:`save_tables`
    retains the caller's original names in its hashed metadata.
    """

    ox = str(oxidizer).strip()
    fu = str(fuel).strip()
    if not ox or not fu:
        raise ValueError("oxidizer and fuel names must be nonempty")
    return _ROCKETCEA_CATALOG_CARD_PAIRS.get(
        (_canonical_name(ox), _canonical_name(fu)), (ox, fu)
    )


def rocketcea_card_notes(oxidizer: str, fuel: str) -> tuple[str, ...]:
    """Return reproducibility/phase notes for non-obvious stock-card choices."""

    pair = (_canonical_name(oxidizer), _canonical_name(fuel))
    if pair == ("lox", "lch4"):
        return (
            "RocketCEA has no stock LCH4 identifier; CH4 is its liquid "
            "methane card (CH4(L), 111.66 K), while GCH4 is a distinct gas card.",
        )
    if pair == ("n2o", "ethanol"):
        return (
            "RocketCEA's stock N2O card is the 298.15 K gas enthalpy card; "
            "N2O_nbp is a distinct 184.4 K card. Pin the oxidizer inlet state "
            "or a custom liquid card before certifying a production table.",
        )
    return ()


def _coerce_cea_point_sample(
    sample: Mapping[str, float], *, Pc: float, OF: float
) -> dict[str, float]:
    missing = [name for name in CEA_PROPERTY_FIELDS if name not in sample]
    if missing:
        raise ValueError(
            "CEA point sampler omitted " + ", ".join(missing)
            + f" at Pc={Pc:.9g} Pa, O/F={OF:.9g}"
        )
    result = {name: float(sample[name]) for name in CEA_PROPERTY_FIELDS}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(
            f"CEA point sampler returned a non-finite value at "
            f"Pc={Pc:.9g} Pa, O/F={OF:.9g}"
        )
    if result["gamma"] <= 1.0 or result["Tc"] <= 0.0 or result["R_gas"] <= 0.0:
        raise ValueError(
            f"CEA point sampler returned an inadmissible chamber state at "
            f"Pc={Pc:.9g} Pa, O/F={OF:.9g}: {result}"
        )
    return result


def sample_cea_chamber_point(
    *, Pc: float, OF: float, oxidizer: str, fuel: str
) -> dict[str, float]:
    """Sample one frozen chamber state from RocketCEA (host-only)."""

    from raosim.cea import cea_propellant  # deferred: optional dependency

    rocket_ox, rocket_fuel = rocketcea_card_names(oxidizer, fuel)
    prop = cea_propellant(
        oxidizer=rocket_ox,
        fuel=rocket_fuel,
        Pc=float(Pc),
        mixture_ratio=float(OF),
    )
    return {
        "gamma": float(prop.gamma),
        "Tc": float(prop.Tc),
        "R_gas": 8.314462618 / float(prop.Mw),
    }


def sample_cea_tables(Pc_grid, OF_grid, *, oxidizer: str, fuel: str,
                      sampler: CEAPointSampler | None = None,
                      ) -> dict[str, np.ndarray]:
    """Sample frozen-chamber properties on the grid via ``raosim.cea``.

    Host-only (RocketCEA + Fortran CEA backend); never called from jitted
    code.  Returns plain arrays for ``save_tables``/``build_chamber_surfaces``.
    Equilibrium sampling is deliberately NOT provided here yet — the
    constant-gamma solver cannot consume it (RQ4-B; evaluation report §D.3.2).

    ``sampler`` is an injection seam for deterministic tests and alternate
    RocketCEA hosts.  It receives resolved RocketCEA card identifiers and must
    return scalar ``gamma``, ``Tc`` and ``R_gas`` values.
    """

    Pc_grid = np.asarray(Pc_grid, float)
    OF_grid = np.asarray(OF_grid, float)
    if Pc_grid.ndim != 1 or OF_grid.ndim != 1:
        raise ValueError("Pc_grid and OF_grid must be one-dimensional")
    if Pc_grid.size < 2 or OF_grid.size < 2:
        raise ValueError("Pc_grid and OF_grid need at least two nodes")
    if not np.all(np.isfinite(Pc_grid)) or not np.all(np.diff(Pc_grid) > 0.0):
        raise ValueError("Pc_grid must be finite and strictly increasing")
    if not np.all(np.isfinite(OF_grid)) or not np.all(np.diff(OF_grid) > 0.0):
        raise ValueError("OF_grid must be finite and strictly increasing")

    point_sampler = sample_cea_chamber_point if sampler is None else sampler
    rocket_ox, rocket_fuel = rocketcea_card_names(oxidizer, fuel)
    gamma = np.zeros((Pc_grid.size, OF_grid.size))
    Tc = np.zeros_like(gamma)
    R_gas = np.zeros_like(gamma)
    for i, Pc in enumerate(Pc_grid):
        for j, OF in enumerate(OF_grid):
            sample = _coerce_cea_point_sample(
                point_sampler(
                    Pc=float(Pc),
                    OF=float(OF),
                    oxidizer=rocket_ox,
                    fuel=rocket_fuel,
                ),
                Pc=float(Pc),
                OF=float(OF),
            )
            gamma[i, j] = sample["gamma"]
            Tc[i, j] = sample["Tc"]
            R_gas[i, j] = sample["R_gas"]
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


def _read_table_payload(
    source: object, *, source_label: str
) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    """Decode one table from a path or an in-memory package resource."""

    required = {
        "Pc_grid", "OF_grid", "gamma", "Tc", "R_gas", "oxidizer", "fuel",
        "schema_version", "interpolator_version", "thermochemistry_mode",
        "of_convention", "units_json", "content_sha256",
    }
    try:
        with np.load(source, allow_pickle=False) as dat:
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
            metadata: dict[str, object] = {
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
        raise ValueError(
            f"could not read chamber property table {source_label}: {exc}"
        ) from exc
    return arrays, metadata, claimed_digest


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


def _validate_table_contract(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    claimed_digest: str,
    *,
    expected_propellant: str | None,
    require_of_dependence: bool,
) -> str:
    """Validate the table contract and return its recomputed logical hash."""

    if metadata["schema_version"] != PROPERTY_TABLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported property table schema_version="
            f"{metadata['schema_version']}; expected "
            f"{PROPERTY_TABLE_SCHEMA_VERSION}"
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
        units = json.loads(str(metadata["units_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("property table units_json is invalid") from exc
    if units != _TABLE_UNITS:
        raise ValueError(
            f"property table units do not match the required SI contract: "
            f"{units!r}"
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
            expected_oxidizer, expected_fuel = str(expected_propellant).split(
                "/", 1
            )
        except ValueError as exc:
            raise ValueError(
                "expected propellant identity must be OXIDIZER/FUEL, got "
                f"{expected_propellant!r}"
            ) from exc
        actual_pair = (
            _canonical_name(metadata["oxidizer"]),
            _canonical_name(metadata["fuel"]),
        )
        expected_pair = (
            _canonical_name(expected_oxidizer),
            _canonical_name(expected_fuel),
        )
        if actual_pair != expected_pair:
            raise ValueError(
                "property table propellant identity mismatch: table stores "
                f"{metadata['oxidizer']}/{metadata['fuel']}, mission requires "
                f"{expected_propellant}"
            )
    return actual_digest


def _catalog_resource(relative_path: str):
    resource = resources.files("raosim")
    for part in relative_path.split("/"):
        resource = resource.joinpath(part)
    return resource


def load_cea_catalog_manifest() -> dict[str, object]:
    """Load and minimally validate the installed CEA catalog manifest."""

    manifest_resource = _catalog_resource(CEA_CATALOG_MANIFEST_RESOURCE)
    try:
        payload = json.loads(manifest_resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "could not read installed CEA catalog manifest "
            f"{CEA_CATALOG_MANIFEST_RESOURCE}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("installed CEA catalog manifest must be a JSON object")
    if payload.get("manifest_schema_version") != (
        CEA_CATALOG_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported CEA catalog manifest_schema_version="
            f"{payload.get('manifest_schema_version')!r}; expected "
            f"{CEA_CATALOG_MANIFEST_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("catalog_version"), str) or not payload[
        "catalog_version"
    ]:
        raise ValueError("CEA catalog manifest needs a nonempty catalog_version")
    if not isinstance(payload.get("tables"), list):
        raise ValueError("CEA catalog manifest tables must be a list")
    if not isinstance(payload.get("blocked_pairs"), list):
        raise ValueError("CEA catalog manifest blocked_pairs must be a list")
    return payload


def _canonical_pair_spelling(propellant: str) -> tuple[str, str]:
    """Canonicalize a manifest-controlled OXIDIZER/FUEL spelling."""

    try:
        oxidizer, fuel = str(propellant).strip().split("/", 1)
    except ValueError as exc:
        raise ValueError(
            "catalog chamber-property resolution requires a selected "
            f"OXIDIZER/FUEL propellant, got {propellant!r}"
        ) from exc
    return _canonical_name(oxidizer), _canonical_name(fuel)


def _canonical_catalog_pair(propellant: str) -> tuple[str, str]:
    """Resolve only repository-catalog names/aliases to a certified pair.

    Reactant spelling aliases alone are not certification.  In particular, a
    runtime-registered propellant must not inherit a packaged table merely
    because its name canonicalizes to the same oxidizer and fuel tokens.
    """

    from raosim.mdo.propellants import get_propellant, registered_propellants

    requested = str(propellant).strip()
    requested_key = requested.lower().replace(" ", "")
    # ``get_propellant`` intentionally expands public aliases before lookup.
    # Check the exact runtime-registration key first so a custom named like a
    # public alias (for example LOX/CH4) cannot be mistaken for the built-in
    # LOX/LCH4 entry and inherit its certification.
    if requested_key in registered_propellants():
        raise ValueError(
            f"catalog chamber-property resolution refuses user-registered "
            f"propellant {requested!r}: certified table identity cannot be "
            "inherited through reactant-name aliases"
        )
    try:
        spec = get_propellant(requested)
    except KeyError as exc:
        raise ValueError(
            "CEA catalog has no validated chamber-property table for "
            f"{propellant!r}; use a repository catalog name or public alias"
        ) from exc
    if spec.provenance != "catalog":
        raise ValueError(
            f"catalog chamber-property resolution refuses user-registered "
            f"propellant {spec.name!r}: certified table identity cannot be "
            "inherited through reactant-name aliases"
        )
    return _canonical_pair_spelling(spec.name)


def _catalog_entry_pair(entry: Mapping[str, object], *, label: str) -> tuple[str, str]:
    propellant = entry.get("propellant")
    pair = entry.get("pair")
    if not isinstance(propellant, str) or not isinstance(pair, dict):
        raise ValueError(
            f"CEA catalog {label} needs string propellant and object pair fields"
        )
    if not isinstance(pair.get("oxidizer"), str) or not isinstance(
        pair.get("fuel"), str
    ):
        raise ValueError(
            f"CEA catalog {label} pair needs string oxidizer and fuel fields"
        )
    pair_from_name = _canonical_pair_spelling(propellant)
    pair_from_fields = (
        _canonical_name(pair["oxidizer"]), _canonical_name(pair["fuel"])
    )
    if pair_from_name != pair_from_fields:
        raise ValueError(
            f"CEA catalog {label} propellant and pair fields disagree"
        )
    return pair_from_name


def _catalog_mapping(
    entry: Mapping[str, object], name: str, *, label: str
) -> Mapping[str, object]:
    value = entry.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"CEA catalog {label} field {name!r} must be an object")
    return value


def _catalog_sha256(value: object, *, label: str) -> str:
    text_value = str(value)
    if len(text_value) != 64 or any(
        char not in "0123456789abcdef" for char in text_value
    ):
        raise ValueError(f"CEA catalog {label} must be a lowercase SHA-256")
    return text_value


def resolve_catalog_cea_table(propellant: str) -> CatalogCEATableResource:
    """Resolve and hash-check an installed production table for ``propellant``.

    Resolution is fail-closed: only manifest entries with a passed complete-cell
    held-out gate are eligible.  Both the exact packaged file hash and the
    logical table content hash are recomputed before the resource is returned.
    """

    manifest = load_cea_catalog_manifest()
    requested_pair = _canonical_catalog_pair(propellant)
    selected: Mapping[str, object] | None = None
    seen: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(manifest["tables"]):
        label = f"tables[{index}]"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"CEA catalog {label} must be an object")
        entry_pair = _catalog_entry_pair(raw_entry, label=label)
        if entry_pair in seen:
            raise ValueError(
                "CEA catalog contains duplicate table identity "
                f"{raw_entry['propellant']!r}"
            )
        seen.add(entry_pair)
        if entry_pair == requested_pair:
            selected = raw_entry

    if selected is None:
        for index, raw_blocked in enumerate(manifest["blocked_pairs"]):
            label = f"blocked_pairs[{index}]"
            if not isinstance(raw_blocked, dict):
                raise ValueError(f"CEA catalog {label} must be an object")
            blocked_name = raw_blocked.get("propellant")
            reason = raw_blocked.get("reason")
            if not isinstance(blocked_name, str) or not isinstance(reason, str):
                raise ValueError(
                    f"CEA catalog {label} needs string propellant and reason"
                )
            if _canonical_pair_spelling(blocked_name) == requested_pair:
                raise ValueError(
                    f"catalog chamber-property table for {blocked_name} is "
                    f"unavailable: {reason}"
                )
        raise ValueError(
            "CEA catalog has no validated chamber-property table for "
            f"{propellant!r}"
        )

    selected_name = str(selected["propellant"])
    label = f"table {selected_name!r}"
    resource_name = selected.get("resource")
    if (
        not isinstance(resource_name, str)
        or not resource_name
        or "/" in resource_name
        or "\\" in resource_name
        or resource_name in {".", ".."}
    ):
        raise ValueError(f"CEA catalog {label} has an unsafe resource name")

    table_info = _catalog_mapping(selected, "table", label=label)
    if table_info.get("schema_version") != PROPERTY_TABLE_SCHEMA_VERSION:
        raise ValueError(
            f"CEA catalog {label} table schema does not match this evaluator"
        )
    if table_info.get("interpolator_version") != INTERPOLATOR_VERSION:
        raise ValueError(
            f"CEA catalog {label} interpolator does not match this evaluator"
        )
    if table_info.get("thermochemistry_mode") != "frozen":
        raise ValueError(f"CEA catalog {label} must select frozen thermochemistry")
    expected_content_sha = _catalog_sha256(
        table_info.get("content_sha256"), label=f"{label} content_sha256"
    )
    expected_file_sha = _catalog_sha256(
        table_info.get("file_sha256"), label=f"{label} file_sha256"
    )

    validation = _catalog_mapping(selected, "validation", label=label)
    if validation.get("status") != "passed" or validation.get(
        "complete_cell_coverage"
    ) is not True:
        raise ValueError(
            f"CEA catalog {label} is not eligible: held-out validation has "
            "not passed with complete-cell coverage"
        )
    if validation.get("gate") != "phase_2_heldout_cea_values_and_gradients":
        raise ValueError(f"CEA catalog {label} has an unsupported validation gate")
    worst_ratio_raw = validation.get("worst_tolerance_ratio")
    try:
        worst_ratio = (
            float("nan")
            if isinstance(worst_ratio_raw, bool)
            else float(worst_ratio_raw)
        )
    except (TypeError, ValueError):
        worst_ratio = float("nan")
    if not np.isfinite(worst_ratio) or not (0.0 <= worst_ratio <= 1.0):
        raise ValueError(
            f"CEA catalog {label} worst_tolerance_ratio must be finite and "
            "within [0, 1] for a passed validation"
        )

    rocketcea = _catalog_mapping(selected, "rocketcea", label=label)
    rocketcea_version = rocketcea.get("version")
    if not isinstance(rocketcea_version, str) or not rocketcea_version:
        raise ValueError(f"CEA catalog {label} needs a RocketCEA version")
    pair = _catalog_mapping(selected, "pair", label=label)
    expected_cards = rocketcea_card_names(
        str(pair["oxidizer"]), str(pair["fuel"])
    )
    stored_cards = (
        rocketcea.get("oxidizer_card"), rocketcea.get("fuel_card")
    )
    if stored_cards != expected_cards:
        raise ValueError(
            f"CEA catalog {label} RocketCEA cards disagree with the registered "
            "catalog mapping"
        )

    package_path = f"raosim/data/cea_tables/{resource_name}"
    table_resource = _catalog_resource(f"data/cea_tables/{resource_name}")
    try:
        payload = table_resource.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(
            f"installed CEA catalog resource is missing: {package_path}"
        ) from exc
    file_sha = hashlib.sha256(payload).hexdigest()
    if file_sha != expected_file_sha:
        raise ValueError(
            f"installed CEA catalog file SHA-256 mismatch for {selected_name}: "
            "the packaged resource does not match the versioned manifest"
        )

    arrays, metadata, claimed_digest = _read_table_payload(
        BytesIO(payload), source_label=package_path
    )
    actual_digest = _validate_table_contract(
        arrays,
        metadata,
        claimed_digest,
        expected_propellant=selected_name,
        require_of_dependence=True,
    )
    if actual_digest != expected_content_sha:
        raise ValueError(
            f"installed CEA catalog content SHA-256 mismatch for {selected_name}: "
            "the table does not match the versioned manifest identity"
        )

    grid = _catalog_mapping(selected, "grid", label=label)
    domain = _catalog_mapping(selected, "domain", label=label)
    pc_domain = _catalog_mapping(domain, "Pc_Pa", label=f"{label} domain")
    of_domain = _catalog_mapping(domain, "OF", label=f"{label} domain")
    pc_grid = np.asarray(arrays["Pc_grid"], dtype=float)
    of_grid = np.asarray(arrays["OF_grid"], dtype=float)
    expected_cells = (pc_grid.size - 1) * (of_grid.size - 1)
    manifest_shape = (grid.get("Pc_nodes"), grid.get("OF_nodes"))
    if manifest_shape != (pc_grid.size, of_grid.size):
        raise ValueError(f"CEA catalog {label} grid metadata disagrees with table")
    manifest_domain = (
        pc_domain.get("min"), pc_domain.get("max"),
        of_domain.get("min"), of_domain.get("max"),
    )
    actual_domain = (pc_grid[0], pc_grid[-1], of_grid[0], of_grid[-1])
    try:
        domain_matches = np.array_equal(
            np.asarray(manifest_domain, dtype=float),
            np.asarray(actual_domain, dtype=float),
        )
    except (TypeError, ValueError):
        domain_matches = False
    if not domain_matches:
        raise ValueError(f"CEA catalog {label} domain metadata disagrees with table")
    if validation.get("cells_checked") != expected_cells:
        raise ValueError(
            f"CEA catalog {label} validation cell count disagrees with its grid"
        )

    return CatalogCEATableResource(
        propellant=selected_name,
        resource_name=resource_name,
        package_path=package_path,
        catalog_version=str(manifest["catalog_version"]),
        table_content_sha256=actual_digest,
        file_sha256=file_sha,
        manifest_entry=dict(selected),
        _payload=payload,
    )


def load_chamber_surfaces(
    path: str | Path,
    *,
    expected_propellant: str | None = None,
    require_of_dependence: bool = False,
) -> ChamberSurfaces:
    """Load and validate a sampled CEA surface before granting model coverage.

    Validation covers schema/interpolator versions, units and O/F convention,
    oxidizer/fuel identity, finite monotone axes, array shapes and physical sign
    conditions, and the content SHA-256.  The explicit ``"catalog"`` sentinel
    resolves a hash-pinned installed resource for ``expected_propellant``; it
    never changes the empty-path constant-property default.  The source name is
    retained solely as provenance; :attr:`ChamberSurfaces.content_sha256` is
    the model identity.
    """

    catalog_resource: CatalogCEATableResource | None = None
    contract_propellant = expected_propellant
    if str(path) == CEA_CATALOG_SENTINEL:
        if expected_propellant is None:
            raise ValueError(
                "chamber_property_table='catalog' requires a selected mission "
                "propellant; no catalog table is assumed implicitly"
            )
        catalog_resource = resolve_catalog_cea_table(expected_propellant)
        # Public catalog aliases (for example ``kerolox``) resolve to one
        # canonical certified identity.  Validate the payload against that
        # identity rather than asking the lower-level explicit-path contract to
        # interpret the alias again.
        contract_propellant = catalog_resource.propellant
        source: object = BytesIO(catalog_resource._payload)
        source_label = catalog_resource.package_path
        provenance = (
            f"cea_frozen_catalog:{catalog_resource.propellant}:"
            f"{catalog_resource.resource_name}"
        )
    else:
        table_path = Path(path).expanduser()
        if not table_path.is_file():
            raise ValueError(
                f"chamber property table does not exist: {table_path}"
            )
        source = table_path
        source_label = str(table_path)
        provenance = f"cea_frozen_table:{table_path}"

    arrays, metadata, claimed_digest = _read_table_payload(
        source, source_label=source_label
    )
    actual_digest = _validate_table_contract(
        arrays,
        metadata,
        claimed_digest,
        expected_propellant=contract_propellant,
        require_of_dependence=require_of_dependence,
    )
    if catalog_resource is not None and actual_digest != (
        catalog_resource.table_content_sha256
    ):
        raise ValueError(
            "installed CEA catalog table changed between resolution and load"
        )

    surfaces = ChamberSurfaces(
        gamma=PropertySurface2D.build(arrays["Pc_grid"], arrays["OF_grid"],
                                      arrays["gamma"], name="gamma"),
        Tc=PropertySurface2D.build(arrays["Pc_grid"], arrays["OF_grid"],
                                   arrays["Tc"], name="Tc"),
        R_gas=PropertySurface2D.build(arrays["Pc_grid"], arrays["OF_grid"],
                                      arrays["R_gas"], name="R_gas"),
        provenance=provenance,
        domain_policy=SAMPLED_BOUNDED,
        oxidizer=str(metadata["oxidizer"]),
        fuel=str(metadata["fuel"]),
        thermochemistry_mode=str(metadata["thermochemistry_mode"]),
        of_convention=str(metadata["of_convention"]),
        units_json=str(metadata["units_json"]),
        schema_version=int(metadata["schema_version"]),
        interpolator_version=str(metadata["interpolator_version"]),
        content_sha256=actual_digest,
        source_path=source_label,
    )
    return surfaces


# --------------------------------------------------------------------------- #
# Held-out authoritative validation (host-only; requires RocketCEA by default) #
# --------------------------------------------------------------------------- #
def select_cea_holdout_points(
    Pc_grid: Sequence[float],
    OF_grid: Sequence[float],
    *,
    max_points: int | None = None,
    seed: int = 20260817,
) -> tuple[tuple[float, float], ...]:
    """Choose deterministic off-node points spread across table cells.

    At most one point is selected per cell.  Randomness is seeded and each
    coordinate lies in the middle 50% of its cell, leaving room for the
    central-difference stencil used by :func:`validate_cea_surface_holdout`.
    None of these points can be a grid node used in the fit.
    """

    Pc = np.asarray(Pc_grid, dtype=float)
    OF = np.asarray(OF_grid, dtype=float)
    if Pc.ndim != 1 or OF.ndim != 1 or Pc.size < 2 or OF.size < 2:
        raise ValueError("holdout selection needs two 1-D grids with >=2 nodes")
    if not np.all(np.diff(Pc) > 0.0) or not np.all(np.diff(OF) > 0.0):
        raise ValueError("holdout grids must be strictly increasing")
    cell_count = (Pc.size - 1) * (OF.size - 1)
    if max_points is None:
        count = cell_count
    else:
        count = int(max_points)
        if count <= 0:
            raise ValueError("max_points must be positive or None")
        count = min(count, cell_count)

    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(cell_count, size=count, replace=False))
    fractions = rng.uniform(0.25, 0.75, size=(count, 2))
    n_of_cells = OF.size - 1
    points: list[tuple[float, float]] = []
    for k, flat_index in enumerate(chosen):
        i, j = divmod(int(flat_index), n_of_cells)
        pc = Pc[i] + fractions[k, 0] * (Pc[i + 1] - Pc[i])
        of = OF[j] + fractions[k, 1] * (OF[j + 1] - OF[j])
        points.append((float(pc), float(of)))
    return tuple(points)


def _open_cell(
    value: float, grid: np.ndarray, *, axis_name: str
) -> tuple[int, float, float]:
    if not np.isfinite(value) or not (grid[0] < value < grid[-1]):
        raise ValueError(
            f"held-out {axis_name}={value!r} must be strictly inside "
            f"({grid[0]:.9g}, {grid[-1]:.9g})"
        )
    node_atol = 32.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(grid))), 1.0
    )
    if np.any(np.isclose(value, grid, rtol=0.0, atol=node_atol)):
        raise ValueError(
            f"held-out {axis_name}={value:.9g} is a fit-grid node"
        )
    index = int(np.searchsorted(grid, value, side="right") - 1)
    return index, float(grid[index]), float(grid[index + 1])


def _normalize_validation_tolerances(
    tolerances: Mapping[str, CEAFieldValidationTolerance] | None,
) -> dict[str, CEAFieldValidationTolerance]:
    selected = dict(
        DEFAULT_CEA_VALIDATION_TOLERANCES if tolerances is None else tolerances
    )
    missing = sorted(set(CEA_PROPERTY_FIELDS) - set(selected))
    extra = sorted(set(selected) - set(CEA_PROPERTY_FIELDS))
    if missing or extra:
        pieces = []
        if missing:
            pieces.append("missing " + ", ".join(missing))
        if extra:
            pieces.append("unknown " + ", ".join(extra))
        raise ValueError("CEA validation tolerances: " + "; ".join(pieces))
    for name, tolerance in selected.items():
        if not isinstance(tolerance, CEAFieldValidationTolerance):
            raise TypeError(
                f"CEA validation tolerance for {name} must be "
                "CEAFieldValidationTolerance"
            )
    return selected


def cea_validation_tolerances_from_mapping(
    value: Mapping[str, Mapping[str, float]],
) -> dict[str, CEAFieldValidationTolerance]:
    """Parse a complete JSON-like field-specific tolerance mapping."""

    parsed = {
        str(name): CEAFieldValidationTolerance(**dict(config))
        for name, config in value.items()
    }
    return _normalize_validation_tolerances(parsed)


_MetricRecord = tuple[
    float, float, float, tuple[float, float], float, float
]


def _metric_summary(records: Sequence[_MetricRecord]) -> CEAMetricValidationSummary:
    if not records:
        raise ValueError("cannot summarize an empty validation metric")
    worst = max(records, key=lambda record: record[2])
    return CEAMetricValidationSummary(
        checked=len(records),
        failures=sum(record[2] > 1.0 for record in records),
        max_abs_error=max(record[0] for record in records),
        max_relative_error=max(record[1] for record in records),
        max_tolerance_ratio=worst[2],
        worst_point=worst[3],
        surrogate_at_worst=worst[4],
        reference_at_worst=worst[5],
    )


def _metric_record(
    surrogate: float,
    reference: float,
    *,
    rtol: float,
    atol: float,
    point: tuple[float, float],
) -> _MetricRecord:
    absolute_error = abs(float(surrogate) - float(reference))
    relative_error = absolute_error / max(abs(float(reference)), 1.0e-300)
    acceptance = float(atol) + float(rtol) * abs(float(reference))
    if acceptance == 0.0:
        # A pure-relative tolerance at an exactly zero reference permits only
        # exact agreement.  Use a finite sentinel for a mismatch so the JSON
        # report remains standards-compliant (rather than emitting Infinity).
        tolerance_ratio = (
            0.0 if absolute_error == 0.0 else np.finfo(float).max
        )
    else:
        tolerance_ratio = absolute_error / acceptance
    return (
        absolute_error,
        relative_error,
        tolerance_ratio,
        point,
        float(surrogate),
        float(reference),
    )


def validate_cea_surface_holdout(
    path: str,
    *,
    expected_propellant: str | None = None,
    sampler: CEAPointSampler | None = None,
    points: Sequence[tuple[float, float]] | None = None,
    max_points: int | None = None,
    seed: int = 20260817,
    finite_difference_fraction: float = 0.20,
    tolerances: Mapping[str, CEAFieldValidationTolerance] | None = None,
) -> CEAHoldoutValidationReport:
    """Validate surface values and gradients against held-out RocketCEA calls.

    The table is loaded through :func:`load_chamber_surfaces`, so schema,
    units, propellant identity and content SHA-256 are verified before any
    comparison.  Values are sampled directly at interior points not used in
    the fit.  Authoritative gradients use centered RocketCEA differences whose
    stencil remains strictly inside the same interpolation cell; those stencil
    points are also not fit nodes.

    Parameters
    ----------
    sampler
        Optional point-sampler injection.  ``None`` invokes live RocketCEA.
        Tests can supply an analytic oracle without installing RocketCEA.
    points
        Optional explicit held-out ``(Pc, O/F)`` points.  Otherwise a seeded,
        cell-stratified selection is generated.
    finite_difference_fraction
        Step as a fraction of the nearer cell-edge distance on each axis.
    """

    fraction = float(finite_difference_fraction)
    if not np.isfinite(fraction) or not (0.0 < fraction < 1.0):
        raise ValueError("finite_difference_fraction must lie strictly in (0, 1)")
    selected_tolerances = _normalize_validation_tolerances(tolerances)
    surfaces = load_chamber_surfaces(
        path, expected_propellant=expected_propellant
    )
    Pc_grid = np.asarray(surfaces.gamma.xg, dtype=float)
    OF_grid = np.asarray(surfaces.gamma.yg, dtype=float)
    if points is None:
        heldout = select_cea_holdout_points(
            Pc_grid, OF_grid, max_points=max_points, seed=seed
        )
        selection_seed: int | None = int(seed)
    else:
        heldout = tuple((float(pc), float(of)) for pc, of in points)
        if not heldout:
            raise ValueError("at least one held-out point is required")
        selection_seed = None

    cells: list[tuple[int, int, float, float, float, float]] = []
    for pc, of in heldout:
        i, pc_left, pc_right = _open_cell(pc, Pc_grid, axis_name="Pc")
        j, of_left, of_right = _open_cell(of, OF_grid, axis_name="O/F")
        cells.append((i, j, pc_left, pc_right, of_left, of_right))

    rocket_ox, rocket_fuel = rocketcea_card_names(
        surfaces.oxidizer, surfaces.fuel
    )
    backend_notes = rocketcea_card_notes(surfaces.oxidizer, surfaces.fuel)
    point_sampler = sample_cea_chamber_point if sampler is None else sampler
    if sampler is None:
        import rocketcea  # type: ignore  # host-only optional dependency

        rocketcea_version = getattr(rocketcea, "__version__", "unknown")
        backend_name = f"RocketCEA {rocketcea_version}"
    else:
        backend_name = getattr(
            sampler, "__qualname__", sampler.__class__.__name__
        )
    cache: dict[tuple[float, float], dict[str, float]] = {}

    def authoritative(pc: float, of: float) -> dict[str, float]:
        key = (float(pc), float(of))
        if key not in cache:
            cache[key] = _coerce_cea_point_sample(
                point_sampler(
                    Pc=key[0], OF=key[1],
                    oxidizer=rocket_ox, fuel=rocket_fuel,
                ),
                Pc=key[0], OF=key[1],
            )
        return cache[key]

    surface_by_field = {
        "gamma": surfaces.gamma,
        "Tc": surfaces.Tc,
        "R_gas": surfaces.R_gas,
    }
    gradient_by_field: dict[str, Callable[[Array, Array], tuple[Array, Array]]] = {
        name: jax.grad(
            lambda pc, of, surface=surface: surface(pc, of),
            argnums=(0, 1),
        )
        for name, surface in surface_by_field.items()
    }
    records: dict[str, dict[str, list[_MetricRecord]]] = {
        name: {"value": [], "dPc": [], "dOF": []}
        for name in CEA_PROPERTY_FIELDS
    }

    for point, cell in zip(heldout, cells):
        pc, of = point
        _, _, pc_left, pc_right, of_left, of_right = cell
        h_pc = fraction * min(pc - pc_left, pc_right - pc)
        h_of = fraction * min(of - of_left, of_right - of)
        pc_resolution = 64.0 * np.finfo(float).eps * max(abs(pc), 1.0)
        of_resolution = 64.0 * np.finfo(float).eps * max(abs(of), 1.0)
        if h_pc <= pc_resolution or h_of <= of_resolution:
            raise ValueError(
                f"held-out point {point!r} is too close to a cell edge for "
                "a stable central-difference stencil"
            )

        center = authoritative(pc, of)
        pc_minus = authoritative(pc - h_pc, of)
        pc_plus = authoritative(pc + h_pc, of)
        of_minus = authoritative(pc, of - h_of)
        of_plus = authoritative(pc, of + h_of)
        for field in CEA_PROPERTY_FIELDS:
            tolerance = selected_tolerances[field]
            surrogate_value = float(surface_by_field[field](pc, of))
            surrogate_dpc, surrogate_dof = (
                float(value)
                for value in gradient_by_field[field](
                    jnp.asarray(pc, dtype=jnp.float64),
                    jnp.asarray(of, dtype=jnp.float64),
                )
            )
            reference_dpc = (
                pc_plus[field] - pc_minus[field]
            ) / (2.0 * h_pc)
            reference_dof = (
                of_plus[field] - of_minus[field]
            ) / (2.0 * h_of)
            records[field]["value"].append(_metric_record(
                surrogate_value,
                center[field],
                rtol=tolerance.value_rtol,
                atol=tolerance.value_atol,
                point=point,
            ))
            records[field]["dPc"].append(_metric_record(
                surrogate_dpc,
                reference_dpc,
                rtol=tolerance.dPc_rtol,
                atol=tolerance.dPc_atol,
                point=point,
            ))
            records[field]["dOF"].append(_metric_record(
                surrogate_dof,
                reference_dof,
                rtol=tolerance.dOF_rtol,
                atol=tolerance.dOF_atol,
                point=point,
            ))

    field_summaries = {
        field: CEAFieldValidationSummary(
            value=_metric_summary(records[field]["value"]),
            dPc=_metric_summary(records[field]["dPc"]),
            dOF=_metric_summary(records[field]["dOF"]),
        )
        for field in CEA_PROPERTY_FIELDS
    }
    passed = all(summary.passed for summary in field_summaries.values())
    return CEAHoldoutValidationReport(
        passed=passed,
        table_path=surfaces.source_path,
        table_content_sha256=surfaces.content_sha256,
        table_schema_version=surfaces.schema_version,
        interpolator_version=surfaces.interpolator_version,
        oxidizer=surfaces.oxidizer,
        fuel=surfaces.fuel,
        rocketcea_oxidizer=rocket_ox,
        rocketcea_fuel=rocket_fuel,
        backend_notes=backend_notes,
        backend=str(backend_name),
        holdout_points=heldout,
        finite_difference_fraction=fraction,
        selection_seed=selection_seed,
        tolerances=selected_tolerances,
        fields=field_summaries,
    )
