"""Weighted droplet-cloud statistics for Lagrangian spray parcels.

Parcel ``weight`` is the physical droplet multiplicity represented by each
computational parcel.  Sauter mean diameter therefore uses number weighting,

``d32 = sum(weight*d**3) / sum(weight*d**2)``,

while Rosin-Rammler fitting uses liquid-mass weighting.  Radhakrishnan et al.
(2021, Eq. following Table 7) write ``exp(-(d/d_bar)**n)``; that expression is
the mass *survival* fraction, not a conventional increasing CDF.  The naming
below keeps that distinction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RosinRammlerFit:
    scale_diameter: float
    shape: float
    r_squared: float
    sample_count: int
    weighting: str = "represented_liquid_mass"
    convention: str = "survival=exp(-(d/scale)**shape)"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "scale_diameter_m": self.scale_diameter,
            "shape": self.shape,
            "r_squared": self.r_squared,
            "sample_count": self.sample_count,
            "weighting": self.weighting,
            "convention": self.convention,
        }


@dataclass(frozen=True)
class SprayStatistics:
    sauter_mean_diameter: float
    number_mean_diameter: float
    mass_d10: float
    mass_d50: float
    mass_d90: float
    represented_droplets: float
    parcel_count: int
    rosin_rammler: RosinRammlerFit | None

    def to_dict(self) -> dict:
        return {
            "sauter_mean_diameter_m": self.sauter_mean_diameter,
            "number_mean_diameter_m": self.number_mean_diameter,
            "mass_d10_m": self.mass_d10,
            "mass_d50_m": self.mass_d50,
            "mass_d90_m": self.mass_d90,
            "represented_droplets": self.represented_droplets,
            "parcel_count": self.parcel_count,
            "rosin_rammler": (
                None if self.rosin_rammler is None
                else self.rosin_rammler.to_dict()
            ),
        }


def _validated_samples(diameter, weight) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(diameter, dtype=float).reshape(-1)
    w = np.asarray(weight, dtype=float).reshape(-1)
    if d.shape != w.shape or d.size == 0:
        raise ValueError("diameter and weight must be non-empty arrays of equal size")
    if np.any(~np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("diameters must be finite and > 0")
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and >= 0")
    if not float(np.sum(w)) > 0.0:
        raise ValueError("at least one parcel weight must be > 0")
    mask = w > 0.0
    return d[mask], w[mask]


def sauter_mean_diameter(diameter, weight) -> float:
    """Return number-weighted ``d32`` for represented droplets."""

    d, w = _validated_samples(diameter, weight)
    denominator = float(np.sum(w * d**2))
    if denominator <= 0.0:
        raise ValueError("SMD denominator must be positive")
    return float(np.sum(w * d**3) / denominator)


def mass_weighted_percentile(diameter, weight, percentile: float) -> float:
    """Return a diameter percentile of represented liquid mass.

    Each parcel's represented mass is proportional to ``weight*d**3``; the
    common ``rho*pi/6`` factor cancels.
    """

    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    d, w = _validated_samples(diameter, weight)
    order = np.argsort(d, kind="mergesort")
    ds = d[order]
    mass = w[order] * ds**3
    cumulative = np.cumsum(mass)
    target = percentile / 100.0 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ds[min(index, ds.size - 1)])


def rosin_rammler_survival(diameter, scale_diameter: float, shape: float):
    """Mass survival fraction used by Radhakrishnan et al. (2021)."""

    if not math.isfinite(scale_diameter) or scale_diameter <= 0.0:
        raise ValueError("scale_diameter must be finite and > 0")
    if not math.isfinite(shape) or shape <= 0.0:
        raise ValueError("shape must be finite and > 0")
    d = np.asarray(diameter, dtype=float)
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("diameter must be finite and >= 0")
    result = np.exp(-np.power(d / scale_diameter, shape))
    return float(result) if result.ndim == 0 else result


def sample_rosin_rammler(
    scale_diameter: float,
    shape: float,
    size: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the increasing mass CDF corresponding to the RR survival law."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    if int(size) != size or size < 1:
        raise ValueError("size must be an integer >= 1")
    # Avoid exact 0/1 so inverse values remain finite and strictly positive.
    u = np.clip(rng.random(int(size)), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return scale_diameter * np.power(-np.log1p(-u), 1.0 / shape)


def fit_rosin_rammler(diameter, weight) -> RosinRammlerFit:
    """Fit the Radhakrishnan/Fluent mass-survival convention.

    Mid-mass plotting positions avoid the singular survival values zero and
    one.  Repeated diameters remain separate observations so computational
    parcel multiplicity and mass are retained without arbitrary binning.
    """

    d, w = _validated_samples(diameter, weight)
    if d.size < 3 or np.unique(d).size < 2:
        raise ValueError("Rosin-Rammler fitting requires >=3 samples and >=2 sizes")
    order = np.argsort(d, kind="mergesort")
    ds = d[order]
    mass = w[order] * ds**3
    total = float(np.sum(mass))
    below_midpoint = np.cumsum(mass) - 0.5 * mass
    survival = np.clip(
        1.0 - below_midpoint / total,
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    x = np.log(ds)
    y = np.log(-np.log(survival))
    # Give each plotting position influence in proportion to represented mass.
    coeff = np.polyfit(x, y, 1, w=np.sqrt(mass / total))
    shape = float(coeff[0])
    if not math.isfinite(shape) or shape <= 0.0:
        raise ValueError("fitted Rosin-Rammler shape is not positive")
    scale = float(math.exp(-coeff[1] / shape))
    fitted = coeff[0] * x + coeff[1]
    y_bar = float(np.average(y, weights=mass))
    ss_res = float(np.sum(mass * (y - fitted) ** 2))
    ss_tot = float(np.sum(mass * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return RosinRammlerFit(
        scale_diameter=scale,
        shape=shape,
        r_squared=float(r2),
        sample_count=int(d.size),
    )


def summarize_spray(diameter, weight, *, fit_rr: bool = True) -> SprayStatistics:
    d, w = _validated_samples(diameter, weight)
    rr = (
        fit_rosin_rammler(d, w)
        if fit_rr and d.size >= 3 and np.unique(d).size >= 2
        else None
    )
    return SprayStatistics(
        sauter_mean_diameter=sauter_mean_diameter(d, w),
        number_mean_diameter=float(np.sum(w * d) / np.sum(w)),
        mass_d10=mass_weighted_percentile(d, w, 10.0),
        mass_d50=mass_weighted_percentile(d, w, 50.0),
        mass_d90=mass_weighted_percentile(d, w, 90.0),
        represented_droplets=float(np.sum(w)),
        parcel_count=int(d.size),
        rosin_rammler=rr,
    )


__all__ = [
    "RosinRammlerFit",
    "SprayStatistics",
    "fit_rosin_rammler",
    "mass_weighted_percentile",
    "rosin_rammler_survival",
    "sample_rosin_rammler",
    "sauter_mean_diameter",
    "summarize_spray",
]
