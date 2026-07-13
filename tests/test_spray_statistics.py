"""Deterministic, multiplicity-aware spray statistics."""

import numpy as np
import pytest

from raosim.spray.statistics import (
    fit_rosin_rammler,
    mass_weighted_percentile,
    rosin_rammler_survival,
    sample_rosin_rammler,
    sauter_mean_diameter,
    summarize_spray,
)


def test_smd_uses_physical_droplet_multiplicity():
    d = np.array([1.0, 2.0])
    weight = np.array([8.0, 1.0])
    assert sauter_mean_diameter(d, weight) == pytest.approx(4.0 / 3.0)


def test_mass_percentiles_use_weight_times_diameter_cubed():
    d = np.array([1.0, 2.0, 3.0])
    # Equal represented liquid mass in all three parcels.
    weight = 1.0 / d**3
    assert mass_weighted_percentile(d, weight, 10.0) == 1.0
    assert mass_weighted_percentile(d, weight, 50.0) == 2.0
    assert mass_weighted_percentile(d, weight, 90.0) == 3.0


def test_rosin_rammler_fit_recovers_known_mass_distribution():
    scale = 80.0e-6
    shape = 1.7
    count = 2000
    u = (np.arange(count, dtype=float) + 0.5) / count
    d = scale * (-np.log1p(-u)) ** (1.0 / shape)
    # Quantiles are uniformly spaced in mass probability.  Set number
    # multiplicity so every computational parcel represents equal mass.
    weight = 1.0 / d**3
    fit = fit_rosin_rammler(d, weight)
    assert fit.scale_diameter == pytest.approx(scale, rel=2.0e-3)
    assert fit.shape == pytest.approx(shape, rel=2.0e-3)
    assert fit.r_squared > 0.9999


def test_rosin_rammler_survival_convention_is_decreasing():
    values = rosin_rammler_survival(
        np.array([0.0, 50.0e-6, 100.0e-6]), 50.0e-6, 2.0
    )
    assert values[0] == 1.0
    assert np.all(np.diff(values) < 0.0)
    assert values[1] == pytest.approx(np.exp(-1.0))


def test_sampling_requires_explicit_rng_and_is_reproducible():
    first = sample_rosin_rammler(
        50.0e-6, 1.5, 32, rng=np.random.default_rng(17)
    )
    second = sample_rosin_rammler(
        50.0e-6, 1.5, 32, rng=np.random.default_rng(17)
    )
    assert np.array_equal(first, second)
    with pytest.raises(TypeError, match="explicit"):
        sample_rosin_rammler(50.0e-6, 1.5, 2, rng=None)


def test_summary_and_input_validation():
    summary = summarize_spray(
        np.array([50.0e-6, 50.0e-6]), np.array([2.0, 3.0]), fit_rr=False
    )
    assert summary.sauter_mean_diameter == pytest.approx(50.0e-6)
    assert summary.represented_droplets == 5.0
    assert summary.rosin_rammler is None

    with pytest.raises(ValueError, match="diameters"):
        sauter_mean_diameter([0.0], [1.0])
    with pytest.raises(ValueError, match="weight"):
        sauter_mean_diameter([1.0], [0.0])


def test_uniform_cloud_has_statistics_without_undefined_rr_fit():
    summary = summarize_spray(
        np.full(4, 25.0e-6), np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert summary.sauter_mean_diameter == pytest.approx(25.0e-6)
    assert summary.rosin_rammler is None
