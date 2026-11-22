"""Canonical coolant identities shared by every thermofluid path."""

from __future__ import annotations


_COOLANT_ALIASES = {
    "rp1": "rp1",
    "rp-1": "rp1",
    "kerosene": "rp1",
    "methane": "methane",
    "ch4": "methane",
    "lch4": "methane",
    "lh2": "hydrogen",
    "hydrogen": "hydrogen",
    "h2": "hydrogen",
    "ethanol": "ethanol",
    "water": "water",
}


def canonical_coolant_name(coolant: str | None) -> str:
    """Return the canonical key used by properties and real-fluid backends."""
    name = str(coolant or "rp1").strip().lower()
    return _COOLANT_ALIASES.get(name, name)
