"""mass_ledger.py — configuration-controlled engine hardware mass ledger.

Why this module exists
----------------------
Until now the repository could size a thrust chamber, a pintle injector and an
electric feed system, but it could not say what any of that hardware *weighed*.
``raosim.mdo.state.MassState`` carried three permanently-``NaN`` slots
(``thrust_chamber_mass``, ``injector_mass``, ``total_dry_mass``) and the shared
host snapshot reported them as ``unavailable``.  The optimizer therefore
minimised an *electric-feed* package mass while calling it engine mass.

The governing modelling decision here is that **mass is integrated from the same
resolved geometry the CAD exporters build**, never from an independent
mass-scaling correlation.  ``raosim.regen_profile.RegenWallProfile`` is what
``raosim.regen_cad`` revolves; ``raosim.injector_cad.resolve_machined_pintle_layout``
is what ``raosim.injector_cad.build_machined_pintle_bodies`` cuts;
``raosim.interface.resolve_bolted_interface_geometry`` is what sets the flange
and bolt circle.  Feeding the ledger from those objects means a reported mass
and an exported STEP part cannot silently disagree, which is the property the
"requirements in, parameterized parts out" workflow actually needs.

Physical basis
--------------
Every entry is a solid-of-revolution or prismatic volume times a catalog
density.  That is the same relation NASA SP-125 uses for preliminary tankage
mass, e.g. the cylindrical shell

    W_c = 2 π a l_c t_c ρ                                   (SP-125 eq. 8-32)

(*Design of Liquid Propellant Rocket Engines*, Huzel & Huang, NASA SP-125,
1971; ``propulsion_texts/19710019929.pdf``, ch. VIII, printed p. 339).  Here
``a`` is the *nominal* (mid-surface) radius, so a thin shell of thickness ``t``
standing off a gas-side radius ``r`` contributes ``2π (r + t/2) t`` per unit
meridional arc — Pappus's centroid theorem, which is what this module applies.
The pre-existing private helper ``raosim.thermal_design._wall_mass`` used the
gas-side radius rather than the mid-surface radius and summed
``hypot(gradient(x), gradient(y))``, which over-counts the path by one grid
interval (see the note in :func:`raosim.regen_profile._nodal_weights_from_segments`).
This module uses the correct centroid radius and the correct trapezoidal nodal
weights.

Preliminary mass estimates in SP-125 also carry an explicit *non-ideal*
allowance rather than pretending a shell is only its membrane: the pressurant
vessel estimate

    W_v = π d² ρ_m (p d / 4 s) + 3 π d ρ_m (0.5 p d / 4 s)   (SP-125 eq. 5-16)

adds a 3-inch weld-land band at half wall thickness on top of the two
hemispherical membranes (SP-125 ch. V, printed p. 173).  This module exposes the
same idea as an explicit ``joint_allowance`` multiplier, but it **defaults to
1.0** — no allowance is invented on the user's behalf.  ``ManufacturingSpec``
already carries ``weld_allowance`` / ``braze_allowance`` fields for callers that
want to set one.

Structural context for the thicknesses being integrated is NASA SP-8087,
*Liquid Rocket Engine Fluid-Cooled Combustion Chambers* (1973;
``propulsion_texts/19730022965.pdf``).  Its §2.1.3 identifies the three
structural jobs the metal in this ledger is doing — hoop support about the
combustion chamber, support at the throat against bending and buckling, and hoop
support about the expansion nozzle against collapse under overexpanded sea-level
operation — and §2.1.3 states the design factors of safety in use: yield 1.0 to
1.32, ultimate 1.3 to 1.8 (printed p. 24).  This module does not *size* against
those; it integrates thicknesses that the thermal/structural screens
(:mod:`raosim.thermal_design`, SP-125 eq. 4-31 via
:func:`raosim.physics.coaxial_shell_wall_stress`) already produced, and records
the closeout sizing basis in its provenance so the load path is auditable.

Honesty rules
-------------
* A component whose geometry or density is missing is emitted as an item with
  ``mass_kg = None`` and an ``unavailable_reason``.  It is never emitted as
  ``0.0``.
* :attr:`MassLedger.complete` is ``False`` whenever any item is unavailable, and
  :attr:`MassLedger.total_mass` is then ``None``.  A partial rollup is available
  separately as :attr:`MassLedger.resolved_mass` and is always labelled as such.
* ``status`` distinguishes ``geometry_resolved`` (integrated from a resolved CAD
  layout) from ``screening_sized`` (a documented first-order shape assumption,
  e.g. hex bolt head/nut envelopes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "MassItem",
    "MassLedger",
    "thrust_chamber_mass_ledger",
    "flange_bolt_mass_ledger",
    "injector_mass_ledger",
    "combine_ledgers",
    "SP125_SHELL_MASS",
    "SP125_WELD_LAND_ALLOWANCE",
    "SP8087_STRUCTURAL",
]

# Stable source identifiers used in every emitted item.
SP125_SHELL_MASS = "NASA SP-125 eq. 8-32 (shell mass = surface x thickness x rho)"
SP125_WELD_LAND_ALLOWANCE = "NASA SP-125 eq. 5-16 (weld-land mass allowance)"
SP8087_STRUCTURAL = "NASA SP-8087 sec. 2.1.3 (chamber reinforcement / FoS)"

# ISO 4014/4032 hex head and hex nut proportions, used only to give the bolt
# head and nut a defensible *envelope* volume.  Across-flats s ~ 1.5 d gives a
# hexagon of area (sqrt(3)/2) s^2; head height ~ 0.65 d and nut height ~ 0.8 d.
# These are machine-design geometry conventions, not propulsion-corpus values,
# and every item built from them is marked ``screening_sized``.
_HEX_AREA_FACTOR = math.sqrt(3.0) / 2.0
_HEX_ACROSS_FLATS_PER_D = 1.5
_HEX_HEAD_HEIGHT_PER_D = 0.65
_HEX_NUT_HEIGHT_PER_D = 0.8
# Matches raosim.interface._THREAD_TENSILE_AREA_FACTOR so the ledger and the
# bolted-joint screen describe the same fastener.
_THREAD_TENSILE_AREA_FACTOR = 0.75


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _finite_positive(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0.0 else None


def _density_of(material: Any, *, name_hint: str | None = None) -> tuple[float | None, str | None]:
    """Return ``(density, material_name)`` for a catalog name or spec object."""

    if material is None:
        return None, name_hint
    if isinstance(material, str):
        try:
            from raosim.materials import get_material

            resolved = get_material(material)
        except Exception:
            return None, material
        return _finite_positive(getattr(resolved, "density", None)), material
    density = _finite_positive(getattr(material, "density", None))
    name = getattr(material, "name", None) or name_hint
    return density, name


# --------------------------------------------------------------------------- #
# ledger containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MassItem:
    """One line of the hardware mass ledger.

    ``mass_kg is None`` means *not known*, and ``unavailable_reason`` says why.
    It never means zero.
    """

    subsystem: str
    component: str
    quantity: int
    material: str | None
    volume_m3: float | None
    density_kg_m3: float | None
    mass_kg: float | None
    status: str
    method: str
    source_ids: tuple[str, ...] = ()
    key_parameters: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.mass_kg is not None and math.isfinite(self.mass_kg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "component": self.component,
            "quantity": int(self.quantity),
            "material": self.material,
            "volume_m3": self.volume_m3,
            "density_kg_m3": self.density_kg_m3,
            "mass_kg": self.mass_kg,
            "status": self.status,
            "method": self.method,
            "source_ids": list(self.source_ids),
            "key_parameters": dict(self.key_parameters),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class MassLedger:
    """A scoped collection of :class:`MassItem` with honest rollups."""

    scope: str
    items: tuple[MassItem, ...]
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    # ---- rollups --------------------------------------------------------- #
    @property
    def complete(self) -> bool:
        """True only when every item resolved to a finite mass."""

        return bool(self.items) and all(item.available for item in self.items)

    @property
    def resolved_mass(self) -> float:
        """Sum of the items that *did* resolve.  Always a partial lower bound
        when :attr:`complete` is ``False``; never presented as a total."""

        return float(sum(
            item.mass_kg * item.quantity
            for item in self.items
            if item.available
        ))

    @property
    def total_mass(self) -> float | None:
        """Total hardware mass, or ``None`` if any item is unavailable."""

        return self.resolved_mass if self.complete else None

    @property
    def unavailable_items(self) -> tuple[MassItem, ...]:
        return tuple(item for item in self.items if not item.available)

    @property
    def unavailable_reason(self) -> str | None:
        """A single stable reason string for the snapshot layer."""

        missing = self.unavailable_items
        if not missing:
            return None
        detail = "; ".join(
            f"{item.subsystem}/{item.component}: {item.unavailable_reason}"
            for item in missing
        )
        return (
            f"the {self.scope} mass ledger is incomplete -- {detail}"
        )

    def by_subsystem(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for item in self.items:
            if item.subsystem not in out:
                out[item.subsystem] = 0.0
            if out[item.subsystem] is None:
                continue
            if item.available:
                out[item.subsystem] = out[item.subsystem] + item.mass_kg * item.quantity
            else:
                out[item.subsystem] = None
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "geometry_integrated_hardware_mass_ledger",
            "scope": self.scope,
            "complete": self.complete,
            "total_mass_kg": self.total_mass,
            "resolved_mass_kg": self.resolved_mass,
            "resolved_mass_is_partial": not self.complete,
            "unavailable_reason": self.unavailable_reason,
            "by_subsystem_kg": self.by_subsystem(),
            "items": [item.to_dict() for item in self.items],
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


def combine_ledgers(
    ledgers: Iterable[MassLedger], *, scope: str
) -> MassLedger:
    """Merge scoped ledgers into one, preserving item-level availability."""

    ledgers = tuple(ledgers)
    items: list[MassItem] = []
    warnings: list[str] = []
    provenance: dict[str, Any] = {"combined_from": []}
    for ledger in ledgers:
        items.extend(ledger.items)
        warnings.extend(ledger.warnings)
        provenance["combined_from"].append(ledger.scope)
        provenance[ledger.scope] = dict(ledger.provenance)
    return MassLedger(
        scope=scope,
        items=tuple(items),
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# thrust chamber (liner + channel lands + structural closeout)
# --------------------------------------------------------------------------- #
def _meridional_weights(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Trapezoidal nodal control lengths along the meridian.

    Delegates to :func:`raosim.regen_profile._nodal_weights_from_segments` so
    the mass integral uses exactly the same quadrature as the cooling model's
    heat-area integral (``sum(weights) == total arc length``).
    """

    from raosim.regen_profile import _nodal_weights_from_segments

    if x.size < 2:
        raise ValueError("a mass integral needs at least two contour stations")
    segments = np.hypot(np.diff(x), np.diff(r))
    return _nodal_weights_from_segments(segments)


def thrust_chamber_mass_ledger(
    profile: Any,
    *,
    liner_material: Any,
    closeout_material: Any | None = None,
    joint_allowance: float = 1.0,
    scope: str = "thrust_chamber",
) -> MassLedger:
    """Integrate liner, channel-land and closeout metal from a wall profile.

    Parameters
    ----------
    profile
        A :class:`raosim.regen_profile.RegenWallProfile` (or any object exposing
        ``x``, ``r_inner``, ``t_hot``, ``channel_width``, ``channel_height``,
        ``land_width``, ``t_jacket`` and ``channel_count``).  This is the object
        :mod:`raosim.regen_cad` revolves into the exported wall, so the mass and
        the CAD describe the same solid.
    liner_material, closeout_material
        Catalog names or ``MaterialSpec``/``MaterialProperties``.  ``closeout``
        defaults to the liner material (a one-piece brazed jacket of the same
        alloy, SP-8087 §2.1.3.1).
    joint_allowance
        Multiplier on the integrated metal for welds, brazes and land build-up,
        after SP-125 eq. 5-16's weld-land treatment.  **Defaults to 1.0**; no
        allowance is applied unless the caller asks for one.

    Notes
    -----
    Per meridional station the metal cross-section is

        A_liner    = 2 pi (r + t_w/2) t_w                     (Pappus / SP-125 8-32)
        A_land     = pi (r_o^2 - r_i^2) * b / (b + w)
        A_closeout = 2 pi (r_o + t_j/2) t_j

    with ``r_i = r + t_w`` and ``r_o = r_i + h``.  Writing the land as an
    *area fraction* of the channel annulus rather than ``N * b * h`` is exact
    for radial ribs at constant angular pitch and is invariant to the helical
    stretch ``dl/ds`` (both ``b`` and ``w`` are widths normal to the coolant
    path, so their ratio is unchanged — see
    :func:`raosim.regen_profile.helix_stretch_factors`).  It is also
    unconditionally non-negative, which ``annulus - N*w*h`` is not at coarse
    resolution near the throat.
    """

    allowance = _finite_positive(joint_allowance)
    if allowance is None:
        raise ValueError("joint_allowance must be a positive finite multiplier")

    x = np.asarray(getattr(profile, "x"), dtype=float)
    r = np.asarray(getattr(profile, "r_inner"), dtype=float)
    t_hot = np.asarray(getattr(profile, "t_hot"), dtype=float)
    w = np.asarray(getattr(profile, "channel_width"), dtype=float)
    h = np.asarray(getattr(profile, "channel_height"), dtype=float)
    b = np.asarray(getattr(profile, "land_width"), dtype=float)
    t_jacket = np.asarray(getattr(profile, "t_jacket"), dtype=float)
    channel_count = int(getattr(profile, "channel_count", 0) or 0)

    for name, arr in (
        ("r_inner", r), ("t_hot", t_hot), ("channel_width", w),
        ("channel_height", h), ("land_width", b), ("t_jacket", t_jacket),
    ):
        if arr.shape != x.shape:
            raise ValueError(f"profile field '{name}' must match the station grid")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"profile field '{name}' contains non-finite values")

    ds = _meridional_weights(x, r)
    r_ch_in = r + t_hot
    r_ch_out = r_ch_in + h
    annulus = np.pi * (r_ch_out ** 2 - r_ch_in ** 2)
    pitch = b + w
    land_fraction = np.where(pitch > 0.0, b / np.maximum(pitch, 1e-30), 0.0)

    a_liner = 2.0 * np.pi * (r + 0.5 * t_hot) * t_hot
    a_land = annulus * land_fraction
    a_closeout = 2.0 * np.pi * (r_ch_out + 0.5 * t_jacket) * t_jacket

    v_liner = float(np.sum(a_liner * ds)) * allowance
    v_land = float(np.sum(a_land * ds)) * allowance
    v_closeout = float(np.sum(a_closeout * ds)) * allowance

    liner_rho, liner_name = _density_of(liner_material)
    close_rho, close_name = _density_of(
        closeout_material if closeout_material is not None else liner_material,
        name_hint=liner_name,
    )

    missing_density = (
        "wall material has no density; pass a raosim.materials catalog name or "
        "a MaterialSpec built with MaterialSpec.from_catalog"
    )

    def item(subsystem, component, volume, rho, mat_name, params, method):
        return MassItem(
            subsystem=subsystem,
            component=component,
            quantity=1,
            material=mat_name,
            volume_m3=float(volume),
            density_kg_m3=rho,
            mass_kg=(float(volume) * rho) if rho is not None else None,
            status="geometry_resolved",
            method=method,
            source_ids=(SP125_SHELL_MASS, SP8087_STRUCTURAL),
            key_parameters=params,
            unavailable_reason=None if rho is not None else missing_density,
        )

    items = (
        item(
            "thrust_chamber", "hot-gas liner", v_liner, liner_rho, liner_name,
            {
                "mean_wall_thickness_m": float(np.mean(t_hot)),
                "min_wall_thickness_m": float(np.min(t_hot)),
                "meridional_length_m": float(np.sum(ds)),
            },
            "solid of revolution: 2*pi*(r + t/2)*t integrated on meridional arc",
        ),
        item(
            "thrust_chamber", "regen channel lands", v_land, liner_rho, liner_name,
            {
                "channel_count": channel_count,
                "mean_land_width_m": float(np.mean(b)),
                "mean_channel_width_m": float(np.mean(w)),
                "mean_channel_height_m": float(np.mean(h)),
                "mean_land_area_fraction": float(np.mean(land_fraction)),
            },
            "channel annulus * land area fraction b/(b+w), integrated on arc",
        ),
        item(
            "thrust_chamber", "structural closeout / jacket", v_closeout,
            close_rho, close_name,
            {
                "mean_closeout_thickness_m": float(np.mean(t_jacket)),
                "mean_outer_radius_m": float(np.mean(r_ch_out)),
            },
            "solid of revolution: 2*pi*(r_o + t_j/2)*t_j integrated on arc",
        ),
    )

    warnings: list[str] = []
    if channel_count <= 0:
        warnings.append(
            "channel_count is not positive; land mass was still integrated from "
            "the land-width profile, which may not describe real hardware"
        )
    if np.any(land_fraction <= 0.0):
        warnings.append(
            "at least one station has zero land width; the closeout there is "
            "unsupported and the integrated land mass is a lower bound"
        )
    if allowance == 1.0:
        warnings.append(
            "no weld/braze joint allowance applied (joint_allowance = 1.0); "
            "SP-125 eq. 5-16 shows preliminary estimates normally carry one"
        )

    provenance = {
        "geometry_source": "raosim.regen_profile.RegenWallProfile",
        "cad_consistency": (
            "the same profile object is revolved by raosim.regen_cad, so this "
            "mass and the exported wall geometry describe one solid"
        ),
        "quadrature": "trapezoidal nodal weights on meridional arc length",
        "joint_allowance": allowance,
        "structural_context": SP8087_STRUCTURAL,
        "excludes": [
            "flanges, bolts and seals (see flange_bolt_mass_ledger)",
            "manifolds, inlet/outlet bosses and instrumentation ports",
            "throat inserts, coatings and thermal barrier layers",
            "gimbal, mounts and thrust take-out structure",
        ],
    }
    return MassLedger(
        scope=scope,
        items=items,
        warnings=tuple(warnings),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# bolted chamber/injector interface: flange ring + fasteners
# --------------------------------------------------------------------------- #
def flange_bolt_mass_ledger(
    resolution: Any,
    *,
    flange_material: Any,
    bolt_material: Any | None = None,
    grip_length: float | None = None,
    include_nuts: bool = True,
    scope: str = "chamber_interface",
) -> MassLedger:
    """Flange ring and fastener mass from the resolved bolted interface.

    ``resolution`` is a :class:`raosim.interface.InterfaceGeometryResolution`
    (or its ``to_dict()``), i.e. the layout that
    :func:`raosim.interface.resolve_bolted_interface_geometry` already produced
    and that :func:`raosim.engine_cad.pump_mount_flange_screen` consumes.

    The flange ring is exact: an annulus from the chamber OD to the flange OD,
    of axial length ``flange_length``, less the through-holes.  The fasteners
    are ``screening_sized``: the shank uses the same 0.75 thread tensile-area
    factor as :mod:`raosim.interface`'s bolted-joint screen, and the head/nut
    use ISO 4014/4032 hex envelope proportions.  Real fastener mass depends on
    the selected part number, thread series and washer stack.
    """

    if isinstance(resolution, Mapping):
        get = lambda key, alt: resolution.get(alt, resolution.get(key))  # noqa: E731
        chamber_od = get("chamber_outer_diameter", "chamber_outer_diameter_m")
        flange_od = get("flange_outer_diameter", "flange_outer_diameter_m")
        flange_len = get("flange_length", "flange_length_m")
        bolt_count = get("bolt_count", "bolt_count")
        hole_d = get("bolt_hole_diameter", "bolt_hole_diameter_m")
        bolt_d = get("bolt_diameter", "bolt_diameter_m")
        face_t = get("face_thickness", "injector_face_thickness_m")
    else:
        chamber_od = getattr(resolution, "chamber_outer_diameter", None)
        flange_od = getattr(resolution, "flange_outer_diameter", None)
        flange_len = getattr(resolution, "flange_length", None)
        bolt_count = getattr(resolution, "bolt_count", None)
        hole_d = getattr(resolution, "bolt_hole_diameter", None)
        bolt_d = getattr(resolution, "bolt_diameter", None)
        face_t = getattr(resolution, "face_thickness", None)

    chamber_od = _finite_positive(chamber_od)
    flange_od = _finite_positive(flange_od)
    flange_len = _finite_positive(flange_len)
    hole_d = _finite_positive(hole_d)
    face_t = _finite_positive(face_t)
    try:
        count = int(bolt_count)
    except (TypeError, ValueError):
        count = 0

    flange_rho, flange_name = _density_of(flange_material)
    bolt_rho, bolt_name = _density_of(
        bolt_material if bolt_material is not None else flange_material,
        name_hint=flange_name,
    )

    items: list[MassItem] = []
    warnings: list[str] = []

    # ---- flange ring ----------------------------------------------------- #
    if (
        chamber_od is not None and flange_od is not None
        and flange_len is not None and flange_od > chamber_od
    ):
        ring_v = 0.25 * math.pi * (flange_od ** 2 - chamber_od ** 2) * flange_len
        hole_v = (
            count * 0.25 * math.pi * (hole_d ** 2) * flange_len
            if hole_d is not None and count > 0 else 0.0
        )
        net_v = max(ring_v - hole_v, 0.0)
        if hole_v > ring_v:
            warnings.append(
                "bolt through-holes remove more material than the flange ring "
                "contains; the flange layout is geometrically inconsistent"
            )
        items.append(MassItem(
            subsystem="chamber_interface",
            component="chamber flange ring",
            quantity=1,
            material=flange_name,
            volume_m3=net_v,
            density_kg_m3=flange_rho,
            mass_kg=(net_v * flange_rho) if flange_rho is not None else None,
            status="geometry_resolved",
            method=(
                "annulus (flange OD to chamber OD) x flange length, less "
                "bolt through-holes"
            ),
            source_ids=(SP125_SHELL_MASS,),
            key_parameters={
                "flange_outer_diameter_m": flange_od,
                "chamber_outer_diameter_m": chamber_od,
                "flange_length_m": flange_len,
                "bolt_count": count,
                "bolt_hole_diameter_m": hole_d,
            },
            unavailable_reason=(
                None if flange_rho is not None
                else "flange material has no density"
            ),
        ))
    else:
        items.append(MassItem(
            subsystem="chamber_interface",
            component="chamber flange ring",
            quantity=1,
            material=flange_name,
            volume_m3=None,
            density_kg_m3=flange_rho,
            mass_kg=None,
            status="unavailable",
            method="annulus x flange length",
            source_ids=(SP125_SHELL_MASS,),
            key_parameters={},
            unavailable_reason=(
                "the bolted interface was not resolved to a flange outer "
                "diameter, chamber outer diameter and flange length"
            ),
        ))

    # ---- fasteners ------------------------------------------------------- #
    shank_d = _finite_positive(bolt_d)
    if shank_d is None and hole_d is not None:
        # raosim.interface uses 0.9 * hole when no bolt diameter is given.
        shank_d = 0.9 * hole_d
        warnings.append(
            "bolt diameter was not specified; the fastener mass uses the same "
            "0.9 x hole-diameter inference as the interface bolted-joint screen"
        )
    grip = _finite_positive(grip_length)
    if grip is None and flange_len is not None:
        grip = flange_len + (face_t or 0.0)

    if shank_d is not None and grip is not None and count > 0:
        tensile_area = _THREAD_TENSILE_AREA_FACTOR * 0.25 * math.pi * shank_d ** 2
        shank_v = tensile_area * grip
        across_flats = _HEX_ACROSS_FLATS_PER_D * shank_d
        hex_area = _HEX_AREA_FACTOR * across_flats ** 2
        head_v = hex_area * _HEX_HEAD_HEIGHT_PER_D * shank_d
        nut_v = hex_area * _HEX_NUT_HEIGHT_PER_D * shank_d if include_nuts else 0.0
        per_bolt_v = shank_v + head_v + nut_v
        items.append(MassItem(
            subsystem="chamber_interface",
            component="flange bolt (with nut)" if include_nuts else "flange bolt",
            quantity=count,
            material=bolt_name,
            volume_m3=per_bolt_v,
            density_kg_m3=bolt_rho,
            mass_kg=(per_bolt_v * bolt_rho) if bolt_rho is not None else None,
            status="screening_sized",
            method=(
                "thread tensile area (0.75 * pi d^2/4, matching "
                "raosim.interface) over the grip, plus ISO 4014/4032 hex "
                "head and nut envelopes"
            ),
            source_ids=("ISO 4014 / ISO 4032 hex proportions (geometry only)",),
            key_parameters={
                "bolt_diameter_m": shank_d,
                "grip_length_m": grip,
                "thread_tensile_area_m2": tensile_area,
                "includes_nut": include_nuts,
            },
            unavailable_reason=(
                None if bolt_rho is not None else "bolt material has no density"
            ),
        ))
    else:
        items.append(MassItem(
            subsystem="chamber_interface",
            component="flange bolt",
            quantity=max(count, 0),
            material=bolt_name,
            volume_m3=None,
            density_kg_m3=bolt_rho,
            mass_kg=None,
            status="unavailable",
            method="thread tensile area over grip + hex head/nut envelope",
            source_ids=(),
            key_parameters={},
            unavailable_reason=(
                "bolt diameter, grip length or bolt count could not be "
                "resolved from the bolted-interface layout"
            ),
        ))

    warnings.append(
        "seals, gaskets, washers and thread inserts are not in this ledger"
    )

    provenance = {
        "geometry_source": "raosim.interface.resolve_bolted_interface_geometry",
        "joint_screen": (
            "strength is screened separately by "
            "raosim.interface.screen_injector_chamber_interface"
        ),
        "structural_context": SP8087_STRUCTURAL,
    }
    return MassLedger(
        scope=scope,
        items=tuple(items),
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# pintle injector
# --------------------------------------------------------------------------- #
def injector_mass_ledger(
    layout: Mapping[str, Any],
    *,
    body_material: Any,
    faceplate_material: Any | None = None,
    post_material: Any | None = None,
    scope: str = "injector",
) -> MassLedger:
    """Injector hardware mass from the resolved machined-pintle layout.

    ``layout`` is the dict returned by
    :func:`raosim.injector_cad.resolve_machined_pintle_layout` — the same object
    :func:`raosim.injector_cad.build_machined_pintle_bodies` cuts the STEP solids
    from.  Every entry below is therefore the actual exported part, not a
    correlation.

    Components
    ----------
    faceplate
        Disc of ``faceplate_radius`` and ``faceplate_thickness``, less the bolt
        through-holes, less the central bore that clears the pintle sleeve, less
        the resolved annular manifold volumes machined into its back face.  The
        faceplate thickness is *not* free: ``resolve_machined_pintle_layout``
        already floors it at the deepest manifold plus the back web, and
        :func:`raosim.interface.screen_injector_chamber_interface` screens it
        against clamped-plate bending, ``sigma ~= 0.75 p a^2 / t^2``.
    pintle post
        Annular tube from ``pintle_inner_radius`` to ``pintle_outer_radius`` over
        the body length.  The radial metering openings are subtracted.
    sleeve
        Annular tube from ``sleeve_inner_radius`` to ``sleeve_outer_radius`` over
        the annulus length.
    """

    resolved = dict(layout.get("resolved") or {})
    if not resolved:
        raise ValueError(
            "layout has no 'resolved' section; pass the dict returned by "
            "raosim.injector_cad.resolve_machined_pintle_layout"
        )
    hydraulic = dict(layout.get("hydraulic_basis") or {})
    roles = dict(layout.get("manifolds") or layout.get("roles") or {})

    body_rho, body_name = _density_of(body_material)
    face_rho, face_name = _density_of(
        faceplate_material if faceplate_material is not None else body_material,
        name_hint=body_name,
    )
    post_rho, post_name = _density_of(
        post_material if post_material is not None else body_material,
        name_hint=body_name,
    )

    g = lambda key: _finite_positive(resolved.get(key))  # noqa: E731
    face_r = g("faceplate_radius_m")
    face_t = g("faceplate_thickness_m")
    sleeve_ro = g("sleeve_outer_radius_m")
    sleeve_ri = g("sleeve_inner_radius_m")
    annulus_len = g("annulus_length_m")
    post_ro = g("pintle_outer_radius_m")
    post_ri = g("pintle_inner_radius_m")
    post_len = g("pintle_body_length_m")
    hole_d = g("bolt_hole_diameter_m")
    try:
        bolt_count = int(resolved.get("bolt_count") or 0)
    except (TypeError, ValueError):
        bolt_count = 0

    items: list[MassItem] = []
    warnings: list[str] = []

    # ---- faceplate ------------------------------------------------------- #
    manifold_volume = 0.0
    manifold_volume_known = True
    for role, data in roles.items():
        if not isinstance(data, Mapping):
            continue
        v = _finite_positive(data.get("manifold_volume_m3"))
        if v is None:
            manifold_volume_known = False
            warnings.append(
                f"manifold volume for role '{role}' was not resolved; the "
                "faceplate mass does not subtract it"
            )
            continue
        manifold_volume += v

    # ---- features the CAD builder adds and removes ------------------------ #
    # ``injector_cad._build_machined_faceplate`` fuses radial inlet bosses onto
    # the disc, then cuts the manifold pockets, the axial inlet holes, the
    # radial side ports, the transfer passages, the bolt holes and the o-ring
    # groove.  Every one of those is metal added or removed, so the ledger has
    # to carry them or it is not describing the part the exporter writes.
    tol = _finite_positive(resolved.get("tolerance_m")) or 0.0
    min_tool = _finite_positive(resolved.get("min_tool_diameter_m")) or 0.0
    boss_volume = 0.0
    port_volume = 0.0
    transfer_volume = 0.0
    for role, data in roles.items():
        if not isinstance(data, Mapping):
            continue
        try:
            count = int(data.get("inlet_count") or 0)
        except (TypeError, ValueError):
            count = 0
        inlet_d = _finite_positive(data.get("inlet_diameter_m"))
        depth = _finite_positive(data.get("manifold_depth_m")) or 0.0
        mean_r = _finite_positive(data.get("manifold_mean_radius_m")) or 0.0
        if count > 0 and inlet_d is not None and face_r is not None:
            boss_od = max(2.4 * inlet_d, inlet_d + 2.0 * min_tool)
            boss_overlap = max(boss_od, 3.0 * min_tool)
            boss_len = (face_r + 2.5 * boss_od) - (face_r - boss_overlap)
            # Net added metal is the boss annulus outside the disc bore it
            # already occupies; the overlap is not new material.
            boss_volume += count * 0.25 * math.pi * boss_od ** 2 * boss_len
            # ...and the through-bore that is then cut back out of it.
            port_len = (face_r + 3.0 * boss_od) - mean_r
            port_volume += count * 0.25 * math.pi * inlet_d ** 2 * (
                port_len + depth
            )
        try:
            n_transfer = int(data.get("transfer_count") or 0)
        except (TypeError, ValueError):
            n_transfer = 0
        transfer_d = _finite_positive(data.get("transfer_diameter_m"))
        outer_r = _finite_positive(data.get("manifold_outer_radius_m")) or 0.0
        if n_transfer > 0 and transfer_d is not None and sleeve_ro is not None:
            run = max(
                (outer_r + 0.5 * transfer_d) - (sleeve_ro - min_tool), 0.0
            )
            transfer_volume += (
                n_transfer * 0.25 * math.pi * transfer_d ** 2 * run
            )

    seal_volume = 0.0
    if str(resolved.get("seal_type") or "") == "o_ring":
        sw = _finite_positive(resolved.get("o_ring_groove_width_m"))
        sd = _finite_positive(resolved.get("o_ring_groove_depth_m"))
        sr = _finite_positive(resolved.get("seal_center_radius_m"))
        if None not in (sw, sd, sr):
            seal_volume = math.pi * (
                (sr + 0.5 * sw) ** 2 - (sr - 0.5 * sw) ** 2
            ) * sd

    igniter_volume = 0.0
    ig_d = _finite_positive(resolved.get("igniter_port_diameter_m"))
    ig_len = _finite_positive(resolved.get("igniter_port_depth_m"))
    if ig_d is not None and ig_len is not None and face_t is not None:
        # Only the part of the central igniter tube that passes through the
        # faceplate removes faceplate metal; the rest runs down the post bore.
        igniter_volume = 0.25 * math.pi * ig_d ** 2 * min(ig_len, face_t)

    if face_r is not None and face_t is not None:
        gross = math.pi * face_r ** 2 * face_t
        bore = (
            math.pi * (sleeve_ro + tol) ** 2 * face_t
            if sleeve_ro is not None else 0.0
        )
        holes = (
            bolt_count * 0.25 * math.pi * hole_d ** 2 * face_t
            if hole_d is not None and bolt_count > 0 else 0.0
        )
        net = (
            gross + boss_volume
            - bore - holes - manifold_volume
            - port_volume - transfer_volume - seal_volume - igniter_volume
        )
        if net <= 0.0:
            warnings.append(
                "resolved faceplate cutouts exceed the disc volume; the "
                "injector layout is geometrically inconsistent"
            )
        net = max(net, 0.0)
        items.append(MassItem(
            subsystem="injector",
            component="faceplate / manifold body",
            quantity=1,
            material=face_name,
            volume_m3=net,
            density_kg_m3=face_rho,
            mass_kg=(net * face_rho) if face_rho is not None else None,
            status="geometry_resolved",
            method=(
                "disc (faceplate radius x thickness) PLUS the fused radial "
                "inlet bosses, LESS the sleeve clearance bore, bolt "
                "through-holes, annular manifold pockets, axial inlet holes "
                "and radial side ports, transfer passages, o-ring groove and "
                "the igniter port -- matching every feature "
                "injector_cad._build_machined_faceplate adds or cuts"
            ),
            source_ids=(SP125_SHELL_MASS,),
            key_parameters={
                "faceplate_radius_m": face_r,
                "faceplate_thickness_m": face_t,
                "faceplate_minimum_thickness_m": resolved.get(
                    "faceplate_minimum_thickness_m"
                ),
                "bolt_count": bolt_count,
                "added_inlet_boss_volume_m3": boss_volume,
                "subtracted_manifold_volume_m3": manifold_volume,
                "subtracted_port_volume_m3": port_volume,
                "subtracted_transfer_volume_m3": transfer_volume,
                "subtracted_seal_groove_volume_m3": seal_volume,
                "subtracted_igniter_port_volume_m3": igniter_volume,
                "manifold_volume_complete": manifold_volume_known,
            },
            unavailable_reason=(
                None if face_rho is not None
                else "faceplate material has no density"
            ),
        ))
    else:
        items.append(MassItem(
            subsystem="injector", component="faceplate / manifold body",
            quantity=1, material=face_name, volume_m3=None,
            density_kg_m3=face_rho, mass_kg=None, status="unavailable",
            method="disc less cutouts", source_ids=(), key_parameters={},
            unavailable_reason=(
                "the machined layout did not resolve a faceplate radius and "
                "thickness"
            ),
        ))

    # ---- pintle post ----------------------------------------------------- #
    if post_ro is not None and post_ri is not None and post_len is not None:
        tube = math.pi * (post_ro ** 2 - post_ri ** 2) * post_len
        style = str(resolved.get("radial_exit_style") or "")
        opening_count = int(hydraulic.get("radial_opening_count") or 0)
        wall = _finite_positive(resolved.get("pintle_wall_thickness_m")) or 0.0
        if style == "holes":
            d_hole = _finite_positive(resolved.get("radial_hole_diameter_m"))
            cut = (
                opening_count * 0.25 * math.pi * d_hole ** 2 * wall
                if d_hole is not None else 0.0
            )
        else:
            slot_w = _finite_positive(resolved.get("slot_width_m"))
            slot_h = _finite_positive(resolved.get("slot_height_m"))
            cut = (
                opening_count * slot_w * slot_h * wall
                if slot_w is not None and slot_h is not None else 0.0
            )
        net = max(tube - cut, 0.0)
        items.append(MassItem(
            subsystem="injector",
            component="pintle post",
            quantity=1,
            material=post_name,
            volume_m3=net,
            density_kg_m3=post_rho,
            mass_kg=(net * post_rho) if post_rho is not None else None,
            status="geometry_resolved",
            method=(
                "annular tube (pintle OD to centre bore) over the body length, "
                "less the radial metering openings through the pintle wall"
            ),
            source_ids=(SP125_SHELL_MASS,),
            key_parameters={
                "pintle_outer_radius_m": post_ro,
                "pintle_inner_radius_m": post_ri,
                "pintle_body_length_m": post_len,
                "radial_exit_style": style or None,
                "radial_opening_count": opening_count,
                "removed_opening_volume_m3": cut,
            },
            unavailable_reason=(
                None if post_rho is not None
                else "pintle material has no density"
            ),
        ))
    else:
        items.append(MassItem(
            subsystem="injector", component="pintle post", quantity=1,
            material=post_name, volume_m3=None, density_kg_m3=post_rho,
            mass_kg=None, status="unavailable", method="annular tube",
            source_ids=(), key_parameters={},
            unavailable_reason=(
                "the machined layout did not resolve the pintle inner/outer "
                "radius and body length"
            ),
        ))

    # ---- sleeve ---------------------------------------------------------- #
    if sleeve_ro is not None and sleeve_ri is not None and annulus_len is not None:
        vol = math.pi * (sleeve_ro ** 2 - sleeve_ri ** 2) * annulus_len
        items.append(MassItem(
            subsystem="injector",
            component="annulus sleeve",
            quantity=1,
            material=body_name,
            volume_m3=vol,
            density_kg_m3=body_rho,
            mass_kg=(vol * body_rho) if body_rho is not None else None,
            status="geometry_resolved",
            method="annular tube (sleeve OD to sleeve ID) over annulus length",
            source_ids=(SP125_SHELL_MASS,),
            key_parameters={
                "sleeve_outer_radius_m": sleeve_ro,
                "sleeve_inner_radius_m": sleeve_ri,
                "annulus_length_m": annulus_len,
                "sleeve_wall_thickness_m": resolved.get(
                    "sleeve_wall_thickness_m"
                ),
            },
            unavailable_reason=(
                None if body_rho is not None
                else "injector body material has no density"
            ),
        ))
    else:
        items.append(MassItem(
            subsystem="injector", component="annulus sleeve", quantity=1,
            material=body_name, volume_m3=None, density_kg_m3=body_rho,
            mass_kg=None, status="unavailable", method="annular tube",
            source_ids=(), key_parameters={},
            unavailable_reason=(
                "the machined layout did not resolve the sleeve radii and "
                "annulus length"
            ),
        ))

    warnings.append(
        "inlet fittings, seals, igniter hardware and the film-cooling ring (if "
        "any) are not in this ledger"
    )
    if not manifold_volume_known:
        warnings.append(
            "the faceplate mass is an upper bound because at least one "
            "manifold pocket volume was unresolved"
        )

    provenance = {
        "geometry_source": (
            "raosim.injector_cad.resolve_machined_pintle_layout"
        ),
        "cad_consistency": (
            "raosim.injector_cad.build_machined_pintle_bodies cuts its solids "
            "from this same layout, so the mass tracks the exported parts"
        ),
        "faceplate_thickness_basis": (
            "floored by the layout at max(deepest manifold + back web, "
            "0.4*Dp, 6*min_tool) and screened against clamped-plate bending "
            "sigma ~= 0.75 p a^2 / t^2 by "
            "raosim.interface.screen_injector_chamber_interface"
        ),
        "layout_status": layout.get("status"),
    }
    return MassLedger(
        scope=scope,
        items=tuple(items),
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
    )
