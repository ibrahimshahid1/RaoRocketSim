"""
cea.py - Optional RocketCEA/NASA CEA thermochemistry integration.

RocketCEA is intentionally optional. When it is not installed, callers can
fall back to the built-in propellant table while preserving an explicit warning
that the values are demo-grade constants rather than CEA-derived properties.
"""

from __future__ import annotations

from raosim.propellants import Propellant, get_propellant


def rocketcea_available() -> bool:
    try:
        import rocketcea  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def cea_propellant(
    *,
    oxidizer: str,
    fuel: str,
    Pc: float,
    mixture_ratio: float,
    eta_Isp: float = 0.95,
) -> Propellant:
    """
    Build a Propellant from RocketCEA chamber properties.

    Parameters use SI units except RocketCEA's mixture ratio convention.
    """
    try:
        from rocketcea.cea_obj_w_units import CEA_Obj  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RocketCEA is not installed; install rocketcea or run without --cea."
        ) from exc

    cea = CEA_Obj(
        oxName=oxidizer,
        fuelName=fuel,
        pressure_units="Pa",
        temperature_units="K",
        cstar_units="m/s",
    )
    mw_gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc, MR=mixture_ratio)
    if isinstance(mw_gamma, tuple) and len(mw_gamma) >= 2:
        mw, gamma = float(mw_gamma[0]), float(mw_gamma[1])
    else:
        raise RuntimeError("RocketCEA did not return chamber molecular weight/gamma.")

    Tc = float(cea.get_Tcomb(Pc=Pc, MR=mixture_ratio))
    c_star = float(cea.get_Cstar(Pc=Pc, MR=mixture_ratio))
    prop = Propellant(
        name=f"CEA {oxidizer}/{fuel}",
        gamma=gamma,
        Mw=mw / 1000.0,
        Tc=Tc,
        eta_Isp=eta_Isp,
        OF=mixture_ratio,
    )
    prop.c_star = c_star
    return prop


def propellant_from_request(
    *,
    propellant_name: str | None,
    use_cea: bool,
    Pc: float,
    mixture_ratio: float | None = None,
    oxidizer: str | None = None,
    fuel: str | None = None,
    eta_Isp: float = 0.95,
) -> tuple[Propellant, list[str]]:
    """Resolve propellant data and return warnings from any fallback path."""
    warnings: list[str] = []

    if use_cea:
        ox = oxidizer
        fu = fuel
        if (ox is None or fu is None) and propellant_name and "/" in propellant_name:
            ox, fu = propellant_name.split("/", 1)
        if ox and fu and mixture_ratio is not None:
            try:
                return cea_propellant(
                    oxidizer=ox,
                    fuel=fu,
                    Pc=Pc,
                    mixture_ratio=mixture_ratio,
                    eta_Isp=eta_Isp,
                ), warnings
            except Exception as exc:
                warnings.append(f"CEA requested but unavailable/failed: {exc}")
        else:
            warnings.append(
                "CEA requested but oxidizer, fuel, or mixture ratio is missing."
            )

    if not propellant_name:
        raise ValueError("propellant_name is required when CEA is unavailable.")
    prop = get_propellant(propellant_name)
    if use_cea:
        warnings.append(
            f"Using built-in {prop.name} constants instead of CEA-derived properties."
        )
    return prop, warnings
