"""Package assets and optional checkout paths, independent of module depth.

Installed numerical fixtures always come from the package. Local literature,
research outputs and the reference case belong to a checkout; a wheel can run
without that corpus and writes its optional outputs relative to the caller.
"""

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repository_root() -> Path:
    for parent in package_root().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/lrekit").is_dir():
            return parent
    return Path.cwd()
