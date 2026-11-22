"""Import aliases for the former package name and the former flat module layout.

The implementation lives once, under :mod:`lrekit` (``lrekit.components``,
``lrekit.optimization``, ``lrekit.numerics`` ...). Two older spellings still import:

* ``raosim.<x>``: the package was named ``raosim`` until 2026-11-21. ``raosim`` is
  now a compatibility package holding only ``__init__`` and ``__main__``.
* the flat module names used before the component packages existed
  (``raosim.cea``, ``raosim.mdo``, ``lrekit.engine`` ...), mapped by
  :data:`FLAT_ALIASES`.

An alias returns the canonical module object itself. Nothing is executed a second
time, so dataclasses, enums, registries and caches exist once, and objects pickled
under an old name still load. Application code imports canonical names only.
"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from importlib.util import find_spec
import sys


CANONICAL = "lrekit"
LEGACY = "raosim"

#: Former flat module name -> canonical module, both relative to the package.
FLAT_ALIASES: dict[str, str] = {
    'altitude_performance': 'performance.altitude',
    'atmosphere': 'core.atmosphere',
    'benchmarks': 'verification.benchmarks',
    'cea': 'thermochemistry.cea',
    'chamber_geometry': 'components.chamber.geometry',
    'cli': 'application.cli',
    'conical': 'components.nozzle.conical',
    'coolants': 'core.coolants',
    'design': 'application.design',
    'electric_technology': 'components.electrical.technology',
    'engine': 'performance.engine',
    'engine_cad': 'assembly.cad',
    'export': 'geometry.export',
    'flange_cad': 'assembly.flange',
    'flow_viz': 'components.nozzle.flow_visualization',
    'frozen_flow': 'thermochemistry.frozen_flow',
    'gas_dynamics': 'core.gas_dynamics',
    'injector': 'components.injector.design',
    'injector_cad': 'components.injector.cad',
    'injector_coaxial_cad': 'components.injector.coaxial',
    'injector_export': 'components.injector.export',
    'injector_plots': 'components.injector.plots',
    'input_validation': 'core.input_validation',
    'interface': 'assembly.interfaces',
    'jax': 'numerics.jax',
    'legacy_io': 'verification.reference_io',
    'manufacturing_limits': 'geometry.manufacturing',
    'mass_ledger': 'assembly.mass',
    'materials': 'core.materials',
    'mdo': 'optimization',
    'moc': 'components.nozzle.solvers.moc',
    'moc_bounded': 'components.nozzle.solvers.bounded',
    'moc_diagrams': 'components.nozzle.diagrams',
    'moc_topology': 'components.nozzle.solvers.topology',
    'model_registry': 'verification.registry',
    'movable_pintle': 'components.injector.movable',
    'nasa_moc': 'components.nozzle.solvers.nasa',
    'nozzle_comparison': 'components.nozzle.comparison',
    'nozzle_geometry': 'components.nozzle.geometry',
    'openfoam': 'simulation.openfoam',
    'physics': 'components.cooling.screens',
    'plotting': 'components.nozzle.plots',
    'propellants': 'thermochemistry.propellants',
    'pump_cad': 'components.pump.cad',
    'pump_cad_brep': 'components.pump.brep',
    'pumps': 'components.pump.design',
    'rao_existence_scan': 'components.nozzle.solvers.existence',
    'rao_optimizer': 'components.nozzle.solvers.wall_optimizer',
    'rao_residuals': 'components.nozzle.solvers.residuals',
    'rao_top_chart': 'components.nozzle.top_chart',
    'rao_variational': 'components.nozzle.solvers.variational',
    'regen_cad': 'components.cooling.cad',
    'regen_geometry': 'components.cooling.geometry',
    'regen_profile': 'components.cooling.profile',
    'regen_volumes': 'components.cooling.volumes',
    'release_readiness': 'verification.release',
    'requirements': 'application.requirements',
    'run_nozzle': 'application.runner',
    'separation': 'components.nozzle.separation',
    'spray': 'components.injector.spray',
    'spray_coupling': 'components.injector.coupling',
    'thermal_correlations': 'components.cooling.correlations',
    'thermal_design': 'components.cooling.design',
    'thermofluids': 'components.cooling.thermofluids',
    'throat_geometry': 'components.nozzle.throat',
    'trade_study': 'application.trade_study',
    'trajectory': 'mission.trajectory',
    'transonic_kernel': 'components.nozzle.solvers.transonic',
    'units': 'core.units',
    'validation': 'verification.validation',
    'wall_model': 'components.nozzle.solvers.wall',
    'wall_pressure': 'components.nozzle.wall_pressure'
}


def canonical_name(name: str) -> str:
    """Return the implementation module for a legacy or flat import name.

    ``raosim.X`` means ``lrekit.X``; former flat module names map through
    :data:`FLAT_ALIASES`. This is the static map only: a module that really
    exists under the requested name always wins (see :class:`_AliasFinder`).
    """
    if name.startswith(LEGACY + "."):
        name = CANONICAL + name[len(LEGACY):]
    if name.startswith(CANONICAL + "."):
        rest = name[len(CANONICAL) + 1:]
        for old, new in FLAT_ALIASES.items():
            if rest == old or rest.startswith(old + "."):
                return f"{CANONICAL}.{new}{rest[len(old):]}"
    return name


class _AliasLoader(Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        module = import_module(self.target)
        self.original_spec = module.__spec__
        self.original_loader = module.__loader__
        self.original_package = module.__package__
        return module

    def exec_module(self, module):
        # Import machinery may assign the alias spec to the shared object.
        # Preserve its canonical identity for relative imports and pickling.
        module.__spec__ = self.original_spec
        module.__loader__ = self.original_loader
        module.__package__ = self.original_package


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith((CANONICAL + ".", LEGACY + ".")):
            return None
        parent_name, _, child = fullname.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and parent.__name__ != parent_name:
            # The parent is itself an alias: its children must be the children
            # of the module it stands for, never a second execution of them.
            canonical = f"{parent.__name__}.{child}"
        elif PathFinder.find_spec(fullname, path) is not None:
            # A module that really lives at this name (lrekit.cli, raosim.__main__,
            # any future package) is never redirected.
            return None
        else:
            canonical = canonical_name(fullname)
        if canonical == fullname:
            return None
        spec = find_spec(canonical)
        if spec is None:
            return None
        return ModuleSpec(fullname, _AliasLoader(canonical),
                          is_package=spec.submodule_search_locations is not None)


def install() -> None:
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        # This must precede PathFinder: an aliased package has a real __path__,
        # otherwise its children would be executed again under the old name.
        sys.meta_path.insert(0, _AliasFinder())
