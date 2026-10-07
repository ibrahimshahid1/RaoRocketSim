"""
moc_bounded.py – Bounded axisymmetric MOC net for a given nozzle wall.

Forward *analysis* march of the characteristic net between the axis and a
prescribed wall (throat arc + bell), built characteristic-by-characteristic
on the NASA/JHU topology so that boundary reflections propagate back into
the interior.  This replaces the shrinking front-marching scheme of
``raosim.moc.march_coupled_net`` (see
``docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md``), whose axis and
wall points were never parents of any later point.

Topology (Rice/NASA manual 20030067852 PDF 7-9, 17; NASA C++
``MOC_GridCalc_BDE.cpp`` ``CalcRRCsAlongArc`` 1030-1116,
``CalcInteriorMeshPoints`` 2466-2534, ``CalcAxialMeshPoint`` 2262-2306;
Anderson MCF 3rd ed. §11.5):

* the net is a stack of right-running characteristics (RRC, C−, slope
  ``tan(θ−μ)``), each stored **wall-first** (index 0 = wall point,
  index −1 = axis point);
* RRC ``j`` is built wall → axis: the wall point from the C+ leaving
  ``[1][j−1]`` (tangency BC ``θ = θ_wall``), interior ``[i][j]`` from the C+
  of ``[i+1][j−1]`` (``[i][j−1]`` after a special-point insertion) and the
  C− of ``[i−1][j]`` — so the wall point **is** the C− parent of the first
  interior point and the previous axis point is a C+ parent — and the
  axis point closes the row by symmetry;
* the point count per RRC is ``iLast[j] = iLast[j−1] + specialFlag`` —
  it never shrinks except by negative-radius truncation at the axis.

On the throat arc the NASA arc unit processes are reused verbatim
(``nasa_moc._calc_arc_wall_point_with_special``), so rows up to the arc
end are identical to :func:`raosim.nasa_moc.build_kernel`.  Beyond the
arc a general wall point (same dθ-form algebra, wall geometry from the
wall object) is used; wall turning larger than ``dtheta_limit`` inserts a
NASA-style special point (``_calc_special_wall_point_at``).  Invalid cells
are never geometrically patched: a cell whose intersection lands upstream
of its parents, or a subsonic/non-convergent state, triggers one
refinement (special half-step) and otherwise stops the march with an
explicit reason (Young 2012 §3.3 eq. 3.10 rejects crossed solutions;
Rice manual PDF 21 prescribes refining DTHETAB).

The exit plane is obtained the way NASA's ``CropNozzleToLength``
(C++ 1341-1429) does it: each RRC is linearly interpolated to ``x = x_E``
and the resulting table is integrated across the plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from raosim.gas_dynamics import (
    isentropic_density_ratio,
    isentropic_temperature_ratio,
    prandtl_meyer,
)
from raosim.moc import CharPoint, CharRow
from raosim.nasa_moc import (
    ArcWall,
    MOCNode,
    _calc_arc_wall_point_with_special,
    _calc_axial_mesh_point,
    _calc_interior_mesh_point,
    _calc_special_wall_point_at,
    _char_point_from_values,
    _nasa_calc_A,
    _nasa_calc_B,
    _nasa_calc_R,
    _nasa_l_dy_dx,
    _nasa_tan_avg,
    _safe_rel_err,
    calc_massflow_along_rrc,
)


class CharacteristicCellError(RuntimeError):
    """A unit process produced a cell that is not a solution (crossing/subsonic)."""


# ----------------------------------------------------------------------
#  Wall geometry
# ----------------------------------------------------------------------


class CompositeWall:
    """Throat arc followed by a bell wall, presented as one ``r(x)``.

    ``arc`` covers ``[0, arc.x_end]`` (the downstream throat arc T→N) and
    ``bell`` covers ``[bell.x_start, bell.x_end]`` with
    ``bell.x_start == arc.x_end`` (N).  The bell must expose ``r``,
    ``dr_dx``, ``theta``, ``x_start`` and ``x_end`` (e.g.
    :class:`raosim.wall_model.SplineWall`).
    """

    def __init__(self, arc: ArcWall, bell, *, seam_tol: float = 1e-6):
        seam_gap = abs(float(bell.x_start) - float(arc.x_end))
        scale = max(abs(float(arc.x_end)), abs(float(bell.x_end)), 1e-12)
        if seam_gap > seam_tol * scale:
            raise ValueError(
                "bell wall must start where the throat arc ends: "
                f"arc.x_end={arc.x_end:.6g} vs bell.x_start={bell.x_start:.6g}"
            )
        self.arc = arc
        self.bell = bell
        self.x_start = float(arc.x_start)
        self.x_seam = float(arc.x_end)
        self.x_end = float(bell.x_end)

    def _on_arc(self, x: float) -> bool:
        return float(x) <= self.x_seam

    def r(self, x: float) -> float:
        return self.arc.r(x) if self._on_arc(x) else float(self.bell.r(x))

    def dr_dx(self, x: float) -> float:
        return self.arc.dr_dx(x) if self._on_arc(x) else float(self.bell.dr_dx(x))

    def theta(self, x: float) -> float:
        return self.arc.theta(x) if self._on_arc(x) else float(self.bell.theta(x))


def intersect_wall_strict(wall, x0: float, r0: float, slope: float,
                          *, x_lo: float | None = None,
                          tol: float = 1e-13, max_iter: int = 200
                          ) -> tuple[float, float] | None:
    """Root of ``r0 + slope·(x−x0) − wall.r(x)`` on ``[x_lo, wall.x_end]``.

    Returns ``None`` when the characteristic does not meet the wall
    before ``wall.x_end`` (it leaves through the exit plane) — it never
    clamps to the wall's parameter range, unlike
    :meth:`raosim.wall_model.SplineWall.intersect_char`.  Bracketed
    bisection with a Newton polish: for a supersonic C+ the line slope
    ``tan(θ+μ)`` exceeds the wall slope everywhere on a bell, so the
    bracketing function is monotone and the root unique.
    """
    x_lo = float(wall.x_start if x_lo is None else max(x_lo, wall.x_start))
    x_hi = float(wall.x_end)
    if x_hi <= x_lo:
        return None

    def f(x: float) -> float:
        return r0 + slope * (x - x0) - wall.r(x)

    f_lo = f(x_lo)
    f_hi = f(x_hi)
    if f_lo > 0.0:
        # Starting point already above the wall at x_lo: no admissible hit.
        return None
    if f_hi < 0.0:
        return None
    if f_lo == 0.0:
        return x_lo, float(wall.r(x_lo))
    a, b = x_lo, x_hi
    x = 0.5 * (a + b)
    span = max(x_hi - x_lo, 1e-300)
    for _ in range(max_iter):
        x = 0.5 * (a + b)
        fx = f(x)
        if fx == 0.0:
            break
        if fx < 0.0:
            a = x
        else:
            b = x
        if (b - a) <= tol * span:
            break
    x = 0.5 * (a + b)
    # Newton polish (bounded to the bracket)
    for _ in range(4):
        df = slope - wall.dr_dx(x)
        if abs(df) < 1e-15:
            break
        x_new = x - f(x) / df
        if not (a <= x_new <= b):
            break
        x = x_new
    return float(x), float(wall.r(x))


def wall_x_at_theta(wall, theta_target: float, x_a: float, x_b: float,
                    *, max_iter: int = 200) -> float:
    """``x`` in ``[x_a, x_b]`` where ``wall.theta(x) == theta_target`` (bisection)."""
    a, b = float(min(x_a, x_b)), float(max(x_a, x_b))
    fa = wall.theta(a) - theta_target
    fb = wall.theta(b) - theta_target
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        # Target not bracketed (non-monotone or out of range): nearest end.
        return a if abs(fa) < abs(fb) else b
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = wall.theta(m) - theta_target
        if fm == 0.0 or (b - a) < 1e-14 * max(abs(b), 1.0):
            return m
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# ----------------------------------------------------------------------
#  General (non-arc) wall point — NASA CalcArcWallPoint algebra
# ----------------------------------------------------------------------


def _calc_wall_point_raw_general(
    prev_wall_first: list[CharPoint],
    wall,
    gamma: float,
    *,
    conv_tol: float = 1e-10,
    max_iter: int = 60,
) -> CharPoint | None | str:
    """Wall point from the C+ leaving ``prev_wall_first[1]`` on a general wall.

    Same fixed-point as :func:`raosim.nasa_moc._calc_arc_wall_point_raw`
    (C++ 835-947) with the arc geometry replaced by
    :func:`intersect_wall_strict` / ``wall.theta``.  Returns the string
    ``"beyond"`` when the C+ exits through the wall end (the caller then
    closes the net at the exit lip), ``None`` on a non-physical state.
    """
    if len(prev_wall_first) < 2:
        return None
    p1 = prev_wall_first[1]
    p_prev_wall = prev_wall_first[0]

    M1 = max(float(p1.M), 1.000001)
    theta1 = float(p1.theta)
    r1 = float(p1.r)
    x1 = float(p1.x)
    mu1 = math.asin(1.0 / M1)

    slrc1 = _nasa_l_dy_dx(theta1, mu1)
    A1 = _nasa_calc_A(M1, gamma)
    B1 = _nasa_calc_B(M1, theta1, r1)
    R1 = _nasa_calc_R(M1, theta1, r1)

    x3 = x1
    r3 = float(p_prev_wall.r)
    M3 = M1
    theta3 = theta1
    slrc3 = slrc1
    A3, B3, R3 = A1, B1, R1

    x3_old = r3_old = M3_old = theta3_old = 9.9
    for _ in range(max_iter):
        slrc13 = _nasa_tan_avg(slrc1, slrc3)
        hit = intersect_wall_strict(wall, x1, r1, slrc13, x_lo=x1)
        if hit is None:
            return "beyond"
        x3, r3 = hit
        theta3 = float(wall.theta(x3))

        if B1 <= R1:
            T1 = (x3 - x1) * (B3 + B1)
        else:
            T1 = (r3 - r1) * (R3 + R1)
        denom = 0.5 * (A1 + A3)
        if abs(denom) < 1e-14:
            return None
        M3 = M1 + (theta3 - theta1 + 0.5 * T1) / denom
        if not math.isfinite(M3) or M3 < 1.0:
            return None

        slrc3 = _nasa_l_dy_dx(theta3, math.asin(1.0 / max(M3, 1.000001)))
        A3 = _nasa_calc_A(M3, gamma)
        B3 = _nasa_calc_B(M3, theta3, r3)
        R3 = _nasa_calc_R(M3, theta3, r3)

        errs = (
            _safe_rel_err(x3, x3_old), _safe_rel_err(r3, r3_old),
            _safe_rel_err(M3, M3_old), _safe_rel_err(theta3, theta3_old),
        )
        x3_old, r3_old, M3_old, theta3_old = x3, r3, M3, theta3
        if all(abs(e) <= conv_tol for e in errs):
            break
    return _char_point_from_values(x3, r3, theta3, M3, gamma)


# ----------------------------------------------------------------------
#  Net container
# ----------------------------------------------------------------------


@dataclass
class BoundedNet:
    """Result of :func:`march_bounded_net` (rows stored wall-first)."""

    rows: list[list[CharPoint]]
    parents: list[list[dict] | None]
    special: list[bool]
    massflow: list[float]
    mdot_rel_err: list[float]
    gamma: float
    x_exit: float
    complete: bool
    stop_reason: str
    n_refined: int = 0
    containment_violations: int = 0
    truncated_rows: int = 0
    wall_x: list[float] = field(default_factory=list)
    # (x, r) of the first cell where RRC j crossed RRC j-1, when the march
    # stopped with ``stop_reason == "rrc_coalescence"``.
    fold_location: tuple[float, float] | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def row_axis_first(self, j: int) -> list[CharPoint]:
        return list(reversed(self.rows[j]))

    def to_char_rows(self) -> list[CharRow]:
        """Adapt to the legacy :class:`raosim.moc.CharRow` view.

        Row 0 is the starting line (``axis=None``, ``wall=None``).  Each
        later row is one RRC: ``wall`` = its wall point, ``axis`` = its axis
        point (``None`` when the row was negative-r truncated), ``interior``
        = the rest in axis-first order.  ``CharRow.parents`` carries the true
        parent map in ``all_points()`` (axis-first) indexing so that
        ``rao_variational.characteristic_net_links`` does not have to guess
        the topology from row order.
        """
        out: list[CharRow] = []
        out.append(CharRow(axis=None, interior=self.row_axis_first(0), wall=None,
                           parents=None))
        for j in range(1, len(self.rows)):
            wf = self.rows[j]
            has_axis = abs(float(wf[-1].r)) <= 1e-12
            wall = wf[0]
            axis = wf[-1] if has_axis else None
            interior_wf = wf[1:-1] if has_axis else wf[1:]
            interior = list(reversed(interior_wf))
            n_pts = len(wf)

            def wf_to_af(row_index: int, k: int) -> int:
                """Wall-first index -> axis-first ``all_points()`` position.

                With an axis point ``all_points()`` is the exact reverse of
                the wall-first row; without one it is ``interior + [wall]``,
                which is the same reversal except that the wall point stays
                last — both cases reduce to ``n-1-k`` for ``k>0`` and
                ``n-1`` for the wall.
                """
                n = len(self.rows[row_index])
                return n - 1 if k == 0 else n - 1 - k

            pmap = self.parents[j]
            parents_af: list[dict | None] = [None] * n_pts
            for k in range(n_pts):
                entry = pmap[k] if pmap is not None else None
                if entry is None:
                    continue
                conv: dict = {}
                for fam in ("cplus", "cminus"):
                    src = entry.get(fam)
                    if src is None:
                        conv[fam] = None
                    else:
                        rj, rk = src
                        conv[fam] = (rj, wf_to_af(rj, rk))
                if "foot" in entry:
                    conv["foot"] = entry["foot"]
                parents_af[wf_to_af(j, k)] = conv
            out.append(CharRow(axis=axis, interior=interior, wall=wall,
                               parents=parents_af))
        return out


# ----------------------------------------------------------------------
#  The march
# ----------------------------------------------------------------------


def _wall_first_nodes(row: list[CharPoint], gamma: float) -> list[MOCNode]:
    return [MOCNode.from_char_point(cp, gamma) for cp in row]


def _signed_cell_area(a: CharPoint, b: CharPoint, c: CharPoint, d: CharPoint) -> float:
    """Shoelace area of the mesh cell ``a→b→c→d`` (sign = orientation)."""
    poly = (a, b, c, d)
    return 0.5 * sum(
        float(p.x) * float(q.r) - float(p.r) * float(q.x)
        for p, q in zip(poly, poly[1:] + poly[:1])
    )


class _CellFold(Exception):
    """A mesh cell has zero/negative area: like characteristics crossed."""

    def __init__(self, i: int, x: float, r: float):
        super().__init__(f"fold at i={i} x={x:.6g} r={r:.6g}")
        self.i, self.x, self.r = i, x, r


def _build_row(
    prev_wall_first: list[CharPoint],
    wall_pt: CharPoint,
    special_flag: bool,
    gamma: float,
    wall,
    *,
    wall_parent: dict,
    row_index: int,
    orientation: float = 0.0,
) -> tuple[list[CharPoint], list[dict], bool, int] | None:
    """Interior + axis closure for one RRC given its wall point.

    Returns ``(row_wall_first, parents, truncated, containment_violations)``
    or ``None`` when a unit process produced an invalid cell; raises
    :class:`_CellFold` when a mesh cell has non-positive area (RRC ``j``
    crossed RRC ``j-1`` — a compression coalescence, i.e. a shock, which
    the isentropic MOC cannot integrate through).
    """
    new_wall_first: list[CharPoint] = [wall_pt]
    parents: list[dict] = [wall_parent]
    i_last_prev = len(prev_wall_first) - 1
    i_end = i_last_prev + (1 if special_flag else 0)
    truncated = False
    violations = 0
    for i in range(1, i_end):
        interior, negative_r = _calc_interior_mesh_point(
            prev_wall_first, new_wall_first, i, special_flag, gamma,
        )
        if negative_r:
            truncated = True
            break
        if interior is None or interior.M < 1.0001:
            return None
        # Fold audit (same test as nasa_moc._calc_remaining_mesh_row): the
        # cell prev[ii-1] -> prev[ii] -> new[i] -> new[i-1] must keep the
        # orientation of the first cell of the row.  NASA's C++ carries
        # this "point behind previous RRC" check commented out because it
        # only ever marches shock-free Rao designs; an analysis of an
        # arbitrary wall must not integrate through coalesced RRCs.
        ii_f = min(i if special_flag else i + 1, i_last_prev)
        if orientation != 0.0 and ii_f >= 1:
            area = _signed_cell_area(
                prev_wall_first[ii_f - 1], prev_wall_first[ii_f],
                interior, new_wall_first[i - 1],
            )
            edge = math.hypot(
                prev_wall_first[ii_f].x - prev_wall_first[ii_f - 1].x,
                prev_wall_first[ii_f].r - prev_wall_first[ii_f - 1].r,
            ) * math.hypot(
                interior.x - prev_wall_first[ii_f].x,
                interior.r - prev_wall_first[ii_f].r,
            )
            if orientation * area <= 1e-12 * max(edge, 1e-24):
                raise _CellFold(i, interior.x, interior.r)
        # The C- from the just-completed neighbour runs at angle theta-mu in
        # (-90 deg, 0), so its intersection must lie downstream of that
        # parent; landing upstream is a crossing of like characteristics,
        # not a solution (Young 2012 eq. 3.10) and is never patched.  (The
        # C+ parent may legitimately sit downstream of the cell near the
        # throat, where theta+mu exceeds 90 deg.)
        ii = min(i if special_flag else i + 1, i_last_prev)
        p2 = new_wall_first[i - 1]
        if interior.x < p2.x - 1e-12 * max(abs(p2.x), 1.0):
            return None
        if interior.x <= wall.x_end and (
            interior.r > wall.r(interior.x) + 1e-9 * max(abs(interior.r), 1.0)
        ):
            violations += 1
        new_wall_first.append(interior)
        parents.append({"cplus": (row_index - 1, ii), "cminus": (row_index, i - 1)})
    # NASA closes every row — including one shortened by a negative-radius
    # point — with CalcAxialMeshPoint from the last retained interior point
    # (C++ 1072 after the r<0 break at 2662-2666; nasa_moc._rrc_march_step).
    axis_pt = _calc_axial_mesh_point(new_wall_first, gamma)
    if axis_pt is None:
        if truncated:
            return new_wall_first, parents, True, violations
        return None
    new_wall_first.append(axis_pt)
    parents.append({"cplus": None, "cminus": (row_index, len(new_wall_first) - 2)})
    return new_wall_first, parents, truncated, violations


def march_bounded_net(
    start_axis_first: list[CharPoint],
    wall,
    gamma: float,
    *,
    arc: ArcWall | None = None,
    dtheta_limit: float = math.radians(0.5),
    mass_tol: float = 0.02,
    on_invalid: str = "refine",
    x_exit: float | None = None,
    max_rrcs: int = 5000,
) -> BoundedNet:
    """March the bounded characteristic net from ``start_axis_first`` to the exit.

    Parameters
    ----------
    start_axis_first : transonic data line, axis first (index 0 = axis,
        index −1 = throat-plane wall point), e.g.
        :func:`raosim.nasa_moc._make_throat_initial_line`.
    wall : the full wall ``r(x)``/``theta(x)``/``dr_dx(x)`` from the throat
        plane to the exit (normally a :class:`CompositeWall`).
    arc : the throat arc portion; while the previous wall point lies on it
        the NASA arc unit processes are used so the kernel rows match
        :func:`raosim.nasa_moc.build_kernel` exactly.
    dtheta_limit : NASA ``DTHETAB`` (radians) — maximum wall turning per
        RRC before a special point is inserted.
    mass_tol : NASA 2 % per-RRC mass-flow abort (C++ 1077-1089).
    on_invalid : ``"refine"`` (retry the row once with a half-step special
        wall point, then stop), ``"reject"`` (raise
        :class:`CharacteristicCellError`) or ``"stop"``.
    x_exit : exit station; default ``wall.x_end``.
    """
    if len(start_axis_first) < 3:
        raise ValueError("Need at least 3 starting-line points")
    if on_invalid not in {"refine", "reject", "stop"}:
        raise ValueError("on_invalid must be 'refine', 'reject' or 'stop'")
    x_exit = float(wall.x_end if x_exit is None else x_exit)

    rows: list[list[CharPoint]] = [list(reversed(start_axis_first))]
    parents: list[list[dict] | None] = [None]
    special: list[bool] = [False]
    massflow: list[float] = [float(calc_massflow_along_rrc(
        _wall_first_nodes(rows[0], gamma), gamma)[0])]
    mdot_rel_err: list[float] = [0.0]
    mdot0 = massflow[0]
    n_refined = 0
    violations = 0
    truncated_rows = 0
    complete = False
    stop_reason = "max_rrcs"

    def fail(reason: str) -> BoundedNet:
        return BoundedNet(rows=rows, parents=parents, special=special,
                          massflow=massflow, mdot_rel_err=mdot_rel_err,
                          gamma=gamma, x_exit=x_exit, complete=False,
                          stop_reason=reason, n_refined=n_refined,
                          containment_violations=violations,
                          truncated_rows=truncated_rows,
                          wall_x=[float(r[0].x) for r in rows[1:]])

    seam = float(arc.x_end) if arc is not None else -math.inf
    # Mesh orientation for the fold audit, fixed from the first row whose
    # cells are non-degenerate.  Row 1 is skipped: the start line is built by
    # stepping along RRC slopes (NASA CalcInitialThroatLine), so RRC 1 nearly
    # coincides with it and its cell areas are sign noise.
    orientation = 0.0

    for j in range(1, max_rrcs + 1):
        prev = rows[-1]
        if len(prev) < 3:
            return fail("row_too_short")
        prev_wall = prev[0]
        on_arc = arc is not None and (
            prev_wall.x < seam - 1e-12 * max(seam, 1.0)
            and prev_wall.theta < arc.theta_max - 1e-9
        )

        attempts = []
        if on_arc:
            res = _calc_arc_wall_point_with_special(prev, arc, gamma, dtheta_limit)
            if res is None:
                return fail("arc_wall_point_failed")
            wall_pt, sflag = res
            foot = "special" if sflag else "raw"
            attempts.append((wall_pt, sflag, foot))
        else:
            raw = _calc_wall_point_raw_general(prev, wall, gamma)
            if raw is None:
                return fail("wall_point_failed")
            if isinstance(raw, str):  # "beyond": close at the exit lip
                x3 = x_exit
                r3, th3 = float(wall.r(x3)), float(wall.theta(x3))
                sp = _calc_special_wall_point_at(prev, x3, r3, th3, gamma)
                if sp is None:
                    return fail("exit_lip_point_failed")
                attempts.append((sp, True, "exit_lip"))
            else:
                dth = raw.theta - prev_wall.theta
                if raw.x <= prev_wall.x + 1e-12 * max(abs(prev_wall.x), 1.0):
                    if on_invalid == "reject":
                        raise CharacteristicCellError(
                            f"wall point of RRC {j} lands upstream of RRC {j-1}")
                    return fail("wall_point_upstream")
                if abs(dth) > dtheta_limit:
                    th_t = prev_wall.theta + math.copysign(0.5 * dtheta_limit, dth)
                    x_t = wall_x_at_theta(wall, th_t, prev_wall.x, raw.x)
                    sp = _calc_special_wall_point_at(
                        prev, x_t, float(wall.r(x_t)), float(wall.theta(x_t)), gamma)
                    if sp is None:
                        return fail("special_wall_point_failed")
                    attempts.append((sp, True, "special"))
                else:
                    attempts.append((raw, False, "raw"))
            # Refinement candidate: half-step special point.
            if on_invalid == "refine" and attempts[0][2] == "raw":
                raw_pt = attempts[0][0]
                th_t = prev_wall.theta + 0.5 * (raw_pt.theta - prev_wall.theta)
                x_t = wall_x_at_theta(wall, th_t, prev_wall.x, raw_pt.x)
                if x_t > prev_wall.x:
                    sp = _calc_special_wall_point_at(
                        prev, x_t, float(wall.r(x_t)), float(wall.theta(x_t)), gamma)
                    if sp is not None:
                        attempts.append((sp, True, "refined"))

        built = None
        used_sflag = False
        fold: _CellFold | None = None
        for k, (wall_pt, sflag, foot) in enumerate(attempts):
            if wall_pt.r < 0.0 or wall_pt.M < 1.0001:
                continue
            wall_parent = {"cplus": (j - 1, 1), "cminus": None, "foot": foot}
            try:
                built = _build_row(prev, wall_pt, sflag, gamma, wall,
                                   wall_parent=wall_parent, row_index=j,
                                   orientation=orientation if j >= 3 else 0.0)
            except _CellFold as exc:
                fold = exc
                built = None
            if built is not None:
                used_sflag = bool(sflag)
                if k > 0:
                    n_refined += 1
                break
        if built is None:
            if fold is not None:
                fold_location = (float(fold.x), float(fold.r))
                if on_invalid == "reject":
                    raise CharacteristicCellError(
                        f"RRC {j} crosses RRC {j-1} at x={fold.x:.6g}, "
                        f"r={fold.r:.6g} (compression coalescence)")
                net = fail("rrc_coalescence")
                net.fold_location = fold_location
                return net
            if on_invalid == "reject":
                raise CharacteristicCellError(f"RRC {j} has an invalid cell")
            return fail("invalid_cell")

        new_row, new_parents, truncated, viol = built
        violations += viol
        if orientation == 0.0 and j >= 2:
            areas = []
            for i in range(1, min(len(new_row), len(prev))):
                ii_f = min(i if used_sflag else i + 1, len(prev) - 1)
                if ii_f >= 1:
                    areas.append(_signed_cell_area(
                        prev[ii_f - 1], prev[ii_f], new_row[i], new_row[i - 1]))
            nz = [a for a in areas if abs(a) > 0.0]
            if nz:
                orientation = math.copysign(1.0, float(np.median(nz)))
        if truncated:
            truncated_rows += 1

        mdot = float(calc_massflow_along_rrc(_wall_first_nodes(new_row, gamma), gamma)[0])
        rel = (mdot - mdot0) / mdot0 if mdot0 > 0 else 0.0
        if abs(rel) > mass_tol:
            return fail("massflow_drift")

        rows.append(new_row)
        parents.append(new_parents)
        special.append(used_sflag)
        massflow.append(mdot)
        mdot_rel_err.append(rel)

        if new_row[0].x >= x_exit - 1e-12 * max(abs(x_exit), 1.0):
            complete = True
            stop_reason = "exit_reached"
            break

    return BoundedNet(rows=rows, parents=parents, special=special,
                      massflow=massflow, mdot_rel_err=mdot_rel_err,
                      gamma=gamma, x_exit=x_exit, complete=complete,
                      stop_reason=stop_reason, n_refined=n_refined,
                      containment_violations=violations,
                      truncated_rows=truncated_rows,
                      wall_x=[float(r[0].x) for r in rows[1:]])


# ----------------------------------------------------------------------
#  Exit plane
# ----------------------------------------------------------------------


def _interp_point(a: CharPoint, b: CharPoint, t: float, gamma: float) -> dict:
    M = max(1.0, a.M + t * (b.M - a.M))
    theta = a.theta + t * (b.theta - a.theta)
    return {
        "r": a.r + t * (b.r - a.r),
        "theta": theta,
        "M": M,
        "nu": prandtl_meyer(M, gamma),
    }


def crop_polyline_to_x(points: list[CharPoint], x_exit: float, gamma: float,
                       tol: float = 1e-12) -> list[dict]:
    """All crossings of ``x = x_exit`` along a characteristic polyline (NASA crop)."""
    hits: list[dict] = []
    for a, b in zip(points[:-1], points[1:]):
        lo, hi = min(a.x, b.x), max(a.x, b.x)
        if not (lo - tol <= x_exit <= hi + tol):
            continue
        dx = b.x - a.x
        if abs(dx) < 1e-15:
            hits.append(_interp_point(a, b, 0.0, gamma))
            continue
        t = (x_exit - a.x) / dx
        t = min(max(t, 0.0), 1.0)
        hits.append(_interp_point(a, b, t, gamma))
    return hits


def crop_to_exit_plane(net: BoundedNet, x_exit: float | None = None) -> list[dict]:
    """Exit-plane table (one sample per RRC crossing ``x = x_exit``).

    Mirrors ``CropNozzleToLength`` (C++ 1341-1429): every RRC is linearly
    interpolated to the exit station.  Samples are sorted by radius; an
    ``r = 0`` sample (``θ = 0``) is added by extrapolation when the
    innermost crossing is off-axis so the plane integral covers the disc.
    """
    x_exit = float(net.x_exit if x_exit is None else x_exit)
    samples: list[dict] = []
    for row in net.rows:
        samples.extend(crop_polyline_to_x(row, x_exit, net.gamma))
    samples = [s for s in samples if s["r"] >= -1e-12]
    samples.sort(key=lambda s: s["r"])
    merged: list[dict] = []
    for s in samples:
        if merged and abs(s["r"] - merged[-1]["r"]) < 1e-12:
            merged[-1] = s
        else:
            merged.append(s)
    if len(merged) >= 2 and merged[0]["r"] > 1e-12:
        a, b = merged[0], merged[1]
        dr = b["r"] - a["r"]
        M0 = a["M"] - (b["M"] - a["M"]) * a["r"] / dr if dr > 0 else a["M"]
        M0 = max(1.0, M0)
        merged.insert(0, {"r": 0.0, "theta": 0.0, "M": M0,
                          "nu": prandtl_meyer(M0, net.gamma)})
    elif len(merged) >= 1 and merged[0]["r"] > 1e-12:
        a = merged[0]
        merged.insert(0, {"r": 0.0, "theta": 0.0, "M": a["M"], "nu": a["nu"]})
    return merged


def exit_plane_mass_flow(samples: list[dict], gamma: float) -> float:
    """``∫ ρ u cosθ dA`` across the exit plane in the stagnation-normalised
    units of :func:`raosim.nasa_moc.calc_massflow_along_rrc` (ρ0 = 1,
    velocities over ``sqrt(R T0)``), so it is directly comparable to
    ``BoundedNet.massflow``."""
    if len(samples) < 2:
        return 0.0
    ordered = sorted(samples, key=lambda s: s["r"])
    r = np.array([s["r"] for s in ordered], dtype=float)
    M = np.array([max(1.0, s["M"]) for s in ordered], dtype=float)
    th = np.array([s["theta"] for s in ordered], dtype=float)
    T = np.array([isentropic_temperature_ratio(m, gamma) for m in M])
    rho = np.array([isentropic_density_ratio(m, gamma) for m in M])
    u = M * np.sqrt(gamma * T) * np.cos(th)
    return float(2.0 * math.pi * np.trapezoid(rho * u * r, r))
