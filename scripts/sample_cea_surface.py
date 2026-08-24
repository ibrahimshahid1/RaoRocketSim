#!/usr/bin/env python3
"""Offline CEA property-table sampler (host-only; needs RocketCEA).

Phase-2 workflow (docs/DIFFERENTIABLE_ENGINE_MDO_PLAN.md §12.2 row 2):
sample frozen chamber properties on a (Pc, O/F) grid, save as .npz, load in
the differentiable layer via ``raosim.mdo.properties.load_chamber_surfaces``.

Example:
    python scripts/sample_cea_surface.py --oxidizer LOX --fuel RP-1 \
        --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 13 \
        --of-min 1.6 --of-max 3.2 --n-of 13 \
        --out builds/cea_tables/lox_rp1_frozen.npz

Validation: ``--holdout`` samples seeded points strictly inside fit-grid cells
and compares both values and gradients to live RocketCEA.  It exits nonzero if
any field-specific tolerance is exceeded.  ``--validate-only`` applies the same
gate to an existing, hash-verified table without resampling its fit grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oxidizer", required=True)
    ap.add_argument("--fuel", required=True)
    ap.add_argument("--pc-min", type=float)
    ap.add_argument("--pc-max", type=float)
    ap.add_argument("--n-pc", type=int, default=13)
    ap.add_argument("--of-min", type=float)
    ap.add_argument("--of-max", type=float)
    ap.add_argument("--n-of", type=int, default=13)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="store_true",
                    help="run the authoritative value+gradient validation gate")
    ap.add_argument(
        "--validate-only", action="store_true",
        help="validate the existing --out table; do not sample or overwrite it",
    )
    ap.add_argument(
        "--holdout-count", type=int, default=0,
        help=("maximum seeded held-out cells to validate; 0 checks every "
              "fit-grid cell (default: 0)"),
    )
    ap.add_argument(
        "--holdout-seed", type=int, default=20260817,
        help="deterministic held-out cell/point selection seed",
    )
    ap.add_argument(
        "--fd-fraction", type=float, default=0.20,
        help="central-difference step as a fraction of nearer cell-edge distance",
    )
    ap.add_argument(
        "--tolerances-json",
        help=("optional JSON file containing complete per-field value/dPc/dOF "
              "rtol+atol settings"),
    )
    ap.add_argument(
        "--report",
        help="write the hash-linked validation report as JSON",
    )
    return ap


def _print_validation_report(report) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(
        f"held-out CEA gate: {status}  points={len(report.holdout_points)} "
        f"schema={report.table_schema_version} "
        f"sha256={report.table_content_sha256}"
    )
    print(
        "RocketCEA cards: "
        f"{report.rocketcea_oxidizer}/{report.rocketcea_fuel} "
        f"(table identity {report.oxidizer}/{report.fuel})"
    )
    for note in report.backend_notes:
        print(f"  CARD NOTE: {note}")
    for field, summary in report.fields.items():
        for quantity in ("value", "dPc", "dOF"):
            metric = getattr(summary, quantity)
            metric_status = "PASS" if metric.passed else "FAIL"
            print(
                f"  {field:5s} {quantity:5s} {metric_status}: "
                f"max |err|={metric.max_abs_error:.6e}, "
                f"max tolerance ratio={metric.max_tolerance_ratio:.6g}, "
                f"failures={metric.failures}/{metric.checked}"
            )


def main(argv: Sequence[str] | None = None, *, sampler=None) -> int:
    ap = _parser()
    args = ap.parse_args(argv)

    from raosim.mdo.properties import (
        cea_validation_tolerances_from_mapping,
        rocketcea_card_names,
        rocketcea_card_notes,
        sample_cea_tables,
        save_tables,
        validate_cea_surface_holdout,
    )

    out = Path(args.out)
    if not args.validate_only:
        missing = [
            flag for flag, value in (
                ("--pc-min", args.pc_min), ("--pc-max", args.pc_max),
                ("--of-min", args.of_min), ("--of-max", args.of_max),
            ) if value is None
        ]
        if missing:
            ap.error(
                "table generation requires " + ", ".join(missing)
                + " (or use --validate-only)"
            )
        if args.n_pc < 2 or args.n_of < 2:
            ap.error("--n-pc and --n-of must each be at least 2")
        Pc_grid = np.linspace(args.pc_min, args.pc_max, args.n_pc)
        OF_grid = np.linspace(args.of_min, args.of_max, args.n_of)
        try:
            tables = sample_cea_tables(
                Pc_grid,
                OF_grid,
                oxidizer=args.oxidizer,
                fuel=args.fuel,
                sampler=sampler,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            save_tables(
                str(out), tables, oxidizer=args.oxidizer, fuel=args.fuel
            )
        except Exception as exc:
            print(f"CEA table sampling failed: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {out}  grid {args.n_pc}x{args.n_of}")
        rocket_ox, rocket_fuel = rocketcea_card_names(
            args.oxidizer, args.fuel
        )
        print(
            f"RocketCEA cards {rocket_ox}/{rocket_fuel}; "
            f"table identity {args.oxidizer}/{args.fuel}"
        )
        for note in rocketcea_card_notes(args.oxidizer, args.fuel):
            print(f"CARD NOTE: {note}")

    run_validation = args.holdout or args.validate_only
    if not run_validation:
        return 0
    if args.holdout_count < 0:
        ap.error("--holdout-count must be nonnegative (0 means every cell)")

    tolerances = None
    if args.tolerances_json:
        try:
            raw_tolerances = json.loads(
                Path(args.tolerances_json).read_text(encoding="utf-8")
            )
            if not isinstance(raw_tolerances, dict):
                raise ValueError("top level must be an object keyed by field")
            tolerances = cea_validation_tolerances_from_mapping(raw_tolerances)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"could not read validation tolerances: {exc}", file=sys.stderr)
            return 2

    try:
        report = validate_cea_surface_holdout(
            str(out),
            expected_propellant=f"{args.oxidizer}/{args.fuel}",
            sampler=sampler,
            max_points=(None if args.holdout_count == 0 else args.holdout_count),
            seed=args.holdout_seed,
            finite_difference_fraction=args.fd_fraction,
            tolerances=tolerances,
        )
    except Exception as exc:
        print(f"held-out CEA validation could not run: {exc}", file=sys.stderr)
        return 2

    _print_validation_report(report)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote validation report {report_path}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
