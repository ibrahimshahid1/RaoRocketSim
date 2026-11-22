# docs/data

Small reference data files kept with the documentation. Nothing in the package reads them. Data that code loads lives
in `raosim/data/` and `raosim/benchmark_data/` and is declared as package data in `pyproject.toml`.

| Document | Status | Type | Stage | What it is |
|---|---|---|---|---|
| [nasa_jhu_file_list.txt](nasa_jhu_file_list.txt) | active | data | S06 | Inventory of the NASA/JHU M3.5Perf fixture files with column layouts and the `raosim/legacy_io.py` parser for each. |
| [rao_nozzle.csv](rao_nozzle.csv) | stub | data | S05 | 200-point `x_m,y_m` wall contour without a provenance header; not referenced by code, tests or docs. |

Notes:

- `requirements.txt` was filed under `docs/` until 2026-11-05; it now sits at the repository root, where
  `requirements-dev.txt` includes it with `-r requirements.txt`.
- A 300-point sibling of `rao_nozzle.csv` (`rao_nozzle_profile.csv`, same first point) is in
  [../archive/](../archive/README.md).

Status, type and stage come from the per-document triage of 2026-11-05. Identifiers, counts and dates quoted in the descriptions were re-checked against each document on 2026-11-09. *Status*: active = current reference or open plan; historical record = dated snapshot kept for traceability (numbers and statuses are as of its date); superseded = replaced by a named later document; stub = empty, or data without provenance. *Stage* is the pipeline stage the document mainly serves (see [../README.md](../README.md)).
