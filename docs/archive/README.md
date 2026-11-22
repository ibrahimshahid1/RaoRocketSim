# docs/archive

Items moved here on 2026-11-05 because they are historical, empty or superseded and nothing cites them as an
authority (checked against code, tests, `pyproject.toml`, `raosim/model_registry.py` `local_source` entries,
`knowledge/kg/` and the live documents). Generated knowledge indexes (`knowledge/repo_documents.json`,
`provenance.json`, `graph/graph.json`) still list some of them under their old paths until `knowledge/kg.sh rebuild`
is run.

| Document | Status | What it is | Why archived | Superseded by / read instead |
|---|---|---|---|---|
| [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md) | superseded | June 2026 hand-off prompt from the characteristic-pairing milestone, with a five-item work queue. | Describes June-era work and test counts; row 22 of `knowledge/reports/pdf_issue_disposition_2026-10-15.md` already says it must be treated as historical. Apart from that report, only generated knowledge indexes name it. | The STATUS head of [JAX_DIFFERENTIABLE_PLAN.md](../program/JAX_DIFFERENTIABLE_PLAN.md) and [AGENTS.md](../agents/AGENTS.md) |
| [PROJECT_AUDIT_2026-08-25.html](PROJECT_AUDIT_2026-08-25.html) | historical record | Styled HTML twin of the 2026-08-25 project audit, section for section. | Presentation-only duplicate; live citations (in `LREKIT_HARDWARE_PROGRAM_2026-09-05.md`) name the `.md`, and only generated knowledge indexes name the `.html`. | [audits/PROJECT_AUDIT_2026-08-25.md](../audits/PROJECT_AUDIT_2026-08-25.md) |
| [Rocket_nozzle_sim_phase2.py](Rocket_nozzle_sim_phase2.py) | superseded | Original single-file interactive Rao bell prototype (contour, plots, simple ascent model, CSV export). | No code, test, `pyproject.toml` entry, registry source or knowledge code imports or cites it. | The `raosim` package (CLI: `lrekit` / `raosim`; legacy toolbox: `main.py`) |
| [deep-research-report.md](deep-research-report.md) | stub | Empty file (0 bytes). | Zero-byte file tracked at the repository root in `HEAD`; nothing to preserve beyond the name. | Nothing |
| [rao_nozzle_profile.csv](rao_nozzle_profile.csv) | stub | 300-point `x_m,y_m` wall profile, apparently a default output of `main.py`. | Example output; the name survives only as an output-file default (`main.py` lines 745 and 1157, and the archived prototype); nothing reads this file. | Nothing (a run writes its own contour CSV into its output folder) |

Documents that are superseded or historical but still cited stay in their folders with a banner or a dated name;
they are listed in Appendix A of the repository guide
(`latex-report/RaoRocketSim_Repository_Guide/chapters/app_a_docs_inventory.tex`).

The repository-level `archive/` folder (gitignored, local only) is unrelated: it holds the former top-level `tmp/`
scratch directory.
