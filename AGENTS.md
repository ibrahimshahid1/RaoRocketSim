# LREKit repository guidance

For scientifically consequential work, read [knowledge/AGENTS.md](knowledge/AGENTS.md)
and use the local scientific memory before changing models, correlations,
coefficients, validity limits, engineering gates, or physics-derived geometry.
Start with `knowledge/kg.sh status` and `knowledge/kg.sh context --task "..."`.
The original PDF is the authority; a registry citation or passing software test
is not physical validation. Preserve explicit uncertainty and unavailable states.

For knowledge-system maintenance, run `.venv/bin/python -m pytest knowledge/tests -q`.
Use `knowledge/kg.sh rebuild` for the complete deterministic refresh. Preserve
literature extraction and review records; derived indexes can be rebuilt.
Resume literature work with `knowledge/kg.sh pending` and a single bounded
`packet`, banking each result before starting another. Do not launch the archived
Claude parallel workflows for routine continuation. Formatting and other
mechanical changes do not require scientific retrieval.
