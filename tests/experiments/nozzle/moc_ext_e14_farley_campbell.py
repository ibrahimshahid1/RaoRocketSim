"""E14 feasibility gate: Farley & Campbell, NASA TN D-293 (1960) wall-pressure benchmark.

Renders the relevant PDF pages at 300 dpi with Ghostscript (OCR is not trusted
for figures) and records the feasibility finding read from those renders.
Digitisation and forward replay run only if the gate passes.

.venv-jax/bin/python tests/experiments/nozzle/moc_ext_e14_farley_campbell.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = str(_repository_root())
PLAN = os.path.join(ROOT, "builds", "moc_extensions_plan_2026-11-21")
PDF = os.path.join(PLAN, "scout4_gap_fill", "downloads", "19980227854.pdf")
OUT_DEFAULT = os.path.join(PLAN, "opus_b", "e14")
PAGES = {
    4: "Printed p. 2: Introduction (coordinates 'obtained from reference 3', digit faint) and Apparatus (initial "
       "coordinates from reference 2, Clippinger; zero throat wall radius of curvature; 8/10 streamlines used)",
    19: "Fig. 2 (printed 17): isentropic nozzle contours incl. boundary-layer displacement correction; plotted "
        "r/rcr vs x/rcr only, inset table of wall angles (initial expansion angle 21.5/23.1/25.0 deg and angles "
        "at A/Acr 10, 15, 20, 25) for M 5.02/5.82/6.85",
    20: "Fig. 3 (printed 18): boundary-layer displacement thickness vs nozzle length ratio (plotted)",
    21: "Fig. 4 (printed 19): 15-deg conical nozzle dimensions and wall static-tap table (expansion ratio of taps)",
    26: "Fig. 10(a,b) (printed 24): wall static-pressure ratio vs A/Acr, measured averages and theoretical curves",
    27: "Fig. 10(c) (printed 25): wall static-pressure ratio vs A/Acr for A/Acr 7.5 to 25",
    31: "Fig. 13(a) (printed 29; OCR caption reads 'Figure 15'): wall pressure distributions with separation, M 5.82",
    32: "Fig. 13(b) (printed 30): wall pressure distributions with separation, M 5.02",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    gd = os.path.join(ROOT, ".git")
    try:
        head = open(os.path.join(gd, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        path = os.path.join(gd, ref)
        if os.path.exists(path):
            return open(path).read().strip()
        for line in open(os.path.join(gd, "packed-refs")):
            if line.strip().endswith(ref):
                return line.split()[0]
    except OSError as exc:
        return f"unavailable: {exc}"
    return "unavailable"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = args.out
    pdir = os.path.join(out, "pages_300dpi")
    os.makedirs(pdir, exist_ok=True)
    gs = shutil.which("gs")
    renders = {}
    for page in sorted(PAGES):
        dest = os.path.join(pdir, f"p{page:02d}.png")
        subprocess.run([gs, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300",
                        f"-dFirstPage={page}", f"-dLastPage={page}", f"-sOutputFile={dest}", PDF], check=True)
        renders[f"PDF {page}"] = dict(file=os.path.relpath(dest, ROOT), sha256=sha256_file(dest), content=PAGES[page])
    script = os.path.abspath(__file__)
    gate = dict(
        question=("Does farley1960 tabulate the MOC nozzle wall coordinates (or the design inputs needed to "
                  "regenerate them) and wall-tap pressures?"),
        wall_coordinates_tabulated=False,
        wall_coordinates_evidence=("Coordinates appear only as a plotted family in Fig. 2 (PDF 19) at about 60 px per "
                                   "x/rcr unit at 300 dpi; the throat region (x/rcr < 1) is not resolvable, and the "
                                   "plotted contours include a boundary-layer displacement correction (Fig. 3, plotted)."),
        design_inputs_sufficient=False,
        design_inputs_evidence=("Printed p. 2 (PDF 4, read from the render): contours are the 8/10 "
                                "streamlines of method-of-characteristics nozzles of Clippinger (ref. 2, BRL Rep. 794, "
                                "1951) designed with zero wall radius of curvature at the throat for uniform exit Mach "
                                "5.018, 5.819, 6.851; the resulting throat radius of curvature is stated only as "
                                "'approximately 8/10 of the nozzle throat diameter'; the Introduction on the same page attributes the "
                                "coordinates to ref. 3 (Rao 1958 IAF; digit faint) while the Apparatus section says ref. 2. The repo "
                                "forward net needs a circular-arc throat start line (KL with Ru, Rd) and a wall contour; "
                                "neither the source design method nor the throat geometry is reproducible from this "
                                "report."),
        wall_pressures_available="plotted only (Fig. 10, PDF 26-27, averages; scatter stated as about +/-3 percent); "
                                 "digitisable but useless without a reconstructible wall",
        spec_figure_correction=("experiment_spec cites 'Fig. 15 (PDF 31-32)'; the page renders show this is Fig. 13 "
                                "(the OCR misreads 13 as 15) and contains separated-flow distributions; the full-flowing "
                                "theory-vs-measurement comparison is Fig. 10 (PDF 26-27)."),
        passes=False,
    )
    result = dict(
        id="E14", status="INFEASIBLE", evidence_class="benchmark agreement (not reached)",
        command=" ".join([".venv-jax/bin/python", os.path.relpath(script, ROOT)] + list(argv if argv is not None else sys.argv[1:])),
        git_head=git_head(), script=os.path.relpath(script, ROOT), script_sha256=sha256_file(script),
        inputs={os.path.relpath(PDF, ROOT): sha256_file(PDF)}, raosim_sources_sha256={},
        configuration=dict(render="Ghostscript png16m 300 dpi, -dSAFER", pages=sorted(PAGES)),
        feasibility_gate=gate, renders=renders,
        reasons=["No tabulated wall coordinates; plotted contours cannot resolve the throat and include a "
                 "boundary-layer displacement correction.",
                 "Design inputs (Clippinger 1951 sharp-throat MOC nozzles, 8/10 streamline) cannot be regenerated by "
                 "the repo forward net, which needs a circular-arc throat and KL start line; any throat radius chosen "
                 "here would be improvised.",
                 "Wall pressures are plotted against A/Acr only, so the comparison would be dominated by contour "
                 "reconstruction error rather than the MOC model."],
        would_make_feasible=["Acquire Clippinger, R. F., Supersonic Axially Symmetric Nozzles, BRL Rep. 794 (1951), "
                             "if it tabulates the design field or wall coordinates, plus a throat model for the 8/10 "
                             "streamline", "Or planner approval of an explicitly DIAGNOSTIC digitise-Fig.2-and-assume-"
                             "throat variant (not done: would require an improvised throat geometry)"],
        hypotheses=dict(benchmark_reconstruction_feasible="not supported"),
        figures={}, wall_time_s=time.time() - t0,
        caption_facts=dict(configuration="Feasibility gate only; no figure produced.",
                           evidence_class="neither", not_established="Any benchmark agreement."),
        not_established="Physical validation of the repo's hot-gas designs; any benchmark agreement.",
    )
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(dict(status=result["status"], renders=list(renders)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
