# Combustion-chamber and injector design literature (provenance + acquisition)

Companion to `docs/thermofluid_literature_provenance.md`. Scope: the
references that govern **combustion-chamber and convergent-section geometry**
(the next build piece after regen), plus the injector and combustion-stability
sources needed for later part exports. Geometry construction is the immediate
target; combustion/injector physics is recorded for the layers after it.

## The documentation question

Unlike the nozzle — which has analytical optimum-shape theory (Rao
calculus-of-variations, method of characteristics) — the combustion chamber
has **no equivalent closed-form optimum**. Chamber geometry is fixed by a small
set of empirically bounded parameters drawn from design handbooks and hot-fire
experience:

- `L*` (characteristic length) = `V_chamber / A_throat` — a residence-time proxy
- `epsilon_c` (contraction ratio) = `A_c / A_t` — sets chamber diameter
- convergent half-angle and throat arc radii — set the converging contour

So the geometry is *thoroughly documented but as design rules + experimental
tables*, not as a flow-PDE optimum. The deep, extensively-researched literature
is the combustion/injector physics that *sets* those parameters (L*-vs-c\*
efficiency, contraction-ratio effects, combustion instability), not the wall
shape itself. This is favorable for construction: the chamber is a bounded,
well-defined build.

SP-125's historical `L*` ranges are starting priors, not universal answers for
a propellant pair.  The minimum viable value also depends on injector design,
chamber pressure, mixture ratio, chamber geometry, atomization/mixing, and the
required combustion efficiency.  Code defaults must therefore remain marked
as preliminary until those inputs and validation evidence are supplied.

## Equation/criterion -> source map (chamber geometry)

| Geometry element | Governing rule | Primary source | In corpus? |
|---|---|---|---|
| Characteristic length `L*` | `L* = V_c / A_t` (eq. 4-4); stay time 0.002-0.040 s, L\* 15-120 in | SP-125 §4 | yes (`19710019929.pdf`) |
| Per-propellant `L*` defaults | Table 4-1 (LOX/RP-1 40-50 in; LOX/LH2 22-40 in; N2O4/hydrazine 30-35 in; LOX/NH3 30-40 in) | SP-125 Table 4-1 | yes |
| Chamber volume / wall area | eq. 4-5 / 4-6 (volume bounded by injector face I-I and throat plane II-II) | SP-125 §4 | yes |
| Contraction ratio `epsilon_c` | 1.3-2.5 turbopump / high-Pc; 2-5 pressure-fed low-thrust | SP-125 §4 | yes |
| Convergent half-angle | 20deg-45deg | SP-125 §4; SP-8120 | yes |
| Throat upstream arc radius | 0.5-1.5 x R_t (circular arc) | SP-125 §4; SP-8120 | yes |
| Chamber shape family | spherical / near-spherical / cylindrical (US standard) | SP-125 §4 | yes |
| Throat-to-exit wall criteria | bell/conic wall from immediately upstream of throat | SP-8120 | yes (`19770009165.pdf`) |

**Net:** the necessary primary sources for the chamber + nozzle geometry build
are already local. Current `raosim/chamber_geometry.py` defaults (Ru = 1.5·Rt,
45deg cone, CR 2.5) sit inside these SP-125 ranges, but are generic preliminary
values rather than a propellant-only prescription.

## Recommended additions (with acquisition links)

All NASA SP documents below are **public-domain U.S. Government** works.
Several are scanned image PDFs (no text layer), so automated text-fetch
returns empty — download the file from the citation page.

| Ref | Title / use | Link | Status |
|---|---|---|---|
| **SP-8089** | Liquid Rocket Engine **Injectors** (Mar 1976) — injector-face geometry + interface for the future injector export | https://ntrs.nasa.gov/citations/19760023196 | public domain; **download** |
| **SP-8113** | Liquid Rocket Engine **Combustion Stabilization Devices** (Nov 1974) — baffles, acoustic absorbers; constrains allowable L\*/CR | https://ntrs.nasa.gov/citations/19750020175 | public domain; **download** |
| **SP-194** | Harrje & Reardon, *Liquid Propellant Rocket Combustion Instability* (1972) — the deep reference on chamber acoustics | https://ui.adsabs.harvard.edu/abs/1972NASSP.194.....H/abstract | large monograph; library/ADS |
| Sutton & Biblarz | *Rocket Propulsion Elements*, Ch. 8 (Thrust Chambers) — cleanest pedagogical treatment of chamber sizing | (copyrighted textbook — acquire separately) | not auto-downloadable |

## Parametric / experimental (DOE) anchors

Empirical data points to validate the geometry + L\*/CR sizing against:

- *Factors Affecting Characteristic Length of the Combustion Chamber of LPREs* —
  https://www.researchgate.net/publication/339188720
- *Characteristic Lengths of LPREs and the Influence of Chemical Reactions* —
  https://www.researchgate.net/publication/356760210
- EUCASS, *Comparison of Single-Element Rocket Combustion* —
  https://www.eucass.eu/component/docindexer/?task=download&id=3679
  (round chamber: 12 mm diameter, 7.6 mm throat, contraction ratio 2.5,
  and L\* = 0.749 m; the paper compares round and rectangular GOX/GCH4
  chambers while controlling contraction ratio, characteristic length, and
  hydraulic diameter)
- UC Irvine *Thrust Chamber Design* project (worked L\*/CR/convergent-angle
  example) —
  https://projects.eng.uci.edu/sites/default/files/Thrust%20Chamber%20Midterm%20Presentation.pdf

## Acquisition checklist

- [ ] SP-8089 injectors -> `propulsion_texts/19760023196.pdf`
- [ ] SP-8113 stabilization devices -> `propulsion_texts/19750020175.pdf`
- [ ] SP-194 combustion instability (if pursuing stability work) -> `propulsion_texts/`
- [x] SP-125 chamber sizing — already `propulsion_texts/19710019929.pdf`
- [x] SP-8120 nozzle wall criteria — already `propulsion_texts/19770009165.pdf`

After the PDFs land in `propulsion_texts/`, fold them into the corpus audit in
`docs/thermofluid_literature_provenance.md` (re-run the SHA-256 dedup + page
count).

## Retrieval note

`web_fetch` in this environment returns extracted page text only (not binary
files) and these SP docs are scanned image PDFs, so the fetch came back empty;
shell download utilities are restricted here. The PDFs therefore need a manual
click from the citation pages above. Drop them into `propulsion_texts/` and they
can be parsed/integrated like the rest of the corpus.
