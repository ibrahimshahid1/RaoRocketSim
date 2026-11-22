# LREKit component data and research plan

Editable LaTeX source for *From Engine Design to Experimental Feedback*,
dated 3 November 2026.

## Files

- main.tex: document setup, typography, and main compilation target.
- plan.tex: complete report text, tables, equations, diagram, and references.

## Build

With a standard TeX Live installation:

~~~sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

Alternatively, run pdflatex main.tex twice to resolve references and the
table of contents. No external images or bibliography processor are required.

The report describes proposed development and research. Its CLI examples
are proposed syntax, and its numerical study endpoints are proposed targets.
