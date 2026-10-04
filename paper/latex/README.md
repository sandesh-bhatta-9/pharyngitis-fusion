# Manuscript (LaTeX, Elsevier `elsarticle`)

`main.tex` is the manuscript for *Computers in Biology and Medicine*; `references.bib` holds the references and
`figures/` the figures (copied from the repository's `figures/` folder).

## Build the PDF

```bash
cd paper/latex
tectonic -X compile main.tex        # brew install tectonic; downloads LaTeX packages on first use
```

or with a standard TeX installation: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

In VS Code, the *LaTeX Workshop* extension shows the PDF next to the source.

## Overleaf

Upload `paper/overleaf_upload.zip` (New Project → Upload Project). Overleaf includes the `elsarticle` class.

## Before submission

- Add `figures/fig1_study_design.pdf` (or `.png`, then change the extension in `main.tex`). Until then the PDF shows a
  "Missing file" box in place of Figure 1.
- Check the references marked `VERIFY` in `references.bib`.
- Elsevier asks for the highlights as a separate file at submission: copy the five `\item` lines from `main.tex`.
