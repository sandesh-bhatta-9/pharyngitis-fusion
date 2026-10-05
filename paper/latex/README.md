# Manuscript (LaTeX, Elsevier `elsarticle`)

`main.tex` is the manuscript for *Computer Methods and Programs in Biomedicine*; `references.bib` holds the references and
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

## Submission files

| File | Upload as |
| --- | --- |
| `main.tex`, `references.bib`, `figures/` (or `main.pdf`) | Manuscript (LaTeX source) |
| `highlights.docx` (`highlights.txt` is the same text) | Highlights |
| `figures/graphical_abstract.png` | Graphical abstract |
| `cover_letter.pdf` | Cover letter |
| `tripod_ai_checklist.pdf` | Supplementary file (reporting checklist) |

Figure 1 is drawn in TikZ inside `main.tex`; adding `figures/fig1_study_design.pdf` replaces it. All references were
checked against Crossref, Europe PMC, arXiv, the NeurIPS proceedings or JSTOR. Page numbers in the TRIPOD+AI checklist
refer to the current `main.pdf`: update them if the text changes.

The GitHub repository is public. To add a Zenodo DOI later (e.g. at revision), enable the repository at
zenodo.org (GitHub login), create a GitHub release, and add the DOI to the Code availability section.
