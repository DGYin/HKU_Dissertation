# Dissertation LaTeX Draft

This folder is an Overleaf-friendly, template-free dissertation draft.

## Compile

Upload `docs/dissertation/` to Overleaf and compile `main.tex`.

Recommended local command if LaTeX is installed:

```bash
latexmk -pdf main.tex
```

The current structure uses standard LaTeX packages and BibTeX with `plainnat`.

## Structure

- `main.tex`: entry point
- `preamble.tex`: packages and macros
- `frontmatter/`: title page, abstract, acknowledgements
- `chapters/`: integrative dissertation chapters
- `papers/`: paper-based manuscript chapters
- `appendices/`: software/configuration appendices
- `references.bib`: starter bibliography

## Migration Notes

When an official HKU or department template is available, migrate content chapter by chapter. Keep the `papers/` folder structure so each manuscript can also be exported independently later.
