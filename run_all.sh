#!/usr/bin/env bash
# Reproduce every result, figure, table and the PDF. Run from the project root.
set -euo pipefail
python -m unittest discover -s tests -v          # 17 validation tests
# python src/tune_hyperparameters.py              # OPTIONAL: re-selects hyper-parameters on development seeds and rewrites config.json
python src/run_experiments.py                    # 20 evaluation seeds x 4 scenarios x 4 controllers (+ passive predictors)
python src/analyze_results.py                    # summaries and paired tests
python src/make_figures.py                       # figures/*.pdf, *.png
python src/make_tables.py                        # paper/tables/*.tex and paper/results_macros.tex
cp figures/*.pdf paper/
(cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
echo "Done. See paper/main.pdf and results/."
