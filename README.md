# CVD65

Analysis code for cardiovascular disease burden among adults aged 65 years and older in 38 OECD countries using GBD 2023 estimates, 1990–2023, with rate extrapolations to 2050.

## Contents

The five scripts in `Result/V1/09_Code_QC/` perform statistical analysis, figure generation, tables and quality checks, Word table export, and independent pooling validation, in that order. `run_all.sh` runs all five stages and stops if a stage fails.

## Setup and execution

Python 3.10.9 was used in the study. Install dependencies in a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Result/V1/09_Code_QC/requirements.txt
bash Result/V1/09_Code_QC/run_all.sh
```

Supply the input files described in [data/README.md](data/README.md) before running. Keep the directory layout intact: scripts locate `data/` relative to their own paths. Generated files are written within `Result/V1/` and the code directory's `analysis_outputs/` and `qc/` folders.

The manuscript records pandas 2.3.3, NumPy 1.26.4, SciPy 1.15.3, statsmodels 0.14.6, scikit-learn 1.2.1 and Matplotlib 3.7.0. The supplied requirements file lists package names; it is not a complete environment lockfile.

## Methods and numbering

The pipeline includes pooled age-specific and age-standardized rates, uncertainty propagation, segmented trends, joint trajectory clustering, Das Gupta decomposition, descriptive SDI smoothing, oldest-old and disability measures, and exponential smoothing extrapolation. UI, regression CI and forecast PI have different meanings. Decomposition and ecological associations are descriptive.

Generated figures and tables retain the original analysis numbering, which differs from the manuscript's six-main-figure arrangement. Generated narrative summaries are analysis outputs, not the revised manuscript.

## Release scope and verification

This is a code-only release. Raw inputs, generated results, manuscript files and internal work logs are excluded. The analysis has not been rerun during repository preparation. All Python scripts were syntax-checked, and the statistical analysis script's AST was checked against the existing study script. The historical pre-UI-update comparison in the pooling checker is optional when its archived input is absent; the raw-data pooling checks remain in place.

No external data access credentials are required by the scripts. Data access and reuse remain subject to source providers' terms.
