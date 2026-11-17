# CVD65

Analysis of cardiovascular disease burden among adults aged 65 years and older in 38 OECD countries using GBD 2023 estimates, 1990–2023, with statistical rate extrapolations to 2050.

## Analysis

The pipeline estimates pooled age-specific and age-standardized rates, propagates uncertainty, fits segmented trends, clusters joint trajectories, decomposes changes in burden, describes SDI associations and oldest-old burden, and fits exponential smoothing models.

OECD rates pool numbers and population within age groups before age standardization. The GBD world-standard weights are renormalized within ages 65 years and older. Uncertainty intervals use 1,000 split-normal simulations with a shared deviate across ages, sexes and years within each country and outcome, independently between countries. These intervals are distinct from official GBD joint intervals, regression confidence intervals and model-based prediction intervals.

Clustering, decomposition and ecological associations are descriptive. Forecasts extrapolate historical rates without demographic, policy or risk-factor scenarios; hold-out prediction interval coverage is below nominal.

## Inputs and execution

Python 3.10.9 was used for the study. The recorded analysis package versions are pandas 2.3.3, NumPy 1.26.4, SciPy 1.15.3, statsmodels 0.14.6, scikit-learn 1.2.1 and Matplotlib 3.7.0. `requirements.txt` lists dependencies rather than a locked environment.

Input sources, schemas and preparation requirements are described in [data/README.md](data/README.md). Raw data are distributed by the source providers and are not included here.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Result/V1/09_Code_QC/requirements.txt
bash Result/V1/09_Code_QC/run_all.sh
```

Paths are resolved relative to the scripts. Inputs belong in `data/`; outputs are written to `Result/V1/` and `Result/V1/09_Code_QC/analysis_outputs/` and `qc/`. The entry point checks input availability and stops if a script or quality check fails.

## Scripts and outputs

| Script | Function |
|---|---|
| `00_run_analysis.py` | Statistical analysis, intermediate tables and figure data |
| `01_make_figures.py` | Main and supplementary figures, CSV and XLSX source data |
| `02_build_tables_text_qc.py` | Tables, legends, summaries, supplementary methods and quality checks |
| `03_build_tables_docx.py` | Word table export and cell-level comparison with CSV sources |
| `04_check_pooling_from_raw.py` | Independent validation of pooled estimates from raw inputs |

Word export requires `python-docx`; the statistical analysis can run independently of that export. [Output correspondence](OUTPUTS.md) relates generated numbering to the paper. [Validation](VALIDATION.md) describes executed checks and their scope.
