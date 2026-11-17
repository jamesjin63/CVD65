# CVD65

Code for the analysis of cardiovascular disease burden among adults aged 65 years and older in 38 OECD countries, using GBD 2023 estimates for 1990–2023 and rate extrapolations to 2050.

## Data

Input files are stored in `data/`. Sources, variables and preparation requirements are described in [data/README.md](data/README.md). Raw data are not included in this repository.

## Analysis

The analysis includes pooled age-specific and age-standardized rates, segmented trends, joint trajectory clustering, Das Gupta decomposition, SDI associations, oldest-old burden, age-threshold sensitivity and exponential smoothing forecasts.

OECD rates are calculated by pooling numbers and population within age groups, then applying the GBD world-standard weights renormalized within ages 65 years and older. Uncertainty intervals use 1,000 split-normal simulations, with a shared deviate across ages, sexes and years within each country and outcome and independence between countries. These differ from official GBD joint intervals, regression confidence intervals and forecast prediction intervals.

Clustering and ecological associations are descriptive, and decomposition is an accounting analysis. Forecasts extrapolate historical rates without demographic, policy or risk-factor scenarios. Hold-out prediction interval coverage is below nominal.

## Requirements

The study used Python 3.10.9, pandas 2.3.3, NumPy 1.26.4, SciPy 1.15.3, statsmodels 0.14.6, scikit-learn 1.2.1 and Matplotlib 3.7.0. Additional dependencies are listed in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
bash run.sh
```

The scripts can also be run separately:

```bash
python3 analysis.py
python3 generate_figures.py
python3 generate_tables.py
```

| File | Description |
|---|---|
| `analysis.py` | Statistical analysis |
| `generate_figures.py` | Figures and CSV/XLSX source data |
| `generate_tables.py` | Main and supplementary tables |

Analysis outputs are saved in `results/analysis_outputs/`. Figures, source data and tables are saved in `results/figures/`, `results/supplementary_figures/`, `results/source_data/`, `results/tables/` and `results/supplementary_tables/`. The relation between generated numbering and paper numbering is listed in [OUTPUTS.md](OUTPUTS.md).
