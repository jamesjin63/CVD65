# Validation

## Environment

Validation used Python 3.10.9, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, statsmodels 0.14.6, scikit-learn 1.2.1, Matplotlib 3.7.0, pyarrow 11.0.0, openpyxl 3.1.5 and python-docx 1.2.0.

## Executed checks

- All five Python modules compiled, and `bash -n Result/V1/09_Code_QC/run_all.sh` passed.
- Structural comparison of the Python modules confirmed unchanged statistical computations and numeric constants. Documentation strings and report labels were compared separately from calculations.
- Figure generation, table/text generation, Word export and independent raw-data pooling validation ran in isolated copies using existing analysis outputs and read-only input data.
- All 333 analysis/output quality checks and all 88 independent pooling checks passed.
- The two Word tables each matched all 611 source CSV cells, with zero differences.
- Output comparison covered 38 CSV files and 114,794 numeric cells: absolute tolerance 0, relative tolerance 0, maximum difference 0, failures 0. Quality-check summaries were aligned by check name.
- All 155,756 cells across 18 XLSX files matched, including sheet names, table notes and source-data panels.
- All 17 PNG figures had identical SHA-256 hashes. Text was extracted from all 17 PDF figures for content inspection.
- The entry point was executed without inputs and returned the expected English missing-input message before analysis.

## Scope

The complete statistical analysis in `00_run_analysis.py` was not rerun. Numerical comparison of generated outputs used the existing study estimates; independent pooling checks recomputed quantities from the raw parquet inputs. The validation does not establish compatibility with package versions other than the environment listed above. Word export was checked for content; page rendering was not repeated.
