#!/usr/bin/env bash
# Rebuild all V1 outputs from the local GBD 2023 data. Stops at the first failure, including any failed QC check.
set -euo pipefail
cd "$(dirname "$0")"
python3 00_run_analysis.py
python3 01_make_figures.py
python3 02_build_tables_text_qc.py
python3 03_build_tables_docx.py
python3 04_check_pooling_from_raw.py
