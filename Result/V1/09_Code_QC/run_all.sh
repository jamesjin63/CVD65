#!/usr/bin/env bash
# Run the analysis, figure exports, tables and quality checks.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PYINPUT'
from pathlib import Path
import sys

data = Path.cwd().parents[2] / "data"
required = ["pop.csv", "sdi.csv", "gdp.csv", "df_world2.geojson"]
missing = [name for name in required if not (data / name).is_file()]
if not any(data.glob("*/*.parquet")):
    missing.append("<cause-directory>/*.parquet")
if missing:
    sys.exit("Missing input files in data/: " + ", ".join(missing) + ". See data/README.md.")
PYINPUT
python3 00_run_analysis.py
python3 01_make_figures.py
python3 02_build_tables_text_qc.py
python3 03_build_tables_docx.py
python3 04_check_pooling_from_raw.py
