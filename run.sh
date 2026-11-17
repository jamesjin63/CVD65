#!/usr/bin/env bash
# Run the analysis and generate figures and tables.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PYINPUT'
from pathlib import Path
import sys

data = Path.cwd() / "data"
required = ["pop.csv", "sdi.csv", "gdp.csv", "df_world2.geojson"]
missing = [name for name in required if not (data / name).is_file()]
if not any(data.glob("*/*.parquet")):
    missing.append("<cause-directory>/*.parquet")
if missing:
    sys.exit("Missing input files in data/: " + ", ".join(missing) + ". See data/README.md.")
PYINPUT
python3 analysis.py
python3 generate_figures.py
python3 generate_tables.py
