# Input data (not included)

Obtain GBD 2023 estimates from https://vizhub.healthdata.org/gbd-results/ and supporting files from https://ghdx.healthdata.org/ under their applicable terms. This repository does not redistribute the input datasets.

The pipeline expects locally prepared files, not unmodified download archives:

- `data/<cause-directory>/*.parquet`: columns `measure`, `location`, `sex`, `age`, `cause`, `metric`, `year`, `val`, `lower`, `upper`. Use the fixed 38-country OECD cohort, years 1990–2023, sexes 1/2/3, ages 18/19/20/30/31/32/235, and cause IDs 491/493/495/496/498. Number is metric 1; rate is metric 3. HHD incidence is unavailable.
- `pop.csv`: GBD population estimates including `location_id`, `location_name`, `sex_id`, `age_id`, `year`, `val`.
- `sdi.csv`: national SDI inputs with `location_id`, `year`, `sdi`; see the filtering in `00_run_analysis.py`.
- `gdp.csv`: national economic context with `year`, `iso3c`, `NY.GDP.PCAP.PP.CD`, `NY.GDP.MKTP.PP.CD`.
- `df_world2.geojson`: country map with feature properties including `location_id`, `brk_a3`, `name_long` and country geometry.

See the script's loading, crosswalk and context functions for exact input transformations. Preparation of these local input files is not included in this release. The 38-country location IDs must match the map and population crosswalk.
