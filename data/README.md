# Input data (not included)

Obtain GBD 2023 estimates from https://vizhub.healthdata.org/gbd-results/ and supporting files from https://ghdx.healthdata.org/ under their applicable terms. This repository does not redistribute the input datasets.

The pipeline expects locally prepared files, not unmodified download archives:

- `data/<cause-directory>/*.parquet`: columns `measure`, `location`, `sex`, `age`, `cause`, `metric`, `year`, `val`, `lower`, `upper`. Use the fixed 38-country OECD cohort, years 1990–2023, sexes 1/2/3, ages 18/19/20/30/31/32/235, and cause IDs 491/493/495/496/498. Number is metric 1; rate is metric 3. HHD incidence is unavailable.
- `pop.csv`: GBD population estimates including `location_id`, `location_name`, `sex_id`, `age_id`, `year`, `val`.
- `sdi.csv`: national SDI inputs with `location_id`, `year`, `sdi`; see the filtering in `00_run_analysis.py`.
- `gdp.csv`: national economic context with `year`, `iso3c`, `NY.GDP.PCAP.PP.CD`, `NY.GDP.MKTP.PP.CD`.
- `df_world2.geojson`: country map with feature properties including `location_id`, `brk_a3`, `name_long` and country geometry.

See the script's loading, crosswalk and context functions for exact input transformations. Preparation of these local input files is not included in this release. The 38-country location IDs must match the map and population crosswalk.

## Input preparation

GBD identifier fields in the parquet files are numeric: `measure`, `location`, `sex`, `age`, `cause` and `metric`. For downloads that use `*_id` column names, rename the corresponding identifiers to these names; retain numeric identifiers rather than category labels. Filter to the cohort and time, age, sex, cause and metric ranges above, combine download parts, and write parquet files under a cause directory. Each cause–measure–location–sex–age–metric–year key occurs once. The outcomes are deaths (1), DALYs (2), YLDs (3), YLLs (4), prevalence (5) and incidence (6), where available.

Population data retain the `*_id` column names. The context table uses all-age (`age_id = 22`), both-sex (`sex_id = 3`) population for 2023. Age-specific analysis denominators are reconstructed from matching GBD numbers and rates. SDI uses one record per location and year, with 1990–2023 observations and 2023 context values. GDP uses 2023 PPP values from the World Bank indicators `NY.GDP.PCAP.PP.CD` and `NY.GDP.MKTP.PP.CD`; `iso3c` matches map ISO3 codes.

The map is a GeoJSON FeatureCollection. Country properties `location_id`, `brk_a3` and `name_long` provide the GBD identifier, ISO3 code and display name. Geometry supplies the map boundaries. The fixed cohort is:

```text
AUS AUT BEL CAN CHE CHL COL CRI CZE DEU DNK ESP EST FIN FRA GBR
GRC HUN IRL ISL ISR ITA JPN KOR LTU LUX LVA MEX NLD NOR NZL POL
PRT SVK SVN SWE TUR USA
```

The scripts consume these prepared files. Downloading, category mapping and map crosswalk preparation are external to the pipeline. Source data access and reuse follow the providers' terms.
