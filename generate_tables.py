#!/usr/bin/env python3
"""Generate cardiovascular disease tables from the analysis results."""

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

CODE = Path(__file__).resolve().parent
OUTPUT = CODE / "results"
AO = OUTPUT / "analysis_outputs"
T, D = AO / "tables", AO / "figure_data"
TAB, TAB_S = OUTPUT / "tables", OUTPUT / "supplementary_tables"
for d in (TAB, TAB_S):
    d.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["Incidence", "Prevalence", "Deaths", "DALYs"]
MINUS = "−"


def rd(folder, name):
    return pd.read_csv(folder / name, low_memory=False)






def write(df, path, *a, **k):
    df.to_csv(path, index=False, encoding="utf-8-sig", *a, **k)


# ----------------------------------------------------------------------------- inputs
rate_fmt = rd(T, "Table1_OECD65plus_CVD_rates_AAPC.csv")
num_fmt = rd(T, "Table1_OECD65plus_CVD_numbers_AAPC.csv")
rate_n = rd(T, "Table1_OECD65plus_CVD_rates_AAPC_numeric.csv")
num_n = rd(T, "Table1_OECD65plus_CVD_numbers_AAPC_numeric.csv")
dec = rd(T, "Table3E_Das_Gupta_Decomposition_Incidence_DALYs.csv")
stats = rd(D, "Figure2_AAPC_APC_data.csv")
clusters = rd(D, "Figure4_trajectory_clusters_incidence_DALYs.csv")
val8 = rd(D, "Figure8_ETS_holdout_validation.csv")
sel8 = rd(D, "Figure8_ETS_model_selection.csv")
fc = rd(D, "Figure9_10_country_ETS_forecast_data.csv")
val = rd(D, "Figure9_10_country_ETS_holdout_validation.csv")
sel = rd(D, "Figure9_10_country_ETS_model_selection.csv")




# ----------------------------------------------------------------------------- main tables
def clean_interval(s):
    """'5,168.1 (5,059.7–5,346.7)' -> '5,168.1 (5,059.7 to 5,346.7)'; '%' dropped from AAPC cells."""
    s = str(s).replace("%", "")
    m = re.match(r"^(\S+) \((.+?)–(.+)\)$", s)
    return f"{m.group(1)} ({m.group(2)} to {m.group(3)})" if m else s


def main_table(frame, kind):
    rows = []
    cols = list(frame.columns)
    for rec in frame.to_dict("records"):
        if rec["section"] == "OECD overall":
            loc = "OECD, ages 65+"
        elif rec["section"] == "OECD age groups":
            loc = f"OECD, ages {rec['age_group'].replace('-', '–')}"
        else:
            loc = rec["location_name"]
        row = {"Location": loc}
        for o in OUTCOMES:
            own = [c for c in cols if c.startswith(o + "_")]
            row[f"{o}: 1990"] = clean_interval(rec[[c for c in own if "_1990_" in c and "AAPC" not in c][0]])
            row[f"{o}: 2023"] = clean_interval(rec[[c for c in own if "_2023_" in c and "AAPC" not in c][0]])
            row[f"{o}: AAPC, %"] = clean_interval(rec[[c for c in own if "AAPC" in c][0]])
        rows.append(row)
    return pd.DataFrame(rows)


NAME_T1, NAME_T1N = "Table1_rates_AAPC_OECD65plus_1990_2023", "Table1_Number_counts_AAPC_OECD65plus_1990_2023"
table1, table1n = main_table(rate_fmt, "rate"), main_table(num_fmt, "number")
write(table1, TAB / f"{NAME_T1}.csv")
write(table1n, TAB / f"{NAME_T1N}.csv")
TITLES = {
    NAME_T1: ("Table 1 | Rates of cardiovascular disease incidence, prevalence, death and DALYs among adults aged 65 years and older in OECD countries, 1990 and 2023, and average annual percent change, 1990–2023",
              "Rates are per 100,000. The OECD row and country rows are age-standardized rates for ages 65 years and older (GBD 2023 world standard renormalized within ages 65+); OECD age-group rows are age-specific rates. OECD values pool numbers and population across the 38 countries within each age group before standardization and are not country averages. Parentheses for 1990 and 2023 give 95% uncertainty intervals propagated from published GBD intervals by 1,000 Monte Carlo draws (split-normal; one shared deviate per country across ages, sexes and years; independent between countries); they are not official GBD joint uncertainty intervals. AAPC, average annual percent change from a BIC-selected continuous segmented log-linear model (at most three joinpoints, segments of at least five years), with regression-based 95% confidence intervals. DALYs, disability-adjusted life years."),
    NAME_T1N: ("Table 1 (Number companion) | Numbers of cardiovascular disease incident cases, prevalent cases, deaths and DALYs among adults aged 65 years and older in OECD countries, 1990 and 2023, and average annual percent change, 1990–2023",
               "Values are absolute numbers at ages 65 years and older (OECD and country rows) or in the age group shown. Intervals and AAPC are defined as in Table 1. Number AAPC is not interchangeable with rate AAPC: numbers also reflect population growth and ageing. DALYs, disability-adjusted life years."),
}
write(pd.DataFrame([{"file": k, "title": v[0], "note": v[1]} for k, v in TITLES.items()]), TAB / "table_titles_notes.csv")
for name, frame in ((NAME_T1, table1), (NAME_T1N, table1n)):
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in frame.values)
    (TAB / f"{name}.md").write_text(f"# {TITLES[name][0]}\n\n{head}{body}\n{TITLES[name][1]}\n", encoding="utf-8")


# ----------------------------------------------------------------------------- supplementary tables
def ets_country_table():
    selected = sel.query("selected == True")[["iso3", "measure_name", "model", "aicc"]]
    last_obs = fc.query("year == 2023 and series == 'Observed'")[["iso3", "measure_name", "observed_asr"]].rename(columns={"observed_asr": "observed_asr_2023"})
    y50 = fc.query("year == 2050")[["iso3", "measure_name", "forecast_asr", "pi_lower_raw", "pi_lower", "pi_upper", "pi_lower_clipped_at_zero"]].rename(columns={"forecast_asr": "forecast_asr_2050", "pi_lower": "pi_lower_2050", "pi_upper": "pi_upper_2050", "pi_lower_raw": "pi_lower_2050_raw", "pi_lower_clipped_at_zero": "pi_lower_2050_clipped_at_zero"})
    out = selected.merge(last_obs).merge(y50).merge(val[["iso3", "measure_name", "mae", "rmse", "mape_percent"]])
    out["relative_change_2050_vs_2023_percent"] = 100 * (out.forecast_asr_2050 / out.observed_asr_2023 - 1)
    return out.sort_values(["measure_name", "iso3"])


context = rd(T, "Table1_Country_Context_2023.csv")
SUPP = [
    ("SuppTable1_data_availability_QC", "Data availability and completeness of the five GBD 2023 cause files", rd(T, "Data_Availability_and_QC.csv"),
     "One row per cause. Rows count country × year × sex × age × measure × metric combinations; missing values, duplicate keys and point estimates outside their bounds were all zero. Hypertensive heart disease has no incidence estimates locally."),
    ("SuppTable2_OECD38_country_context_2023", "OECD-38 countries: identifiers and 2023 demographic and development context", context,
     "SDI, Socio-demographic Index. Fixed cohort of the 38 current OECD members, applied retrospectively to all years; GDP is contextual and was not used in inferential analyses."),
    ("SuppTable3_age_standard_weights_65plus", "GBD 2023 world standard weights renormalized within ages 65 years and older", rd(T, "GBD2023_age_weights_65plus.csv"),
     "The seven weights sum to 1. The 65+ age-standardized rate is the weighted sum of seven age-specific rates."),
    ("SuppTable4_OECD_summary_selected_years", "OECD aggregate deaths and DALYs for selected years", rd(T, "Table2B_OECD_Summary_Selected_Years.csv"),
     "Number and age-standardized rate with Monte Carlo-propagated 95% uncertainty intervals, plus the median and quartiles of the 38 country rates. Country quartiles describe between-country dispersion."),
    ("SuppTable5_OECD_APC_AAPC_segments", "Segmented trend results for OECD incidence and DALY rates, 1990–2023 (Fig. 2)", stats,
     "APC, annual percent change; AAPC, average annual percent change, both with regression-based 95% confidence intervals. Joinpoint-selection uncertainty is not included."),
    ("SuppTable6_Table1_rates_numeric", "Numeric long form of Table 1 (rates)", rate_n, "One row per entity and outcome (46 × 4); suitable for re-analysis."),
    ("SuppTable7_Table1_numbers_numeric", "Numeric long form of the Table 1 Number companion", num_n, "One row per entity and outcome (46 × 4)."),
    ("SuppTable8_age_sex_cause_2023", "OECD number, population and age-specific rate by age, sex and cause, 2023", rd(T, "TableS1_Age_Sex_Subtype_Rates.csv"),
     "Pooled OECD-38. Cause levels are not mutually exhaustive; total CVD includes CVD causes not available locally."),
    ("SuppTable9_trajectory_clusters", "Joint incidence–DALY trajectory clusters (Fig. 4)", clusters,
     "Index = 2023 age-standardized rate relative to 1990 (= 100). Clusters are ordered by overall speed of decline and are descriptive, not performance grades."),
    ("SuppTable10_DasGupta_decomposition", "Three-factor Das Gupta decomposition of change in incident cases and DALYs, 1990–2023 (Fig. 5)", dec,
     "Contributions are averaged over all six replacement orders and sum exactly to the total change. Accounting identity, not causal effects."),
    ("SuppTable11_oldest_old_disability", "Oldest-old concentration and YLD/DALY indices by cause, 1990–2023 (Fig. 7)", rd(T, "TableS2_OldestOld_and_Disability_Transition.csv"),
     "Based on OECD-38 summed numbers without age standardization. Ratios are point estimates; draw-level covariance was unavailable."),
    ("SuppTable12_OECD_ETS_selection_validation", "OECD ETS candidate models and hold-out validation (Fig. 8)", sel8.merge(val8[["measure", "model", "training_years", "validation_years", "mae", "rmse", "mape_percent", "prediction_interval_coverage_percent"]], how="left", on=["measure", "model"]),
     "Model selected by minimum AICc among 10 non-seasonal candidates. Hold-out columns are filled for the selected model only. Nominal 95% interval coverage in the hold-out period was below nominal."),
    ("SuppTable13_country_ETS_selected_2050", "Country ETS models, 2050 forecasts and hold-out accuracy (Figs. 9–10)", ets_country_table(),
     "One row per country and outcome. Negative lower limits are retained in the raw column and truncated at zero in the plotted column. Prediction intervals are conditional on the model and are not GBD uncertainty intervals. Not for country ranking."),
    ("SuppTable14_age_threshold_sensitivity", "Sensitivity of total CVD deaths and DALYs to the older-age threshold", rd(T, "TableS3_Age_Threshold_Sensitivity.csv"),
     "EAPC, estimated annual percent change from a log-linear model, reported for this sensitivity analysis only."),
    ("SuppTable15_OECD_propagated_UI_annual", "Monte Carlo-propagated 95% uncertainty intervals for OECD total CVD deaths and DALYs by year", rd(T, "Approximate_UI_OECD_Total_CVD.csv"),
     "Method as in Table 1 and Supplementary Methods 4; not official GBD joint uncertainty intervals."),
]
for name, _, frame, _note in SUPP:
    write(frame, TAB_S / f"{name}.csv")
with pd.ExcelWriter(TAB_S / "All_Supplementary_Tables_OECD65plus_CVD.xlsx", engine="openpyxl") as xl:
    pd.DataFrame([{"Sheet": f"S{i}", "Supplementary Table": f"Supplementary Table {i}", "Title": t, "Note": n} for i, (_, t, _, n) in enumerate(SUPP, 1)]).to_excel(xl, sheet_name="Contents", index=False)
    for i, (_, _, frame, _) in enumerate(SUPP, 1):
        frame.to_excel(xl, sheet_name=f"S{i}", index=False)
