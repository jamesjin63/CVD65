#!/usr/bin/env python3
"""OECD-38 age 65+ CVD burden, GBD 2023: tables, legends, summary, supplement and QC (V1, step 3 of 4).

Run after 01_make_figures.py. Every number in the legends and summary is read from
./analysis_outputs; none is typed by hand. Hard-coded values appear only in design_lock(),
where they are the numbers stated in the design document and are compared with the results.
"""

import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
V1 = CODE.parent
AO = CODE / "analysis_outputs"
T, D = AO / "tables", AO / "figure_data"
FIG, FIG_S = V1 / "02_主图_Figures", V1 / "05_附图_Supplementary_Figures"
TAB, TAB_S, SD, QC = V1 / "04_主表_Tables", V1 / "06_附表_Supplementary_Tables", V1 / "07_源数据_SourceData", CODE / "qc"
for d in (TAB, TAB_S, QC):
    d.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["Incidence", "Prevalence", "Deaths", "DALYs"]
AGE_ORDER = ["65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+"]
MINUS = "−"


def rd(folder, name):
    return pd.read_csv(folder / name, low_memory=False)


def num(x, k=0):
    return f"{x:,.{k}f}".replace("-", MINUS)


def sgn(x, k=1):
    return f"{x:+,.{k}f}".replace("-", MINUS)


def write(df, path, *a, **k):
    df.to_csv(path, index=False, encoding="utf-8-sig", *a, **k)


# ----------------------------------------------------------------------------- inputs
rate_fmt = rd(T, "Table1_OECD65plus_CVD_rates_AAPC.csv")
num_fmt = rd(T, "Table1_OECD65plus_CVD_numbers_AAPC.csv")
rate_n = rd(T, "Table1_OECD65plus_CVD_rates_AAPC_numeric.csv")
num_n = rd(T, "Table1_OECD65plus_CVD_numbers_AAPC_numeric.csv")
diag = json.loads((AO / "diagnostics.json").read_text(encoding="utf-8"))
dec = rd(T, "Table3E_Das_Gupta_Decomposition_Incidence_DALYs.csv")
stats = rd(D, "Figure2_AAPC_APC_data.csv")
clusters = rd(D, "Figure4_trajectory_clusters_incidence_DALYs.csv")
series = rd(D, "Figure4D_cluster_series_data.csv")
sdi = rd(D, "Figure6_incidence_DALY_SDI_data.csv")
conc = rd(D, "Figure7_oldest_old_concentration_data.csv")
dis = rd(D, "Figure7_disability_transition_data.csv")
fc8 = rd(D, "Figure8_ETS_forecast_data.csv")
val8 = rd(D, "Figure8_ETS_holdout_validation.csv")
sel8 = rd(D, "Figure8_ETS_model_selection.csv")
fc = rd(D, "Figure9_10_country_ETS_forecast_data.csv")
val = rd(D, "Figure9_10_country_ETS_holdout_validation.csv")
sel = rd(D, "Figure9_10_country_ETS_model_selection.csv")
fig1 = rd(D, "Figure1_four_map_data.csv")


def t1(frame, outcome, col, section="OECD overall"):
    return float(frame.query("section == @section and measure_name == @outcome")[col].iloc[0])


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


# ----------------------------------------------------------------------------- derived key numbers
def dg(scope_iso, m, col):
    return float(dec.query("iso3 == @scope_iso and measure == @m")[col].iloc[0])


def apc_text(m):
    block = stats.query("measure == @m")
    a = block.query("statistic == 'AAPC'").iloc[0]
    segs = "; ".join(f"{int(s.start_year)}–{int(s.end_year)}, {sgn(s.estimate, 2)}%" for s in block.query("statistic == 'APC'").sort_values("start_year").itertuples())
    return a, segs


def cluster_index(c, m):
    return float(series.query("cluster == @c and measure == @m and year == 2023").mean_index.iloc[0])


def ooci(c, m, y):
    return float(conc.query("cause == @c and measure == @m and year == @y").oldest_old_concentration_percent.iloc[0])


def dti(c, y):
    return float(dis.query("cause == @c and year == @y").yld_to_daly_percent.iloc[0])


def ets8(m):
    block = fc8.query("measure == @m")
    obs = float(block.query("year == 2023").observed_asr.iloc[0])
    r = block.query("year == 2050").iloc[0]
    v = val8.query("measure == @m").iloc[0]
    return obs, r, v


def country_summary(m):
    b = ets_country_table().query("measure_name == @m")
    n = fc.query("measure_name == @m and series == 'Forecast'")
    clipped = n[n.pi_lower_clipped_at_zero.astype(bool)]
    return {"lower": int((b.forecast_asr_2050 < b.observed_asr_2023).sum()), "higher": int((b.forecast_asr_2050 > b.observed_asr_2023).sum()),
            "median_change": float(b.relative_change_2050_vs_2023_percent.median()), "median_mape": float(b.mape_percent.median()),
            "clip_countries": int(clipped.iso3.nunique()), "clip_rows": int(len(clipped))}


medians = fig1.groupby("measure_name").asr.median()
a6, seg6 = apc_text(6)
a2, seg2 = apc_text(2)
O6, E6, V6 = ets8(6)
O2, E2, V2 = ets8(2)
C6, C2 = country_summary("Incidence"), country_summary("DALYs")
cd4 = diag["figure4_cluster_diagnostics"]
sil = {int(r["k"]): r["silhouette"] for r in cd4["silhouette_by_k"]}

# ----------------------------------------------------------------------------- legends
FILES = {p.stem: p for p in list(FIG.glob("*.pdf")) + list(FIG_S.glob("*.pdf"))}


def fname(key):
    return next(s for s in FILES if s.startswith(key + "_"))


def legend(key, head, body, colours, note=None):
    label = key.replace("SuppFig", "Supplementary Fig. ").replace("Fig", "Figure ")
    txt = f"## {label} | {head}\n*File:* `{fname(key)}.pdf` · *Source data:* `{key}_SourceData.xlsx`\n\n{body}\n"
    if colours:
        txt += f"\n*Colours:* {colours}\n"
    if note:
        txt += f"\n**Note.** {note}\n"
    return txt + "\n---\n\n"


MAIN_LEG = [
    legend("Fig1", "Age-standardized rates of cardiovascular disease at ages 65 years and older in 38 OECD countries, 2023.",
           "**a**, Incidence. **b**, Prevalence. **c**, Deaths. **d**, DALYs. Total CVD, both sexes. Rates per 100,000 are standardized with the GBD 2023 world standard renormalized within ages 65+. The inset enlarges Europe. Each panel has its own scale, so colours are not comparable between panels. Non-OECD areas are light grey. Point estimates; intervals are in Table 1.",
           "one single-hue scale per outcome.",
           f"Country medians were {num(medians['Incidence'])}, {num(medians['Prevalence'])}, {num(medians['Deaths'])} and {num(medians['DALYs'])} per 100,000 for incidence, prevalence, deaths and DALYs."),
    legend("Fig2", "Trends in cardiovascular disease incidence and DALY rates at ages 65 years and older in OECD countries, 1990–2023.",
           "**a**,**b**, OECD aggregate incidence (**a**) and DALY (**b**) rates: points, GBD-derived rates; line, BIC-selected continuous segmented log-linear fit; dashed lines, selected joinpoints; numbers, annual percent change (APC) by segment; AAPC with regression-based 95% CI (segment CIs in Supplementary Table 5). **c**,**d**, Country rates by year, countries in ISO3 order, with outcome-specific colour scales.",
           "blue, incidence; orange, DALYs.",
           f"Incidence: AAPC {sgn(a6.estimate, 2)}% (95% CI {num(a6.lower, 2)} to {num(a6.upper, 2)}); segments {seg6}. DALYs: AAPC {sgn(a2.estimate, 2)}% (95% CI {num(a2.lower, 2)} to {num(a2.upper, 2)}); segments {seg2}."),
    legend("Fig3", "Age, sex and cause structure of cardiovascular disease burden at ages 65 years and older in OECD countries, 2023.",
           "**a**,**b**, Age-specific incidence (**a**) and DALY (**b**) rates per 100,000 by sex. **c**,**d**, Percentage distribution of each sex's 65+ incident cases (**c**) and DALYs (**d**) across age groups. **e**,**f**, Share of total CVD incidence (**e**) and DALYs (**f**) by available cause within each age group, both sexes. Other CVD is total CVD minus the causes shown and is a residual, not a disease entity. HHD has no local incidence estimates. Bands in **a**,**b** are Monte Carlo-propagated 95% UIs (Supplementary Methods 4). No intervals are shown for the proportions in **c**–**f**, because correlations between strata in the original GBD draws are unavailable.",
           "blue, males; vermillion, females; panels **e**,**f** use a fixed cause palette."),
    legend("Fig4", "Joint trajectories of cardiovascular disease incidence and DALY rates at ages 65 years and older, 38 OECD countries, 1990–2023.",
           f"**a**, Incidence. **b**, DALYs. Index, 1990 = 100. Countries were grouped by k-means after PCA on standardized log changes of both outcomes for 1991–2023 (K = 3, 4, 5 tested; silhouette {sil[3]:.3f}, {sil[4]:.3f}, {sil[5]:.3f}; K = 5 selected; {cd4['pca_components_for_90pct_variance']} principal components). Lines, cluster means; bands, between-country interquartile range, not confidence intervals. Numbers at line ends identify clusters. Clusters are numbered from fastest (1) to slowest (5) combined decline and are not performance grades.",
           "one colour per cluster, the same in Fig. 6 and Supplementary Figs. 3–4.",
           f"Bootstrap mean adjusted Rand index {cd4['bootstrap_mean_adjusted_rand_index']:.3f} (2.5th–97.5th percentile {cd4['bootstrap_ari_2_5_percentile']:.3f}–{cd4['bootstrap_ari_97_5_percentile']:.3f}; 500 resamples). 2023 indices (incidence, DALYs): " + "; ".join(f"cluster {c}, {cluster_index(c, 6):.1f}, {cluster_index(c, 2):.1f}" for c in range(1, 6)) + "."),
    legend("Fig5", "Decomposition of the change in cardiovascular disease incident cases and DALYs at ages 65 years and older, 38 OECD countries and OECD aggregate, 1990–2023.",
           "**a**, Incident cases. **b**, DALYs. Bars show the contribution of population growth (total population size), population ageing (share of total population in each 65+ age group) and epidemiological change (age-specific rates) as a percentage of the 1990 burden, from a symmetric three-factor Das Gupta decomposition averaged over all six replacement orders. Diamonds, net change. Countries are sorted by net change within each panel; the OECD aggregate is on top. Contributions sum exactly to net change and are an accounting identity, not causal effects.",
           "grey, population growth; amber, population ageing; blue, epidemiological change.",
           f"OECD aggregate, incidence: population growth {sgn(dg('OECD38', 6, 'population_growth_percent_of_1990'), 2)}%, ageing {sgn(dg('OECD38', 6, 'population_ageing_percent_of_1990'), 2)}%, epidemiological change {sgn(dg('OECD38', 6, 'epidemiological_change_percent_of_1990'), 2)}%, net {sgn(dg('OECD38', 6, 'total_change_percent_of_1990'), 2)}%. DALYs: {sgn(dg('OECD38', 2, 'population_growth_percent_of_1990'), 2)}%, {sgn(dg('OECD38', 2, 'population_ageing_percent_of_1990'), 2)}%, {sgn(dg('OECD38', 2, 'epidemiological_change_percent_of_1990'), 2)}%, net {sgn(dg('OECD38', 2, 'total_change_percent_of_1990'), 2)}%."),
    legend("Fig6", "Relationship of cardiovascular disease incidence and DALY rates at ages 65 years and older with the Socio-demographic Index, 38 OECD countries, 1990–2023.",
           f"**a**, Incidence. **b**, DALYs. Each thin line joins one country's annual values, coloured by its Fig. 4 cluster. The black line is a pooled robust LOWESS of all {len(sdi.query('measure == 6')):,} country-years (span 0.30; three robustness iterations), without confidence band, country or year effects, population weights or confounder adjustment. The curve mixes between-country and within-country variation and does not imply that a change in SDI causes a change in burden.",
           "cluster colours as in Fig. 4; black, pooled LOWESS.",
           f"Observed SDI ranged from {sdi.sdi.min():.3f} to {sdi.sdi.max():.3f}."),
    legend("Fig7", "Concentration of cardiovascular disease burden at ages 80 years and older and the YLD share of DALYs, OECD-38, 1990–2023.",
           "**a**–**e**, Percentage of the 65+ number of deaths, DALYs and YLDs occurring at ages 80+. **f**–**j**, YLDs as a percentage of DALYs at ages 65+. Columns: total CVD, IHD, ischaemic stroke, ICH and HHD. Summed OECD-38 numbers, without age standardization; point estimates only, because draw-level covariance was unavailable. A rising YLD/DALY ratio does not by itself indicate compression of morbidity or treatment benefit.",
           "purple, deaths; amber, DALYs; green, YLDs; black, YLDs/DALYs.",
           f"For total CVD, ages 80+ held {ooci(491, 1, 1990):.2f}% of deaths, {ooci(491, 2, 1990):.2f}% of DALYs and {ooci(491, 3, 1990):.2f}% of YLDs in 1990, and {ooci(491, 1, 2023):.2f}%, {ooci(491, 2, 2023):.2f}% and {ooci(491, 3, 2023):.2f}% in 2023; YLDs/DALYs rose from {dti(491, 1990):.2f}% to {dti(491, 2023):.2f}%."),
    legend("Fig8", "Statistical extrapolation of OECD cardiovascular disease incidence and DALY rates at ages 65 years and older to 2050.",
           "**a**, Incidence. **b**, DALYs. Black, 1990–2023 rates; coloured line, exponential smoothing (ETS) forecast of the age-standardized rate; band, nominal 95% prediction interval from 20,000 simulated paths. The model was selected by AICc among 10 non-seasonal candidates and checked by training on 1990–2013 and validating on 2014–2023. The extrapolation assumes that historical patterns continue and includes no future population, risk factor, policy, diagnostic, treatment or structural-break scenario. It forecasts rates, not numbers.",
           "blue, incidence; orange, DALYs.",
           f"Incidence 2050: {num(E6.forecast_asr)} per 100,000 (95% PI {num(E6.pi_lower)}–{num(E6.pi_upper)}), {sgn(100 * (E6.forecast_asr / O6 - 1))}% from 2023. DALYs 2050: {num(E2.forecast_asr)} (95% PI {num(E2.pi_lower)}–{num(E2.pi_upper)}), {sgn(100 * (E2.forecast_asr / O2 - 1))}%. Hold-out MAPE was {V6.mape_percent:.1f}% and {V2.mape_percent:.1f}%, but nominal 95% PI coverage was only {V6.prediction_interval_coverage_percent:.0f}% and {V2.prediction_interval_coverage_percent:.0f}%, indicating under-calibrated intervals; both 2050 intervals include values above the 2023 rate."),
    legend("Fig9", "Country-specific extrapolation of cardiovascular disease incidence rates at ages 65 years and older to 2050, 38 OECD countries.",
           "Each panel shows one country (ISO3): black, 1990–2023 age-standardized rate; blue, ETS forecast selected separately for that country; band, nominal 95% prediction interval from 10,000 simulated paths. Vertical line, 2023. Y axes differ between countries, so panel height does not allow comparison of absolute levels. Lower limits below zero are truncated at zero for display; raw limits are in the source data.",
           None,
           f"2050 point forecasts were below the 2023 rate in {C6['lower']} countries and above it in {C6['higher']}; median relative change {sgn(C6['median_change'])}%; median hold-out MAPE {C6['median_mape']:.2f}%. Lower limits were truncated in {C6['clip_rows']} country-years across {C6['clip_countries']} countries. Intervals are wide at long horizons, so countries are not ranked."),
    legend("Fig10", "Country-specific extrapolation of cardiovascular disease DALY rates at ages 65 years and older to 2050, 38 OECD countries.",
           "As Fig. 9, for DALY rates; orange, ETS forecast.", None,
           f"2050 point forecasts were below the 2023 rate in {C2['lower']} countries and above it in {C2['higher']}; median relative change {sgn(C2['median_change'])}%; median hold-out MAPE {C2['median_mape']:.2f}%. Lower limits were truncated in {C2['clip_rows']} country-years across {C2['clip_countries']} countries."),
]
SUPP_LEG = [
    legend("SuppFig1", "Trends in cardiovascular disease prevalence and death rates at ages 65 years and older in OECD countries, 1990–2023.",
           "**a**,**b**, OECD aggregate prevalence (**a**) and death (**b**) rates with AAPC (95% CI, regression-based; from Table 1). **c**,**d**, Country rates by year, in ISO3 order.", "green, prevalence; purple, deaths."),
    legend("SuppFig2", "Rates and numbers indexed to 1990, OECD aggregate, ages 65 years and older.",
           "**a**, Incidence. **b**, Prevalence. **c**, Deaths. **d**, DALYs. Age-standardized rate and absolute number, each with 1990 = 100, on one axis.", "vermillion, number; blue, rate."),
    legend("SuppFig3", "Country trajectories within each joint incidence–DALY cluster.",
           "**a**–**e**, Incidence rate index (1990 = 100). **f**–**j**, DALY rate index. Grey lines, member countries; coloured line, mean of the countries in the cluster.", "cluster colours as in Fig. 4."),
    legend("SuppFig4", "Diagnostics of the trajectory clustering.",
           "**a**, Silhouette width for K = 3, 4 and 5, with bootstrap stability of the selected solution. K = 5 is the largest value tested. **b**, Each country's 2023 incidence and DALY rate indices (1990 = 100).", "cluster colours as in Fig. 4."),
    legend("SuppFig5", "Sensitivity of total CVD trends to the older-age threshold, OECD aggregate, 1990–2023.",
           "**a**, Percentage change in the age-standardized rate. **b**, Percentage change in number. Deaths and DALYs at ages 65+, 70+ and 80+.", "purple circles, deaths; amber squares, DALYs."),
    legend("SuppFig6", "Hold-out accuracy of country ETS models, 2014–2023.",
           "**a**, Incidence. **b**, DALYs. Mean absolute percentage error of models trained on 1990–2013. Dashed line, median of 38 countries.", "blue, incidence; orange, DALYs."),
    legend("SuppFig7", "Observed 2023 rates and 2050 country forecasts.",
           "**a**, Incidence. **b**, DALYs. Open circles, observed 2023 age-standardized rate; filled circles, 2050 ETS point forecast; bars, nominal 95% prediction interval. Logarithmic axis; lower limits that fall left of the axis (including those truncated at zero) end at the axis. Countries are in ISO3 order and are not ranked.", "blue, incidence; orange, DALYs."),
]
HEAD = ("# Figure Legends — Cardiovascular disease burden among adults aged 65 years and older in 38 OECD countries, 1990–2023, with statistical extrapolation to 2050\n\n"
        "> Nature Medicine-style legends generated by `09_代码与质控_Code_QC/02_build_tables_text_qc.py`. Abbreviations: **AAPC** average annual percent change; **APC** annual percent change; **ASR** age-standardized rate for ages 65+ (GBD 2023 world standard renormalized within ages 65+), per 100,000; **CI** confidence interval; **CVD** cardiovascular disease; **DALYs** disability-adjusted life years; **ETS** exponential smoothing; **HHD** hypertensive heart disease; **ICH** intracerebral haemorrhage; **IHD** ischaemic heart disease; **MAPE** mean absolute percentage error; **PI** prediction interval; **SDI** Socio-demographic Index; **UI** uncertainty interval; **YLDs** years lived with disability. Data: GBD 2023 (local extract), 38 current OECD members as a fixed cohort. UIs (Monte Carlo-propagated, not official GBD intervals), CIs (regression), IQRs and PIs answer different questions and are named accordingly. Source data are in `07_源数据_SourceData/<Figure>_SourceData.xlsx` (sheet = panel) and `.csv` (column `panel`).\n\n---\n\n")
(V1 / "03_图注_Figure_Legends.md").write_text(HEAD + "".join(MAIN_LEG) + "# Supplementary Figures\n\n" + "".join(SUPP_LEG), encoding="utf-8")

# ----------------------------------------------------------------------------- results summary
tbl = lambda f, o, c: t1(f, o, c)
rows = ""
for o in OUTCOMES:
    rows += (f"| {o} | {num(tbl(rate_n, o, 'rate_1990'), 1)} | {num(tbl(rate_n, o, 'rate_2023'), 1)} | {sgn(tbl(rate_n, o, 'aapc_1990_2023_percent'), 2)}% "
             f"| {num(tbl(num_n, o, 'number_1990'))} | {num(tbl(num_n, o, 'number_2023'))} | {sgn(tbl(num_n, o, 'aapc_1990_2023_percent'), 2)}% |\n")
SUMMARY = f"""# 结果摘要（OECD 38 国 65+ CVD，V1，自动生成）

> 数据：GBD 2023 本地结果；1990–2023 年，预测至 2050 年。所有数字由 `09_代码与质控_Code_QC/02_build_tables_text_qc.py` 从分析结果读取。解释边界和不确定性命名以设计文档为准。

## ① OECD 总体（Table 1）

| 结局 | 率 1990 | 率 2023 | 率 AAPC | Number 1990 | Number 2023 | Number AAPC |
|---|---:|---:|---:|---:|---:|---:|
{rows}
四项年龄标化率均下降；发病数、患病数和死亡数增加，DALY 绝对数仅轻度下降。

## ② 趋势与国家分布（Fig. 1–2）

- 2023 年 38 国中位数：发病 {num(medians['Incidence'])}，患病 {num(medians['Prevalence'])}，死亡 {num(medians['Deaths'])}，DALY {num(medians['DALYs'])}（/10 万）。
- 发病率分段 APC：{seg6}；DALY 率分段 APC：{seg2}。

## ③ 联合轨迹（Fig. 4）

K = 5（silhouette {sil[3]:.3f} / {sil[4]:.3f} / {sil[5]:.3f}），bootstrap ARI {cd4['bootstrap_mean_adjusted_rand_index']:.3f}（{cd4['bootstrap_ari_2_5_percentile']:.3f}–{cd4['bootstrap_ari_97_5_percentile']:.3f}）。各类国家数：{', '.join(f'第{c}类 {n}' for c, n in clusters.cluster.value_counts().sort_index().items())}。

## ④ 分解（Fig. 5，占 1990 年负担的百分比）

| 结局 | 人口增长 | 人口老龄化 | 流行病学变化 | 净变化 |
|---|---:|---:|---:|---:|
| Incidence | {sgn(dg('OECD38', 6, 'population_growth_percent_of_1990'), 2)} | {sgn(dg('OECD38', 6, 'population_ageing_percent_of_1990'), 2)} | {sgn(dg('OECD38', 6, 'epidemiological_change_percent_of_1990'), 2)} | {sgn(dg('OECD38', 6, 'total_change_percent_of_1990'), 2)} |
| DALYs | {sgn(dg('OECD38', 2, 'population_growth_percent_of_1990'), 2)} | {sgn(dg('OECD38', 2, 'population_ageing_percent_of_1990'), 2)} | {sgn(dg('OECD38', 2, 'epidemiological_change_percent_of_1990'), 2)} | {sgn(dg('OECD38', 2, 'total_change_percent_of_1990'), 2)} |

## ⑤ 最老年组集中与伤残转移（Fig. 7，Total CVD）

80+/65+：死亡 {ooci(491, 1, 1990):.2f}% → {ooci(491, 1, 2023):.2f}%，DALYs {ooci(491, 2, 1990):.2f}% → {ooci(491, 2, 2023):.2f}%，YLDs {ooci(491, 3, 1990):.2f}% → {ooci(491, 3, 2023):.2f}%。YLD/DALY {dti(491, 1990):.2f}% → {dti(491, 2023):.2f}%。

## ⑥ ETS 外推（Fig. 8–10）

- OECD 发病率 2050：{num(E6.forecast_asr)}（95% PI {num(E6.pi_lower)}–{num(E6.pi_upper)}），较 2023 年 {sgn(100 * (E6.forecast_asr / O6 - 1))}%；留出期 MAPE {V6.mape_percent:.1f}%，PI 覆盖率 {V6.prediction_interval_coverage_percent:.0f}%。
- OECD DALY 率 2050：{num(E2.forecast_asr)}（95% PI {num(E2.pi_lower)}–{num(E2.pi_upper)}），较 2023 年 {sgn(100 * (E2.forecast_asr / O2 - 1))}%；留出期 MAPE {V2.mape_percent:.1f}%，PI 覆盖率 {V2.prediction_interval_coverage_percent:.0f}%。
- 38 国发病率：{C6['lower']} 国低于 2023 年，{C6['higher']} 国高于；相对变化中位数 {sgn(C6['median_change'])}%；MAPE 中位数 {C6['median_mape']:.2f}%；{C6['clip_countries']} 国共 {C6['clip_rows']} 个国家-年份下限截 0。
- 38 国 DALY 率：{C2['lower']} 国低于 2023 年，{C2['higher']} 国高于；相对变化中位数 {sgn(C2['median_change'])}%；MAPE 中位数 {C2['median_mape']:.2f}%；{C2['clip_countries']} 国共 {C2['clip_rows']} 个国家-年份下限截 0。
- 预测区间校准不足：不能表述为“到 2050 年必然下降”，也不做国家排名。
"""
(V1 / "08_结果摘要_Results_summary.md").write_text(SUMMARY, encoding="utf-8")

# ----------------------------------------------------------------------------- supplement document
SM = f"""# Supplementary Information

> Generated by `09_代码与质控_Code_QC/02_build_tables_text_qc.py` from the V1 results and the study design. Part A, Supplementary Methods. Part B, Supplementary Figure legends. Part C, Supplementary Table titles and notes (tables are in `06_附表_Supplementary_Tables/`).

# Part A. Supplementary Methods

### Supplementary Methods 1. Data sources and scope
We used local GBD 2023 results for total cardiovascular disease (cause 491), ischaemic heart disease (493), ischaemic stroke (495), intracerebral haemorrhage (496) and hypertensive heart disease (498), for the 38 current OECD members applied as a fixed cohort to every year from 1990 to 2023, seven age groups from 65–69 to 95+ years, and males, females and both sexes. Hypertensive heart disease has no incidence estimates in the local files. Total CVD includes causes not available locally and is not the sum of the listed causes; "Other CVD" in Fig. 3 is a residual. 1990 is the first year with complete coverage and is an analytical baseline only.

### Supplementary Methods 2. Recovery of population denominators
The local population file lacks ages 90–94 years and has incomplete sex coverage, so denominators were recovered from the identity rate = number / population × 100,000 using total CVD deaths: P = D(number) / D(rate) × 100,000. No population was interpolated. Recovered populations agreed with the {diag['population_qc']['matched_population_rows']:,} overlapping rows of the population file (see the quality checks below).

### Supplementary Methods 3. Age standardization and OECD aggregation
For each country c, age a, sex s and year t, the denominator is N_(c,a,s,t) = B_(c,a,s,t) / r_(c,a,s,t) × 100,000. Pooled age-specific rates are calculated by summing numbers and denominators across the 38 countries within each age–sex–year cell: r_(G,a,s,t) = Σ_c B_(c,a,s,t) / Σ_c N_(c,a,s,t) × 100,000. The pooled age-standardized rate is ASR_(G,s,t) = Σ_a w_a r_(G,a,s,t), where w_a are the GBD 2023 world standard weights for the seven age groups renormalized to sum to 1 (Supplementary Table 3). Country rates are not averaged. The same pooled counts and denominators are used for trends, decomposition and forecasts. Summing numbers and denominators was verified independently from the raw files (`qc/pooling_checks.csv`).

### Supplementary Methods 4. Propagation of uncertainty
Our local GBD 2023 extract holds point estimates with 95% bounds for each country, age, sex and year; it has neither draws nor an aggregate for ages 65 years and older. To obtain intervals for OECD totals, country totals and the 65+ age-standardized rates, we simulated 1,000 values for every stratum from a split-normal distribution centred on the published estimate, whose lower and upper standard deviations were the distances to the published bounds divided by 1.96: y^(m) = y + z^(m)_(c,k) × [(y^U − y) 1(z > 0) + (y − y^L) 1(z ≤ 0)] / 1.96, with z ~ N(0, 1). Each country and outcome received one deviate per simulation, applied to all its ages, sexes and years, so strata within a country move together while countries vary independently (seed 20260730). Each quantity was computed within each simulation and summarized by its 2.5th and 97.5th percentiles. These are simulated intervals and are not official GBD joint intervals. Age shares, cause shares and YLD/DALY ratios are shown without intervals, because the correlations between their numerators and denominators are unknown.

### Supplementary Methods 5. Segmented trends
log(r_t) = β0 + β1(t − t0) + Σ_j δ_j (t − κ_j)+ + ε_t, with at most three joinpoints and segments of at least five years, selected by minimum BIC. APC_s = 100 × (exp(b_s) − 1). AAPC = 100 × (exp(Σ_s L_s b_s / Σ_s L_s) − 1). Intervals are regression-based 95% CIs from the model covariance and do not include joinpoint-selection uncertainty. Official NCI Joinpoint permutation tests were not run.

### Supplementary Methods 6. Joint trajectory clustering
z_(c,m,t) = log(ASR_(c,m,t) / ASR_(c,m,1990)) for 1991–2023 and m in incidence and DALYs. Concatenated series were standardized, reduced by PCA to at least 90% of variance and clustered by k-means with K in 3, 4, 5, selected by silhouette. Stability was assessed with 500 bootstrap resamples using the adjusted Rand index. The selected K = 5 is the largest candidate, and silhouette values are moderate, so clusters should be read as descriptive groupings.

### Supplementary Methods 7. Three-factor Das Gupta decomposition
Y = N Σ_a w_a r_a, where N is the all-age population, w_a = P_a / N for each 65+ age group (Σ w_a is the 65+ population share and need not equal 1) and r_a is the age-specific rate. Each factor's contribution is the mean of its marginal effect over all 3! = 6 replacement orders; contributions sum exactly to Y_2023 − Y_1990.

### Supplementary Methods 8. SDI association, oldest-old and disability indices
The SDI curve is a pooled robust LOWESS (span 0.30, three iterations) over 1,292 country-years per outcome. OOCI = B(80+) / B(65+) × 100%. DTI = YLD(65+) / DALY(65+) × 100%. Both use summed OECD numbers without age standardization.

### Supplementary Methods 9. ETS forecasting
For each annual series, 10 non-seasonal ETS candidates (additive or multiplicative error; none, additive or multiplicative trend; damped or not) were compared, and the model with minimum AICc that converged with positive point forecasts was retained. Simulation gave nominal 95% prediction intervals (20,000 paths for the OECD aggregate; 10,000 per country and outcome; seed 20260729). Validation used training on 1990–2013 and a 2014–2023 hold-out. Country lower limits below zero are truncated at zero for display and kept in raw form in the data. The approach extrapolates rates under continuation of historical patterns and does not forecast numbers.

### Supplementary Methods 10. Quality checks
Automated checks are listed in `09_代码与质控_Code_QC/qc/qc_checks.csv` and in `qc/pooling_checks.csv` (independent recomputation of all pooled quantities from the raw files), and cover data completeness, denominators, weights, aggregation identities, interval ordering, decomposition additivity, clustering and forecast validity, one-to-one figure outputs, and agreement with the numbers stated in the study design.

# Part B. Supplementary Figures

{''.join(SUPP_LEG)}
# Part C. Supplementary Tables

All tables are in `All_Supplementary_Tables_OECD65plus_CVD.xlsx` (sheet S1–S15) and as individual CSV files.

""" + "\n".join(f"**Supplementary Table {i} | {t}.** {n}\n" for i, (_, t, _, n) in enumerate(SUPP, 1))
(V1 / "10_附件_Supplementary_Information.md").write_text(SM, encoding="utf-8")

# ----------------------------------------------------------------------------- QC
QC_ROWS = []


def qc(check, detail, diff, tol=0.0):
    diff = float(diff)
    QC_ROWS.append({"check": check, "detail": detail, "difference": diff, "tolerance": tol, "pass": bool(abs(diff) <= tol)})


avail = rd(T, "Data_Availability_and_QC.csv")
for r in avail.itertuples():
    qc("Data: 38 countries, 34 years, 7 ages", r.cause, abs(r.locations - 38) + abs(r.years - 34) + abs(r.age_groups - 7))
    qc("Data: no missing/duplicate/out-of-bound values", r.cause, r.missing_values + r.duplicate_keys + r.bound_violations)
pq = diag["population_qc"]
qc("Denominator recovery vs pop.csv (max relative difference)", "total CVD deaths", max(v for k, v in pq.items() if "max" in k), 1e-12)
w = rd(T, "GBD2023_age_weights_65plus.csv")
qc("Age weights sum to 1", "7 age groups", w.renormalized_weight_within_65plus.sum() - 1, 1e-12)
for name, frame in (("rates", table1), ("numbers", table1n)):
    qc("Table 1 has 46 rows", name, len(frame) - 46)
    qc("Table 1 has 13 columns", name, frame.shape[1] - 13)
    qc("Table 1 has no empty cell", name, int(frame.isna().sum().sum()))
wt = dict(zip(w.age_group, w.renormalized_weight_within_65plus))
for o in OUTCOMES:
    for y in (1990, 2023):
        tot = t1(num_n, o, f"number_{y}")
        a = num_n.query("section == 'OECD age groups' and measure_name == @o")[f"number_{y}"].sum()
        c = num_n.query("section == 'OECD countries' and measure_name == @o")[f"number_{y}"].sum()
        qc("OECD number = sum of 7 age groups", f"{o} {y}", (a - tot) / tot, 1e-9)
        qc("OECD number = sum of 38 countries", f"{o} {y}", (c - tot) / tot, 1e-9)
        ar = rate_n.query("section == 'OECD age groups' and measure_name == @o").set_index("age_group")[f"rate_{y}"]
        std = sum(wt[k] * ar[k] for k in AGE_ORDER)
        qc("OECD ASR = weighted sum of age-specific rates", f"{o} {y}", (std - t1(rate_n, o, f"rate_{y}")) / std, 1e-9)
for label, frame, stem in (("rate", rate_n, "rate"), ("number", num_n, "number")):
    for y in (1990, 2023):
        bad = ((frame[f"{stem}_{y}_lower"] > frame[f"{stem}_{y}"]) | (frame[f"{stem}_{y}"] > frame[f"{stem}_{y}_upper"])).sum()
        qc("Propagated UI contains estimate", f"{label} {y}", int(bad))
    bad = ((frame.aapc_lower > frame.aapc_1990_2023_percent) | (frame.aapc_1990_2023_percent > frame.aapc_upper)).sum()
    qc("AAPC CI contains estimate", label, int(bad))
for r in dec.itertuples():
    qc("Das Gupta: factors sum to total change", f"{r.iso3}, {r.measure_name}", (r.population_growth + r.population_ageing + r.epidemiological_change - r.total_change) / max(abs(r.total_change), 1), 1e-9)
    qc("Das Gupta: all 6 orders agree with closed form", f"{r.iso3}, {r.measure_name}", r.explicit_permutation_max_abs_difference, 1e-5)
for m, name in ((6, "Incidence"), (2, "DALYs")):
    qc("Das Gupta OECD burden 1990 = Table 1 Number", name, (dg("OECD38", m, "burden_1990_reconstructed") - t1(num_n, name, "number_1990")) / t1(num_n, name, "number_1990"), 1e-6)
    qc("Das Gupta OECD burden 2023 = Table 1 Number", name, (dg("OECD38", m, "burden_2023_reconstructed") - t1(num_n, name, "number_2023")) / t1(num_n, name, "number_2023"), 1e-6)
qc("Clusters: 38 countries assigned once", "Fig. 4", clusters.iso3.nunique() - 38)
speed = [cluster_index(c, 6) + cluster_index(c, 2) for c in range(1, 6)]
qc("Clusters numbered fastest to slowest", "sum of 2023 indices increases", int(sum(np.diff(speed) <= 0)))
for m in (6, 2):
    qc("SDI: 1,292 complete country-years", f"measure {m}", len(sdi.query("measure == @m").dropna(subset=["sdi", "rate_per_100000"])) - 1292)
qc("OOCI within 0–100%", "Fig. 7", int((~conc.oldest_old_concentration_percent.between(0, 100)).sum()))
qc("YLD/DALY within 0–100%", "Fig. 7", int((~dis.yld_to_daly_percent.between(0, 100)).sum()))
for name, frame in (("OECD", sel8), ("countries", sel)):
    keys = ["measure"] if name == "OECD" else ["measure", "location"]
    qc("ETS: exactly one model selected per series", name, int((frame[frame.selected == True].groupby(keys).size() != 1).sum()))
qc("ETS: 76 country series forecast 2024–2050", "27 years", len(fc.query("series == 'Forecast'")) - 76 * 27)
qc("ETS: all point forecasts positive", "countries", int((fc.query("series == 'Forecast'").forecast_asr <= 0).sum()))
qc("ETS: interval lower ≤ point ≤ upper", "countries", int(((fc.pi_lower > fc.forecast_asr) | (fc.forecast_asr > fc.pi_upper)).sum()))
EXPECTED_FIG = [f"Fig{i}" for i in range(1, 11)] + [f"SuppFig{i}" for i in range(1, 8)]
for k in EXPECTED_FIG:
    folder = FIG_S if k.startswith("Supp") else FIG
    stem = fname(k)
    qc("Figure outputs complete (PDF, PNG, xlsx, csv)", k, 4 - sum(p.exists() for p in (folder / f"{stem}.pdf", folder / f"{stem}.png", SD / f"{k}_SourceData.xlsx", SD / f"{k}_SourceData.csv")))
for k, panels in (("Fig1", 4), ("Fig2", 4), ("Fig3", 6), ("Fig5", 2), ("Fig6", 2), ("Fig7", 10), ("Fig8", 2), ("Fig9", 38), ("Fig10", 38)):
    qc("Source data: one panel per expected panel", k, rd(SD, f"{k}_SourceData.csv").panel.astype(str).nunique() - panels)
s8 = rd(SD, "Fig8_SourceData.csv").query("year == 2050")
qc("Source data equals analysis output", "Fig. 8, incidence 2050", float(s8.query("measure_name == 'Incidence'").forecast_asr.iloc[0]) - float(E6.forecast_asr), 1e-6)


def design_lock():
    """Numbers stated in OECD38_65岁以上_CVD_GBD2023_研究设计.md (2026-07-30), compared at the printed precision."""
    pairs = [
        ("Incidence rate 1990", t1(rate_n, "Incidence", "rate_1990"), 5168.1, .05), ("Incidence rate 2023", t1(rate_n, "Incidence", "rate_2023"), 3731.0, .05),
        ("Incidence rate AAPC", t1(rate_n, "Incidence", "aapc_1990_2023_percent"), -0.96, .005), ("Incidence number 1990", t1(num_n, "Incidence", "number_1990"), 6405640, .5),
        ("Incidence number 2023", t1(num_n, "Incidence", "number_2023"), 9830589, .5), ("Incidence number AAPC", t1(num_n, "Incidence", "aapc_1990_2023_percent"), 1.33, .005),
        ("Prevalence rate 1990", t1(rate_n, "Prevalence", "rate_1990"), 49972.7, .05), ("Prevalence rate 2023", t1(rate_n, "Prevalence", "rate_2023"), 43774.2, .05),
        ("Prevalence rate AAPC", t1(rate_n, "Prevalence", "aapc_1990_2023_percent"), -0.39, .005), ("Prevalence number 2023", t1(num_n, "Prevalence", "number_2023"), 114559583, .5),
        ("Deaths rate 1990", t1(rate_n, "Deaths", "rate_1990"), 2573.7, .05), ("Deaths rate 2023", t1(rate_n, "Deaths", "rate_2023"), 1107.7, .05),
        ("Deaths rate AAPC", t1(rate_n, "Deaths", "aapc_1990_2023_percent"), -2.53, .005), ("Deaths number AAPC", t1(num_n, "Deaths", "aapc_1990_2023_percent"), 0.09, .005),
        ("DALY rate 1990", t1(rate_n, "DALYs", "rate_1990"), 39409.0, .05), ("DALY rate 2023", t1(rate_n, "DALYs", "rate_2023"), 17556.0, .05),
        ("DALY rate AAPC", t1(rate_n, "DALYs", "aapc_1990_2023_percent"), -2.44, .005), ("DALY number 1990", t1(num_n, "DALYs", "number_1990"), 48680349, .5),
        ("DALY number 2023", t1(num_n, "DALYs", "number_2023"), 48090066, .5), ("DALY number AAPC", t1(num_n, "DALYs", "aapc_1990_2023_percent"), -0.06, .005),
        ("Median incidence 2023", medians["Incidence"], 3507, .5), ("Median prevalence 2023", medians["Prevalence"], 39894, .5), ("Median deaths 2023", medians["Deaths"], 1114, .5), ("Median DALYs 2023", medians["DALYs"], 17198, .5),
        ("Das Gupta inc. growth", dg("OECD38", 6, "population_growth_percent_of_1990"), 28.00, .005), ("Das Gupta inc. ageing", dg("OECD38", 6, "population_ageing_percent_of_1990"), 69.34, .005),
        ("Das Gupta inc. epi", dg("OECD38", 6, "epidemiological_change_percent_of_1990"), -43.87, .005), ("Das Gupta inc. net", dg("OECD38", 6, "total_change_percent_of_1990"), 53.47, .005),
        ("Das Gupta DALY growth", dg("OECD38", 2, "population_growth_percent_of_1990"), 23.39, .005), ("Das Gupta DALY ageing", dg("OECD38", 2, "population_ageing_percent_of_1990"), 62.02, .005),
        ("Das Gupta DALY epi", dg("OECD38", 2, "epidemiological_change_percent_of_1990"), -86.63, .005), ("Das Gupta DALY net", dg("OECD38", 2, "total_change_percent_of_1990"), -1.21, .005),
        ("Silhouette K=3", sil[3], .329, .0005), ("Silhouette K=4", sil[4], .339, .0005), ("Silhouette K=5", sil[5], .364, .0005),
        ("ARI mean", cd4["bootstrap_mean_adjusted_rand_index"], .831, .0005), ("ARI low", cd4["bootstrap_ari_2_5_percentile"], .514, .0005), ("ARI high", cd4["bootstrap_ari_97_5_percentile"], 1.0, .0005),
        ("SDI min", sdi.sdi.min(), .478, .0005), ("SDI max", sdi.sdi.max(), .946, .0005),
        ("OOCI deaths 1990", ooci(491, 1, 1990), 55.60, .005), ("OOCI DALYs 1990", ooci(491, 2, 1990), 40.42, .005), ("OOCI YLDs 1990", ooci(491, 3, 1990), 33.63, .005),
        ("OOCI deaths 2023", ooci(491, 1, 2023), 69.47, .005), ("OOCI DALYs 2023", ooci(491, 2, 2023), 51.94, .005), ("OOCI YLDs 2023", ooci(491, 3, 2023), 41.03, .005),
        ("DTI 1990", dti(491, 1990), 8.41, .005), ("DTI 2023", dti(491, 2023), 15.05, .005),
        ("ETS inc. 2050", E6.forecast_asr, 2800, .5), ("ETS inc. lower", E6.pi_lower, 1745, .5), ("ETS inc. upper", E6.pi_upper, 4278, .5), ("ETS DALY 2050", E2.forecast_asr, 15462, .5), ("ETS DALY lower", E2.pi_lower, 6694, .5), ("ETS DALY upper", E2.pi_upper, 30851, .5),
        ("ETS inc. change", 100 * (E6.forecast_asr / O6 - 1), -24.9, .06), ("ETS DALY change", 100 * (E2.forecast_asr / O2 - 1), -11.9, .05),
        ("Hold-out MAPE inc.", V6.mape_percent, 4.2, .05), ("Hold-out MAPE DALY", V2.mape_percent, 7.4, .05), ("PI coverage inc.", V6.prediction_interval_coverage_percent, 20, .01), ("PI coverage DALY", V2.prediction_interval_coverage_percent, 30, .01),
        ("Fig 9 lower", C6["lower"], 25, 0), ("Fig 9 higher", C6["higher"], 13, 0), ("Fig 9 median change", C6["median_change"], -4.8, .05), ("Fig 9 median MAPE", C6["median_mape"], 4.12, .005), ("Fig 9 clip rows", C6["clip_rows"], 16, 0), ("Fig 9 clip countries", C6["clip_countries"], 3, 0),
        ("Fig 10 lower", C2["lower"], 33, 0), ("Fig 10 higher", C2["higher"], 5, 0), ("Fig 10 median change", C2["median_change"], -28.1, .05), ("Fig 10 median MAPE", C2["median_mape"], 6.16, .005), ("Fig 10 clip rows", C2["clip_rows"], 62, 0), ("Fig 10 clip countries", C2["clip_countries"], 7, 0),
    ]
    for c, (inc, dal) in {1: (57.5, 28.0), 2: (53.8, 46.0), 3: (68.5, 38.1), 4: (79.8, 44.2), 5: (70.9, 58.3)}.items():
        pairs += [(f"Cluster {c} incidence index", cluster_index(c, 6), inc, .05), (f"Cluster {c} DALY index", cluster_index(c, 2), dal, .05)]
    for c, n in {1: 4, 2: 4, 3: 16, 4: 7, 5: 7}.items():
        pairs.append((f"Cluster {c} size", int((clusters.cluster == c).sum()), n, 0))
    for name, got, want, tol in pairs:
        qc("Agrees with design document", name, float(got) - want, tol)


design_lock()
report = pd.DataFrame(QC_ROWS)
write(report, QC / "qc_checks.csv")
summary = report.groupby("check")["pass"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "passed"}).reset_index()
write(summary, QC / "qc_summary.csv")
print(f"QC: {int(report['pass'].sum())} of {len(report)} passed")
if not report["pass"].all():
    print(report[~report["pass"]].to_string())
    raise SystemExit(1)
