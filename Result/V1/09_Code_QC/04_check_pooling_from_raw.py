#!/usr/bin/env python3
"""Independent validation of OECD pooling from GBD parquet inputs.

Population = number / rate x 100,000 within each location-age-sex-year.
Pooled age-specific rate = summed numbers / summed populations x 100,000.
Pooled ASR = weighted sum of age-specific rates using renormalized world-standard
weights. Split-normal intervals use one deviate per location-measure-draw and
are compared with the row-level implementation used for Table 1.
Writes qc/pooling_checks.csv and exits with a nonzero status on failure."""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
V1 = CODE.parent
DATA = V1.parents[1] / "data"
AO = CODE / "analysis_outputs"
T, D = AO / "tables", AO / "figure_data"
SD = V1 / "07_SourceData"
AGES = [18, 19, 20, 30, 31, 32, 235]
W_PCT = np.array([2.98509, 2.26526, 1.59758, 1.09729, 0.604519, 0.246663, 0.078509])
WEIGHT = dict(zip(AGES, W_PCT / W_PCT.sum()))
OECD = sorted("AUS AUT BEL CAN CHE CHL COL CRI CZE DEU DNK ESP EST FIN FRA GBR GRC HUN IRL ISL ISR ITA JPN KOR LTU LUX LVA MEX NLD NOR NZL POL PRT SVK SVN SWE TUR USA".split())
ROWS = []


def check(group, item, diff, tol=1e-9):
    diff = float(diff)
    ROWS.append({"group": group, "item": item, "max_difference": diff, "tolerance": tol, "pass": bool(np.isfinite(diff) and abs(diff) <= tol)})


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def rd(folder, name):
    return pd.read_csv(folder / name, low_memory=False)


# ----------------------------------------------------------------------------- raw data
gbd = pd.concat([pd.read_parquet(p) for p in sorted(DATA.glob("*/*.parquet"))], ignore_index=True)
keys = ["cause", "measure", "location", "sex", "age", "year", "metric"]
cw = rd(T, "OECD38_crosswalk.csv")

check("A. Population scope", "38 raw locations match the crosswalk", len(set(gbd.location) ^ set(cw.location_id)), 0)
check("A. Population scope", "Crosswalk ISO3 matches the fixed 38-country OECD cohort", len(set(cw.iso3) ^ set(OECD)), 0)
check("A. Population scope", "Crosswalk ISO3 and names are unique", (38 - cw.iso3.nunique()) + (38 - cw.location_name.nunique()), 0)
check("A. Population scope", "Years are 1990-2023", abs(gbd.year.min() - 1990) + abs(gbd.year.max() - 2023) + abs(gbd.year.nunique() - 34), 0)
check("A. Population scope", "Only seven 65+ age groups", len(set(gbd.age) ^ set(AGES)), 0)
check("B. Completeness", "No duplicated key", int(gbd.duplicated(keys).sum()), 0)
expected = 0
for (cause, measure), b in gbd.groupby(["cause", "measure"]):
    want = 38 * 3 * 7 * 34 * 2
    expected += want
    check("B. Completeness", f"cause {cause}, measure {measure}: every stratum present", len(b) - want, 0)
check("B. Completeness", "HHD (498) has no incidence and all else is complete", len(gbd.query("cause == 498 and measure == 6")) + (len(gbd) - expected), 0)
check("B. Completeness", "No missing val/lower/upper", int(gbd[["val", "lower", "upper"]].isna().sum().sum()), 0)
check("C. Bounds", "lower <= val <= upper", int(((gbd.lower > gbd.val) | (gbd.val > gbd.upper)).sum()), 0)
check("C. Bounds", "All values positive", int((gbd.val <= 0).sum()), 0)

# ----------------------------------------------------------------------------- denominators
wide = gbd.pivot_table(index=["cause", "measure", "location", "sex", "age", "year"], columns="metric", values="val")
wide["N"] = wide[1] / wide[3] * 1e5
ref = wide.loc[(491, 1)].N
pop = ref.rename("N").reset_index()
aligned = wide.N.reset_index().merge(pop, on=["location", "sex", "age", "year"], suffixes=("", "_ref"))
ratio = (aligned.N / aligned.N_ref - 1).abs()
check("D. Denominators", "Denominator from every cause and measure = total CVD deaths", ratio.max(), 1e-6)
ROWS[-1]["note"] = f"median {ratio.median():.2e}"
p_all = pop.pivot_table(index=["location", "age", "year"], columns="sex", values="N")
check("D. Denominators", "Population: males + females = both sexes", rel(p_all[1] + p_all[2], p_all[3]), 1e-4)
popcsv = rd(DATA, "pop.csv")
m = pop.merge(popcsv, left_on=["location", "sex", "age", "year"], right_on=["location_id", "sex_id", "age_id", "year"])
check("D. Denominators", f"Recovered denominators = pop.csv ({len(m):,} rows)", rel(m.N, m.val), 1e-12)
num3 = gbd.query("metric == 1")
g_sex = num3.pivot_table(index=["cause", "measure", "location", "age", "year"], columns="sex", values="val")
sex_gap = ((g_sex[1] + g_sex[2]) / g_sex[3] - 1).abs()
check("D. Denominators", "Numbers: males + females = both sexes", sex_gap.max(), 1e-4)
ROWS[-1]["note"] = f"median {sex_gap.median():.2e}"
# ----------------------------------------------------------------------------- pooled point estimates
num = gbd.query("metric == 1").merge(pop, on=["location", "sex", "age", "year"], validate="many_to_one")
num["w"] = num.age.map(WEIGHT)
num["age_rate"] = num.val / num.N * 1e5
cell = num.groupby(["cause", "measure", "sex", "age", "year"], as_index=False).agg(number=("val", "sum"), N=("N", "sum"), w=("w", "first"))
cell["rate"] = cell.number / cell.N * 1e5
cell["wr"] = cell.w * cell.rate
oecd = cell.groupby(["cause", "measure", "sex", "year"], as_index=False).agg(number=("number", "sum"), asr=("wr", "sum"))
num["wr"] = num.w * num.age_rate
country = num.groupby(["cause", "measure", "location", "sex", "year"], as_index=False).agg(number=("val", "sum"), asr=("wr", "sum"))
via_rate = gbd.query("metric == 3").assign(w=lambda d: d.age.map(WEIGHT)).assign(wr=lambda d: d.w * d.val).groupby(["cause", "measure", "location", "sex", "year"]).wr.sum()
check("E. Pooling identities", "Country ASR: published rate = number / denominator", rel(via_rate, country.set_index(["cause", "measure", "location", "sex", "year"]).asr.reindex(via_rate.index)), 1e-9)
check("E. Pooling identities", "Weights sum to 1", sum(WEIGHT.values()) - 1, 1e-12)
naive = country.query("cause == 491 and sex == 3").groupby(["measure", "year"]).asr.mean()
pooled = oecd.query("cause == 491 and sex == 3").set_index(["measure", "year"]).asr
gap = (naive / pooled.reindex(naive.index) - 1).abs()
check("E. Pooling identities", "OECD ASR differs from simple country mean (pooling applied)", 0 if gap.max() > 0.01 else 1, 0)
ROWS[-1]["note"] = f"largest |mean / pooled - 1| = {gap.max():.1%}"


def compare(group, item, left, right, on, lcol, rcol, tol=1e-9):
    j = left.merge(right, on=on, validate="one_to_one", suffixes=("_l", "_r"))
    if len(j) != len(right):
        check(group, item + " (row count)", len(right) - len(j), 0)
        return
    l, r = (lcol + "_l", rcol + "_r") if lcol == rcol else (lcol, rcol)
    check(group, f"{item} ({len(j):,} rows)", rel(j[l], j[r]), tol)


G = "F. Outputs vs raw recomputation"
long_asr, long_n = rd(D, "Long_OECD_ASR_total_CVD.csv"), rd(D, "Long_OECD_number_total_CVD.csv")
O3 = oecd.query("cause == 491 and sex == 3")
compare(G, "OECD ASR, 6 measures", long_asr, O3, ["cause", "measure", "sex", "year"], "asr", "asr")
compare(G, "OECD number, 6 measures", long_n, O3, ["cause", "measure", "sex", "year"], "number", "number")
C3 = country.query("cause == 491 and sex == 3")
compare(G, "Country ASR, 38 countries", rd(D, "Long_country_ASR_total_CVD.csv"), C3, ["cause", "measure", "location", "sex", "year"], "asr", "asr")
compare(G, "Country number, 38 countries", rd(D, "Long_country_number_total_CVD.csv"), C3, ["cause", "measure", "location", "sex", "year"], "number", "number")

MEAS = {"Incidence": 6, "Prevalence": 5, "Deaths": 1, "DALYs": 2}
rn, nn = rd(T, "Table1_OECD65plus_CVD_rates_AAPC_numeric.csv"), rd(T, "Table1_OECD65plus_CVD_numbers_AAPC_numeric.csv")
for frame, stem in ((rn, "rate"), (nn, "number")):
    for y in (1990, 2023):
        rows, want = [], []
        for r in frame.itertuples():
            mcode = MEAS[r.measure_name]
            if r.section == "OECD overall":
                v = O3.query("measure == @mcode and year == @y")[("asr" if stem == "rate" else "number")].iloc[0]
            elif r.section == "OECD age groups":
                a = int(r.age_id)
                v = cell.query("cause == 491 and sex == 3 and measure == @mcode and year == @y and age == @a")[("rate" if stem == "rate" else "number")].iloc[0]
            else:
                l = int(r.location_id)
                v = C3.query("measure == @mcode and year == @y and location == @l")[("asr" if stem == "rate" else "number")].iloc[0]
            rows.append(getattr(r, f"{stem}_{y}")); want.append(v)
        check(G, f"Table 1 {stem} {y}: 46 rows x 4 outcomes", rel(rows, want))

s1 = rd(T, "TableS1_Age_Sex_Subtype_Rates.csv")
j = s1.merge(cell, on=["cause", "measure", "sex", "age", "year"], validate="one_to_one")
check(G, f"Fig. 3 / Supp. Table 8 age-specific rates and numbers ({len(j)} rows)", max(rel(j.age_specific_rate, j.rate), rel(j.number_x, j.number_y)))

s2 = rd(T, "TableS2_OldestOld_and_Disability_Transition.csv")
c80 = cell.query("sex == 3").assign(old=lambda d: d.age >= 30)
gr = c80.groupby(["cause", "measure", "year"]).apply(lambda d: pd.Series({"n65": d.number.sum(), "n80": d.loc[d.old, "number"].sum()})).reset_index()
j = s2.merge(gr, on=["cause", "measure", "year"], validate="one_to_one")
check(G, f"Fig. 7 numbers at 65+ and 80+ ({len(j)} rows)", max(rel(j.number, j.n65), rel(j.number_80plus, j.n80)))
yld = gr.query("measure == 3").merge(gr.query("measure == 2"), on=["cause", "year"], suffixes=("_y", "_d"))
yld["dti"] = yld.n65_y / yld.n65_d * 100
fig7 = rd(D, "Figure7_disability_transition_data.csv").merge(yld, on=["cause", "year"])
check(G, f"Fig. 7 YLD/DALY ({len(fig7)} rows)", rel(fig7.yld_to_daly_percent, fig7.dti))

thr = rd(T, "TableS3_Age_Threshold_Sensitivity.csv")
for label, ages in (("65+", AGES), ("70+", AGES[1:]), ("80+", AGES[3:])):
    wts = np.array([WEIGHT[a] for a in ages]); wts /= wts.sum()
    for m_ in (1, 2):
        for y in (1990, 2023):
            b = cell.query("cause == 491 and sex == 3 and measure == @m_ and year == @y and age in @ages").set_index("age").reindex(ages)
            got = thr.query("age_threshold == @label and measure == @m_").iloc[0]
            check(G, f"Age threshold {label}, measure {m_}, {y}: ASR and number", max(rel([got[f"asr_{y}"]], [np.dot(wts, b.rate)]), rel([got[f"number_{y}"]], [b.number.sum()])))

# ----------------------------------------------------------------------------- Das Gupta
dec = rd(T, "Table3E_Das_Gupta_Decomposition_Incidence_DALYs.csv")
total_pop = popcsv.query("sex_id == 3 and age_id == 22 and location_id in @cw.location_id")
for code, name in ((6, "Incidence"), (2, "DALYs")):
    y0, y1 = 1990, 2023
    def state(y):
        n = float(total_pop.query("year == @y").val.sum())
        b = cell.query("cause == 491 and sex == 3 and measure == @code and year == @y").set_index("age").reindex(AGES)
        return n, (b.N / n).to_numpy(), (b.number / b.N).to_numpy()
    s0, s1_ = state(y0), state(y1)
    def Y(a, b, c):
        return a * np.sum(b * c)
    from itertools import permutations
    eff = [0.0, 0.0, 0.0]
    for order in permutations(range(3)):
        cur = list(s0)
        for f in order:
            before = Y(*cur); cur[f] = s1_[f]; eff[f] += (Y(*cur) - before) / 6
    r = dec.query("iso3 == 'OECD38' and measure == @code").iloc[0]
    check("G. Decomposition", f"OECD {name}: three factors reproduced independently", rel(eff, [r.population_growth, r.population_ageing, r.epidemiological_change]), 1e-9)
    check("G. Decomposition", f"OECD {name}: factors sum to change in pooled number", rel([sum(eff)], [O3.query("measure == @code and year == 2023").number.iloc[0] - O3.query("measure == @code and year == 1990").number.iloc[0]]), 1e-6)
cn = dec.query("scope == 'Country'").merge(cw[["iso3", "location_id"]], on="iso3", suffixes=("", "_c"))
b90 = C3.query("year == 1990")[["location", "measure", "number"]]
j = cn.merge(b90, left_on=["location_id_c", "measure"], right_on=["location", "measure"])
check("G. Decomposition", f"Country 1990 burden = pooled ages ({len(j)} rows)", rel(j.burden_1990_reconstructed, j.number), 1e-6)

# ----------------------------------------------------------------------------- uncertainty
SEED, M = 20260730, 1000
rng = np.random.default_rng(SEED)
Z = {(mc, int(l)): rng.standard_normal(M) for mc in sorted(gbd.measure.unique()) for l in sorted(gbd.location.unique())}
base = gbd.query("cause == 491 and sex == 3 and metric == 1 and year in [1990, 2023] and measure in [1, 2, 5, 6]").merge(pop.query("sex == 3")[["location", "age", "year", "N"]], on=["location", "age", "year"])
pg = base.groupby(["measure", "age", "year"]).N.transform("sum")
base = base.assign(Pg=pg, w=base.age.map(WEIGHT), hu=(base.upper - base.val) / 1.96, hl=(base.val - base.lower) / 1.96)
qmax = {"rate": 0.0, "number": 0.0}


def quantiles(frame, ccol, ucol, lcol):
    """Sum over locations of value + z * half-width."""
    t = np.zeros(M)
    for loc, row in frame.groupby("location")[[ccol, ucol, lcol]].sum().iterrows():
        z = Z[(mc, int(loc))]
        t += row[ccol] + z * np.where(z > 0, row[ucol], row[lcol])
    return np.quantile(t, [0.025, 0.975])


for r_ in rn.itertuples():
    mc = MEAS[r_.measure_name]
    n_row = nn.query("entity_order == @r_.entity_order and measure_name == @r_.measure_name").iloc[0]
    for y in (1990, 2023):
        b = base.query("measure == @mc and year == @y")
        if r_.section == "OECD age groups":
            b = b[b.age == int(r_.age_id)]
            frame = b.assign(c=b.val / b.Pg * 1e5, u=b.hu / b.Pg * 1e5, l=b.hl / b.Pg * 1e5)
        elif r_.section == "OECD countries":
            b = b[b.location == int(r_.location_id)]
            frame = b.assign(c=b.w * b.val / b.N * 1e5, u=b.w * b.hu / b.N * 1e5, l=b.w * b.hl / b.N * 1e5)
        else:
            frame = b.assign(c=b.w * b.val / b.Pg * 1e5, u=b.w * b.hu / b.Pg * 1e5, l=b.w * b.hl / b.Pg * 1e5)
        q = quantiles(frame, "c", "u", "l")
        qmax["rate"] = max(qmax["rate"], rel(q, [getattr(r_, f"rate_{y}_lower"), getattr(r_, f"rate_{y}_upper")]))
        q = quantiles(b, "val", "hu", "hl")
        qmax["number"] = max(qmax["number"], rel(q, [n_row[f"number_{y}_lower"], n_row[f"number_{y}_upper"]]))
check("H. Uncertainty", "Table 1 rate UI: aggregate formula = row-level draws (368 pairs)", qmax["rate"], 1e-8)
check("H. Uncertainty", "Table 1 number UI: aggregate formula = row-level draws (368 pairs)", qmax["number"], 1e-8)
check("H. Uncertainty", "Same deviates across functions: first deviate reproducible", 0 if abs(np.random.default_rng(SEED).standard_normal(1)[0] - Z[(1, min(gbd.location))][0]) < 1e-15 else 1, 0)
# single country: with one shared deviate, each limit is exactly total + z-quantile x summed half-width
exact, noise = 0.0, 0.0
for o, mc in MEAS.items():
    for l in cw.location_id:
        qz = np.quantile(Z[(mc, int(l))], [0.025, 0.975])
        got = nn.query("location_id == @l and measure_name == @o").iloc[0]
        for y in (1990, 2023):
            b = base.query("measure == @mc and year == @y and location == @l")
            lo, up, c = b.lower.sum(), b.upper.sum(), b.val.sum()
            exact = max(exact, rel([got[f"number_{y}_lower"], got[f"number_{y}_upper"]], [c + qz[0] * (c - lo) / 1.96, c + qz[1] * (up - c) / 1.96]))
            noise = max(noise, abs(got[f"number_{y}_lower"] - lo) / (c - lo), abs(got[f"number_{y}_upper"] - up) / (up - c))
check("H. Uncertainty", "Country number UI = total + z quantile x summed half-width (304 pairs x 2 limits)", exact, 1e-9)
ROWS[-1]["note"] = f"largest distance from the summed published limits: {noise:.1%} of the half-width (Monte Carlo error of the 2.5%/97.5% quantiles with 1,000 draws)"

# ----------------------------------------------------------------------------- cross-output consistency
G = "I. Consistency between outputs"
f1 = rd(D, "Figure1_four_map_data.csv").merge(rn.query("section == 'OECD countries'"), left_on=["location", "measure_name"], right_on=["location_id", "measure_name"])
check(G, f"Fig. 1 maps = Table 1 2023 rates ({len(f1)} rows)", rel(f1.asr, f1.rate_2023))
f2 = rd(D, "Figure2_trend_data.csv").query("scope == 'OECD aggregate'").merge(O3, on=["measure", "year"])
check(G, f"Fig. 2 trend = pooled ASR ({len(f2)} rows)", rel(f2.asr_x, f2.asr_y))
f8 = rd(D, "Figure8_ETS_forecast_data.csv").query("series == 'Observed'").merge(O3, on=["measure", "year"])
check(G, f"Fig. 8 observed series = pooled ASR ({len(f8)} rows)", rel(f8.observed_asr, f8.asr))
f9 = rd(D, "Figure9_10_country_ETS_forecast_data.csv").query("series == 'Observed'").merge(C3, on=["measure", "location", "year"])
check(G, f"Figs. 9-10 observed = country ASR ({len(f9)} rows)", rel(f9.observed_asr, f9.asr))
f6 = rd(D, "Figure6_incidence_DALY_SDI_data.csv").merge(C3, left_on=["location_id", "measure", "year"], right_on=["location", "measure", "year"])
check(G, f"Fig. 6 rates = country ASR ({len(f6)} rows)", rel(f6.rate_per_100000, f6.asr))
idx = C3.query("year in [1990, 2023]").pivot_table(index=["location", "measure"], columns="year", values="asr")
idx = (idx[2023] / idx[1990] * 100).unstack("measure")
f4 = rd(D, "Figure4_trajectory_clusters_incidence_DALYs.csv").set_index("location_id")
check(G, "Fig. 4 2023 indices = ASR ratios (38 countries)", max(rel(f4.asir_index_2023, idx[6].reindex(f4.index)), rel(f4.asdr_index_2023, idx[2].reindex(f4.index))))
a_ui = rd(T, "Approximate_UI_OECD_Total_CVD.csv").query("year in [1990, 2023]").assign(measure_name=lambda d: d.measure.map({1: "Deaths", 2: "DALYs"}))
ov = rn.query("section == 'OECD overall'").set_index("measure_name")
for r in a_ui.itertuples():
    check(G, f"Supp. Table 15 UI = Table 1 UI: {r.measure_name} {r.year}", rel([r.asr_lower, r.asr_upper], [ov.loc[r.measure_name, f"rate_{r.year}_lower"], ov.loc[r.measure_name, f"rate_{r.year}_upper"]]), 1e-9)

report = pd.DataFrame(ROWS)
report.to_csv(CODE / "qc" / "pooling_checks.csv", index=False, encoding="utf-8-sig")
print(f"Pooling checks: {int(report['pass'].sum())} of {len(report)} passed")
print(report.groupby("group")["pass"].agg(["size", "sum"]).to_string())
for r in report[report.get("note").notna()].itertuples():
    print("note:", r.item, "|", r.note)
if not report["pass"].all():
    print(report[~report["pass"]].to_string())
    raise SystemExit(1)
