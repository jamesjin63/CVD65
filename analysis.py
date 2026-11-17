#!/usr/bin/env python3
"""OECD-38 cardiovascular disease burden at ages 65 years and older, GBD 2023.

Computes statistical estimates and exports analysis tables and figure data.
Publication figures are rendered separately by generate_figures.py."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR
DATA = PROJECT / "data"
RESULT = PROJECT / "results" / "analysis_outputs"
TABLES = RESULT / "tables"
FIGURES = RESULT / "_draft_figures"  # removed at the end of main()
FIGURE_DATA = RESULT / "figure_data"

for directory in (TABLES, FIGURES, FIGURE_DATA):
    directory.mkdir(parents=True, exist_ok=True)


MEASURE_NAMES = {
    1: "Deaths",
    2: "DALYs",
    3: "YLDs",
    4: "YLLs",
    5: "Prevalence",
    6: "Incidence",
}
CAUSE_NAMES = {
    491: "Cardiovascular diseases",
    493: "Ischemic heart disease",
    495: "Ischemic stroke",
    496: "Intracerebral hemorrhage",
    498: "Hypertensive heart disease",
}
CAUSE_SHORT = {
    491: "Total CVD",
    493: "IHD",
    495: "Ischemic stroke",
    496: "ICH",
    498: "HHD",
}
SEX_NAMES = {1: "Male", 2: "Female", 3: "Both"}
AGE_ORDER = [18, 19, 20, 30, 31, 32, 235]
AGE_NAMES = {
    18: "65-69",
    19: "70-74",
    20: "75-79",
    30: "80-84",
    31: "85-89",
    32: "90-94",
    235: "95+",
}

# GBD 2023 world population age standard, percent of total population.
# Values are renormalized across the seven age groups from 65 years upward.
GBD_WORLD_STANDARD_PERCENT = {
    18: 2.98509,
    19: 2.26526,
    20: 1.59758,
    30: 1.09729,
    31: 0.604519,
    32: 0.246663,
    235: 0.078509,
}
_weight_sum = sum(GBD_WORLD_STANDARD_PERCENT.values())
AGE_WEIGHTS = {age: value / _weight_sum for age, value in GBD_WORLD_STANDARD_PERCENT.items()}

SELECTED_YEARS = [1990, 2000, 2010, 2019, 2021, 2023]
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#7A5195",
    "sky": "#56B4E9",
    "gray": "#6B7280",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": .8, "lines.linewidth": 1.6,
})


def panel_label(ax: plt.Axes, label: str, *, wide: bool = False, fontsize: int = 14) -> None:
    letter = str(label).strip("()").lower()
    ax.text(
        -0.16 if wide else -0.12,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
        ha="left",
        fontsize=fontsize,
    )


def save_csv(df: pd.DataFrame, name: str, figure_data: bool = False) -> None:
    target = (FIGURE_DATA if figure_data else TABLES) / name
    df.to_csv(target, index=False, encoding="utf-8-sig")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frames = []
    for parquet_file in sorted(DATA.glob("*/*.parquet")):
        frame = pd.read_parquet(parquet_file)
        frame["source_file"] = parquet_file.name
        frames.append(frame)
    gbd = pd.concat(frames, ignore_index=True)
    pop = pd.read_csv(DATA / "pop.csv")
    sdi = pd.read_csv(DATA / "sdi.csv")
    gdp = pd.read_csv(DATA / "gdp.csv")
    with (DATA / "df_world2.geojson").open(encoding="utf-8") as stream:
        geo = json.load(stream)
    return gbd, pop, sdi, gdp, geo


def build_crosswalk(gbd: pd.DataFrame, pop: pd.DataFrame, geo: dict) -> pd.DataFrame:
    location_ids = sorted(gbd["location"].unique())
    loc = (
        pop[["location_id", "location_name"]]
        .drop_duplicates()
        .query("location_id in @location_ids")
        .copy()
    )
    geo_props = pd.DataFrame([feature["properties"] for feature in geo["features"]])
    loc = loc.merge(
        geo_props[["location_id", "brk_a3", "name_long"]],
        on="location_id",
        how="left",
    )
    loc = loc.rename(columns={"brk_a3": "iso3", "name_long": "map_name"})
    if len(loc) != 38 or loc["iso3"].isna().any():
        raise ValueError("OECD crosswalk did not resolve to 38 ISO3-coded countries")
    return loc.sort_values("iso3").reset_index(drop=True)


def validate_gbd(gbd: pd.DataFrame) -> pd.DataFrame:
    key = ["measure", "location", "sex", "age", "cause", "metric", "year"]
    records = []
    for cause_id, group in gbd.groupby("cause"):
        records.append(
            {
                "cause_id": cause_id,
                "cause": CAUSE_NAMES[cause_id],
                "rows": len(group),
                "locations": group["location"].nunique(),
                "years": group["year"].nunique(),
                "year_min": group["year"].min(),
                "year_max": group["year"].max(),
                "sexes": group["sex"].nunique(),
                "age_groups": group["age"].nunique(),
                "measures": ", ".join(MEASURE_NAMES[x] for x in sorted(group["measure"].unique())),
                "metrics": ", ".join(str(x) for x in sorted(group["metric"].unique())),
                "missing_values": int(group[["val", "lower", "upper"]].isna().sum().sum()),
                "duplicate_keys": int(group.duplicated(key).sum()),
                "bound_violations": int(((group["lower"] > group["val"]) | (group["val"] > group["upper"])).sum()),
            }
        )
    return pd.DataFrame(records)


def derive_population(gbd: pd.DataFrame, pop: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = gbd.query("cause == 491 and measure == 1").copy()
    number = base.query("metric == 1")[
        ["location", "sex", "age", "year", "val"]
    ].rename(columns={"val": "number"})
    rate = base.query("metric == 3")[
        ["location", "sex", "age", "year", "val"]
    ].rename(columns={"val": "rate"})
    derived = number.merge(rate, on=["location", "sex", "age", "year"], validate="one_to_one")
    if (derived["rate"] <= 0).any():
        raise ValueError("Cannot derive population from non-positive total CVD death rates")
    derived["population"] = derived["number"] / derived["rate"] * 100000.0

    check = derived.merge(
        pop[["location_id", "sex_id", "age_id", "year", "val"]],
        left_on=["location", "sex", "age", "year"],
        right_on=["location_id", "sex_id", "age_id", "year"],
        how="inner",
        suffixes=("", "_pop_file"),
    )
    check["relative_difference"] = (check["population"] - check["val"]).abs() / check["val"].clip(lower=1e-12)
    qc = {
        "matched_population_rows": int(len(check)),
        "median_relative_difference": float(check["relative_difference"].median()),
        "max_relative_difference": float(check["relative_difference"].max()),
        "note": "Age-specific population recovered as number/rate*100000; this restores the missing 90-94 group.",
    }
    return derived, qc


def summarize_country(gbd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    number = (
        gbd.query("metric == 1")
        .groupby(["cause", "measure", "location", "sex", "year"], as_index=False)
        .agg(number=("val", "sum"))
    )
    rate = gbd.query("metric == 3").copy()
    rate["weight"] = rate["age"].map(AGE_WEIGHTS)
    rate["weighted_rate"] = rate["val"] * rate["weight"]
    asr = (
        rate.groupby(["cause", "measure", "location", "sex", "year"], as_index=False)
        .agg(asr=("weighted_rate", "sum"))
    )
    return number, asr


def summarize_oecd(
    gbd: pd.DataFrame,
    derived_pop: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    number_age = (
        gbd.query("metric == 1")
        .groupby(["cause", "measure", "sex", "age", "year"], as_index=False)
        .agg(number=("val", "sum"))
    )
    pop_age = (
        derived_pop.groupby(["sex", "age", "year"], as_index=False)
        .agg(population=("population", "sum"))
    )
    age_rate = number_age.merge(pop_age, on=["sex", "age", "year"], validate="many_to_one")
    age_rate["rate"] = age_rate["number"] / age_rate["population"] * 100000.0
    age_rate["weight"] = age_rate["age"].map(AGE_WEIGHTS)
    age_rate["weighted_rate"] = age_rate["rate"] * age_rate["weight"]
    oecd_asr = (
        age_rate.groupby(["cause", "measure", "sex", "year"], as_index=False)
        .agg(asr=("weighted_rate", "sum"))
    )
    oecd_number = (
        number_age.groupby(["cause", "measure", "sex", "year"], as_index=False)
        .agg(number=("number", "sum"))
    )
    return oecd_number, oecd_asr, age_rate


UI_DRAWS = 1000
UI_SEED = 20260730
UI_ASSUMPTION = (
    "Monte Carlo propagation of published GBD intervals: split-normal draws with one standard-normal deviate "
    "per location-measure-draw (perfectly correlated across ages, sexes and years within a country; independent "
    "between countries). Not official GBD joint uncertainty intervals."
)


def location_deviates(locations) -> dict:
    """One deviate per measure, location and draw; identical wherever pooled uncertainty is computed."""
    rng = np.random.default_rng(UI_SEED)
    return {(measure, int(location)): rng.standard_normal(UI_DRAWS)
            for measure in sorted(MEASURE_NAMES) for location in sorted(int(x) for x in locations)}


def split_normal_samples(group: pd.DataFrame, measure: int, deviates: dict) -> np.ndarray:
    """Draws of `val` for each row: val + z * (upper half-width if z > 0, else lower half-width) / 1.96."""
    z = np.vstack([deviates[(int(measure), int(location))] for location in group["location"]])
    value = group["val"].to_numpy()[:, None]
    upper = (group["upper"].to_numpy()[:, None] - value) / 1.96
    lower = (value - group["lower"].to_numpy()[:, None]) / 1.96
    return value + z * np.where(z > 0, upper, lower)


def approximate_oecd_ui(
    gbd: pd.DataFrame,
    derived_pop: pd.DataFrame,
    draws: int = 1000,
) -> pd.DataFrame:
    deviates = location_deviates(gbd["location"].unique())
    source = gbd.query("cause == 491 and measure in [1, 2] and sex == 3 and metric == 1").copy()
    population = (
        derived_pop.query("sex == 3")
        .groupby(["age", "year"], as_index=False)
        .agg(population=("population", "sum"))
    )
    rows = []
    for (measure, year), group in source.groupby(["measure", "year"]):
        group = group.sort_values(["age", "location"]).copy()
        samples = split_normal_samples(group, measure, deviates)
        count_draws = samples.sum(axis=0)
        asr_draws = np.zeros(draws)
        for age in AGE_ORDER:
            idx = group["age"].to_numpy() == age
            age_population = float(population.query("age == @age and year == @year")["population"].iloc[0])
            asr_draws += AGE_WEIGHTS[age] * samples[idx].sum(axis=0) / age_population * 100000.0
        rows.append(
            {
                "measure": measure,
                "year": year,
                "number_lower": float(np.quantile(count_draws, 0.025)),
                "number_upper": float(np.quantile(count_draws, 0.975)),
                "asr_lower": float(np.quantile(asr_draws, 0.025)),
                "asr_upper": float(np.quantile(asr_draws, 0.975)),
                "draws": draws,
                "assumption": UI_ASSUMPTION,
            }
        )
    return pd.DataFrame(rows)


def fit_eapc(series: pd.DataFrame, value_col: str) -> dict:
    series = series.sort_values("year")
    y = np.log(series[value_col].to_numpy())
    x = sm.add_constant(series["year"].to_numpy())
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    low, high = model.conf_int(alpha=0.05)[1]
    return {
        "eapc": 100 * (math.exp(beta) - 1),
        "eapc_lower": 100 * (math.exp(low) - 1),
        "eapc_upper": 100 * (math.exp(high) - 1),
        "p_value": float(model.pvalues[1]),
        "r_squared": float(model.rsquared),
    }


def calculate_country_trends(
    country_asr: pd.DataFrame,
    country_number: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    source = country_asr.query("cause == 491 and measure in [1, 2] and sex == 3")
    rows = []
    for (measure, location), group in source.groupby(["measure", "location"]):
        result = fit_eapc(group, "asr")
        n = country_number.query(
            "cause == 491 and measure == @measure and sex == 3 and location == @location and year in [1990, 2023]"
        ).set_index("year")["number"]
        r = group.query("year in [1990, 2023]").set_index("year")["asr"]
        rows.append(
            {
                "measure": measure,
                "location_id": location,
                "asr_1990": r.loc[1990],
                "asr_2023": r.loc[2023],
                "asr_percent_change": (r.loc[2023] / r.loc[1990] - 1) * 100,
                "number_1990": n.loc[1990],
                "number_2023": n.loc[2023],
                "number_percent_change": (n.loc[2023] / n.loc[1990] - 1) * 100,
                **result,
            }
        )
    trends = pd.DataFrame(rows)
    trends["fdr_q_value"] = np.nan
    for measure, idx in trends.groupby("measure").groups.items():
        trends.loc[idx, "fdr_q_value"] = multipletests(trends.loc[idx, "p_value"], method="fdr_bh")[1]
    trends["trend"] = np.where(
        trends["eapc_upper"] < 0,
        "Decreasing",
        np.where(trends["eapc_lower"] > 0, "Increasing", "Stable"),
    )
    trends = trends.merge(
        crosswalk[["location_id", "location_name", "iso3"]], on="location_id", how="left"
    )
    trends["measure_name"] = trends["measure"].map(MEASURE_NAMES)
    columns = [
        "location_id",
        "location_name",
        "iso3",
        "measure",
        "measure_name",
        "asr_1990",
        "asr_2023",
        "asr_percent_change",
        "number_1990",
        "number_2023",
        "number_percent_change",
        "eapc",
        "eapc_lower",
        "eapc_upper",
        "p_value",
        "fdr_q_value",
        "r_squared",
        "trend",
    ]
    return trends[columns].sort_values(["measure", "asr_2023"], ascending=[True, False])


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def convergence_analysis(
    country_asr: pd.DataFrame,
    derived_pop: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p65 = (
        derived_pop.query("sex == 3")
        .groupby(["location", "year"], as_index=False)
        .agg(population_65plus=("population", "sum"))
    )
    source = country_asr.query("cause == 491 and measure in [1, 2] and sex == 3").merge(
        p65, on=["location", "year"], how="left", validate="many_to_one"
    )
    time_rows = []
    model_rows = []
    for measure, block in source.groupby("measure"):
        for year, group in block.groupby("year"):
            log_rate = np.log(group["asr"].to_numpy())
            raw_rate = group["asr"].to_numpy()
            weights = group["population_65plus"].to_numpy()
            time_rows.append(
                {
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": year,
                    "sd_log_rate": float(np.std(log_rate, ddof=1)),
                    "iqr_log_rate": float(np.subtract(*np.percentile(log_rate, [75, 25]))),
                    "cv_rate": float(np.std(raw_rate, ddof=1) / np.mean(raw_rate)),
                    "population_weighted_sd_log_rate": weighted_sd(log_rate, weights),
                }
            )

        wide = block.pivot(index="location", columns="year", values="asr")
        beta_df = pd.DataFrame(
            {
                "location": wide.index,
                "log_baseline": np.log(wide[1990].to_numpy()),
                "annualized_log_change_percent": 100
                * (np.log(wide[2023].to_numpy()) - np.log(wide[1990].to_numpy()))
                / 33,
            }
        )
        beta_model = sm.OLS(
            beta_df["annualized_log_change_percent"],
            sm.add_constant(beta_df["log_baseline"]),
        ).fit(cov_type="HC3")
        ci = beta_model.conf_int().loc["log_baseline"]
        model_rows.append(
            {
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "model": "Cross-sectional beta convergence, 1990-2023",
                "coefficient": beta_model.params["log_baseline"],
                "lower_95ci": ci.iloc[0],
                "upper_95ci": ci.iloc[1],
                "p_value": beta_model.pvalues["log_baseline"],
                "n": len(beta_df),
                "interpretation": "Negative coefficient indicates faster proportional improvement from higher baseline burden.",
            }
        )

        panel = block.merge(crosswalk[["location_id", "iso3"]], left_on="location", right_on="location_id")
        panel = panel.sort_values(["location", "year"])
        panel["log_rate"] = np.log(panel["asr"])
        panel["lag_log_rate"] = panel.groupby("location")["log_rate"].shift(1)
        panel["delta_log_rate"] = panel.groupby("location")["log_rate"].diff()
        panel = panel.dropna(subset=["lag_log_rate", "delta_log_rate"])
        panel_model = smf.ols(
            "delta_log_rate ~ lag_log_rate + C(location) + C(year)", data=panel
        ).fit(cov_type="cluster", cov_kwds={"groups": panel["location"]})
        ci2 = panel_model.conf_int().loc["lag_log_rate"]
        model_rows.append(
            {
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "model": "Country-year panel beta convergence with country and year fixed effects",
                "coefficient": panel_model.params["lag_log_rate"],
                "lower_95ci": ci2.iloc[0],
                "upper_95ci": ci2.iloc[1],
                "p_value": panel_model.pvalues["lag_log_rate"],
                "n": int(panel_model.nobs),
                "interpretation": "Negative coefficient indicates conditional convergence; standard errors clustered by country.",
            }
        )
    return pd.DataFrame(time_rows), pd.DataFrame(model_rows)


def trajectory_clustering(
    country_asr: pd.DataFrame,
    crosswalk: pd.DataFrame,
    bootstrap_reps: int = 500,
    measures: tuple[int, int] = (1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    blocks = []
    years = list(range(1991, 2024))
    for measure in measures:
        wide = (
            country_asr.query("cause == 491 and measure == @measure and sex == 3")
            .pivot(index="location", columns="year", values="asr")
            .sort_index()
        )
        change = np.log(wide[years].to_numpy()) - np.log(wide[[1990]].to_numpy())
        blocks.append(change)
    features = np.concatenate(blocks, axis=1)
    standardized = StandardScaler().fit_transform(features)
    pca_full = PCA().fit(standardized)
    n_components = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.90) + 1)
    scores = PCA(n_components=n_components).fit_transform(standardized)
    candidates = []
    labels_by_k = {}
    for k in (3, 4, 5):
        labels = KMeans(n_clusters=k, n_init=100, random_state=42).fit_predict(scores)
        silhouette = silhouette_score(scores, labels)
        candidates.append({"k": k, "silhouette": silhouette})
        labels_by_k[k] = labels
    best_k = max(candidates, key=lambda x: x["silhouette"])["k"]
    raw_labels = labels_by_k[best_k]

    primary_measure = measures[0]
    locations = (
        country_asr.query("cause == 491 and measure == @primary_measure and sex == 3")
        .pivot(index="location", columns="year", values="asr")
        .sort_index()
        .index.to_numpy()
    )
    improvement = features[:, len(years) - 1] + features[:, -1]
    order = sorted(np.unique(raw_labels), key=lambda lab: improvement[raw_labels == lab].mean())
    remap = {old: new + 1 for new, old in enumerate(order)}
    labels = np.array([remap[x] for x in raw_labels])

    rng = np.random.default_rng(42)
    ari_values = []
    n_years = len(years)
    for _ in range(bootstrap_reps):
        sampled = rng.integers(0, n_years, size=n_years)
        sampled_features = np.concatenate([blocks[0][:, sampled], blocks[1][:, sampled]], axis=1)
        sampled_standard = StandardScaler().fit_transform(sampled_features)
        n_pc = min(n_components, sampled_standard.shape[0] - 1, sampled_standard.shape[1])
        sampled_scores = PCA(n_components=n_pc).fit_transform(sampled_standard)
        sampled_labels = KMeans(n_clusters=best_k, n_init=30, random_state=int(rng.integers(1, 2**31 - 1))).fit_predict(sampled_scores)
        ari_values.append(adjusted_rand_score(raw_labels, sampled_labels))

    cluster_df = pd.DataFrame({"location_id": locations, "cluster": labels}).merge(
        crosswalk[["location_id", "location_name", "iso3"]], on="location_id", how="left"
    )
    index_labels = {1: "asmr_index_2023", 2: "asdr_index_2023", 6: "asir_index_2023"}
    for measure in measures:
        label = index_labels[measure]
        wide = (
            country_asr.query("cause == 491 and measure == @measure and sex == 3")
            .pivot(index="location", columns="year", values="asr")
        )
        idx = (wide[2023] / wide[1990] * 100).rename(label)
        cluster_df = cluster_df.merge(idx, left_on="location_id", right_index=True)
    cluster_df = cluster_df.sort_values(["cluster", "iso3"])

    cluster_series = []
    source = country_asr.query("cause == 491 and measure in @measures and sex == 3").merge(
        cluster_df[["location_id", "cluster"]], left_on="location", right_on="location_id"
    )
    source["index"] = source["asr"] / source.groupby(["measure", "location"])["asr"].transform("first") * 100
    for keys, group in source.groupby(["measure", "cluster", "year"]):
        cluster_series.append(
            {
                "measure": keys[0],
                "measure_name": MEASURE_NAMES[keys[0]],
                "cluster": keys[1],
                "year": keys[2],
                "mean_index": group["index"].mean(),
                "lower_quartile": group["index"].quantile(0.25),
                "upper_quartile": group["index"].quantile(0.75),
                "countries": group["location"].nunique(),
            }
        )
    diagnostics = {
        "measures": [MEASURE_NAMES[measure] for measure in measures],
        "selected_k": int(best_k),
        "silhouette_by_k": candidates,
        "pca_components_for_90pct_variance": n_components,
        "bootstrap_repetitions": bootstrap_reps,
        "bootstrap_mean_adjusted_rand_index": float(np.mean(ari_values)),
        "bootstrap_ari_2_5_percentile": float(np.quantile(ari_values, 0.025)),
        "bootstrap_ari_97_5_percentile": float(np.quantile(ari_values, 0.975)),
    }
    return cluster_df, pd.DataFrame(cluster_series), diagnostics


def segmented_joinpoint(years: np.ndarray, values: np.ndarray, max_breaks: int = 2, min_segment: int = 5) -> list[dict]:
    x = years.astype(float)
    y = np.log(values.astype(float))
    candidates: list[tuple[int, ...]] = [tuple()]
    interior = list(range(min_segment, len(x) - min_segment + 1))
    candidates += [(idx,) for idx in interior]
    for first in interior:
        for second in interior:
            if second - first >= min_segment and len(x) - second >= min_segment:
                candidates.append((first, second))

    best = None
    for breaks_idx in candidates:
        knots = [x[idx] for idx in breaks_idx]
        design_cols = [np.ones(len(x)), x]
        design_cols += [np.maximum(0, x - knot) for knot in knots]
        design = np.column_stack(design_cols)
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        rss = max(float(residual @ residual), 1e-15)
        p = design.shape[1]
        bic = len(x) * math.log(rss / len(x)) + p * math.log(len(x))
        if best is None or bic < best["bic"]:
            best = {"bic": bic, "knots": knots, "design": design, "coef": coef, "rss": rss}

    design = best["design"]
    coef = best["coef"]
    dof = len(x) - design.shape[1]
    sigma2 = best["rss"] / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    knots = best["knots"]
    boundaries = [x.min()] + knots + [x.max()]
    segments = []
    for i in range(len(boundaries) - 1):
        contrast = np.zeros(len(coef))
        contrast[1] = 1
        if i > 0:
            contrast[2 : 2 + i] = 1
        slope = float(contrast @ coef)
        se = float(np.sqrt(contrast @ cov @ contrast))
        low = slope - 1.96 * se
        high = slope + 1.96 * se
        segments.append(
            {
                "segment": i + 1,
                "start_year": int(boundaries[i]),
                "end_year": int(boundaries[i + 1]),
                "apc": 100 * (math.exp(slope) - 1),
                "apc_lower": 100 * (math.exp(low) - 1),
                "apc_upper": 100 * (math.exp(high) - 1),
                "selected_breakpoints": ";".join(str(int(k)) for k in knots) if knots else "None",
                "bic": best["bic"],
            }
        )
    return segments


def joinpoint_table(
    country_asr: pd.DataFrame,
    oecd_asr: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    source = country_asr.query("cause == 491 and measure in [1, 2] and sex == 3")
    for (measure, location), group in source.groupby(["measure", "location"]):
        for segment in segmented_joinpoint(group["year"].to_numpy(), group["asr"].to_numpy()):
            rows.append({"scope": "Country", "location_id": location, "measure": measure, **segment})
    for measure, group in oecd_asr.query("cause == 491 and measure in [1, 2] and sex == 3").groupby("measure"):
        for segment in segmented_joinpoint(group["year"].to_numpy(), group["asr"].to_numpy()):
            rows.append({"scope": "OECD aggregate", "location_id": 0, "measure": measure, **segment})
    table = pd.DataFrame(rows).merge(
        crosswalk[["location_id", "location_name", "iso3"]], on="location_id", how="left"
    )
    table.loc[table["scope"] == "OECD aggregate", ["location_name", "iso3"]] = ["OECD 38 aggregate", "OECD38"]
    table["measure_name"] = table["measure"].map(MEASURE_NAMES)
    table["method_note"] = "Exploratory continuous segmented log-linear regression selected by BIC; not the NCI permutation-test Joinpoint software."
    return table.sort_values(["measure", "scope", "iso3", "segment"])


def das_gupta_burden(factors: dict) -> float:
    return float(
        factors["total_population"]
        * np.dot(factors["age_structure"], factors["age_specific_rate"])
    )


def _das_gupta_permutation_effects(start: dict, end: dict) -> dict:
    names = ["total_population", "age_structure", "age_specific_rate"]
    contributions = dict.fromkeys(names, 0.0)
    for order in itertools.permutations(names):
        state = {
            "total_population": start["total_population"],
            "age_structure": start["age_structure"].copy(),
            "age_specific_rate": start["age_specific_rate"].copy(),
        }
        previous = das_gupta_burden(state)
        for name in order:
            state[name] = end[name].copy() if isinstance(end[name], np.ndarray) else end[name]
            current = das_gupta_burden(state)
            contributions[name] += current - previous
            previous = current
    for name in names:
        contributions[name] /= math.factorial(len(names))
    return contributions


def das_gupta_contributions(start: dict, end: dict) -> dict:
    """Symmetric three-factor Das Gupta decomposition over all 3! orders."""
    n0, n1 = start["total_population"], end["total_population"]
    w0, w1 = start["age_structure"], end["age_structure"]
    r0, r1 = start["age_specific_rate"], end["age_specific_rate"]
    permutation = _das_gupta_permutation_effects(start, end)

    m00, m11 = np.dot(w0, r0), np.dot(w1, r1)
    m10, m01 = np.dot(w1, r0), np.dot(w0, r1)
    explicit = {
        "total_population": (n1 - n0) * ((m00 + m11) / 3 + (m10 + m01) / 6),
        "age_structure": float(
            np.dot(
                w1 - w0,
                (n0 * r0 + n1 * r1) / 3 + (n1 * r0 + n0 * r1) / 6,
            )
        ),
        "age_specific_rate": float(
            np.dot(
                r1 - r0,
                (n0 * w0 + n1 * w1) / 3 + (n1 * w0 + n0 * w1) / 6,
            )
        ),
    }
    formula_difference = max(abs(permutation[name] - explicit[name]) for name in explicit)
    if formula_difference > 1e-6:
        raise ValueError(f"Das Gupta explicit and permutation formulas disagree by {formula_difference}")

    reverse = _das_gupta_permutation_effects(end, start)
    reversal_error = max(abs(permutation[name] + reverse[name]) for name in permutation)
    if reversal_error > 1e-6:
        raise ValueError(f"Das Gupta reversal symmetry failed by {reversal_error}")

    burden_1990 = das_gupta_burden(start)
    burden_2023 = das_gupta_burden(end)
    total_change = burden_2023 - burden_1990
    additivity_error = sum(permutation.values()) - total_change
    if abs(additivity_error) > 1e-6:
        raise ValueError(f"Das Gupta additivity failed by {additivity_error}")

    growth_state = {
        "total_population": n1,
        "age_structure": w0,
        "age_specific_rate": r0,
    }
    growth_ageing_state = {
        "total_population": n1,
        "age_structure": w1,
        "age_specific_rate": r0,
    }
    growth_counterfactual = das_gupta_burden(growth_state)
    growth_ageing_counterfactual = das_gupta_burden(growth_ageing_state)
    return {
        "population_growth": permutation["total_population"],
        "population_ageing": permutation["age_structure"],
        "epidemiological_change": permutation["age_specific_rate"],
        "burden_1990_reconstructed": burden_1990,
        "burden_2023_reconstructed": burden_2023,
        "total_change": total_change,
        "additivity_error": additivity_error,
        "explicit_permutation_max_abs_difference": formula_difference,
        "reversal_symmetry_max_abs_error": reversal_error,
        "fixed_order_population_growth": growth_counterfactual - burden_1990,
        "fixed_order_population_ageing": growth_ageing_counterfactual - growth_counterfactual,
        "fixed_order_epidemiological_change": burden_2023 - growth_ageing_counterfactual,
    }


def make_das_gupta_state(
    gbd: pd.DataFrame,
    derived_pop: pd.DataFrame,
    pop: pd.DataFrame,
    cause: int,
    measure: int,
    year: int,
    location: int | None,
) -> dict:
    pop_age = derived_pop.query("sex == 3 and year == @year")
    if location is not None:
        pop_age = pop_age.query("location == @location")
        n_total = float(
            pop.query("location_id == @location and sex_id == 3 and age_id == 22 and year == @year")["val"].iloc[0]
        )
    else:
        n_total = float(
            pop.query("location_id in @derived_pop.location.unique() and sex_id == 3 and age_id == 22 and year == @year")["val"].sum()
        )
    population_vector = pop_age.groupby("age")["population"].sum().reindex(AGE_ORDER).to_numpy()
    number_age = gbd.query(
        "cause == @cause and measure == @measure and sex == 3 and metric == 1 and year == @year"
    )
    if location is not None:
        number_age = number_age.query("location == @location")
    number_vector = number_age.groupby("age")["val"].sum().reindex(AGE_ORDER).to_numpy()
    if np.isnan(population_vector).any() or np.isnan(number_vector).any() or (population_vector <= 0).any():
        raise ValueError("Incomplete or invalid age-specific inputs for Das Gupta decomposition")
    return {
        "total_population": n_total,
        "age_structure": population_vector / n_total,
        "age_specific_rate": number_vector / population_vector,
    }


def das_gupta_table(
    gbd: pd.DataFrame,
    derived_pop: pd.DataFrame,
    pop: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    cause = 491
    for measure in [6, 2]:
        for location in list(crosswalk["location_id"]) + [None]:
            start = make_das_gupta_state(gbd, derived_pop, pop, cause, measure, 1990, location)
            end = make_das_gupta_state(gbd, derived_pop, pop, cause, measure, 2023, location)
            values = das_gupta_contributions(start, end)
            rows.append(
                {
                    "scope": "OECD aggregate" if location is None else "Country",
                    "location_id": 0 if location is None else location,
                    "cause_id": cause,
                    "cause": CAUSE_NAMES[cause],
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    **values,
                }
            )
    table = pd.DataFrame(rows).merge(
        crosswalk[["location_id", "location_name", "iso3"]], on="location_id", how="left"
    )
    table.loc[table["scope"] == "OECD aggregate", ["location_name", "iso3"]] = ["OECD 38 aggregate", "OECD38"]
    for col in ["population_growth", "population_ageing", "epidemiological_change", "total_change"]:
        table[col + "_percent_of_1990"] = table[col] / table["burden_1990_reconstructed"] * 100
    table["method"] = "Symmetric three-factor Das Gupta decomposition averaged over all 3! replacement orders"
    return table.sort_values(["measure", "scope", "iso3"])


def context_table(
    derived_pop: pd.DataFrame,
    pop: pd.DataFrame,
    sdi: pd.DataFrame,
    gdp: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    age = (
        derived_pop.query("sex == 3 and year == 2023")
        .assign(is_80plus=lambda x: x["age"].isin([30, 31, 32, 235]))
        .groupby("location", as_index=False)
        .agg(population_65plus=("population", "sum"), population_80plus=("population", lambda x: x[derived_pop.loc[x.index, "age"].isin([30, 31, 32, 235])].sum()))
    )
    total = pop.query("sex_id == 3 and age_id == 22 and year == 2023")[["location_id", "val"]].rename(columns={"val": "total_population"})
    table = crosswalk.merge(age, left_on="location_id", right_on="location").merge(total, on="location_id")
    table["share_65plus_percent"] = table["population_65plus"] / table["total_population"] * 100
    table["share_80plus_percent"] = table["population_80plus"] / table["total_population"] * 100
    sdi_2023 = sdi.query("year == 2023")[["location_id", "sdi"]].dropna(subset=["location_id"]).copy()
    sdi_2023["location_id"] = sdi_2023["location_id"].astype(int)
    gdp_2023 = gdp.query("year == 2023")[["iso3c", "NY.GDP.PCAP.PP.CD", "NY.GDP.MKTP.PP.CD"]].rename(
        columns={"iso3c": "iso3", "NY.GDP.PCAP.PP.CD": "gdp_per_capita_ppp_current_intl_dollar", "NY.GDP.MKTP.PP.CD": "gdp_ppp_current_intl_dollar"}
    )
    table = table.merge(sdi_2023, on="location_id", how="left").merge(gdp_2023, on="iso3", how="left")
    return table[
        [
            "location_id",
            "location_name",
            "iso3",
            "total_population",
            "population_65plus",
            "population_80plus",
            "share_65plus_percent",
            "share_80plus_percent",
            "sdi",
            "gdp_per_capita_ppp_current_intl_dollar",
            "gdp_ppp_current_intl_dollar",
        ]
    ].sort_values("iso3")


def performance_quadrants(trends: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    table = trends.merge(clusters[["location_id", "cluster"]], on="location_id", how="left")
    decline = table["eapc_upper"] < 0
    count_up = table["number_percent_change"] >= 0
    table["quadrant"] = np.select(
        [decline & ~count_up, decline & count_up, ~decline & count_up, ~decline & ~count_up],
        [
            "A: Rate down, count down",
            "B: Rate down, count up",
            "C: Rate stable/up, count up",
            "D: Rate stable/up, count down",
        ],
    )
    return table.sort_values(["measure", "quadrant", "iso3"])


def figure6_sdi_data(
    country_asr: pd.DataFrame,
    sdi: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Join 65+ standardized CVD rates to annual SDI for Figure 6."""
    rates = country_asr.query("cause == 491 and measure in [2, 6] and sex == 3").copy()
    oecd_location_ids = set(crosswalk["location_id"])
    sdi_source = sdi.query(
        "1990 <= year <= 2023 and location_id in @oecd_location_ids"
    )[["location_id", "year", "sdi"]].dropna().copy()
    sdi_source["location_id"] = sdi_source["location_id"].astype(int)
    table = (
        rates.merge(
            crosswalk[["location_id", "location_name", "iso3"]],
            left_on="location",
            right_on="location_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(sdi_source, on=["location_id", "year"], how="inner", validate="many_to_one")
        .rename(columns={"asr": "rate_per_100000"})
    )
    table["measure_name"] = table["measure"].map(MEASURE_NAMES)
    expected_rows = 38 * 34 * 2
    if len(table) != expected_rows:
        raise ValueError(f"Figure 6 expected {expected_rows} country-year-measure rows; found {len(table)}")
    coverage = table.groupby(["measure", "location_id"])["year"].nunique()
    if len(coverage) != 76 or not coverage.eq(34).all():
        raise ValueError("Figure 6 does not have complete 1990-2023 coverage for all OECD-38 countries")
    return table[
        [
            "location_id",
            "location_name",
            "iso3",
            "year",
            "sdi",
            "measure",
            "measure_name",
            "rate_per_100000",
        ]
    ].sort_values(["measure", "iso3", "year"])


def oecd_selected_summary(
    oecd_number: pd.DataFrame,
    oecd_asr: pd.DataFrame,
    country_asr: pd.DataFrame,
    ui: pd.DataFrame,
) -> pd.DataFrame:
    base = oecd_number.query("cause == 491 and measure in [1,2] and sex == 3 and year in @SELECTED_YEARS").merge(
        oecd_asr.query("cause == 491 and measure in [1,2] and sex == 3 and year in @SELECTED_YEARS"),
        on=["cause", "measure", "sex", "year"],
    ).merge(ui, on=["measure", "year"], how="left")
    med = (
        country_asr.query("cause == 491 and measure in [1,2] and sex == 3 and year in @SELECTED_YEARS")
        .groupby(["measure", "year"])["asr"]
        .agg(country_median_asr="median", country_q1_asr=lambda x: x.quantile(0.25), country_q3_asr=lambda x: x.quantile(0.75))
        .reset_index()
    )
    base = base.merge(med, on=["measure", "year"])
    base["measure_name"] = base["measure"].map(MEASURE_NAMES)
    return base.sort_values(["measure", "year"])


def subtype_age_sex_table(gbd: pd.DataFrame, derived_pop: pd.DataFrame) -> pd.DataFrame:
    cause_ids = [491, 493, 495, 496, 498]
    number = (
        gbd.query("cause in @cause_ids and measure in [2,6] and metric == 1 and year == 2023")
        .groupby(["cause", "measure", "sex", "age", "year"], as_index=False)
        .agg(number=("val", "sum"))
    )
    population = (
        derived_pop.groupby(["sex", "age", "year"], as_index=False)
        .agg(population=("population", "sum"))
    )
    table = number.merge(population, on=["sex", "age", "year"])
    table["age_specific_rate"] = table["number"] / table["population"] * 100000
    table["cause_name"] = table["cause"].map(CAUSE_NAMES)
    table["cause_short"] = table["cause"].map(CAUSE_SHORT)
    table["measure_name"] = table["measure"].map(MEASURE_NAMES)
    table["sex_name"] = table["sex"].map(SEX_NAMES)
    table["age_name"] = table["age"].map(AGE_NAMES)
    return table.sort_values(["measure", "year", "cause", "sex", "age"])


def figure3_age_sex_uncertainty(
    gbd: pd.DataFrame,
    derived_pop: pd.DataFrame,
    draws: int = 1000,
) -> pd.DataFrame:
    deviates = location_deviates(gbd["location"].unique())
    source = gbd.query(
        "cause == 491 and measure in [2,6] and sex in [1,2] and metric == 1 and year == 2023"
    ).copy()
    population = (
        derived_pop.query("sex in [1,2] and year == 2023")
        .groupby(["sex", "age"], as_index=False)
        .agg(population=("population", "sum"))
        .set_index(["sex", "age"])["population"]
    )
    rows = []
    for (measure, sex), group in source.groupby(["measure", "sex"]):
        group = group.sort_values(["age", "location"]).copy()
        samples = split_normal_samples(group, measure, deviates)
        age_count_draws = np.vstack(
            [samples[group["age"].to_numpy() == age].sum(axis=0) for age in AGE_ORDER]
        )
        age_share_draws = age_count_draws / age_count_draws.sum(axis=0) * 100
        for age_index, age in enumerate(AGE_ORDER):
            rate_draws = age_count_draws[age_index] / float(population.loc[(sex, age)]) * 100000
            rows.append(
                {
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "sex": sex,
                    "sex_name": SEX_NAMES[sex],
                    "age": age,
                    "age_name": AGE_NAMES[age],
                    "rate_lower": float(np.quantile(rate_draws, 0.025)),
                    "rate_upper": float(np.quantile(rate_draws, 0.975)),
                    "age_share_lower": float(np.quantile(age_share_draws[age_index], 0.025)),
                    "age_share_upper": float(np.quantile(age_share_draws[age_index], 0.975)),
                    "draws": draws,
                    "assumption": UI_ASSUMPTION,
                }
            )
    return pd.DataFrame(rows).sort_values(["measure", "sex", "age"])


def transition_indices(gbd: pd.DataFrame) -> pd.DataFrame:
    count = (
        gbd.query("metric == 1 and sex == 3 and measure in [1,2,3,4]")
        .groupby(["cause", "measure", "age", "year"], as_index=False)
        .agg(number=("val", "sum"))
    )
    totals = count.groupby(["cause", "measure", "year"], as_index=False)["number"].sum()
    old = (
        count.query("age in [30,31,32,235]")
        .groupby(["cause", "measure", "year"], as_index=False)["number"]
        .sum()
        .rename(columns={"number": "number_80plus"})
    )
    wide = totals.merge(old, on=["cause", "measure", "year"])
    wide["oldest_old_concentration"] = wide["number_80plus"] / wide["number"]
    ratio = totals.pivot(index=["cause", "year"], columns="measure", values="number")
    required_measures = {1, 2, 3, 4}
    if not required_measures.issubset(ratio.columns):
        raise ValueError("Figure 7 requires Deaths, DALYs, YLDs, and YLLs for every cause")
    ratio = ratio.assign(
        yld_to_daly_ratio=ratio[3] / ratio[2],
        yll_to_daly_ratio=ratio[4] / ratio[2],
    )[["yld_to_daly_ratio", "yll_to_daly_ratio"]].reset_index()
    table = wide.merge(ratio, on=["cause", "year"], validate="many_to_one")
    table["cause_name"] = table["cause"].map(CAUSE_NAMES)
    table["cause_short"] = table["cause"].map(CAUSE_SHORT)
    table["measure_name"] = table["measure"].map(MEASURE_NAMES)
    return table.sort_values(["cause", "measure", "year"])


def sensitivity_table(gbd: pd.DataFrame, derived_pop: pd.DataFrame) -> pd.DataFrame:
    thresholds = {"65+": AGE_ORDER, "70+": [19, 20, 30, 31, 32, 235], "80+": [30, 31, 32, 235]}
    rows = []
    for label, ages in thresholds.items():
        weights = np.array([AGE_WEIGHTS[a] for a in ages])
        weights = weights / weights.sum()
        for measure in (1, 2):
            number = (
                gbd.query("cause == 491 and measure == @measure and sex == 3 and metric == 1 and age in @ages")
                .groupby(["age", "year"], as_index=False)["val"].sum()
            )
            population = (
                derived_pop.query("sex == 3 and age in @ages")
                .groupby(["age", "year"], as_index=False)["population"].sum()
            )
            rate = number.merge(population, on=["age", "year"])
            rate["age_rate"] = rate["val"] / rate["population"] * 100000
            rates = []
            counts = []
            for year, group in rate.groupby("year"):
                aligned = group.set_index("age").reindex(ages)
                rates.append({"year": year, "asr": float(np.dot(weights, aligned["age_rate"]))})
                counts.append({"year": year, "number": float(aligned["val"].sum())})
            rate_df = pd.DataFrame(rates)
            count_df = pd.DataFrame(counts).set_index("year")
            fit = fit_eapc(rate_df, "asr")
            rate_index = rate_df.set_index("year")["asr"]
            rows.append(
                {
                    "age_threshold": label,
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "asr_1990": rate_index.loc[1990],
                    "asr_2023": rate_index.loc[2023],
                    "asr_percent_change": (rate_index.loc[2023] / rate_index.loc[1990] - 1) * 100,
                    "number_1990": count_df.loc[1990, "number"],
                    "number_2023": count_df.loc[2023, "number"],
                    "number_percent_change": (count_df.loc[2023, "number"] / count_df.loc[1990, "number"] - 1) * 100,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def input_hashes() -> list[dict]:
    rows = []
    for path in sorted(DATA.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append({"file": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return rows


def _geometry_patches(feature: dict) -> list[MplPolygon]:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates", [])
    patches = []
    if geometry.get("type") == "Polygon":
        polygons = [coordinates]
    elif geometry.get("type") == "MultiPolygon":
        polygons = coordinates
    else:
        polygons = []
    for polygon in polygons:
        if polygon and polygon[0]:
            patches.append(MplPolygon(np.asarray(polygon[0]), closed=True))
    return patches


def figure1_baseline(
    country_asr: pd.DataFrame,
    geo: dict,
    crosswalk: pd.DataFrame,
) -> None:
    panel_specs = [
        (6, "Incidence", "A"),
        (5, "Prevalence", "B"),
        (1, "Deaths", "C"),
        (2, "DALYs", "D"),
    ]
    cvd = (
        country_asr.query("cause == 491 and year == 2023 and sex == 3 and measure in [1, 2, 5, 6]")
        .merge(crosswalk, left_on="location", right_on="location_id", how="left")
    )
    if cvd.groupby("measure")["location"].nunique().to_dict() != {1: 38, 2: 38, 5: 38, 6: 38}:
        raise ValueError("Figure 1 requires 2023 both-sex data for four measures across all OECD-38 countries")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.8))
    cmap = mpl.colormaps["YlOrRd"]
    output_rows = []
    for ax, (measure, measure_name, label) in zip(axes.flat, panel_specs):
        block = cvd.query("measure == @measure").copy()
        value_map = block.set_index("iso3")["asr"].to_dict()
        norm = colors.Normalize(vmin=float(block["asr"].min()), vmax=float(block["asr"].max()))
        world_patches = []
        world_colors = []
        for feature in geo["features"]:
            iso3 = feature["properties"].get("brk_a3")
            for patch in _geometry_patches(feature):
                world_patches.append(patch)
                world_colors.append(cmap(norm(value_map[iso3])) if iso3 in value_map and pd.notna(value_map[iso3]) else "#FFFFFF")
        collection = PatchCollection(
            world_patches,
            facecolor=world_colors,
            edgecolor="#111111",
            linewidth=0.28,
            antialiased=True,
        )
        ax.add_collection(collection)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 84)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_title(f"({label.lower()}) {measure_name}", loc="left", pad=8, fontweight="bold")
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            fraction=0.048,
            pad=0.025,
            aspect=35,
        )
        colorbar.set_label(f"{measure_name} (per 100,000)")
        colorbar.outline.set_edgecolor("#111111")
        colorbar.outline.set_linewidth(0.6)
        panel_data = block[["location", "location_name", "iso3", "asr"]].copy()
        panel_data["measure_id"] = measure
        panel_data["measure_name"] = measure_name
        panel_data["panel"] = label
        output_rows.append(panel_data)

    fig.text(
        0.5,
        0.006,
        "2023, both sexes. Rates use the GBD 2023 world age standard renormalized within age 65+. Countries without data are shown in white.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.995], h_pad=1.8, w_pad=1.5)
    fig.savefig(FIGURES / "Figure1_2023_OECD65plus_CVD_baseline.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(pd.concat(output_rows, ignore_index=True), "Figure1_four_map_data.csv", True)


def bic_segmented_aapc_apc(
    years: np.ndarray,
    values: np.ndarray,
    max_joinpoints: int = 3,
    min_segment_years: int = 5,
) -> dict:
    years = np.asarray(years, dtype=int)
    values = np.asarray(values, dtype=float)
    order = np.argsort(years)
    years = years[order]
    values = values[order]
    if not np.array_equal(years, np.arange(years.min(), years.max() + 1)):
        raise ValueError("AAPC/APC analysis requires a complete annual series")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("ASR values must be finite and greater than zero")

    first_year = int(years.min())
    last_year = int(years.max())
    candidate_years = range(first_year + min_segment_years, last_year - min_segment_years + 1)
    knot_sets: list[tuple[int, ...]] = [tuple()]
    for n_knots in range(1, max_joinpoints + 1):
        for knots in itertools.combinations(candidate_years, n_knots):
            if np.all(np.diff([first_year, *knots, last_year]) >= min_segment_years):
                knot_sets.append(knots)

    x = years - first_year
    log_values = np.log(values)
    best = None
    for knots in knot_sets:
        design_columns = [np.ones(len(years)), x.astype(float)]
        design_columns.extend(np.maximum(0, x - (knot - first_year)) for knot in knots)
        design = np.column_stack(design_columns)
        coefficients, _, rank, _ = np.linalg.lstsq(design, log_values, rcond=None)
        residuals = log_values - design @ coefficients
        rss = float(residuals @ residuals)
        n_obs, n_parameters = design.shape
        if not np.isfinite(rss) or rss <= 0 or rank < n_parameters or n_obs <= n_parameters:
            continue
        try:
            covariance = rss / (n_obs - n_parameters) * np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError:
            continue
        bic = n_obs * math.log(rss / n_obs) + n_parameters * math.log(n_obs)
        if best is None or bic < best["bic"]:
            best = {
                "knots": knots,
                "design": design,
                "coefficients": coefficients,
                "covariance": covariance,
                "residual_df": n_obs - n_parameters,
                "bic": bic,
            }
    if best is None:
        raise ValueError("No valid segmented regression model could be fitted")

    def contrast(gradient: np.ndarray) -> dict:
        estimate = float(gradient @ best["coefficients"])
        se = float(np.sqrt(gradient @ best["covariance"] @ gradient))
        critical = float(stats.t.ppf(0.975, df=best["residual_df"]))
        p_value = float(2 * stats.t.sf(abs(estimate / se), df=best["residual_df"]))
        return {
            "estimate": 100 * (math.exp(estimate) - 1),
            "lower": 100 * (math.exp(estimate - critical * se) - 1),
            "upper": 100 * (math.exp(estimate + critical * se) - 1),
            "p_value": p_value,
        }

    boundaries = [first_year, *best["knots"], last_year]
    durations = np.diff(boundaries)
    gradients = []
    segments = []
    for index, duration in enumerate(durations):
        gradient = np.zeros(len(best["coefficients"]))
        gradient[1] = 1
        if index > 0:
            gradient[2 : 2 + index] = 1
        gradients.append(gradient)
        result = contrast(gradient)
        segments.append(
            {
                "segment": index + 1,
                "segment_start": int(boundaries[index]),
                "segment_end": int(boundaries[index + 1]),
                "duration_years": int(duration),
                "apc": result["estimate"],
                "apc_lower": result["lower"],
                "apc_upper": result["upper"],
                "apc_p_value": result["p_value"],
            }
        )
    aapc_gradient = sum(
        gradient * (duration / durations.sum())
        for gradient, duration in zip(gradients, durations)
    )
    aapc_result = contrast(aapc_gradient)
    return {
        "knots": best["knots"],
        "bic": best["bic"],
        "residual_df": best["residual_df"],
        "fitted_asr": np.exp(best["design"] @ best["coefficients"]),
        "fitted_segment": np.searchsorted(best["knots"], years, side="right") + 1,
        "segments": segments,
        "aapc": aapc_result["estimate"],
        "aapc_lower": aapc_result["lower"],
        "aapc_upper": aapc_result["upper"],
        "aapc_p_value": aapc_result["p_value"],
    }


def figure2_heatmaps(
    oecd_asr: pd.DataFrame,
    country_asr: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> None:
    measures = [
        (6, "Incidence", "ASIR", "YlGnBu", "#4C72B0", "A", "C"),
        (2, "DALYs", "ASDR", "YlOrRd", "#55A868", "B", "D"),
    ]
    aggregate = oecd_asr.query("cause == 491 and measure in [2, 6] and sex == 3").copy()
    aggregate["scope"] = "OECD aggregate"
    aggregate["location"] = 0
    aggregate["location_name"] = "OECD-38 aggregate"
    aggregate["iso3"] = "OECD38"
    countries = (
        country_asr.query("cause == 491 and measure in [2, 6] and sex == 3")
        .merge(crosswalk, left_on="location", right_on="location_id", how="left")
    )
    countries["scope"] = "OECD country"
    years = sorted(aggregate["year"].unique())
    country_order = sorted(countries["iso3"].dropna().unique())
    if len(years) != 34 or len(country_order) != 38:
        raise ValueError("Figure 2 requires 1990-2023 data for all OECD-38 countries")

    fig = plt.figure(figsize=(14.5, 12.2))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.8, 6.2, 0.20],
        hspace=0.22,
        wspace=0.18,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    tick_years = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023]
    tick_positions = [years.index(year) for year in tick_years]
    trend_statistics = []
    aggregate["fitted_asr"] = np.nan
    aggregate["segment"] = np.nan

    for column, (measure, measure_name, rate_name, cmap_name, line_color, top_label, bottom_label) in enumerate(measures):
        aggregate_block = aggregate.query("measure == @measure").set_index("year").reindex(years)
        country_matrix = (
            countries.query("measure == @measure")
            .pivot(index="iso3", columns="year", values="asr")
            .reindex(index=country_order, columns=years)
        )
        country_values = country_matrix.to_numpy().ravel()
        norm = colors.Normalize(vmin=float(np.nanmin(country_values)), vmax=float(np.nanmax(country_values)))
        cmap = mpl.colormaps[cmap_name].copy()
        cmap.set_bad("#FFFFFF")

        top_ax = axes[0, column]
        x_positions = np.arange(len(years))
        trend = bic_segmented_aapc_apc(years, aggregate_block["asr"].to_numpy())
        top_ax.plot(
            x_positions,
            aggregate_block["asr"].to_numpy(),
            color=line_color,
            linewidth=1.45,
            marker="o",
            markersize=2.2,
            label="Observed ASR",
        )
        top_ax.plot(x_positions, trend["fitted_asr"], color="#C44E52", linewidth=1.9, label="BIC segmented fit")
        for knot in trend["knots"]:
            top_ax.axvline(knot - years[0], color="#6B7280", linewidth=0.7, linestyle="--")
        top_ax.set_xlim(-0.5, len(years) - 0.5)
        top_ax.set_xticks(tick_positions, tick_years)
        top_ax.set_xlabel("Year")
        top_ax.set_ylabel(f"{rate_name} (per 100,000)")
        top_ax.set_title(f"OECD - {measure_name}", loc="left", fontweight="bold", pad=8)
        top_ax.grid(color="#E5E7EB", linewidth=0.6)
        if measure == 6:
            top_ax.legend(frameon=False, loc="upper right", ncol=2)
        y_span = float(max(aggregate_block["asr"].max(), trend["fitted_asr"].max()) - min(aggregate_block["asr"].min(), trend["fitted_asr"].min()))
        for index, segment in enumerate(trend["segments"]):
            midpoint_year = (segment["segment_start"] + segment["segment_end"]) / 2
            midpoint_x = midpoint_year - years[0]
            midpoint_y = float(np.interp(midpoint_year, years, trend["fitted_asr"]))
            label = (
                f"{segment['segment_start']}-{segment['segment_end']}\n"
                f"APC {segment['apc']:.2f}%\n"
                f"({segment['apc_lower']:.2f}% to {segment['apc_upper']:.2f}%)"
            )
            top_ax.text(
                midpoint_x,
                midpoint_y + y_span * (0.08 + 0.07 * (index % 2)),
                label,
                ha="center",
                va="bottom",
                fontsize=5.8,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#9CA3AF", linewidth=0.45, alpha=0.92),
            )
        joinpoints = ";".join(str(year) for year in trend["knots"]) if trend["knots"] else "None"
        top_ax.text(
            0.015,
            0.04,
            f"AAPC {trend['aapc']:.2f}% ({trend['aapc_lower']:.2f}% to {trend['aapc_upper']:.2f}%)\nJoinpoints: {joinpoints}",
            transform=top_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#9CA3AF", linewidth=0.45, alpha=0.92),
        )
        top_ax.set_ylim(
            min(aggregate_block["asr"].min(), trend["fitted_asr"].min()) - 0.08 * y_span,
            max(aggregate_block["asr"].max(), trend["fitted_asr"].max()) + 0.30 * y_span,
        )
        panel_label(top_ax, top_label, fontsize=13)

        aggregate_mask = aggregate["measure"].eq(measure)
        aggregate.loc[aggregate_mask, "fitted_asr"] = trend["fitted_asr"]
        aggregate.loc[aggregate_mask, "segment"] = trend["fitted_segment"]
        trend_statistics.append(
            {
                "measure": measure,
                "measure_name": measure_name,
                "statistic": "AAPC",
                "segment": np.nan,
                "start_year": years[0],
                "end_year": years[-1],
                "estimate": trend["aapc"],
                "lower": trend["aapc_lower"],
                "upper": trend["aapc_upper"],
                "p_value": trend["aapc_p_value"],
                "joinpoints": joinpoints,
                "bic": trend["bic"],
            }
        )
        for segment in trend["segments"]:
            trend_statistics.append(
                {
                    "measure": measure,
                    "measure_name": measure_name,
                    "statistic": "APC",
                    "segment": segment["segment"],
                    "start_year": segment["segment_start"],
                    "end_year": segment["segment_end"],
                    "estimate": segment["apc"],
                    "lower": segment["apc_lower"],
                    "upper": segment["apc_upper"],
                    "p_value": segment["apc_p_value"],
                    "joinpoints": joinpoints,
                    "bic": trend["bic"],
                }
            )

        bottom_ax = axes[1, column]
        bottom_ax.imshow(
            country_matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        bottom_ax.set_yticks(np.arange(len(country_order)), country_order, fontsize=6.5)
        bottom_ax.set_xticks(tick_positions, tick_years)
        bottom_ax.set_xlabel("Year")
        bottom_ax.set_ylabel("OECD-38 countries")
        bottom_ax.set_title(f"Country trends - {measure_name}", loc="left", fontweight="bold", pad=8)
        bottom_ax.grid(False)
        panel_label(bottom_ax, bottom_label, fontsize=13)

        colorbar_ax = fig.add_subplot(grid[2, column])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax, orientation="horizontal")
        colorbar.set_label(f"{rate_name} (per 100,000)")
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.045)
    fig.savefig(FIGURES / "Figure2_1990_2023_OECD65plus_CVD_trends.pdf", bbox_inches="tight")
    plt.close(fig)

    output = pd.concat(
        [
            aggregate[["scope", "location", "location_name", "iso3", "measure", "year", "asr", "fitted_asr", "segment"]],
            countries.assign(fitted_asr=np.nan, segment=np.nan)[["scope", "location", "location_name", "iso3", "measure", "year", "asr", "fitted_asr", "segment"]],
        ],
        ignore_index=True,
    )
    output["measure_name"] = output["measure"].map({6: "Incidence", 2: "DALYs"})
    save_csv(output.sort_values(["measure", "scope", "iso3", "year"]), "Figure2_trend_data.csv", True)
    save_csv(pd.DataFrame(trend_statistics), "Figure2_AAPC_APC_data.csv", True)


def _annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, image, fmt: str = ".0f") -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=6.5, color="#111827" if luminance > 0.58 else "white")


def figure3_age_sex_subtype(subtype: pd.DataFrame, uncertainty: pd.DataFrame) -> None:
    ages = AGE_ORDER
    age_labels = [AGE_NAMES[age] for age in ages]
    x = np.arange(len(ages))
    measures = [(6, "Incidence", "A", "C", "E"), (2, "DALYs", "B", "D", "F")]
    sex_colors = {1: "#4C72B0", 2: "#C44E52"}
    cause_colors = {
        493: "#E69F00",
        495: "#0072B2",
        496: "#56B4E9",
        498: "#D55E00",
        -1: "#B8BDC7",
    }
    cause_labels = {
        493: "IHD",
        495: "Ischemic stroke",
        496: "ICH",
        498: "HHD (DALYs only)",
        -1: "Other CVD",
    }
    expected_total = subtype.query("cause == 491").groupby("measure").agg(ages=("age", "nunique"), sexes=("sex", "nunique")).to_dict("index")
    if expected_total != {2: {"ages": 7, "sexes": 3}, 6: {"ages": 7, "sexes": 3}}:
        raise ValueError("Figure 3 requires total CVD incidence and DALY data for seven age groups and three sexes")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 13),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]},
    )
    derived_rows = []

    for column, (measure, measure_name, rate_label, share_label, composition_label) in enumerate(measures):
        total = subtype.query("measure == @measure and cause == 491").copy()

        rate_ax = axes[0, column]
        for sex in [2, 1]:
            block = total.query("sex == @sex").set_index("age").reindex(ages)
            interval = uncertainty.query("measure == @measure and sex == @sex").set_index("age").reindex(ages)
            rate_ax.fill_between(
                x,
                interval["rate_lower"].to_numpy(),
                interval["rate_upper"].to_numpy(),
                color=sex_colors[sex],
                alpha=0.16,
                linewidth=0,
            )
            rate_ax.plot(
                x,
                block["age_specific_rate"],
                color=sex_colors[sex],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=SEX_NAMES[sex],
            )
        rate_ax.set_xticks(x, age_labels)
        rate_ax.set_xlabel("Age group")
        rate_ax.set_ylabel(f"Age-specific {measure_name.lower()} rate (per 100,000)")
        rate_ax.set_title(f"Total CVD - {measure_name}", loc="left", fontweight="bold")
        rate_ax.grid(False)
        if measure == 6:
            handles, labels = rate_ax.get_legend_handles_labels()
            handles.append(Patch(facecolor="#9CA3AF", edgecolor="none", alpha=0.22))
            labels.append("Approx. 95% UI")
            rate_ax.legend(handles, labels, frameon=False, loc="upper left", ncol=3)
        panel_label(rate_ax, rate_label)

        share_ax = axes[1, column]
        for sex in [2, 1]:
            block = total.query("sex == @sex").set_index("age").reindex(ages)
            age_share = block["number"] / block["number"].sum() * 100
            interval = uncertainty.query("measure == @measure and sex == @sex").set_index("age").reindex(ages)
            share_ax.fill_between(
                x,
                interval["age_share_lower"].to_numpy(),
                interval["age_share_upper"].to_numpy(),
                color=sex_colors[sex],
                alpha=0.16,
                linewidth=0,
            )
            share_ax.plot(
                x,
                age_share,
                color=sex_colors[sex],
                marker="o",
                markersize=4,
                linewidth=1.8,
            )
            for age, value in zip(ages, age_share):
                derived_rows.append(
                    {
                        "panel_data": "age_distribution",
                        "measure": measure,
                        "sex": sex,
                        "age": age,
                        "cause": 491,
                        "value_percent": value,
                    }
                )
        share_ax.set_xticks(x, age_labels)
        share_ax.set_xlabel("Age group")
        share_ax.set_ylabel(f"Share of sex-specific {measure_name.lower()} (%)")
        share_ax.set_title(f"Age distribution - {measure_name}", loc="left", fontweight="bold")
        share_ax.grid(False)
        share_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
        panel_label(share_ax, share_label)

        composition_ax = axes[2, column]
        both = subtype.query("measure == @measure and sex == 3")
        total_by_age = both.query("cause == 491").set_index("age").reindex(ages)["number"]
        available_causes = [cause for cause in [493, 495, 496, 498] if cause in both["cause"].unique()]
        subtype_matrix = (
            both.query("cause in @available_causes")
            .pivot(index="cause", columns="age", values="number")
            .reindex(index=available_causes, columns=ages)
            .fillna(0)
        )
        residual = total_by_age - subtype_matrix.sum(axis=0)
        if (residual < -1e-6).any():
            raise ValueError(f"Figure 3 {measure_name} subtype sum exceeds total CVD")
        stack_order = [493, 495, 496] + ([498] if 498 in available_causes else []) + [-1]
        bottom = np.zeros(len(ages))
        for cause in stack_order:
            values = residual.to_numpy() if cause == -1 else subtype_matrix.loc[cause].to_numpy()
            shares = values / total_by_age.to_numpy() * 100
            composition_ax.bar(
                x,
                shares,
                bottom=bottom,
                color=cause_colors[cause],
                edgecolor="white",
                linewidth=0.45,
                width=0.78,
            )
            bottom += shares
            for age, value in zip(ages, shares):
                derived_rows.append(
                    {
                        "panel_data": "cause_composition",
                        "measure": measure,
                        "sex": 3,
                        "age": age,
                        "cause": cause,
                        "value_percent": value,
                    }
                )
        composition_ax.set_xticks(x, age_labels)
        composition_ax.set_xlabel("Age group")
        composition_ax.set_ylabel(f"Cause share within age-specific {measure_name.lower()} (%)")
        composition_ax.set_ylim(0, 100)
        composition_ax.set_title(f"Cause composition - {measure_name}", loc="left", fontweight="bold")
        composition_ax.grid(False)
        composition_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
        panel_label(composition_ax, composition_label)

    legend_order = [493, 495, 496, 498, -1]
    fig.legend(
        handles=[Patch(facecolor=cause_colors[cause], edgecolor="none", label=cause_labels[cause]) for cause in legend_order],
        labels=[cause_labels[cause] for cause in legend_order],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.995], h_pad=2.0, w_pad=2.0)
    fig.savefig(FIGURES / "Figure3_age_sex_subtype_structure.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(subtype, "Figure3_age_sex_subtype_data.csv", True)
    save_csv(pd.DataFrame(derived_rows), "Figure3_age_distribution_and_cause_share_data.csv", True)
    save_csv(uncertainty, "Figure3_age_sex_uncertainty_data.csv", True)


def figure4_trajectories(
    clusters: pd.DataFrame,
    cluster_series: pd.DataFrame,
) -> None:
    cluster_colors = dict(
        zip(
            sorted(clusters["cluster"].unique()),
            [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"], PALETTE["red"]],
        )
    )
    cluster_ids = sorted(clusters["cluster"].unique())
    membership_labels = {}
    for cluster in cluster_ids:
        block = clusters.query("cluster == @cluster").sort_values("iso3")
        label = f"Cluster {cluster} (n={len(block)}): " + ", ".join(block["iso3"])
        membership_labels[cluster] = textwrap.fill(label, width=58, subsequent_indent="    ")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    panels = [(6, "Incidence rate index (1990=100)", "Incidence trajectories", "A"), (2, "DALY rate index (1990=100)", "DALY trajectories", "B")]

    for ax, (measure, ylabel, title, panel) in zip(axes, panels):
        for cluster, block in cluster_series.query("measure == @measure").groupby("cluster"):
            ax.fill_between(
                block["year"].to_numpy(),
                block["lower_quartile"].to_numpy(),
                block["upper_quartile"].to_numpy(),
                color=cluster_colors[cluster],
                alpha=0.14,
                linewidth=0,
            )
            ax.plot(
                block["year"],
                block["mean_index"],
                color=cluster_colors[cluster],
                linewidth=2,
                label=membership_labels[cluster],
            )
        ax.axhline(100, color="#111827", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        if measure == 6:
            ax.legend(
                frameon=False,
                loc="lower left",
                ncol=1,
                fontsize=6.4,
                handlelength=2.2,
                handletextpad=0.7,
                labelspacing=0.55,
                borderaxespad=0.6,
            )
        panel_label(ax, panel, fontsize=13)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.13, wspace=0.16)
    fig.savefig(FIGURES / "Figure4_convergence_and_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(clusters, "Figure4_trajectory_clusters_incidence_DALYs.csv", True)
    save_csv(cluster_series, "Figure4D_cluster_series_data.csv", True)


def figure5_das_gupta(decomposition: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 11.2))
    component_cols = [
        "population_growth_percent_of_1990",
        "population_ageing_percent_of_1990",
        "epidemiological_change_percent_of_1990",
    ]
    component_labels = ["Population growth", "Population ageing", "Epidemiological change"]
    component_colors = [PALETTE["gray"], PALETTE["orange"], PALETTE["blue"]]
    panels = ((6, "Incidence", "C"), (2, "DALYs", "D"))

    for ax, (measure, title, panel) in zip(axes, panels):
        countries = decomposition.query("scope == 'Country' and measure == @measure").sort_values(
            "total_change_percent_of_1990"
        )
        oecd = decomposition.query("scope == 'OECD aggregate' and measure == @measure")
        block = pd.concat([countries, oecd], ignore_index=True)
        row_labels = block["iso3"].replace({"OECD38": "OECD"})
        y = np.arange(len(block))
        positive = np.zeros(len(block))
        negative = np.zeros(len(block))
        ax.axhspan(len(block) - 1.45, len(block) - 0.55, color="#F3F4F6", zorder=0)
        for col, name, color in zip(component_cols, component_labels, component_colors):
            values = block[col].to_numpy()
            left = np.where(values >= 0, positive, negative)
            ax.barh(y, values, left=left, color=color, label=name, height=0.70, zorder=2)
            positive += np.where(values >= 0, values, 0)
            negative += np.where(values < 0, values, 0)
        ax.scatter(
            block["total_change_percent_of_1990"],
            y,
            color="#111827",
            marker="D",
            s=12,
            label="Observed net change",
            zorder=4,
        )
        extent = max(float(positive.max()), float(abs(negative.min())))
        ax.set_xlim(-1.06 * extent, 1.06 * extent)
        ax.axvline(0, color="#111827", lw=0.8, zorder=3)
        ax.axhline(len(block) - 1.5, color="#9CA3AF", lw=0.7)
        ax.set_yticks(y, row_labels, fontsize=6.4)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.set_xlabel("Contribution to change (% of 1990 burden)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.55)
        panel_label(ax, panel, fontsize=13)

    handles = [Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(component_colors, component_labels)]
    handles.append(Line2D([0], [0], color="#111827", marker="D", linestyle="none", markersize=4, label="Observed net change"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.006))
    fig.tight_layout(rect=[0, 0.035, 1, 0.995], w_pad=2.5)
    fig.savefig(FIGURES / "Figure5_four_factor_Shapley_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(decomposition, "Figure5_Das_Gupta_decomposition_data.csv", True)


def figure6_sdi_relationships(data: pd.DataFrame) -> None:
    iso3_codes = sorted(data["iso3"].unique())
    categorical_colors = list(mpl.colormaps["tab20"].colors) + list(mpl.colormaps["tab20b"].colors)
    country_colors = dict(zip(iso3_codes, categorical_colors[: len(iso3_codes)]))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    fit_rows = []
    panels = (
        (6, "Incidence", "Age-standardized incidence rate per 100,000", "A"),
        (2, "DALYs", "Age-standardized DALY rate per 100,000", "B"),
    )

    for ax, (measure, title, ylabel, panel) in zip(axes, panels):
        block = data.query("measure == @measure")
        for iso3, country in block.groupby("iso3", sort=True):
            country = country.sort_values("year")
            ax.plot(
                country["sdi"],
                country["rate_per_100000"],
                color=country_colors[iso3],
                linewidth=1.05,
                marker="o",
                markersize=1.8,
                alpha=0.70,
                zorder=2,
            )

        fitted = sm.nonparametric.lowess(
            endog=block["rate_per_100000"].to_numpy(),
            exog=block["sdi"].to_numpy(),
            frac=0.30,
            it=3,
            return_sorted=True,
        )
        fitted = pd.DataFrame(fitted, columns=["sdi", "fitted_rate_per_100000"])
        fitted = fitted.groupby("sdi", as_index=False)["fitted_rate_per_100000"].mean()
        fitted["measure"] = measure
        fitted["measure_name"] = MEASURE_NAMES[measure]
        fit_rows.append(fitted)
        ax.plot(
            fitted["sdi"],
            fitted["fitted_rate_per_100000"],
            color="#111827",
            linewidth=3.0,
            zorder=5,
        )
        ax.set_xlabel("Socio-demographic Index (SDI)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} and SDI, 1990-2023", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        ax.margins(x=0.025, y=0.05)
        panel_label(ax, panel, fontsize=13)

    legend_handles = [
        Line2D([0], [0], color=country_colors[iso3], marker="o", markersize=3.0, linewidth=1.3, label=iso3)
        for iso3 in iso3_codes
    ]
    legend_handles.append(Line2D([0], [0], color="#111827", linewidth=3.0, label="Overall LOWESS fit"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=10,
        frameon=False,
        fontsize=6.7,
        columnspacing=1.1,
        handlelength=1.9,
        handletextpad=0.45,
        labelspacing=0.7,
    )
    fig.tight_layout(rect=[0, 0.19, 1, 0.995], w_pad=2.5)
    fig.savefig(FIGURES / "Figure6_country_performance_quadrants.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(data, "Figure6_incidence_DALY_SDI_data.csv", True)
    save_csv(pd.concat(fit_rows, ignore_index=True), "Figure6_LOWESS_fit_data.csv", True)


def figure7_transition_indices(transitions: pd.DataFrame) -> None:
    concentration = transitions.loc[
        transitions["measure"].isin([1, 2, 3]),
        [
            "cause",
            "cause_name",
            "cause_short",
            "measure",
            "measure_name",
            "year",
            "number",
            "number_80plus",
            "oldest_old_concentration",
        ],
    ].copy()
    concentration = concentration.rename(
        columns={"number": "burden_65plus", "number_80plus": "burden_80plus"}
    )
    concentration["oldest_old_concentration_percent"] = (
        100 * concentration["oldest_old_concentration"]
    )
    disability = (
        transitions[["cause", "cause_name", "cause_short", "year", "yld_to_daly_ratio"]]
        .drop_duplicates(["cause", "year"])
        .sort_values(["cause", "year"])
        .copy()
    )
    disability["yld_to_daly_percent"] = 100 * disability["yld_to_daly_ratio"]

    causes = [491, 493, 495, 496, 498]
    fig, axes = plt.subplots(
        2,
        len(causes),
        figsize=(17.2, 7.6),
        sharex="col",
        sharey="row",
    )
    measure_styles = {
        1: ("Deaths", PALETTE["blue"]),
        2: ("DALYs", PALETTE["orange"]),
        3: ("YLDs", PALETTE["green"]),
    }
    panel_letters = iter("abcdefghij")
    for column, cause in enumerate(causes):
        ax = axes[0, column]
        for measure, (label, color) in measure_styles.items():
            block = concentration.query(
                "cause == @cause and measure == @measure"
            ).sort_values("year")
            ax.plot(
                block["year"],
                block["oldest_old_concentration_percent"],
                color=color,
                linewidth=1.9,
                zorder=3,
            )
            endpoint = block.iloc[-1]
            ax.scatter(
                endpoint["year"],
                endpoint["oldest_old_concentration_percent"],
                color=color,
                s=18,
                zorder=4,
            )
        ax.set_title(CAUSE_SHORT[cause], fontweight="bold", pad=8)
        ax.set_xlim(1990, 2023.8)
        ax.set_ylim(20, 82)
        ax.set_yticks([20, 40, 60, 80])
        ax.tick_params(axis="x", labelbottom=False)
        ax.grid(False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        panel_label(ax, next(panel_letters), fontsize=11)

    for column, cause in enumerate(causes):
        ax = axes[1, column]
        block = disability.query("cause == @cause").sort_values("year")
        ax.plot(
            block["year"],
            block["yld_to_daly_percent"],
            color=PALETTE["purple"],
            linewidth=2.1,
            zorder=3,
        )
        endpoint = block.iloc[-1]
        ax.scatter(
            endpoint["year"],
            endpoint["yld_to_daly_percent"],
            color=PALETTE["purple"],
            s=19,
            zorder=4,
        )
        ax.set_xlim(1990, 2023.8)
        ax.set_ylim(0, 24)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_xticks([1990, 2000, 2010, 2023])
        ax.grid(False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        panel_label(ax, next(panel_letters), fontsize=11)

    axes[0, 0].set_ylabel("80+ share of 65+ burden (%)")
    axes[1, 0].set_ylabel("YLDs as a share of DALYs (%)")
    fig.supxlabel("Year", y=0.065, fontsize=9)
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=label)
        for label, color in measure_styles.values()
    ]
    legend_handles.append(
        Line2D([0], [0], color=PALETTE["purple"], linewidth=2.2, label="YLD / DALY")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=2.0,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        top=0.89,
        bottom=0.12,
        hspace=0.24,
        wspace=0.10,
    )
    fig.savefig(FIGURES / "Figure7_oldest_old_and_disability_transition.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(concentration, "Figure7_oldest_old_concentration_data.csv", True)
    save_csv(disability, "Figure7_disability_transition_data.csv", True)


def ets_model_label(error: str, trend: str | None, damped: bool) -> str:
    error_code = {"add": "A", "mul": "M"}[error]
    if trend is None:
        trend_code = "N"
    else:
        trend_code = {"add": "A", "mul": "M"}[trend] + ("d" if damped else "")
    return f"ETS({error_code},{trend_code},N)"


def ets_forecast_analysis(
    oecd_asr: pd.DataFrame,
    forecast_end: int = 2050,
    interval_repetitions: int = 20000,
    seed: int = 20260729,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = (
        oecd_asr.query("cause == 491 and measure in [2, 6] and sex == 3")
        .sort_values(["measure", "year"])
        .copy()
    )
    expected_years = list(range(1990, 2024))
    candidates = []
    for error in ("add", "mul"):
        candidates.append((error, None, False))
        for trend in ("add", "mul"):
            candidates.extend([(error, trend, False), (error, trend, True)])

    forecast_rows = []
    selection_rows = []
    validation_rows = []
    for measure in (6, 2):
        block = source.query("measure == @measure").sort_values("year")
        if block["year"].tolist() != expected_years or (block["asr"] <= 0).any():
            raise ValueError(
                f"Figure 8 requires positive annual {MEASURE_NAMES[measure]} ASRs for 1990-2023"
            )
        values = block["asr"].reset_index(drop=True)
        fitted_candidates = []
        for error, trend, damped in candidates:
            label = ets_model_label(error, trend, damped)
            record = {
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "model": label,
                "error": error,
                "trend": "none" if trend is None else trend,
                "damped_trend": damped,
                "seasonal": "none",
                "converged": False,
                "aic": np.nan,
                "aicc": np.nan,
                "bic": np.nan,
                "selected": False,
                "status": "fit_failed",
            }
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ETSModel(
                        values,
                        error=error,
                        trend=trend,
                        damped_trend=damped,
                        seasonal=None,
                        initialization_method="estimated",
                    ).fit(maxiter=2000, disp=False)
                converged = bool(fit.mle_retvals.get("converged", True))
                record.update(
                    {
                        "converged": converged,
                        "aic": float(fit.aic),
                        "aicc": float(fit.aicc),
                        "bic": float(fit.bic),
                        "status": "ok" if converged else "not_converged",
                    }
                )
                if converged and np.isfinite(fit.aicc):
                    fitted_candidates.append((float(fit.aicc), fit, error, trend, damped, label))
            except Exception as exc:
                record["status"] = f"fit_failed: {type(exc).__name__}"
            selection_rows.append(record)

        if not fitted_candidates:
            raise ValueError(f"No converged ETS candidate for {MEASURE_NAMES[measure]}")
        _, selected_fit, error, trend, damped, selected_label = min(
            fitted_candidates, key=lambda item: item[0]
        )
        selection_rows[-len(candidates) + next(
            index
            for index, candidate in enumerate(candidates)
            if candidate == (error, trend, damped)
        )]["selected"] = True

        forecast_years = np.arange(2024, forecast_end + 1)
        prediction = selected_fit.get_prediction(
            start=len(values),
            end=len(values) + len(forecast_years) - 1,
            method="simulated",
            simulate_repetitions=interval_repetitions,
            random_state=seed + measure,
        ).summary_frame(alpha=0.05)
        if (
            prediction[["mean", "pi_lower", "pi_upper"]].isna().any().any()
            or (prediction["pi_lower"] <= 0).any()
            or (prediction["pi_upper"] < prediction["pi_lower"]).any()
        ):
            raise ValueError(f"Invalid ETS prediction interval for {MEASURE_NAMES[measure]}")

        for year, observed in zip(block["year"], block["asr"]):
            forecast_rows.append(
                {
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": int(year),
                    "series": "Observed",
                    "observed_asr": float(observed),
                    "forecast_asr": np.nan,
                    "pi_lower": np.nan,
                    "pi_upper": np.nan,
                    "model": selected_label,
                    "interval": "Not applicable",
                }
            )
        for row_index, year in enumerate(forecast_years):
            forecast_rows.append(
                {
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": int(year),
                    "series": "Forecast",
                    "observed_asr": np.nan,
                    "forecast_asr": float(prediction["mean"].iloc[row_index]),
                    "pi_lower": float(prediction["pi_lower"].iloc[row_index]),
                    "pi_upper": float(prediction["pi_upper"].iloc[row_index]),
                    "model": selected_label,
                    "interval": "95% model-based prediction interval",
                }
            )

        validation_start = 2014
        training = values.iloc[: validation_start - 1990]
        holdout = values.iloc[validation_start - 1990 :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validation_fit = ETSModel(
                training,
                error=error,
                trend=trend,
                damped_trend=damped,
                seasonal=None,
                initialization_method="estimated",
            ).fit(maxiter=2000, disp=False)
        validation_prediction = validation_fit.get_prediction(
            start=len(training),
            end=len(values) - 1,
            method="simulated",
            simulate_repetitions=interval_repetitions,
            random_state=seed + 100 + measure,
        ).summary_frame(alpha=0.05)
        predicted = validation_prediction["mean"].to_numpy()
        actual = holdout.to_numpy()
        lower = validation_prediction["pi_lower"].to_numpy()
        upper = validation_prediction["pi_upper"].to_numpy()
        errors = predicted - actual
        validation_rows.append(
            {
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "model": selected_label,
                "training_years": "1990-2013",
                "validation_years": "2014-2023",
                "validation_n": len(actual),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mape_percent": float(np.mean(np.abs(errors / actual)) * 100),
                "prediction_interval_coverage_percent": float(
                    np.mean((actual >= lower) & (actual <= upper)) * 100
                ),
            }
        )

    selection = pd.DataFrame(selection_rows)
    validation = pd.DataFrame(validation_rows)
    forecast = pd.DataFrame(forecast_rows).sort_values(["measure", "year"])
    return forecast, selection, validation


def ets_country_forecast_analysis(
    country_asr: pd.DataFrame,
    crosswalk: pd.DataFrame,
    forecast_end: int = 2050,
    interval_repetitions: int = 10000,
    seed: int = 20260729,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = (
        country_asr.query("cause == 491 and measure in [2, 6] and sex == 3")
        .merge(
            crosswalk[["location_id", "location_name", "iso3"]],
            left_on="location",
            right_on="location_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["measure", "iso3", "year"])
    )
    expected_years = list(range(1990, 2024))
    iso3_codes = sorted(source["iso3"].dropna().unique())
    if len(iso3_codes) != 38:
        raise ValueError("Country ETS forecasts require all 38 OECD ISO3 codes")
    candidates = []
    for error in ("add", "mul"):
        candidates.append((error, None, False))
        for trend in ("add", "mul"):
            candidates.extend([(error, trend, False), (error, trend, True)])

    forecast_rows = []
    selection_rows = []
    validation_rows = []
    for measure in (6, 2):
        for iso3 in iso3_codes:
            block = source.query("measure == @measure and iso3 == @iso3").sort_values("year")
            if block["year"].tolist() != expected_years or (block["asr"] <= 0).any():
                raise ValueError(
                    f"Country ETS requires positive annual {MEASURE_NAMES[measure]} ASRs "
                    f"for {iso3}, 1990-2023"
                )
            values = block["asr"].reset_index(drop=True)
            location_id = int(block["location"].iloc[0])
            location_name = str(block["location_name"].iloc[0])
            candidate_records = []
            fitted_candidates = []
            for candidate_index, (error, trend, damped) in enumerate(candidates):
                label = ets_model_label(error, trend, damped)
                record = {
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "location": location_id,
                    "location_name": location_name,
                    "iso3": iso3,
                    "model": label,
                    "error": error,
                    "trend": "none" if trend is None else trend,
                    "damped_trend": damped,
                    "seasonal": "none",
                    "converged": False,
                    "positive_point_forecast_to_2050": False,
                    "aic": np.nan,
                    "aicc": np.nan,
                    "bic": np.nan,
                    "selected": False,
                    "status": "fit_failed",
                }
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = ETSModel(
                            values,
                            error=error,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=None,
                            initialization_method="estimated",
                        ).fit(maxiter=2000, disp=False)
                    converged = bool(fit.mle_retvals.get("converged", True))
                    candidate_forecast = np.asarray(
                        fit.predict(
                            start=len(values),
                            end=len(values) + (forecast_end - 2023) - 1,
                        )
                    )
                    positive_forecast = bool(
                        np.isfinite(candidate_forecast).all()
                        and (candidate_forecast > 0).all()
                    )
                    record.update(
                        {
                            "converged": converged,
                            "positive_point_forecast_to_2050": positive_forecast,
                            "aic": float(fit.aic),
                            "aicc": float(fit.aicc),
                            "bic": float(fit.bic),
                            "status": "ok" if converged else "not_converged",
                        }
                    )
                    if converged and positive_forecast and np.isfinite(fit.aicc):
                        fitted_candidates.append(
                            (
                                float(fit.aicc),
                                candidate_index,
                                fit,
                                error,
                                trend,
                                damped,
                                label,
                            )
                        )
                except Exception as exc:
                    record["status"] = f"fit_failed: {type(exc).__name__}"
                candidate_records.append(record)

            if not fitted_candidates:
                raise ValueError(
                    f"No converged ETS candidate for {iso3} {MEASURE_NAMES[measure]}"
                )
            (
                _,
                selected_index,
                selected_fit,
                error,
                trend,
                damped,
                selected_label,
            ) = min(fitted_candidates, key=lambda item: item[0])
            candidate_records[selected_index]["selected"] = True
            selection_rows.extend(candidate_records)

            forecast_years = np.arange(2024, forecast_end + 1)
            prediction = selected_fit.get_prediction(
                start=len(values),
                end=len(values) + len(forecast_years) - 1,
                method="simulated",
                simulate_repetitions=interval_repetitions,
                random_state=seed + measure * 1000 + location_id,
            ).summary_frame(alpha=0.05)
            if (
                prediction[["mean", "pi_lower", "pi_upper"]].isna().any().any()
                or (prediction["mean"] <= 0).any()
                or (prediction["pi_upper"] < prediction["pi_lower"]).any()
            ):
                raise ValueError(
                    f"Invalid country ETS prediction for {iso3} {MEASURE_NAMES[measure]}"
                )

            for year, observed in zip(block["year"], block["asr"]):
                forecast_rows.append(
                    {
                        "measure": measure,
                        "measure_name": MEASURE_NAMES[measure],
                        "location": location_id,
                        "location_name": location_name,
                        "iso3": iso3,
                        "year": int(year),
                        "series": "Observed",
                        "observed_asr": float(observed),
                        "forecast_asr": np.nan,
                        "pi_lower_raw": np.nan,
                        "pi_lower": np.nan,
                        "pi_upper": np.nan,
                        "pi_lower_clipped_at_zero": False,
                        "model": selected_label,
                        "interval": "Not applicable",
                    }
                )
            for row_index, year in enumerate(forecast_years):
                raw_lower = float(prediction["pi_lower"].iloc[row_index])
                forecast_rows.append(
                    {
                        "measure": measure,
                        "measure_name": MEASURE_NAMES[measure],
                        "location": location_id,
                        "location_name": location_name,
                        "iso3": iso3,
                        "year": int(year),
                        "series": "Forecast",
                        "observed_asr": np.nan,
                        "forecast_asr": float(prediction["mean"].iloc[row_index]),
                        "pi_lower_raw": raw_lower,
                        "pi_lower": max(0.0, raw_lower),
                        "pi_upper": float(prediction["pi_upper"].iloc[row_index]),
                        "pi_lower_clipped_at_zero": raw_lower < 0,
                        "model": selected_label,
                        "interval": "95% model-based prediction interval",
                    }
                )

            validation_start = 2014
            training = values.iloc[: validation_start - 1990]
            actual = values.iloc[validation_start - 1990 :].to_numpy()
            validation_record = {
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "location": location_id,
                "location_name": location_name,
                "iso3": iso3,
                "model": selected_label,
                "training_years": "1990-2013",
                "validation_years": "2014-2023",
                "validation_n": len(actual),
                "mae": np.nan,
                "rmse": np.nan,
                "mape_percent": np.nan,
                "status": "fit_failed",
            }
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    validation_fit = ETSModel(
                        training,
                        error=error,
                        trend=trend,
                        damped_trend=damped,
                        seasonal=None,
                        initialization_method="estimated",
                    ).fit(maxiter=2000, disp=False)
                predicted = np.asarray(
                    validation_fit.predict(start=len(training), end=len(values) - 1)
                )
                errors = predicted - actual
                validation_record.update(
                    {
                        "mae": float(np.mean(np.abs(errors))),
                        "rmse": float(np.sqrt(np.mean(errors**2))),
                        "mape_percent": float(np.mean(np.abs(errors / actual)) * 100),
                        "status": (
                            "ok"
                            if bool(validation_fit.mle_retvals.get("converged", True))
                            else "not_converged"
                        ),
                    }
                )
            except Exception as exc:
                validation_record["status"] = f"fit_failed: {type(exc).__name__}"
            validation_rows.append(validation_record)

    forecast = pd.DataFrame(forecast_rows).sort_values(["measure", "iso3", "year"])
    selection = pd.DataFrame(selection_rows).sort_values(["measure", "iso3", "aicc"])
    validation = pd.DataFrame(validation_rows).sort_values(["measure", "iso3"])
    return forecast, selection, validation


def figure8_ets_forecast(
    forecast: pd.DataFrame,
    selection: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    panels = (
        (6, "Incidence", PALETTE["blue"], "A"),
        (2, "DALYs", PALETTE["orange"], "B"),
    )
    for ax, (measure, title, color, panel) in zip(axes, panels):
        observed = forecast.query("measure == @measure and series == 'Observed'").sort_values("year")
        future = forecast.query("measure == @measure and series == 'Forecast'").sort_values("year")
        selected = selection.query("measure == @measure and selected").iloc[0]
        diagnostic = validation.query("measure == @measure").iloc[0]
        anchor_year = int(observed["year"].iloc[-1])
        anchor_value = float(observed["observed_asr"].iloc[-1])
        forecast_years = np.r_[anchor_year, future["year"].to_numpy()]
        forecast_values = np.r_[anchor_value, future["forecast_asr"].to_numpy()]

        ax.plot(
            observed["year"],
            observed["observed_asr"],
            color="#374151",
            linewidth=1.9,
            label="Observed ASR",
            zorder=3,
        )
        ax.fill_between(
            future["year"].to_numpy(),
            future["pi_lower"].to_numpy(),
            future["pi_upper"].to_numpy(),
            color=color,
            alpha=0.18,
            linewidth=0,
            label="95% prediction interval",
            zorder=1,
        )
        ax.plot(
            forecast_years,
            forecast_values,
            color=color,
            linewidth=2.4,
            label="ETS forecast",
            zorder=4,
        )
        endpoint = future.iloc[-1]
        ax.scatter(
            endpoint["year"],
            endpoint["forecast_asr"],
            color=color,
            s=28,
            zorder=5,
        )
        ax.axvline(2023, color="#6B7280", linewidth=0.8, linestyle="--", zorder=2)
        ax.text(
            0.97,
            0.96,
            (
                f"2050: {endpoint['forecast_asr']:,.0f}\n"
                f"95% PI: {endpoint['pi_lower']:,.0f}-{endpoint['pi_upper']:,.0f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.03,
            0.05,
            (
                f"{selected['model']} selected by AICc\n"
                f"2014-2023 holdout MAPE: {diagnostic['mape_percent']:.1f}%"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#4B5563",
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Age-standardized rate per 100,000")
        ax.set_xlim(1990, 2052)
        ax.set_xticks([1990, 2000, 2010, 2023, 2030, 2040, 2050])
        ax.grid(False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        ax.margins(y=0.10)
        panel_label(ax, panel, fontsize=13)

    handles = [
        Line2D([0], [0], color="#374151", linewidth=1.9, label="Observed ASR"),
        Line2D([0], [0], color=PALETTE["blue"], linewidth=2.4, label="ETS forecast"),
        Patch(facecolor=PALETTE["blue"], alpha=0.18, edgecolor="none", label="95% prediction interval"),
        Line2D([0], [0], color="#6B7280", linewidth=0.8, linestyle="--", label="Forecast origin (2023)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.99], w_pad=2.8)
    fig.savefig(FIGURES / "Figure8_ETS_forecast_incidence_DALYs_2050.pdf", bbox_inches="tight")
    plt.close(fig)
    save_csv(forecast, "Figure8_ETS_forecast_data.csv", True)
    save_csv(selection, "Figure8_ETS_model_selection.csv", True)
    save_csv(validation, "Figure8_ETS_holdout_validation.csv", True)


def country_ets_forecast_figure(
    forecast: pd.DataFrame,
    measure: int,
    filename: str,
) -> None:
    block = forecast.query("measure == @measure").copy()
    iso3_codes = sorted(block["iso3"].unique())
    if len(iso3_codes) != 38:
        raise ValueError("Country ETS figure requires 38 country panels")
    color = PALETTE["blue"] if measure == 6 else PALETTE["orange"]
    rows, columns = 7, 6
    fig, axes = plt.subplots(rows, columns, figsize=(17.5, 17.2))
    axes_flat = axes.ravel()
    panel_positions = list(range(36)) + [38, 39]

    for axis_index, iso3 in zip(panel_positions, iso3_codes):
        ax = axes_flat[axis_index]
        country = block.query("iso3 == @iso3")
        observed = country.query("series == 'Observed'").sort_values("year")
        future = country.query("series == 'Forecast'").sort_values("year")
        anchor_year = int(observed["year"].iloc[-1])
        anchor_value = float(observed["observed_asr"].iloc[-1])
        forecast_years = np.r_[anchor_year, future["year"].to_numpy()]
        forecast_values = np.r_[anchor_value, future["forecast_asr"].to_numpy()]

        ax.plot(
            observed["year"],
            observed["observed_asr"],
            color="#374151",
            linewidth=1.0,
            zorder=3,
        )
        ax.fill_between(
            future["year"].to_numpy(),
            future["pi_lower"].to_numpy(),
            future["pi_upper"].to_numpy(),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            forecast_years,
            forecast_values,
            color=color,
            linewidth=1.3,
            zorder=4,
        )
        endpoint = future.iloc[-1]
        ax.scatter(
            endpoint["year"],
            endpoint["forecast_asr"],
            color=color,
            s=9,
            zorder=5,
        )
        ax.axvline(2023, color="#9CA3AF", linewidth=0.55, linestyle="--", zorder=2)
        ax.set_title(iso3, loc="left", fontweight="bold", fontsize=7.5, pad=2)
        ax.set_xlim(1990, 2052)
        ax.set_xticks([1990, 2023, 2050])
        ax.tick_params(axis="x", labelsize=5.5, pad=1.5)
        ax.tick_params(axis="y", labelsize=5.5, pad=1.5)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.grid(False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.4)
        ax.margins(y=0.08)

    used_positions = set(panel_positions)
    for axis_index, unused in enumerate(axes_flat):
        if axis_index not in used_positions:
            unused.remove()

    outcome = "Incidence" if measure == 6 else "DALYs"
    fig.suptitle(
        f"OECD-38 country {outcome} ETS forecasts, 1990-2050",
        y=0.992,
        fontsize=13,
        fontweight="bold",
    )
    fig.supxlabel("Year", y=0.018, fontsize=9)
    fig.supylabel(
        "Age-standardized rate per 100,000 (country-specific scales)",
        x=0.012,
        fontsize=9,
    )
    legend_handles = [
        Line2D([0], [0], color="#374151", linewidth=1.2, label="Observed ASR"),
        Line2D([0], [0], color=color, linewidth=1.5, label="ETS forecast"),
        Patch(facecolor=color, alpha=0.18, edgecolor="none", label="95% prediction interval"),
        Line2D(
            [0],
            [0],
            color="#9CA3AF",
            linewidth=0.6,
            linestyle="--",
            label="Forecast origin (2023)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.973),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.048,
        right=0.992,
        top=0.942,
        bottom=0.045,
        hspace=0.35,
        wspace=0.22,
    )
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)


def _main_figure_csv_block(
    frame: pd.DataFrame,
    figure_id: str,
    data_block: str,
    source_csv: str,
    panel: str | None = None,
) -> pd.DataFrame:
    """Add stable provenance fields to one block of a main-figure CSV export."""
    block = frame.copy()
    if panel is not None:
        block["panel"] = panel
    if "panel" not in block.columns:
        block["panel"] = ""
    block.insert(0, "source_csv", source_csv)
    block.insert(0, "data_block", data_block)
    block.insert(0, "figure_id", figure_id)
    metadata = ["figure_id", "panel", "data_block", "source_csv"]
    return block[metadata + [column for column in block.columns if column not in metadata]]


def write_main_figure_csv_exports() -> pd.DataFrame:
    """Write one same-stem CSV for each fixed main-text Figure 1-Figure 10."""
    read = lambda name: pd.read_csv(FIGURE_DATA / name)
    exports: list[tuple[str, str, list[pd.DataFrame], str]] = []

    figure1_source = "Figure1_four_map_data.csv"
    figure1 = read(figure1_source)
    exports.append(
        (
            "Figure 1",
            "Figure1_2023_OECD65plus_CVD_baseline",
            [_main_figure_csv_block(figure1, "Figure 1", "map_rate", figure1_source)],
            "Four 2023 maps; one row per OECD country and measure.",
        )
    )

    figure2_trend_source = "Figure2_trend_data.csv"
    figure2_trend = read(figure2_trend_source)
    figure2_trend["panel"] = np.select(
        [
            figure2_trend["measure"].eq(6) & figure2_trend["scope"].eq("OECD aggregate"),
            figure2_trend["measure"].eq(2) & figure2_trend["scope"].eq("OECD aggregate"),
            figure2_trend["measure"].eq(6) & figure2_trend["scope"].eq("OECD country"),
            figure2_trend["measure"].eq(2) & figure2_trend["scope"].eq("OECD country"),
        ],
        ["A", "B", "C", "D"],
        default="",
    )
    figure2_stats_source = "Figure2_AAPC_APC_data.csv"
    figure2_stats = read(figure2_stats_source)
    figure2_stats["panel"] = figure2_stats["measure"].map({6: "A", 2: "B"})
    exports.append(
        (
            "Figure 2",
            "Figure2_1990_2023_OECD65plus_CVD_trends",
            [
                _main_figure_csv_block(
                    figure2_trend,
                    "Figure 2",
                    "oecd_trend_or_country_heatmap",
                    figure2_trend_source,
                ),
                _main_figure_csv_block(
                    figure2_stats,
                    "Figure 2",
                    "aapc_apc_statistics",
                    figure2_stats_source,
                ),
            ],
            "Panels A-B contain OECD trends and AAPC/APC; panels C-D contain country heatmaps.",
        )
    )

    figure3_base_source = "Figure3_age_sex_subtype_data.csv"
    figure3_base = read(figure3_base_source)
    figure3_uncertainty_source = "Figure3_age_sex_uncertainty_data.csv"
    figure3_uncertainty = read(figure3_uncertainty_source)
    figure3_rate = figure3_base.query("cause == 491 and sex in [1, 2]").merge(
        figure3_uncertainty,
        on=["measure", "measure_name", "sex", "sex_name", "age", "age_name"],
        how="left",
        validate="one_to_one",
    )
    figure3_rate["panel"] = figure3_rate["measure"].map({6: "A", 2: "B"})
    figure3_derived_source = "Figure3_age_distribution_and_cause_share_data.csv"
    figure3_derived = read(figure3_derived_source)
    figure3_age_share = figure3_derived.query("panel_data == 'age_distribution'").merge(
        figure3_uncertainty[
            [
                "measure",
                "sex",
                "age",
                "age_share_lower",
                "age_share_upper",
                "draws",
                "assumption",
            ]
        ],
        on=["measure", "sex", "age"],
        how="left",
        validate="one_to_one",
    )
    figure3_age_share["panel"] = figure3_age_share["measure"].map({6: "C", 2: "D"})
    figure3_composition = figure3_derived.query("panel_data == 'cause_composition'").copy()
    figure3_composition["panel"] = figure3_composition["measure"].map({6: "E", 2: "F"})
    exports.append(
        (
            "Figure 3",
            "Figure3_age_sex_subtype_structure",
            [
                _main_figure_csv_block(
                    figure3_rate,
                    "Figure 3",
                    "age_specific_rate_with_approximate_ui",
                    f"{figure3_base_source};{figure3_uncertainty_source}",
                ),
                _main_figure_csv_block(
                    figure3_age_share,
                    "Figure 3",
                    "sex_specific_age_distribution_with_approximate_ui",
                    f"{figure3_derived_source};{figure3_uncertainty_source}",
                ),
                _main_figure_csv_block(
                    figure3_composition,
                    "Figure 3",
                    "cause_composition",
                    figure3_derived_source,
                ),
            ],
            "Panels A-B are age-specific rates, C-D age distributions, and E-F cause composition.",
        )
    )

    figure4_series_source = "Figure4D_cluster_series_data.csv"
    figure4_series = read(figure4_series_source)
    figure4_series["panel"] = figure4_series["measure"].map({6: "A", 2: "B"})
    figure4_members_source = "Figure4_trajectory_clusters_incidence_DALYs.csv"
    figure4_members = read(figure4_members_source)
    exports.append(
        (
            "Figure 4",
            "Figure4_convergence_and_trajectories",
            [
                _main_figure_csv_block(
                    figure4_series,
                    "Figure 4",
                    "cluster_trajectory_summary",
                    figure4_series_source,
                ),
                _main_figure_csv_block(
                    figure4_members,
                    "Figure 4",
                    "country_cluster_membership",
                    figure4_members_source,
                    panel="A-B",
                ),
            ],
            "Cluster mean and IQR trajectories plus country membership used by both panels.",
        )
    )

    figure5_source = "Figure5_Das_Gupta_decomposition_data.csv"
    figure5 = read(figure5_source)
    figure5["panel"] = figure5["measure"].map({6: "C", 2: "D"})
    exports.append(
        (
            "Figure 5",
            "Figure5_four_factor_Shapley_decomposition",
            [
                _main_figure_csv_block(
                    figure5,
                    "Figure 5",
                    "das_gupta_decomposition",
                    figure5_source,
                )
            ],
            "Symmetric three-factor Das Gupta decomposition.",
        )
    )

    figure6_observed_source = "Figure6_incidence_DALY_SDI_data.csv"
    figure6_observed = read(figure6_observed_source)
    figure6_observed["panel"] = figure6_observed["measure"].map({6: "A", 2: "B"})
    figure6_fit_source = "Figure6_LOWESS_fit_data.csv"
    figure6_fit = read(figure6_fit_source)
    figure6_fit["panel"] = figure6_fit["measure"].map({6: "A", 2: "B"})
    exports.append(
        (
            "Figure 6",
            "Figure6_country_performance_quadrants",
            [
                _main_figure_csv_block(
                    figure6_observed,
                    "Figure 6",
                    "country_year_observations",
                    figure6_observed_source,
                ),
                _main_figure_csv_block(
                    figure6_fit,
                    "Figure 6",
                    "overall_lowess_fit",
                    figure6_fit_source,
                ),
            ],
            "Country-year SDI observations and pooled LOWESS fits.",
        )
    )

    figure7_concentration_source = "Figure7_oldest_old_concentration_data.csv"
    figure7_concentration = read(figure7_concentration_source)
    cause_panel = {491: "A", 493: "B", 495: "C", 496: "D", 498: "E"}
    figure7_concentration["panel"] = figure7_concentration["cause"].map(cause_panel)
    figure7_disability_source = "Figure7_disability_transition_data.csv"
    figure7_disability = read(figure7_disability_source)
    disability_panel = {491: "F", 493: "G", 495: "H", 496: "I", 498: "J"}
    figure7_disability["panel"] = figure7_disability["cause"].map(disability_panel)
    exports.append(
        (
            "Figure 7",
            "Figure7_oldest_old_and_disability_transition",
            [
                _main_figure_csv_block(
                    figure7_concentration,
                    "Figure 7",
                    "oldest_old_concentration",
                    figure7_concentration_source,
                ),
                _main_figure_csv_block(
                    figure7_disability,
                    "Figure 7",
                    "disability_transition",
                    figure7_disability_source,
                ),
            ],
            "Panels A-E contain 80+/65+ concentration and F-J contain YLD/DALY transition.",
        )
    )

    figure8_forecast_source = "Figure8_ETS_forecast_data.csv"
    figure8_forecast = read(figure8_forecast_source)
    figure8_forecast["panel"] = figure8_forecast["measure"].map({6: "A", 2: "B"})
    figure8_selection_source = "Figure8_ETS_model_selection.csv"
    figure8_selection = read(figure8_selection_source)
    figure8_selection["panel"] = figure8_selection["measure"].map({6: "A", 2: "B"})
    figure8_validation_source = "Figure8_ETS_holdout_validation.csv"
    figure8_validation = read(figure8_validation_source)
    figure8_validation["panel"] = figure8_validation["measure"].map({6: "A", 2: "B"})
    exports.append(
        (
            "Figure 8",
            "Figure8_ETS_forecast_incidence_DALYs_2050",
            [
                _main_figure_csv_block(
                    figure8_forecast,
                    "Figure 8",
                    "observed_and_forecast_series",
                    figure8_forecast_source,
                ),
                _main_figure_csv_block(
                    figure8_selection,
                    "Figure 8",
                    "ets_model_selection",
                    figure8_selection_source,
                ),
                _main_figure_csv_block(
                    figure8_validation,
                    "Figure 8",
                    "temporal_holdout_validation",
                    figure8_validation_source,
                ),
            ],
            "OECD observed/forecast series, model candidates, and temporal validation.",
        )
    )

    country_forecast_source = "Figure9_10_country_ETS_forecast_data.csv"
    country_forecast = read(country_forecast_source)
    country_selection_source = "Figure9_10_country_ETS_model_selection.csv"
    country_selection = read(country_selection_source)
    country_validation_source = "Figure9_10_country_ETS_holdout_validation.csv"
    country_validation = read(country_validation_source)
    for figure_id, stem, measure, note in [
        (
            "Figure 9",
            "Figure9_OECD38_country_ETS_incidence_forecast_2050",
            6,
            "Country-specific Incidence observations, forecasts, candidates, and validation.",
        ),
        (
            "Figure 10",
            "Figure10_OECD38_country_ETS_DALY_forecast_2050",
            2,
            "Country-specific DALY observations, forecasts, candidates, and validation.",
        ),
    ]:
        forecast_block = country_forecast.query("measure == @measure").copy()
        forecast_block["panel"] = forecast_block["iso3"]
        selection_block = country_selection.query("measure == @measure").copy()
        selection_block["panel"] = selection_block["iso3"]
        validation_block = country_validation.query("measure == @measure").copy()
        validation_block["panel"] = validation_block["iso3"]
        exports.append(
            (
                figure_id,
                stem,
                [
                    _main_figure_csv_block(
                        forecast_block,
                        figure_id,
                        "country_observed_and_forecast_series",
                        country_forecast_source,
                    ),
                    _main_figure_csv_block(
                        selection_block,
                        figure_id,
                        "country_ets_model_selection",
                        country_selection_source,
                    ),
                    _main_figure_csv_block(
                        validation_block,
                        figure_id,
                        "country_temporal_holdout_validation",
                        country_validation_source,
                    ),
                ],
                note,
            )
        )

    index_rows = []
    for figure_id, stem, blocks, note in exports:
        combined = pd.concat(blocks, ignore_index=True, sort=False)
        metadata = ["figure_id", "panel", "data_block", "source_csv"]
        combined = combined[metadata + [column for column in combined.columns if column not in metadata]]
        csv_name = f"{stem}.csv"
        save_csv(combined, csv_name, True)
        index_rows.append(
            {
                "figure_id": figure_id,
                "pdf_file": f"figures/{stem}.pdf",
                "explanation_file": f"figures/{stem}_explanation.md",
                "csv_file": f"figure_data/{csv_name}",
                "csv_rows": len(combined),
                "panel_labels": ";".join(sorted(str(value) for value in combined["panel"].dropna().unique())),
                "data_blocks": ";".join(combined["data_block"].dropna().drop_duplicates()),
                "note": note,
            }
        )

    index = pd.DataFrame(index_rows)
    save_csv(index, "MainFigure_CSV_index.csv", True)
    if len(index) != 10 or index["csv_file"].nunique() != 10:
        raise ValueError("Main-figure CSV export must contain exactly one unique CSV for each Figure 1-Figure 10")
    return index


def _format_table1_rate(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:,.1f} ({lower:,.1f}–{upper:,.1f})"


def _format_table1_percent(estimate: float, lower: float, upper: float) -> str:
    def value(number: float) -> str:
        return f"{number:.2f}%".replace("-", "−")

    return f"{value(estimate)} ({value(lower)}–{value(upper)})"


def _format_table1_number(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:,.0f} ({lower:,.0f}–{upper:,.0f})"


def table1_oecd_country_age_rates_aapc(
    gbd: pd.DataFrame,
    derived_pop: pd.DataFrame,
    country_asr: pd.DataFrame,
    oecd_asr: pd.DataFrame,
    oecd_age_rate: pd.DataFrame,
    crosswalk: pd.DataFrame,
    draws: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create manuscript Table 1 with endpoint rates/UI and segmented-regression AAPC/CI."""
    measures = [(6, "Incidence"), (5, "Prevalence"), (1, "Deaths"), (2, "DALYs")]
    endpoints = [1990, 2023]
    source = gbd.query(
        "cause == 491 and measure in [1, 2, 5, 6] and sex == 3 and metric == 1 and year in @endpoints"
    ).copy()
    population = derived_pop.query("sex == 3 and year in @endpoints")[
        ["location", "age", "year", "population"]
    ].copy()
    source = source.merge(
        population,
        on=["location", "age", "year"],
        how="left",
        validate="many_to_one",
    )
    expected_rows = 4 * 2 * 38 * len(AGE_ORDER)
    if len(source) != expected_rows or source["population"].isna().any():
        raise ValueError(
            f"Table 1 requires {expected_rows} complete measure-year-country-age endpoint rows"
        )
    if (source[["val", "lower", "upper", "population"]] <= 0).any().any():
        raise ValueError("Table 1 endpoint values, uncertainty bounds, and populations must be positive")

    entity_rows = [
        {
            "entity_order": 0,
            "entity_id": "OECD_65PLUS",
            "section": "OECD overall",
            "scope": "OECD total",
            "location_id": 0,
            "location_name": "OECD",
            "iso3": "OECD",
            "age_id": np.nan,
            "age_group": "65+",
            "rate_type": "Age-standardized rate per 100,000 (65+)",
        }
    ]
    for index, age in enumerate(AGE_ORDER, start=1):
        entity_rows.append(
            {
                "entity_order": index,
                "entity_id": f"OECD_AGE_{age}",
                "section": "OECD age groups",
                "scope": "OECD age group",
                "location_id": 0,
                "location_name": "OECD",
                "iso3": "OECD",
                "age_id": age,
                "age_group": AGE_NAMES[age],
                "rate_type": "Age-specific rate per 100,000",
            }
        )
    for index, row in enumerate(crosswalk.sort_values("iso3").itertuples(), start=1 + len(AGE_ORDER)):
        entity_rows.append(
            {
                "entity_order": index,
                "entity_id": f"COUNTRY_{int(row.location_id)}",
                "section": "OECD countries",
                "scope": "Country",
                "location_id": int(row.location_id),
                "location_name": row.location_name,
                "iso3": row.iso3,
                "age_id": np.nan,
                "age_group": "65+",
                "rate_type": "Age-standardized rate per 100,000 (65+)",
            }
        )
    entities = pd.DataFrame(entity_rows)
    if len(entities) != 46:
        raise ValueError("Table 1 must contain OECD overall, seven OECD age groups, and 38 countries")

    deviates = location_deviates(gbd["location"].unique())
    endpoint_rows = []
    uncertainty_assumption = UI_ASSUMPTION
    for (measure, year), group in source.groupby(["measure", "year"], sort=True):
        group = group.sort_values(["location", "age"]).reset_index(drop=True)
        if len(group) != 38 * len(AGE_ORDER):
            raise ValueError(f"Incomplete Table 1 endpoint block for measure={measure}, year={year}")
        samples = split_normal_samples(group, measure, deviates)
        ages = group["age"].to_numpy()
        locations = group["location"].to_numpy()

        oecd_draws = np.zeros(draws)
        oecd_point = 0.0
        for age in AGE_ORDER:
            mask = ages == age
            age_population = float(group.loc[mask, "population"].sum())
            oecd_draws += AGE_WEIGHTS[age] * samples[mask].sum(axis=0) / age_population * 100000.0
            oecd_point += AGE_WEIGHTS[age] * float(group.loc[mask, "val"].sum()) / age_population * 100000.0
        endpoint_rows.append(
            {
                "entity_id": "OECD_65PLUS",
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "year": year,
                "rate": oecd_point,
                "rate_lower": float(np.quantile(oecd_draws, 0.025)),
                "rate_upper": float(np.quantile(oecd_draws, 0.975)),
                "uncertainty_draws": draws,
                "uncertainty_assumption": uncertainty_assumption,
            }
        )

        for age in AGE_ORDER:
            mask = ages == age
            age_population = float(group.loc[mask, "population"].sum())
            age_draws = samples[mask].sum(axis=0) / age_population * 100000.0
            age_point = float(group.loc[mask, "val"].sum()) / age_population * 100000.0
            endpoint_rows.append(
                {
                    "entity_id": f"OECD_AGE_{age}",
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": year,
                    "rate": age_point,
                    "rate_lower": float(np.quantile(age_draws, 0.025)),
                    "rate_upper": float(np.quantile(age_draws, 0.975)),
                    "uncertainty_draws": draws,
                    "uncertainty_assumption": uncertainty_assumption,
                }
            )

        for location_id in sorted(crosswalk["location_id"]):
            mask_location = locations == location_id
            country_draws = np.zeros(draws)
            country_point = 0.0
            for age in AGE_ORDER:
                mask = mask_location & (ages == age)
                if mask.sum() != 1:
                    raise ValueError(
                        f"Table 1 requires one row for location={location_id}, age={age}, measure={measure}, year={year}"
                    )
                age_population = float(group.loc[mask, "population"].iloc[0])
                country_draws += AGE_WEIGHTS[age] * samples[mask][0] / age_population * 100000.0
                country_point += AGE_WEIGHTS[age] * float(group.loc[mask, "val"].iloc[0]) / age_population * 100000.0
            endpoint_rows.append(
                {
                    "entity_id": f"COUNTRY_{int(location_id)}",
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": year,
                    "rate": country_point,
                    "rate_lower": float(np.quantile(country_draws, 0.025)),
                    "rate_upper": float(np.quantile(country_draws, 0.975)),
                    "uncertainty_draws": draws,
                    "uncertainty_assumption": uncertainty_assumption,
                }
            )

    endpoint_table = pd.DataFrame(endpoint_rows)
    expected_endpoint_rows = 46 * 4 * 2
    if len(endpoint_table) != expected_endpoint_rows:
        raise ValueError(f"Table 1 expected {expected_endpoint_rows} endpoint rows")

    numeric_rows = []
    expected_years = list(range(1990, 2024))
    for entity in entities.sort_values("entity_order").itertuples():
        for measure, measure_name in measures:
            if entity.scope == "OECD total":
                series = oecd_asr.query(
                    "cause == 491 and measure == @measure and sex == 3"
                ).sort_values("year")[["year", "asr"]].rename(columns={"asr": "rate"})
            elif entity.scope == "OECD age group":
                age_id = int(entity.age_id)
                series = oecd_age_rate.query(
                    "cause == 491 and measure == @measure and sex == 3 and age == @age_id"
                ).sort_values("year")[["year", "rate"]]
            else:
                location_id = int(entity.location_id)
                series = country_asr.query(
                    "cause == 491 and measure == @measure and sex == 3 and location == @location_id"
                ).sort_values("year")[["year", "asr"]].rename(columns={"asr": "rate"})
            if series["year"].tolist() != expected_years or (series["rate"] <= 0).any():
                raise ValueError(
                    f"Incomplete positive annual Table 1 series for {entity.entity_id}, {measure_name}"
                )
            trend = bic_segmented_aapc_apc(
                series["year"].to_numpy(),
                series["rate"].to_numpy(),
                max_joinpoints=3,
                min_segment_years=5,
            )
            endpoint = endpoint_table.query(
                "entity_id == @entity.entity_id and measure == @measure"
            ).set_index("year")
            numeric_rows.append(
                {
                    "entity_order": entity.entity_order,
                    "section": entity.section,
                    "scope": entity.scope,
                    "location_id": entity.location_id,
                    "location_name": entity.location_name,
                    "iso3": entity.iso3,
                    "age_id": entity.age_id,
                    "age_group": entity.age_group,
                    "rate_type": entity.rate_type,
                    "measure": measure,
                    "measure_name": measure_name,
                    "rate_1990": float(endpoint.loc[1990, "rate"]),
                    "rate_1990_lower": float(endpoint.loc[1990, "rate_lower"]),
                    "rate_1990_upper": float(endpoint.loc[1990, "rate_upper"]),
                    "rate_2023": float(endpoint.loc[2023, "rate"]),
                    "rate_2023_lower": float(endpoint.loc[2023, "rate_lower"]),
                    "rate_2023_upper": float(endpoint.loc[2023, "rate_upper"]),
                    "aapc_1990_2023_percent": trend["aapc"],
                    "aapc_lower": trend["aapc_lower"],
                    "aapc_upper": trend["aapc_upper"],
                    "aapc_p_value": trend["aapc_p_value"],
                    "aapc_joinpoints": ";".join(str(year) for year in trend["knots"])
                    if trend["knots"]
                    else "None",
                    "aapc_bic": trend["bic"],
                    "endpoint_uncertainty_draws": draws,
                    "endpoint_uncertainty_assumption": uncertainty_assumption,
                    "aapc_method": (
                        "BIC-selected continuous segmented log-linear regression; maximum 3 joinpoints; "
                        "minimum segment length 5 years"
                    ),
                    "aapc_interval_type": "Regression-based 95% confidence interval",
                }
            )

    numeric = pd.DataFrame(numeric_rows).sort_values(["entity_order", "measure"]).reset_index(drop=True)
    if len(numeric) != 46 * 4:
        raise ValueError("Table 1 numeric output must contain 184 entity-measure rows")

    formatted_rows = []
    for entity in entities.sort_values("entity_order").itertuples():
        row = {
            "section": entity.section,
            "location_name": entity.location_name,
            "iso3": entity.iso3,
            "age_group": entity.age_group,
            "rate_type": entity.rate_type,
        }
        for measure, measure_name in measures:
            value = numeric.query(
                "entity_order == @entity.entity_order and measure == @measure"
            ).iloc[0]
            row[f"{measure_name}_1990_rate_per_100000_approx_95_UI"] = _format_table1_rate(
                value["rate_1990"], value["rate_1990_lower"], value["rate_1990_upper"]
            )
            row[f"{measure_name}_2023_rate_per_100000_approx_95_UI"] = _format_table1_rate(
                value["rate_2023"], value["rate_2023_lower"], value["rate_2023_upper"]
            )
            row[f"{measure_name}_1990_2023_AAPC_percent_95_CI"] = _format_table1_percent(
                value["aapc_1990_2023_percent"], value["aapc_lower"], value["aapc_upper"]
            )
        formatted_rows.append(row)
    formatted = pd.DataFrame(formatted_rows)

    save_csv(formatted, "Table1_OECD65plus_CVD_rates_AAPC.csv")
    save_csv(numeric, "Table1_OECD65plus_CVD_rates_AAPC_numeric.csv")
    return formatted, numeric


def table1_oecd_country_age_numbers_aapc(
    gbd: pd.DataFrame,
    country_number: pd.DataFrame,
    oecd_number: pd.DataFrame,
    oecd_age_rate: pd.DataFrame,
    entity_metadata: pd.DataFrame,
    draws: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the Number companion to Table 1 using the same 46 entities."""
    measures = [(6, "Incidence"), (5, "Prevalence"), (1, "Deaths"), (2, "DALYs")]
    endpoints = [1990, 2023]
    metadata_columns = [
        "entity_order",
        "section",
        "scope",
        "location_id",
        "location_name",
        "iso3",
        "age_id",
        "age_group",
    ]
    entities = (
        entity_metadata[metadata_columns]
        .drop_duplicates("entity_order")
        .sort_values("entity_order")
        .copy()
    )
    entities["entity_id"] = np.select(
        [
            entities["scope"].eq("OECD total"),
            entities["scope"].eq("OECD age group"),
            entities["scope"].eq("Country"),
        ],
        [
            "OECD_65PLUS",
            "OECD_AGE_" + entities["age_id"].fillna(0).astype(int).astype(str),
            "COUNTRY_" + entities["location_id"].astype(int).astype(str),
        ],
        default="",
    )
    entities["quantity_type"] = np.where(
        entities["scope"].eq("OECD age group"),
        "Age-group-specific number",
        "Number at ages 65+",
    )
    if len(entities) != 46 or (entities["entity_id"] == "").any():
        raise ValueError("Number Table 1 requires the same 46 entities as the rate Table 1")

    source = gbd.query(
        "cause == 491 and measure in [1, 2, 5, 6] and sex == 3 and metric == 1 and year in @endpoints"
    ).copy()
    expected_rows = 4 * 2 * 38 * len(AGE_ORDER)
    if len(source) != expected_rows:
        raise ValueError(f"Number Table 1 requires {expected_rows} endpoint rows")
    if (source[["val", "lower", "upper"]] <= 0).any().any():
        raise ValueError("Number Table 1 endpoint values and bounds must be positive")

    deviates = location_deviates(gbd["location"].unique())
    uncertainty_assumption = UI_ASSUMPTION
    endpoint_rows = []
    for (measure, year), group in source.groupby(["measure", "year"], sort=True):
        group = group.sort_values(["location", "age"]).reset_index(drop=True)
        samples = split_normal_samples(group, measure, deviates)
        ages = group["age"].to_numpy()
        locations = group["location"].to_numpy()

        oecd_draws = samples.sum(axis=0)
        endpoint_rows.append(
            {
                "entity_id": "OECD_65PLUS",
                "measure": measure,
                "measure_name": MEASURE_NAMES[measure],
                "year": year,
                "number": float(group["val"].sum()),
                "number_lower": float(np.quantile(oecd_draws, 0.025)),
                "number_upper": float(np.quantile(oecd_draws, 0.975)),
                "uncertainty_draws": draws,
                "uncertainty_assumption": uncertainty_assumption,
            }
        )

        for age in AGE_ORDER:
            mask = ages == age
            age_draws = samples[mask].sum(axis=0)
            endpoint_rows.append(
                {
                    "entity_id": f"OECD_AGE_{age}",
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": year,
                    "number": float(group.loc[mask, "val"].sum()),
                    "number_lower": float(np.quantile(age_draws, 0.025)),
                    "number_upper": float(np.quantile(age_draws, 0.975)),
                    "uncertainty_draws": draws,
                    "uncertainty_assumption": uncertainty_assumption,
                }
            )

        for location_id in sorted(entities.query("scope == 'Country'")["location_id"].astype(int)):
            mask = locations == location_id
            country_draws = samples[mask].sum(axis=0)
            endpoint_rows.append(
                {
                    "entity_id": f"COUNTRY_{location_id}",
                    "measure": measure,
                    "measure_name": MEASURE_NAMES[measure],
                    "year": year,
                    "number": float(group.loc[mask, "val"].sum()),
                    "number_lower": float(np.quantile(country_draws, 0.025)),
                    "number_upper": float(np.quantile(country_draws, 0.975)),
                    "uncertainty_draws": draws,
                    "uncertainty_assumption": uncertainty_assumption,
                }
            )

    endpoint_table = pd.DataFrame(endpoint_rows)
    if len(endpoint_table) != 46 * 4 * 2:
        raise ValueError("Number Table 1 must contain 368 endpoint rows")

    expected_years = list(range(1990, 2024))
    numeric_rows = []
    for entity in entities.itertuples():
        for measure, measure_name in measures:
            if entity.scope == "OECD total":
                series = oecd_number.query(
                    "cause == 491 and measure == @measure and sex == 3"
                ).sort_values("year")[["year", "number"]]
            elif entity.scope == "OECD age group":
                age_id = int(entity.age_id)
                series = oecd_age_rate.query(
                    "cause == 491 and measure == @measure and sex == 3 and age == @age_id"
                ).sort_values("year")[["year", "number"]]
            else:
                location_id = int(entity.location_id)
                series = country_number.query(
                    "cause == 491 and measure == @measure and sex == 3 and location == @location_id"
                ).sort_values("year")[["year", "number"]]
            if series["year"].tolist() != expected_years or (series["number"] <= 0).any():
                raise ValueError(
                    f"Incomplete positive annual Number series for {entity.entity_id}, {measure_name}"
                )
            trend = bic_segmented_aapc_apc(
                series["year"].to_numpy(),
                series["number"].to_numpy(),
                max_joinpoints=3,
                min_segment_years=5,
            )
            endpoint = endpoint_table.query(
                "entity_id == @entity.entity_id and measure == @measure"
            ).set_index("year")
            numeric_rows.append(
                {
                    "entity_order": entity.entity_order,
                    "section": entity.section,
                    "scope": entity.scope,
                    "location_id": entity.location_id,
                    "location_name": entity.location_name,
                    "iso3": entity.iso3,
                    "age_id": entity.age_id,
                    "age_group": entity.age_group,
                    "quantity_type": entity.quantity_type,
                    "measure": measure,
                    "measure_name": measure_name,
                    "number_1990": float(endpoint.loc[1990, "number"]),
                    "number_1990_lower": float(endpoint.loc[1990, "number_lower"]),
                    "number_1990_upper": float(endpoint.loc[1990, "number_upper"]),
                    "number_2023": float(endpoint.loc[2023, "number"]),
                    "number_2023_lower": float(endpoint.loc[2023, "number_lower"]),
                    "number_2023_upper": float(endpoint.loc[2023, "number_upper"]),
                    "aapc_1990_2023_percent": trend["aapc"],
                    "aapc_lower": trend["aapc_lower"],
                    "aapc_upper": trend["aapc_upper"],
                    "aapc_p_value": trend["aapc_p_value"],
                    "aapc_joinpoints": ";".join(str(year) for year in trend["knots"])
                    if trend["knots"]
                    else "None",
                    "aapc_bic": trend["bic"],
                    "endpoint_uncertainty_draws": draws,
                    "endpoint_uncertainty_assumption": uncertainty_assumption,
                    "aapc_method": (
                        "BIC-selected continuous segmented log-linear regression; maximum 3 joinpoints; "
                        "minimum segment length 5 years"
                    ),
                    "aapc_interval_type": "Regression-based 95% confidence interval",
                }
            )

    numeric = pd.DataFrame(numeric_rows).sort_values(["entity_order", "measure"]).reset_index(drop=True)
    if len(numeric) != 46 * 4:
        raise ValueError("Number Table 1 numeric output must contain 184 rows")

    formatted_rows = []
    for entity in entities.itertuples():
        row = {
            "section": entity.section,
            "location_name": entity.location_name,
            "iso3": entity.iso3,
            "age_group": entity.age_group,
            "quantity_type": entity.quantity_type,
        }
        for measure, measure_name in measures:
            value = numeric.query(
                "entity_order == @entity.entity_order and measure == @measure"
            ).iloc[0]
            row[f"{measure_name}_1990_number_approx_95_UI"] = _format_table1_number(
                value["number_1990"], value["number_1990_lower"], value["number_1990_upper"]
            )
            row[f"{measure_name}_2023_number_approx_95_UI"] = _format_table1_number(
                value["number_2023"], value["number_2023_lower"], value["number_2023_upper"]
            )
            row[f"{measure_name}_number_1990_2023_AAPC_percent_95_CI"] = _format_table1_percent(
                value["aapc_1990_2023_percent"], value["aapc_lower"], value["aapc_upper"]
            )
        formatted_rows.append(row)
    formatted = pd.DataFrame(formatted_rows)

    save_csv(formatted, "Table1_OECD65plus_CVD_numbers_AAPC.csv")
    save_csv(numeric, "Table1_OECD65plus_CVD_numbers_AAPC_numeric.csv")
    return formatted, numeric


def main() -> None:
    gbd, pop, sdi, gdp, geo = load_data()
    crosswalk = build_crosswalk(gbd, pop, geo)
    data_qc = validate_gbd(gbd)
    derived_pop, population_qc = derive_population(gbd, pop)
    country_number, country_asr = summarize_country(gbd)
    oecd_number, oecd_asr, oecd_age_rate = summarize_oecd(gbd, derived_pop)
    ui = approximate_oecd_ui(gbd, derived_pop)
    trends = calculate_country_trends(country_asr, country_number, crosswalk)
    convergence, convergence_models = convergence_analysis(country_asr, derived_pop, crosswalk)
    clusters, cluster_series, cluster_diagnostics = trajectory_clustering(country_asr, crosswalk)
    figure4_clusters, figure4_cluster_series, figure4_cluster_diagnostics = trajectory_clustering(
        country_asr, crosswalk, measures=(6, 2)
    )
    joinpoints = joinpoint_table(country_asr, oecd_asr, crosswalk)
    das_gupta = das_gupta_table(gbd, derived_pop, pop, crosswalk)
    context = context_table(derived_pop, pop, sdi, gdp, crosswalk)
    quadrants = performance_quadrants(trends, clusters)
    figure6_data = figure6_sdi_data(country_asr, sdi, crosswalk)
    selected_summary = oecd_selected_summary(oecd_number, oecd_asr, country_asr, ui)
    subtype = subtype_age_sex_table(gbd, derived_pop)
    figure3_uncertainty = figure3_age_sex_uncertainty(gbd, derived_pop)
    transitions = transition_indices(gbd)
    ets_forecast, ets_selection, ets_validation = ets_forecast_analysis(oecd_asr)
    country_ets_forecast, country_ets_selection, country_ets_validation = (
        ets_country_forecast_analysis(country_asr, crosswalk)
    )
    sensitivity = sensitivity_table(gbd, derived_pop)

    age_weight_table = pd.DataFrame(
        {
            "age_id": AGE_ORDER,
            "age_group": [AGE_NAMES[a] for a in AGE_ORDER],
            "gbd_world_standard_percent": [GBD_WORLD_STANDARD_PERCENT[a] for a in AGE_ORDER],
            "renormalized_weight_within_65plus": [AGE_WEIGHTS[a] for a in AGE_ORDER],
        }
    )
    save_csv(age_weight_table, "GBD2023_age_weights_65plus.csv")
    save_csv(country_asr.query("cause == 491 and sex == 3"), "Long_country_ASR_total_CVD.csv", True)
    save_csv(country_number.query("cause == 491 and sex == 3"), "Long_country_number_total_CVD.csv", True)
    save_csv(oecd_asr.query("cause == 491 and sex == 3"), "Long_OECD_ASR_total_CVD.csv", True)
    save_csv(oecd_number.query("cause == 491 and sex == 3"), "Long_OECD_number_total_CVD.csv", True)
    json_dump = {"population_qc": population_qc, "cluster_diagnostics": cluster_diagnostics,
                 "figure4_cluster_diagnostics": figure4_cluster_diagnostics}
    (RESULT / "diagnostics.json").write_text(json.dumps(json_dump, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    save_csv(data_qc, "Data_Availability_and_QC.csv")
    save_csv(crosswalk, "OECD38_crosswalk.csv")
    save_csv(context, "Table1_Country_Context_2023.csv")
    save_csv(trends, "Table2_Total_CVD_Country_Trends.csv")
    save_csv(selected_summary, "Table2B_OECD_Summary_Selected_Years.csv")
    save_csv(convergence_models, "Table3A_Convergence_Models.csv")
    save_csv(convergence, "Table3B_Convergence_Time_Series.csv")
    save_csv(clusters, "Table3C_Trajectory_Clusters.csv")
    save_csv(figure4_clusters, "Table3C_Figure4_Incidence_DALY_Trajectory_Clusters.csv")
    save_csv(joinpoints, "Table3D_Joinpoint_Segments.csv")
    save_csv(das_gupta, "Table3E_Das_Gupta_Decomposition_Incidence_DALYs.csv")
    save_csv(quadrants, "Table4_Country_Performance_Quadrants.csv")
    save_csv(subtype, "TableS1_Age_Sex_Subtype_Rates.csv")
    save_csv(transitions, "TableS2_OldestOld_and_Disability_Transition.csv")
    save_csv(sensitivity, "TableS3_Age_Threshold_Sensitivity.csv")
    save_csv(ui, "Approximate_UI_OECD_Total_CVD.csv")
    save_csv(country_ets_forecast, "Figure9_10_country_ETS_forecast_data.csv", True)
    save_csv(country_ets_selection, "Figure9_10_country_ETS_model_selection.csv", True)
    save_csv(country_ets_validation, "Figure9_10_country_ETS_holdout_validation.csv", True)

    print("Rendering Figure 1", flush=True)
    figure1_baseline(country_asr, geo, crosswalk)
    print("Rendering Figure 2", flush=True)
    figure2_heatmaps(oecd_asr, country_asr, crosswalk)
    print("Rendering Figure 3", flush=True)
    figure3_age_sex_subtype(subtype, figure3_uncertainty)
    print("Rendering Figure 4", flush=True)
    figure4_trajectories(figure4_clusters, figure4_cluster_series)
    print("Rendering Figure 5", flush=True)
    figure5_das_gupta(das_gupta)
    print("Rendering Figure 6", flush=True)
    figure6_sdi_relationships(figure6_data)
    print("Rendering Figure 7", flush=True)
    figure7_transition_indices(transitions)
    print("Rendering Figure 8", flush=True)
    figure8_ets_forecast(ets_forecast, ets_selection, ets_validation)
    print("Rendering Figure 9", flush=True)
    country_ets_forecast_figure(
        country_ets_forecast,
        6,
        "Figure9_OECD38_country_ETS_incidence_forecast_2050.pdf",
    )
    print("Rendering Figure 10", flush=True)
    country_ets_forecast_figure(
        country_ets_forecast,
        2,
        "Figure10_OECD38_country_ETS_DALY_forecast_2050.pdf",
    )

    print("Writing one-to-one main-figure CSV exports", flush=True)
    main_figure_csv_index = write_main_figure_csv_exports()
    print("Writing Table 1", flush=True)
    table1_formatted, table1_numeric = table1_oecd_country_age_rates_aapc(
        gbd,
        derived_pop,
        country_asr,
        oecd_asr,
        oecd_age_rate,
        crosswalk,
    )
    print("Writing Table 1 Number companion", flush=True)
    table1_number_formatted, table1_number_numeric = table1_oecd_country_age_numbers_aapc(
        gbd,
        country_number,
        oecd_number,
        oecd_age_rate,
        table1_numeric,
    )

    requirements = "pandas\nnumpy\npyarrow\nscipy\nstatsmodels\nscikit-learn\nmatplotlib\n"
    (RESULT / "requirements.txt").write_text(requirements, encoding="utf-8")
    import shutil
    shutil.rmtree(FIGURES, ignore_errors=True)
    manifest = {
        "analysis_date": "2026-11-09",
        "project": "OECD-38 age 65+ CVD burden, GBD 2023",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": mpl.__version__,
        },
        "parameters": {
            "years": [1990, 2023],
            "locations": 38,
            "ages": AGE_NAMES,
            "age_weights": AGE_WEIGHTS,
            "ui_draws": 1000,
            "cluster_k_candidates": [3, 4, 5],
            "cluster_bootstrap_reps": 500,
            "joinpoint_max_breaks": 2,
            "joinpoint_min_segment_years": 5,
            "figure2_aapc_apc_max_joinpoints": 3,
            "figure2_aapc_apc_min_segment_years": 5,
            "figure2_aapc_apc_selection": "BIC continuous segmented log-linear regression",
            "figure3_uncertainty_draws": 1000,
            "figure3_uncertainty_seed": UI_SEED,
            "figure3_uncertainty_assumption": UI_ASSUMPTION,
            "figure5_decomposition_method": "Symmetric three-factor Das Gupta decomposition averaged over all 3! replacement orders",
            "figure5_outcomes": ["Incidence", "DALYs"],
            "figure5_factors": ["Population growth", "Population ageing", "Epidemiological change"],
            "figure5_display_scale": "Factor contribution as percent of 1990 burden",
            "figure7_oldest_old_concentration": "Burden number at ages 80+ divided by burden number at ages 65+",
            "figure7_disability_transition": "YLD number divided by DALY number",
            "figure7_causes": ["Total CVD", "IHD", "Ischemic stroke", "ICH", "HHD"],
            "figure8_historical_years": [1990, 2023],
            "figure8_forecast_years": [2024, 2050],
            "figure8_outcomes": ["Incidence ASR", "DALY ASR"],
            "figure8_model_selection": "Minimum AICc among 10 non-seasonal ETS candidates",
            "figure8_interval": "95% model-based prediction interval from 20,000 simulated ETS paths",
            "figure8_interval_seed": 20260729,
            "figure8_validation": "Train 1990-2013; validate 2014-2023",
            "figure9_10_locations": 38,
            "figure9_10_forecast_years": [2024, 2050],
            "figure9_10_model_selection": "Country-specific minimum AICc among 10 non-seasonal ETS candidates",
            "figure9_10_interval": "95% model-based prediction interval from 10,000 simulated ETS paths per country and outcome",
            "figure9_10_interval_seed": 20260729,
            "figure9_10_validation": "Country-specific train 1990-2013; validate 2014-2023",
            "figure9_10_y_scales": "Country-specific within each panel",
            "main_figure_csv_exports": main_figure_csv_index.to_dict("records"),
            "table1_rows": len(table1_formatted),
            "table1_numeric_rows": len(table1_numeric),
            "table1_outcomes": ["Incidence", "Prevalence", "Deaths", "DALYs"],
            "table1_endpoint_years": [1990, 2023],
            "table1_endpoint_interval": UI_ASSUMPTION,
            "table1_aapc_period": [1990, 2023],
            "table1_aapc_method": "BIC-selected continuous segmented log-linear regression; maximum 3 joinpoints; minimum segment length 5 years",
            "table1_aapc_interval": "Regression-based 95% confidence interval",
            "table1_number_rows": len(table1_number_formatted),
            "table1_number_numeric_rows": len(table1_number_numeric),
            "table1_number_endpoint_interval": UI_ASSUMPTION,
            "table1_number_aapc_period": [1990, 2023],
            "table1_number_aapc_method": "BIC-selected continuous segmented log-linear regression; maximum 3 joinpoints; minimum segment length 5 years",
            "table1_number_aapc_interval": "Regression-based 95% confidence interval",
        },
        "population_qc": population_qc,
        "cluster_diagnostics": cluster_diagnostics,
        "figure4_cluster_diagnostics": figure4_cluster_diagnostics,
        "input_files": input_hashes(),
        "output_files": sorted(str(path.relative_to(RESULT)) for path in RESULT.rglob("*") if path.is_file()),
    }
    (RESULT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": len(list(FIGURES.glob("*.pdf"))), "tables": len(list(TABLES.glob("*.csv"))), "population_qc": population_qc, "cluster_diagnostics": cluster_diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
