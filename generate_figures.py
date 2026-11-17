#!/usr/bin/env python3
"""Main and supplementary figures for OECD-38 cardiovascular disease burden.

Reads analysis_outputs without recomputing statistical estimates. Exports 300 dpi
PNG and PDF figures, one source-data workbook per figure with a sheet per panel,
and one CSV per figure with a panel column. Colours are consistent across outcomes,
clusters, sexes and causes."""

import json
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator

CODE = Path(__file__).resolve().parent
OUTPUT = CODE / "results"
AO = OUTPUT / "analysis_outputs"
T, D = AO / "tables", AO / "figure_data"
DATA = CODE / "data"
FIG, FIG_S, SD = OUTPUT / "figures", OUTPUT / "supplementary_figures", OUTPUT / "source_data"
for d in (FIG, FIG_S, SD):
    d.mkdir(parents=True, exist_ok=True)

WIDTH = 7.2  # inches, about 183 mm double-column
INK, MUTED, GRID = "#1F2933", "#6B7280", "#E5E7EB"
M_COLOR = {6: "#0072B2", 5: "#009E73", 1: "#7A5195", 2: "#E69F00", 3: "#009E73"}
M_NAME = {6: "Incidence", 5: "Prevalence", 1: "Deaths", 2: "DALYs", 3: "YLDs"}
M_ADJ = {6: "Incidence", 5: "Prevalence", 1: "Death", 2: "DALY", 3: "YLD"}  # used before "rate"/"ASR"
M_CMAP = {6: "Blues", 5: "Greens", 1: "Purples", 2: "Oranges"}
CL_COLOR = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#7A5195", 5: "#D55E00"}
SEX_COLOR = {1: "#0072B2", 2: "#D55E00"}
SEX_NAME = {1: "Male", 2: "Female"}
CAUSE_COLOR = {493: "#E69F00", 495: "#0072B2", 496: "#56B4E9", 498: "#D55E00", -1: "#C9CDD3"}
CAUSE_NAME = {493: "IHD", 495: "Ischaemic stroke", 496: "ICH", 498: "HHD", -1: "Other CVD (residual)"}
AGE_IDS = [18, 19, 20, 30, 31, 32, 235]
AGES = ["65–69", "70–74", "75–79", "80–84", "85–89", "90–94", "95+"]
CAUSES_F7 = [491, 493, 495, 496, 498]
CAUSE_SHORT = {491: "Total CVD", 493: "IHD", 495: "Ischaemic stroke", 496: "ICH", 498: "HHD"}
TICK_YEARS = [1990, 2000, 2010, 2023]

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.3,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": INK, "legend.frameon": False,
})

SOURCE: dict = defaultdict(dict)


def read(folder, name):
    return pd.read_csv(folder / name, low_memory=False)


def num(x, k=0):
    return f"{x:,.{k}f}".replace("-", "−")


def pct(x, k=2, sign=False):
    return (f"{x:+.{k}f}%" if sign else f"{x:.{k}f}%").replace("-", "−")


WHITE_BOX = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9)


def spread(values, min_gap):
    """End-label positions: original order kept, neighbours at least min_gap apart."""
    order = np.argsort(values)
    out = np.asarray(values, float).copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def title(ax, letter, text=""):
    ax.set_title(f"{letter}  {text}".rstrip(), loc="left", fontweight="bold", pad=4)


def ygrid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    folder = FIG_S if name.startswith("SuppFig") else FIG
    fig.savefig(folder / f"{name}.png")
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)
    print("saved", name, flush=True)


def source(key, panel, frame, block=None):
    frame = frame.copy()
    if block is not None:
        frame.insert(0, "data_block", block)
    SOURCE[key].setdefault(str(panel), []).append(frame)


def write_source_data():
    for key, panels in SOURCE.items():
        merged = {p: pd.concat(fr, ignore_index=True) for p, fr in panels.items()}
        with pd.ExcelWriter(SD / f"{key}_SourceData.xlsx", engine="openpyxl") as xl:
            for p, frame in merged.items():
                frame.to_excel(xl, sheet_name=p[:31], index=False)
        long = pd.concat([f.assign(panel=p) for p, f in merged.items()], ignore_index=True)
        long = long[["panel"] + [c for c in long.columns if c != "panel"]]
        long.to_csv(SD / f"{key}_SourceData.csv", index=False, encoding="utf-8-sig")


crosswalk = read(T, "OECD38_crosswalk.csv")
ISO = crosswalk.set_index("location_id")["iso3"].to_dict()
ISO_ORDER = sorted(crosswalk["iso3"])
clusters = read(D, "Figure4_trajectory_clusters_incidence_DALYs.csv")
CLUSTER_OF = clusters.set_index("iso3")["cluster"].to_dict()
country_asr = read(D, "Long_country_ASR_total_CVD.csv")
country_asr["iso3"] = country_asr["location"].map(ISO)
oecd_asr = read(D, "Long_OECD_ASR_total_CVD.csv")
oecd_number = read(D, "Long_OECD_number_total_CVD.csv")


# ----------------------------------------------------------------------------- Fig 1
def geometry(feature):
    geom = feature.get("geometry") or {}
    rings = []
    if geom.get("type") == "Polygon":
        rings = [geom["coordinates"][0]]
    elif geom.get("type") == "MultiPolygon":
        rings = [poly[0] for poly in geom["coordinates"]]
    return [np.asarray(r)[:, :2] for r in rings if len(r) >= 3]


def draw_map(ax, feats, values, norm, cmap, xlim, ylim, lw):
    patches, fills = [], []
    for iso, rings in feats:
        fill = cmap(norm(values[iso])) if iso in values else "#F3F4F6"
        for ring in rings:
            patches.append(Polygon(ring, closed=True))
            fills.append(fill)
    ax.add_collection(PatchCollection(patches, facecolor=fills, edgecolor="#6B7280", linewidth=lw))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


def fig1():
    with (DATA / "df_world2.geojson").open(encoding="utf-8") as fh:
        geo = json.load(fh)
    feats = [(f["properties"].get("brk_a3"), geometry(f)) for f in geo["features"]]
    data = read(D, "Figure1_four_map_data.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5))
    for ax, m, letter in zip(axes.flat, [6, 5, 1, 2], "abcd"):
        block = data.query("measure_id == @m")
        values = block.set_index("iso3")["asr"].to_dict()
        norm = colors.Normalize(block["asr"].min(), block["asr"].max())
        cmap = mpl.colormaps[M_CMAP[m]]
        cmap = colors.LinearSegmentedColormap.from_list(M_CMAP[m], cmap(np.linspace(0.15, 0.95, 256)))
        draw_map(ax, feats, values, norm, cmap, (-170, 180), (-57, 78), 0.15)
        ax.axis("off")
        title(ax, letter, M_NAME[m])
        inset = ax.inset_axes([0.385, 0.015, 0.30, 0.56])
        draw_map(inset, feats, values, norm, cmap, (-12, 42), (34.5, 71), 0.18)
        inset.set_xticks([]); inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5); spine.set_edgecolor(MUTED)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.02, aspect=38)
        cb.set_label(f"{M_ADJ[m]} ASR per 100,000", fontsize=7, labelpad=1.5)
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        cb.outline.set_linewidth(0.4)
        cb.locator = MaxNLocator(nbins=5); cb.update_ticks()
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
        source("Fig1", letter, block[["location", "location_name", "iso3", "measure_name", "asr"]])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, hspace=0.16, wspace=0.03)
    save(fig, "Fig1_ASR_maps_OECD65plus_2023")


# ----------------------------------------------------------------------------- Fig 2 / SuppFig 1
def heatmap(ax, fig, matrix, m, letter, name, cb_ax):
    norm = colors.Normalize(np.nanmin(matrix.values), np.nanmax(matrix.values))
    cmap = mpl.colormaps[M_CMAP[m]]
    ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    years = list(matrix.columns)
    ax.set_xticks([years.index(y) for y in [1990, 1995, 2000, 2005, 2010, 2015, 2023]], [1990, 1995, 2000, 2005, 2010, 2015, 2023])
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=6.2)
    ax.tick_params(axis="y", length=0, pad=1.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Year")
    title(ax, letter, name)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="horizontal")
    cb.set_label(f"{M_ADJ[m]} ASR per 100,000", fontsize=7, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))


def trend_axis(ax):
    ax.set_xlim(1989.5, 2023.8)
    ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2023])
    ax.set_xlabel("Year")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
    ygrid(ax)


def fig2():
    trend = read(D, "Figure2_trend_data.csv")
    stats = read(D, "Figure2_AAPC_APC_data.csv")
    fig = plt.figure(figsize=(WIDTH, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.75, 6.3, 0.13], hspace=0.27, wspace=0.17,
                          left=0.075, right=0.985, top=0.97, bottom=0.05)
    for col, (m, top, low) in enumerate([(6, "a", "c"), (2, "b", "d")]):
        ax = fig.add_subplot(gs[0, col])
        block = trend.query("measure == @m and scope == 'OECD aggregate'").sort_values("year")
        st = stats.query("measure == @m")
        ax.scatter(block.year, block.asr, s=7, color=MUTED, zorder=3, linewidths=0, label="GBD-derived ASR")
        ax.plot(block.year, block.fitted_asr, color=M_COLOR[m], lw=1.5, zorder=4, label="Segmented fit")
        aapc = st.query("statistic == 'AAPC'").iloc[0]
        span = block.asr.max() - block.asr.min()
        segments = st.query("statistic == 'APC'").sort_values("start_year")
        for i, s in enumerate(segments.itertuples()):
            if i:
                ax.axvline(s.start_year, color=MUTED, lw=0.5, ls="--", zorder=1)
            mid = (s.start_year + s.end_year) / 2
            y = np.interp(mid, block.year, block.fitted_asr)
            ax.text(mid, y + span * 0.10, pct(s.estimate), ha="center", va="bottom", fontsize=6.3)
        ax.text(0.02, 0.05, f"AAPC {pct(aapc.estimate)} ({pct(aapc.lower)} to {pct(aapc.upper)})",
                transform=ax.transAxes, fontsize=6.6, fontweight="bold", bbox=WHITE_BOX, zorder=6)
        ax.set_ylim(block.asr.min() - span * 0.12, block.asr.max() + span * 0.2)
        ax.set_ylabel("ASR per 100,000")
        trend_axis(ax)
        title(ax, top, f"OECD {M_NAME[m]}")
        if col == 0:
            ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0, labelspacing=0.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
        source("Fig2", top, block[["year", "asr", "fitted_asr", "segment"]].assign(measure=M_NAME[m]), "trend")
        source("Fig2", top, st, "apc_aapc")
        matrix = trend.query("measure == @m and scope == 'OECD country'").pivot(index="iso3", columns="year", values="asr").reindex(ISO_ORDER)
        hm = fig.add_subplot(gs[1, col])
        heatmap(hm, fig, matrix, m, low, f"Country {M_ADJ[m]} ASR", fig.add_subplot(gs[2, col]))
        source("Fig2", low, matrix.reset_index())
    save(fig, "Fig2_trends_APC_AAPC_heatmaps_1990_2023")


def suppfig1():
    table = read(T, "Table1_OECD65plus_CVD_rates_AAPC_numeric.csv").query("section == 'OECD overall'")
    fig = plt.figure(figsize=(WIDTH, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.75, 6.3, 0.13], hspace=0.27, wspace=0.17,
                          left=0.075, right=0.985, top=0.97, bottom=0.05)
    for col, (m, top, low) in enumerate([(5, "a", "c"), (1, "b", "d")]):
        ax = fig.add_subplot(gs[0, col])
        block = oecd_asr.query("measure == @m").sort_values("year")
        ax.plot(block.year, block.asr, color=M_COLOR[m], lw=1.5, zorder=3)
        row = table.query("measure == @m").iloc[0]
        ax.text(0.02, 0.05, f"AAPC {pct(row.aapc_1990_2023_percent)} ({pct(row.aapc_lower)} to {pct(row.aapc_upper)})",
                transform=ax.transAxes, fontsize=6.6, fontweight="bold", bbox=WHITE_BOX, zorder=6)
        span = block.asr.max() - block.asr.min()
        ax.set_ylim(block.asr.min() - span * 0.2, block.asr.max() + span * 0.08)
        ax.set_ylabel("ASR per 100,000")
        trend_axis(ax)
        title(ax, top, f"OECD {M_NAME[m]}")
        source("SuppFig1", top, block[["year", "asr"]].assign(measure=M_NAME[m]))
        matrix = country_asr.query("measure == @m").pivot(index="iso3", columns="year", values="asr").reindex(ISO_ORDER)
        hm = fig.add_subplot(gs[1, col])
        heatmap(hm, fig, matrix, m, low, f"Country {M_ADJ[m]} ASR", fig.add_subplot(gs[2, col]))
        source("SuppFig1", low, matrix.reset_index())
    save(fig, "SuppFig1_prevalence_deaths_trends_heatmaps")


# ----------------------------------------------------------------------------- Fig 3
def fig3():
    base = read(D, "Figure3_age_sex_subtype_data.csv")
    unc = read(D, "Figure3_age_sex_uncertainty_data.csv")
    derived = read(D, "Figure3_age_distribution_and_cause_share_data.csv")
    x = np.arange(7)
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 7.6))
    for col, m in enumerate([6, 2]):
        ax = axes[0, col]
        f3 = base[(base.cause == 491) & (base.measure == m) & (base.age == 235)].set_index("sex").age_specific_rate
        ends = [f3[1], f3[2]]
        label_y = spread(ends, (max(ends) * 1.1) * 0.06)
        for s in (1, 2):
            b = base.query("cause == 491 and measure == @m and sex == @s").set_index("age").reindex(AGE_IDS)
            u = unc.query("measure == @m and sex == @s").set_index("age").reindex(AGE_IDS)
            ax.fill_between(x, u.rate_lower, u.rate_upper, color=SEX_COLOR[s], alpha=0.2, lw=0)
            ax.plot(x, b.age_specific_rate, color=SEX_COLOR[s], marker="o", ms=2.8)
            ax.text(6.08, label_y[s - 1], SEX_NAME[s], fontsize=6.8, va="center")
            source("Fig3", "ab"[col], b.reset_index()[["age_name", "sex_name", "measure_name", "number", "population", "age_specific_rate"]].assign(
                rate_lower=u.rate_lower.values, rate_upper=u.rate_upper.values))
        ax.set_ylabel(f"{M_ADJ[m]} rate per 100,000")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
        title(ax, "ab"[col], f"Age-specific {M_ADJ[m]} rate")
        ax.set_xlim(-0.2, 6.85)
        ygrid(ax)
        ax = axes[1, col]
        for s in (1, 2):
            b = derived.query("panel_data == 'age_distribution' and measure == @m and sex == @s").set_index("age").reindex(AGE_IDS)
            ax.plot(x, b.value_percent, color=SEX_COLOR[s], marker="o", ms=2.8)
            ax.text(6.08, b.value_percent.iloc[-1], SEX_NAME[s], fontsize=6.8, va="center")
            source("Fig3", "cd"[col], pd.DataFrame({"age": AGES, "sex": SEX_NAME[s], "measure": M_NAME[m],
                   "share_percent": b.value_percent.values}))
        ax.set_ylabel(f"Share of 65+ {M_NAME[m]} number (%)")
        title(ax, "cd"[col], f"Age distribution of {M_NAME[m]} number")
        ax.set_xlim(-0.2, 6.85)
        ax.set_ylim(bottom=0)
        ygrid(ax)
        ax = axes[2, col]
        comp = derived.query("panel_data == 'cause_composition' and measure == @m")
        bottom = np.zeros(7)
        for c in [493, 495, 496, 498, -1]:
            v = comp.query("cause == @c").set_index("age").reindex(AGE_IDS).value_percent.fillna(0).values
            ax.bar(x, v, 0.76, bottom=bottom, color=CAUSE_COLOR[c], edgecolor="white", linewidth=0.5, zorder=2)
            bottom += v
        source("Fig3", "ef"[col], comp.assign(cause_name=comp.cause.map(CAUSE_NAME), measure_name=M_NAME[m])[["age", "cause_name", "measure_name", "value_percent"]])
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"Share of {M_NAME[m]} within age (%)")
        title(ax, "ef"[col], f"Cause composition of {M_NAME[m]}")
        ygrid(ax)
        for row in axes[:, col]:
            row.set_xticks(x, AGES, fontsize=6.6)
        ax.set_xlabel("Age group (years)")
    handles = [Patch(color=CAUSE_COLOR[c], label=CAUSE_NAME[c] + (" (DALYs only)" if c == 498 else "")) for c in [493, 495, 496, 498, -1]]
    fig.legend(handles=handles, loc="lower center", ncol=5, handlelength=1.1, columnspacing=1.3, bbox_to_anchor=(0.5, 0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.975, bottom=0.095, hspace=0.33, wspace=0.22)
    save(fig, "Fig3_age_sex_cause_structure_2023")


# ----------------------------------------------------------------------------- Fig 4 / SuppFig 3-4
def cluster_labels():
    out = {}
    for c, b in clusters.sort_values("iso3").groupby("cluster"):
        out[c] = f"Cluster {c} (n={len(b)}): " + ", ".join(b.iso3)
    return out


def fig4():
    series = read(D, "Figure4D_cluster_series_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.2))
    for ax, m, letter in zip(axes, [6, 2], "ab"):
        block = series.query("measure == @m")
        last = block.query("year == 2023").sort_values("cluster")
        label_y = dict(zip(last.cluster, spread(last.mean_index.values, 2.6)))
        for c, b in block.groupby("cluster"):
            ax.fill_between(b.year, b.lower_quartile, b.upper_quartile, color=CL_COLOR[c], alpha=0.16, lw=0)
            ax.plot(b.year, b.mean_index, color=CL_COLOR[c], lw=1.5)
            ax.text(2023.5, label_y[c], f"{c}", fontsize=7, fontweight="bold", va="center")
        ax.axhline(100, color=INK, lw=0.6, ls=":")
        ax.set_xlim(1990, 2025)
        ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2023])
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{M_ADJ[m]} ASR index (1990 = 100)")
        title(ax, letter, f"{M_ADJ[m]} trajectories")
        ygrid(ax)
        source("Fig4", letter, block[["measure_name", "cluster", "year", "mean_index", "lower_quartile", "upper_quartile", "countries"]], "cluster_series")
    source("Fig4", "membership", clusters, "cluster_membership")
    handles = [Line2D([0], [0], color=CL_COLOR[c], lw=1.8, label=t) for c, t in cluster_labels().items()]
    fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=6.3, bbox_to_anchor=(0.04, -0.01), columnspacing=1.5, handlelength=1.4, labelspacing=0.3)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.235, wspace=0.2)
    save(fig, "Fig4_incidence_DALY_trajectory_clusters")


def suppfig3():
    base = country_asr.query("measure in [6, 2]").copy()
    ref = base.query("year == 1990").set_index(["iso3", "measure"]).asr
    base["index"] = 100 * base.asr / [ref[(i, m)] for i, m in zip(base.iso3, base.measure)]
    base["cluster"] = base.iso3.map(CLUSTER_OF)
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 3.9), sharex=True, sharey="row")
    letters = iter("abcdefghij")
    for row, m in enumerate([6, 2]):
        for col, c in enumerate(range(1, 6)):
            ax = axes[row, col]
            block = base.query("measure == @m and cluster == @c")
            for _, b in block.groupby("iso3"):
                ax.plot(b.year, b["index"], color="#AEB4BC", lw=0.6)
            mean = block.groupby("year")["index"].mean()
            ax.plot(mean.index, mean.values, color=CL_COLOR[c], lw=1.6)
            ax.axhline(100, color=INK, lw=0.5, ls=":")
            letter = next(letters)
            title(ax, letter, f"Cluster {c}")
            ax.set_xticks([1990, 2023])
            ygrid(ax)
            source("SuppFig3", letter, block[["iso3", "year", "measure", "index", "cluster"]])
        axes[row, 0].set_ylabel(f"{M_ADJ[m]} ASR index")
    fig.supxlabel("Year", fontsize=8, y=0.01)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.11, hspace=0.3, wspace=0.12)
    save(fig, "SuppFig3_cluster_country_trajectories")


def suppfig4():
    diag = json.loads((AO / "diagnostics.json").read_text(encoding="utf-8"))["figure4_cluster_diagnostics"]
    sil = pd.DataFrame(diag["silhouette_by_k"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), gridspec_kw={"width_ratios": [1, 2.3]})
    ax = axes[0]
    ax.bar(sil.k.astype(str), sil.silhouette, width=0.55, color="#0072B2", zorder=2)
    for k, v in zip(sil.k.astype(str), sil.silhouette):
        ax.text(k, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.8)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Candidate number of clusters")
    ax.set_ylabel("Silhouette")
    title(ax, "a", "Selection of K")
    ygrid(ax)
    ax.text(0.03, 0.95, f"Bootstrap ARI {diag['bootstrap_mean_adjusted_rand_index']:.3f}\n({diag['bootstrap_ari_2_5_percentile']:.3f}–{diag['bootstrap_ari_97_5_percentile']:.3f}), 500 resamples",
            transform=ax.transAxes, fontsize=6.3, va="top")
    source("SuppFig4", "a", sil)
    ax = axes[1]
    for c, b in clusters.groupby("cluster"):
        ax.scatter(b.asir_index_2023, b.asdr_index_2023, s=16, color=CL_COLOR[c], zorder=3, linewidths=0, label=f"Cluster {c}")
    fig.canvas.draw()
    placed, offsets = [], [(3, 2), (3, -6), (-15, 2), (-15, -6), (-6, 4.5), (-6, -9), (3, 7), (-15, -11), (3, -11), (-15, 7)]
    pts = ax.transData.transform(clusters[["asir_index_2023", "asdr_index_2023"]].values) * 72 / fig.dpi
    for r, (px, py) in sorted(zip(clusters.itertuples(), pts), key=lambda t: t[1][0]):
        def free(dx, dy):
            box = (px + dx, py + dy, px + dx + 13, py + dy + 6)
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
            dot = any(box[0] - 2 < qx < box[2] + 2 and box[1] - 2 < qy < box[3] + 2 for qx, qy in pts if (qx, qy) != (px, py))
            return not hit and not dot
        dx, dy = next((o for o in offsets if free(*o)), offsets[0])
        placed.append((px + dx, py + dy, px + dx + 13, py + dy + 6))
        ax.annotate(r.iso3, (r.asir_index_2023, r.asdr_index_2023), xytext=(dx, dy), textcoords="offset points", fontsize=5.6)
    ax.set_xlabel("Incidence ASR index in 2023 (1990 = 100)")
    ax.set_ylabel("DALY ASR index in 2023 (1990 = 100)")
    title(ax, "b", "Country positions in 2023")
    ygrid(ax, "both")
    ax.legend(loc="upper left", handletextpad=0.1, labelspacing=0.2, borderaxespad=0.2)
    source("SuppFig4", "b", clusters)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.14, wspace=0.24)
    save(fig, "SuppFig4_cluster_diagnostics")


# ----------------------------------------------------------------------------- Fig 5
def fig5():
    dec = read(D, "Figure5_Das_Gupta_decomposition_data.csv")
    comps = [("population_growth_percent_of_1990", "Population growth", "#8A929C"),
             ("population_ageing_percent_of_1990", "Population ageing", "#E69F00"),
             ("epidemiological_change_percent_of_1990", "Epidemiological change", "#0072B2")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 8.3))
    for ax, m, letter in zip(axes, [6, 2], "ab"):
        countries = dec.query("scope == 'Country' and measure == @m").sort_values("total_change_percent_of_1990")
        block = pd.concat([countries, dec.query("scope == 'OECD aggregate' and measure == @m")], ignore_index=True)
        y = np.arange(len(block))
        pos, neg = np.zeros(len(block)), np.zeros(len(block))
        ax.axhspan(len(block) - 1.5, len(block) - 0.5, color="#F3F4F6", zorder=0)
        for col, name, color in comps:
            v = block[col].to_numpy()
            ax.barh(y, v, left=np.where(v >= 0, pos, neg), height=0.72, color=color, edgecolor="white", linewidth=0.3, zorder=2)
            pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
        ax.scatter(block.total_change_percent_of_1990, y, marker="D", s=11, color=INK, edgecolors="white", linewidths=0.35, zorder=4)
        ax.axvline(0, color=INK, lw=0.6, zorder=3)
        ax.set_yticks(y, block.iso3.replace({"OECD38": "OECD"}), fontsize=6.4)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(block) - 0.3)
        ax.set_xlabel("Contribution (% of 1990 burden)")
        title(ax, letter, M_NAME[m])
        ygrid(ax, "x")
        source("Fig5", letter, block[["iso3", "location_name", "measure_name"] + [c for c, _, _ in comps] + ["total_change_percent_of_1990", "population_growth", "population_ageing", "epidemiological_change", "total_change", "additivity_error"]])
    handles = [Patch(color=c, label=n) for _, n, c in comps] + [Line2D([0], [0], marker="D", color=INK, ls="none", ms=3.6, label="Net change")]
    fig.legend(handles=handles, loc="lower center", ncol=4, handlelength=1.1, bbox_to_anchor=(0.5, 0))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.975, bottom=0.075, wspace=0.17)
    save(fig, "Fig5_DasGupta_decomposition_1990_2023")


# ----------------------------------------------------------------------------- Fig 6
def fig6():
    data = read(D, "Figure6_incidence_DALY_SDI_data.csv")
    fit = read(D, "Figure6_LOWESS_fit_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.7))
    for ax, m, letter in zip(axes, [6, 2], "ab"):
        block = data.query("measure == @m")
        for iso, b in block.groupby("iso3"):
            b = b.sort_values("year")
            ax.plot(b.sdi, b.rate_per_100000, color=CL_COLOR[CLUSTER_OF[iso]], lw=0.65, alpha=0.62, zorder=2)
        f = fit.query("measure == @m").sort_values("sdi")
        ax.plot(f.sdi, f.fitted_rate_per_100000, color=INK, lw=2, zorder=4)
        ax.set_xlabel("Socio-demographic Index")
        ax.set_ylabel(f"{M_ADJ[m]} ASR per 100,000")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
        title(ax, letter, f"{M_ADJ[m]} ASR and SDI")
        ygrid(ax)
        source("Fig6", letter, block.assign(cluster=block.iso3.map(CLUSTER_OF))[["iso3", "year", "sdi", "measure_name", "rate_per_100000", "cluster"]], "country_year")
        source("Fig6", letter, f, "pooled_lowess")
    handles = [Line2D([0], [0], color=CL_COLOR[c], lw=1.2, label=f"Cluster {c}") for c in range(1, 6)] + [Line2D([0], [0], color=INK, lw=2, label="Pooled LOWESS")]
    fig.legend(handles=handles, loc="lower center", ncol=6, handlelength=1.5, columnspacing=1.2, bbox_to_anchor=(0.5, 0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.2, wspace=0.26)
    save(fig, "Fig6_SDI_association_1990_2023")


# ----------------------------------------------------------------------------- Fig 7
def fig7():
    conc = read(D, "Figure7_oldest_old_concentration_data.csv")
    dis = read(D, "Figure7_disability_transition_data.csv")
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 3.9), sharex=True, sharey="row")
    top, low = iter("abcde"), iter("fghij")
    for col, c in enumerate(CAUSES_F7):
        ax = axes[0, col]
        letter = next(top)
        block = conc.query("cause == @c")
        for m in (1, 2, 3):
            b = block.query("measure == @m").sort_values("year")
            ax.plot(b.year, b.oldest_old_concentration_percent, color=M_COLOR[m], lw=1.3)
        ax.set_ylim(20, 85)
        title(ax, letter, CAUSE_SHORT[c])
        ygrid(ax)
        source("Fig7", letter, block[["cause_short", "measure_name", "year", "burden_65plus", "burden_80plus", "oldest_old_concentration_percent"]])
        ax = axes[1, col]
        letter = next(low)
        b = dis.query("cause == @c").sort_values("year")
        ax.plot(b.year, b.yld_to_daly_percent, color=INK, lw=1.4)
        ax.set_ylim(0, 25)
        ax.set_xticks(TICK_YEARS[::1], [1990, "", 2010, 2023])
        title(ax, letter, CAUSE_SHORT[c])
        ygrid(ax)
        source("Fig7", letter, b[["cause_short", "year", "yld_to_daly_percent"]])
    axes[0, 0].set_ylabel("Ages 80+ share of 65+ (%)")
    axes[1, 0].set_ylabel("YLDs / DALYs (%)")
    fig.supxlabel("Year", fontsize=8, y=0.075)
    handles = [Line2D([0], [0], color=M_COLOR[m], lw=1.5, label=M_NAME[m]) for m in (1, 2, 3)] + [Line2D([0], [0], color=INK, lw=1.5, label="YLDs / DALYs")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.2, hspace=0.3, wspace=0.12)
    save(fig, "Fig7_oldest_old_disability_transition")


# ----------------------------------------------------------------------------- Fig 8-10
def fig8():
    fc = read(D, "Figure8_ETS_forecast_data.csv")
    val = read(D, "Figure8_ETS_holdout_validation.csv").set_index("measure")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3))
    for ax, m, letter in zip(axes, [6, 2], "ab"):
        block = fc.query("measure == @m")
        obs, fut = block.query("series == 'Observed'").sort_values("year"), block.query("series == 'Forecast'").sort_values("year")
        ax.fill_between(fut.year, fut.pi_lower, fut.pi_upper, color=M_COLOR[m], alpha=0.2, lw=0)
        ax.plot(obs.year, obs.observed_asr, color=INK, lw=1.4)
        ax.plot(np.r_[2023, fut.year], np.r_[obs.observed_asr.iloc[-1], fut.forecast_asr], color=M_COLOR[m], lw=1.5)
        ax.axvline(2023, color=MUTED, lw=0.5, ls="--")
        end, v = fut.iloc[-1], val.loc[m]
        ax.text(0.98, 0.97, f"2050: {num(end.forecast_asr)} (95% PI {num(end.pi_lower)}–{num(end.pi_upper)})", transform=ax.transAxes, ha="right", va="top", fontsize=6.6, fontweight="bold", bbox=WHITE_BOX, zorder=6)
        ax.text(0.03, 0.05, f"{v.model}; hold-out 2014–2023\nMAPE {v.mape_percent:.1f}%; nominal 95% PI coverage {v.prediction_interval_coverage_percent:.0f}%", transform=ax.transAxes, fontsize=6.3)
        ax.set_xlim(1990, 2051)
        ax.set_xticks([1990, 2000, 2010, 2023, 2040, 2050])
        ax.set_ylim(bottom=0, top=max(fut.pi_upper.max(), obs.observed_asr.max()) * 1.12)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{M_ADJ[m]} ASR per 100,000")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: num(x)))
        title(ax, letter, M_NAME[m])
        ygrid(ax)
        source("Fig8", letter, block[["measure_name", "year", "series", "observed_asr", "forecast_asr", "pi_lower", "pi_upper", "model", "interval"]])
    lines = (Line2D([0], [0], color=M_COLOR[6], lw=1.5), Line2D([0], [0], color=M_COLOR[2], lw=1.5))
    bands = (Patch(color=M_COLOR[6], alpha=0.25), Patch(color=M_COLOR[2], alpha=0.25))
    fig.legend([Line2D([0], [0], color=INK, lw=1.4), lines, bands], ["Observed ASR", "ETS forecast", "95% prediction interval"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0), handlelength=2.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.21, wspace=0.26)
    save(fig, "Fig8_OECD_ETS_forecast_2050")


COUNTRY_FC = read(D, "Figure9_10_country_ETS_forecast_data.csv")


def country_grid(m, key, name):
    block = COUNTRY_FC.query("measure == @m")
    fig, axes = plt.subplots(7, 6, figsize=(WIDTH, 9.1))
    flat = axes.ravel()
    for i, iso in enumerate(ISO_ORDER):
        ax = flat[i]
        c = block.query("iso3 == @iso")
        obs, fut = c.query("series == 'Observed'").sort_values("year"), c.query("series == 'Forecast'").sort_values("year")
        ax.fill_between(fut.year, fut.pi_lower, fut.pi_upper, color=M_COLOR[m], alpha=0.22, lw=0)
        ax.plot(obs.year, obs.observed_asr, color=INK, lw=0.9)
        ax.plot(np.r_[2023, fut.year], np.r_[obs.observed_asr.iloc[-1], fut.forecast_asr], color=M_COLOR[m], lw=1.1)
        ax.axvline(2023, color=MUTED, lw=0.4, ls="--")
        ax.set_title(iso, loc="left", fontweight="bold", fontsize=7, pad=2)
        ax.set_xlim(1990, 2051)
        ax.set_xticks([1990, 2023, 2050])
        ax.tick_params(labelsize=5.8, pad=1.2, length=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:g}k"))
        ax.set_ylim(bottom=0)
        ygrid(ax)
        source(key, iso, c[["iso3", "measure_name", "year", "series", "observed_asr", "forecast_asr", "pi_lower_raw", "pi_lower", "pi_upper", "pi_lower_clipped_at_zero", "model"]])
    for ax in flat[38:]:
        ax.remove()
    handles = [Line2D([0], [0], color=INK, lw=1.2, label="Observed ASR"), Line2D([0], [0], color=M_COLOR[m], lw=1.4, label="ETS forecast"),
               Patch(color=M_COLOR[m], alpha=0.25, label="95% prediction interval")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.74, 0.075), ncol=1, handlelength=1.6)
    fig.supylabel(f"{M_ADJ[m]} ASR per 100,000 (country-specific scale)", fontsize=8, x=0.008)
    fig.supxlabel("Year", fontsize=8, y=0.012)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.975, bottom=0.05, hspace=0.42, wspace=0.34)
    save(fig, name)


# ----------------------------------------------------------------------------- SuppFig 2, 5-7
def suppfig2():
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.5), sharey=True)
    for ax, m, letter in zip(axes, [6, 5, 1, 2], "abcd"):
        r = oecd_asr.query("measure == @m").sort_values("year")
        n = oecd_number.query("measure == @m").sort_values("year")
        ri, ni = 100 * r.asr.values / r.asr.iloc[0], 100 * n.number.values / n.number.iloc[0]
        ax.plot(r.year, ni, color="#D55E00", lw=1.5)
        ax.plot(r.year, ri, color="#0072B2", lw=1.5)
        ax.axhline(100, color=INK, lw=0.5, ls=":")
        ax.set_xticks([1990, 2005, 2023])
        ax.set_xlabel("Year")
        title(ax, letter, M_NAME[m])
        ygrid(ax)
        source("SuppFig2", letter, pd.DataFrame({"year": r.year.values, "measure": M_NAME[m], "asr": r.asr.values, "number": n.number.values, "asr_index": ri, "number_index": ni}))
    axes[0].set_ylabel("Index (1990 = 100)")
    fig.legend(handles=[Line2D([0], [0], color="#D55E00", lw=1.5, label="Number at ages 65+"), Line2D([0], [0], color="#0072B2", lw=1.5, label="65+ ASR")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.27, wspace=0.1)
    save(fig, "SuppFig2_rate_vs_number_index")


def suppfig5():
    t = read(T, "TableS3_Age_Threshold_Sensitivity.csv")
    labels = ["65+", "70+", "80+"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, col, name, letter in zip(axes, ["asr_percent_change", "number_percent_change"], ["ASR change, 1990–2023 (%)", "Number change, 1990–2023 (%)"], "ab"):
        for m, off, mark in [(1, 0.12, "o"), (2, -0.12, "s")]:
            b = t.query("measure == @m").set_index("age_threshold").reindex(labels)
            ax.scatter(b[col], np.arange(3) + off, s=26, marker=mark, color=M_COLOR[m], zorder=3, linewidths=0, label=M_NAME[m])
            for v, y in zip(b[col], np.arange(3) + off):
                ax.annotate(pct(v, 1, True), (v, y), xytext=(5, 0), textcoords="offset points", fontsize=6.5, va="center")
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(range(3), [f"Ages {x}" for x in labels])
        ax.margins(x=0.22)
        ax.invert_yaxis()
        ax.set_xlabel(name)
        title(ax, letter, name.split(",")[0])
        ygrid(ax, "x")
        source("SuppFig5", letter, t[["age_threshold", "measure_name", col]])
    axes[0].legend(loc="lower right")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.2, wspace=0.08)
    save(fig, "SuppFig5_age_threshold_sensitivity")


def suppfig6():
    v = read(D, "Figure9_10_country_ETS_holdout_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 6.6), sharey=True)
    y = np.arange(len(ISO_ORDER))
    for ax, m, letter in zip(axes, [6, 2], "ab"):
        b = v.query("measure == @m").set_index("iso3").reindex(ISO_ORDER)
        ax.hlines(y, 0, b.mape_percent, color=GRID, lw=0.8, zorder=1)
        ax.scatter(b.mape_percent, y, s=15, color=M_COLOR[m], zorder=3, linewidths=0)
        med = b.mape_percent.median()
        ax.axvline(med, color=INK, lw=0.7, ls="--")
        ax.text(med, -1.1, f" median {med:.2f}%", fontsize=6.6, va="bottom")
        ax.set_yticks(y, ISO_ORDER, fontsize=6.4)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(38, -1.3)
        ax.set_xlim(left=0)
        ax.set_xlabel("Hold-out MAPE, 2014–2023 (%)")
        title(ax, letter, M_NAME[m])
        source("SuppFig6", letter, b.reset_index()[["iso3", "measure_name", "model", "mae", "rmse", "mape_percent"]])
    fig.subplots_adjust(left=0.065, right=0.985, top=0.96, bottom=0.075, wspace=0.08)
    save(fig, "SuppFig6_ETS_holdout_country_MAPE")


def suppfig7():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 6.6), sharey=True)
    y = np.arange(len(ISO_ORDER))
    for ax, m, letter in zip(axes, [6, 2], "ab"):
        b = COUNTRY_FC.query("measure == @m")
        o = b.query("year == 2023 and series == 'Observed'").set_index("iso3").reindex(ISO_ORDER)
        f = b.query("year == 2050").set_index("iso3").reindex(ISO_ORDER)
        left = 300 if m == 6 else 1500
        ax.hlines(y, f.pi_lower.clip(lower=left), f.pi_upper, color=M_COLOR[m], alpha=0.4, lw=2.2, zorder=1)
        ax.scatter(f.forecast_asr, y, s=15, color=M_COLOR[m], zorder=3, linewidths=0)
        ax.scatter(o.observed_asr, y, s=15, facecolors="white", edgecolors=INK, linewidths=0.7, zorder=4)
        ax.set_yticks(y, ISO_ORDER, fontsize=6.4)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(37.7, -0.7)
        ax.set_xscale("log")
        ax.set_xlim(left=left)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: num(x)))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: num(x) if str(int(x))[0] in "25" else ""))
        ax.tick_params(axis="x", which="minor", labelsize=6)
        ax.set_xlabel(f"{M_ADJ[m]} ASR per 100,000 (log scale)")
        title(ax, letter, M_NAME[m])
        ygrid(ax, "x")
        source("SuppFig7", letter, pd.DataFrame({"iso3": ISO_ORDER, "measure": M_NAME[m], "observed_asr_2023": o.observed_asr.values, "forecast_asr_2050": f.forecast_asr.values,
                                                  "pi_lower_2050": f.pi_lower.values, "pi_upper_2050": f.pi_upper.values, "pi_lower_clipped_at_zero": f.pi_lower_clipped_at_zero.values}))
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="none", mfc="white", mec=INK, ms=4, label="Observed 2023"), Line2D([0], [0], marker="o", ls="none", color=MUTED, ms=4, label="ETS forecast 2050"),
                        Line2D([0], [0], color=MUTED, alpha=0.5, lw=2.2, label="95% prediction interval 2050")], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.095, wspace=0.08)
    save(fig, "SuppFig7_country_2050_vs_2023")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    country_grid(6, "Fig9", "Fig9_country_ETS_incidence_2050")
    country_grid(2, "Fig10", "Fig10_country_ETS_DALY_2050")
    suppfig1(); suppfig2(); suppfig3(); suppfig4(); suppfig5(); suppfig6(); suppfig7()
    write_source_data()
    print("figures and source data complete")
