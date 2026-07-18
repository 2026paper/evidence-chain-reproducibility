"""Render Fig. 4 with the exact-text bridge, reader changes, and failure probes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "experiments" / "api_reader_bridge_18"
ANALYSIS = ROOT / "analysis"
FIG_DIR = ROOT / "output" / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

CELLS_FILE = BRIDGE / "reader_bridge_cells_18.csv"
ASSOC_FILE = BRIDGE / "reader_bridge_associations.csv"
ABC_FILE = ANALYSIS / "public_abc_analysis_results.json"
PROBE_FILE = ANALYSIS / "controlled_ab_machine_human_case_summary.csv"

INK = "#17212B"
MUTED = "#667381"
GRID = "#D8E0E7"
BLUE = "#4F83AD"
ORANGE = "#D55E00"
GREY = "#737E89"
RED = "#B5442A"
VERSION_COLORS = {"A": GREY, "B": BLUE, "C": ORANGE}
HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "soft_blue",
    ["#F7F9FA", "#E5EFF2", "#C9DEE4", "#9FC1CC", "#6F9FAD"],
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.25,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.35,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.hashsalt": "adma2026-fig4-v1",
            "ps.fonttype": 42,
        }
    )


def validate() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    cells = pd.read_csv(CELLS_FILE)
    assoc = pd.read_csv(ASSOC_FILE)
    abc = json.loads(ABC_FILE.read_text(encoding="utf-8"))
    probes = pd.read_csv(PROBE_FILE)
    assert cells.shape[0] == 18
    assert cells.groupby("concept_key")["version"].nunique().eq(3).all()
    assert cells["n_judges"].eq(9).all()
    primary = assoc[(assoc["family"] == "primary_all_6_concepts") & (assoc["aggregation"] == "mean")]
    assert set(primary["predictor"]) == {"api_quality_mean", "api_risk_mean"}
    assert probes.shape[0] == 8 and probes["n_human_respondents"].eq(60).all() and probes["n_machine_judges"].eq(9).all()
    assert abc["primary"]["n_responses"] == 178
    return cells, assoc, abc, probes


def centered_bridge(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.copy()
    for col in ["api_quality_mean", "api_risk_mean", "accuracy_change"]:
        out[f"{col}_centered"] = out[col] - out.groupby("concept_key")[col].transform("mean")
    out["accuracy_change_centered_pp"] = 100 * out["accuracy_change_centered"]
    return out


def draw_bridge_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    assoc_row: pd.Series,
    predictor: str,
    title: str,
    x_label: str,
    *,
    show_y: bool,
) -> None:
    xcol = f"{predictor}_centered"
    ycol = "accuracy_change_centered_pp"
    for _, frame in data.groupby("concept_key", sort=True):
        frame = frame.set_index("version").loc[["A", "B", "C"]].reset_index()
        ax.plot(frame[xcol], frame[ycol], color="#C8D0D8", lw=0.65, zorder=1)
    for version in ["A", "B", "C"]:
        frame = data[data["version"] == version]
        ax.scatter(
            frame[xcol],
            frame[ycol],
            s=24,
            color=VERSION_COLORS[version],
            edgecolor="white",
            linewidth=0.45,
            label=version,
            zorder=3,
        )
    slope_pp = 100 * float(assoc_row["within_concept_slope"])
    xx = np.array([data[xcol].min(), data[xcol].max()])
    ax.plot(xx, slope_pp * xx, color=INK, lw=0.9, zorder=2)
    ax.axhline(0, color=GRID, lw=0.7, zorder=0)
    ax.axvline(0, color=GRID, lw=0.7, zorder=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.set_xlabel(x_label, labelpad=2)
    if show_y:
        ax.set_ylabel("Accuracy-change deviation (pp)", labelpad=2)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    r = float(assoc_row["centered_pearson_r"])
    p = float(assoc_row["exact_block_permutation_p_two_sided"])
    ph = float(assoc_row["p_holm_within_family"])
    ax.text(
        0.03,
        0.96,
        rf"$r={r:.3f}$; exact $p={p:.4f}$" + "\n" + rf"Holm $p={ph:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.45,
        color=INK,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)


def draw_reader_change(ax: plt.Axes, abc: dict) -> pd.DataFrame:
    frame = pd.DataFrame(abc["primary"]["marginal"]["changes"])
    frame["change_pp"] = 100 * frame["change"]
    frame["low_pp"] = 100 * frame["ci_low"]
    frame["high_pp"] = 100 * frame["ci_high"]
    frame = frame.set_index("version").loc[["A", "B", "C"]].reset_index()
    y = np.array([2, 1, 0])
    for i, row in frame.iterrows():
        value = row["change_pp"]
        ax.errorbar(
            value,
            y[i],
            xerr=np.array([[value - row["low_pp"]], [row["high_pp"] - value]]),
            fmt="o",
            color=VERSION_COLORS[row["version"]],
            markersize=5.5,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            elinewidth=1.0,
            capsize=2.2,
        )
        ax.text(row["high_pp"] + 0.55, y[i], f"{value:.1f}", va="center", fontsize=5.6, color=INK)
    ax.set_yticks(y, ["A", "B", "C"])
    ax.set_xlim(0, 18.8)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Immediate correctness change (pp)")
    ax.set_title("C   Reader version × phase", loc="left", fontweight="bold", pad=4)
    omnibus = abc["primary"]["omnibus_interaction"]
    ax.text(
        0.03,
        0.04,
        rf"Omnibus $\chi^2_2={omnibus['statistic']:.2f}$, $p={omnibus['p_value']:.4f}$" + "\nNo probability-scale pairwise Holm $p<.05$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.35,
        color=MUTED,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)
    return frame


def draw_probe_heatmap(ax: plt.Axes, probes: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "human_suitability_accuracy",
        "machine_quality_alignment",
        "human_misleading_accuracy",
        "machine_risk_alignment",
    ]
    labels = ["Reader\nquality", "API\nquality", "Reader\nrisk", "API\nrisk"]
    matrix = probes.set_index("case_id").loc[[f"A{i:02d}" for i in range(1, 9)], cols].to_numpy()
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0.4, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(4), labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=1.5)
    ax.set_yticks(np.arange(8), [f"A{i:02d}" for i in range(1, 9)])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.get_yticklabels()[0].set_color(RED)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_title("D   Expected-direction probes", loc="left", fontweight="bold", pad=15)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            ax.text(c, r, f"{value:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.1, color=INK, fontweight="semibold")
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((-0.49, -0.49), 3.98, 0.98, fill=False, edgecolor=RED, linewidth=1.15, clip_on=False))
    ax.text(
        0.0,
        -0.075,
        "Red outline: A01 omits validity conditions",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=RED,
        fontsize=5.35,
        fontweight="bold",
    )
    return probes[["case_id", *cols]].copy()


def main() -> None:
    cells, assoc, abc, probes = validate()
    bridge = centered_bridge(cells)
    primary = assoc[(assoc["family"] == "primary_all_6_concepts") & (assoc["aggregation"] == "mean")].set_index("predictor")
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    width_in, height_in = 4.80, 3.72
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.03],
        width_ratios=[1.0, 1.06],
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.90,
        hspace=0.54,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_bridge_panel(ax_a, bridge, primary.loc["api_quality_mean"], "api_quality_mean", "A   Exact-text quality bridge", "Quality-score deviation", show_y=True)
    draw_bridge_panel(ax_b, bridge, primary.loc["api_risk_mean"], "api_risk_mean", "B   Exact-text risk bridge", "Risk-score deviation", show_y=False)
    changes = draw_reader_change(ax_c, abc)
    probe_source = draw_probe_heatmap(ax_d, probes)

    # One centered legend for version markers; it does not compete with panel titles.
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=VERSION_COLORS[v], label=f"Version {v}")
        for v in ["A", "B", "C"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.0)

    bridge.to_csv(SOURCE_DIR / "fig4_reader_bridge_cells_source.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(SOURCE_DIR / "fig4_reader_change_source.csv", index=False, encoding="utf-8-sig")
    probe_source.to_csv(SOURCE_DIR / "fig4_probe_source.csv", index=False, encoding="utf-8-sig")

    pdf = FIG_DIR / "fig4_public_validation.pdf"
    png = FIG_DIR / "fig4_public_validation.png"
    fig.savefig(pdf, bbox_inches=None, pad_inches=0)
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    page = PdfReader(pdf).pages[0]
    dims = [float(page.mediabox.width) / 72, float(page.mediabox.height) / 72]
    manifest = {
        "status": "PASS",
        "dominant_claim": "the exact-text bridge is directionally favorable but familywise inconclusive, and aggregate probe success still hides A01",
        "text_cells": int(cells.shape[0]),
        "concept_blocks": int(cells["concept_key"].nunique()),
        "api_calls": int(cells["n_judges"].sum()),
        "quality_r": float(primary.loc["api_quality_mean", "centered_pearson_r"]),
        "quality_holm_p": float(primary.loc["api_quality_mean", "p_holm_within_family"]),
        "risk_r": float(primary.loc["api_risk_mean", "centered_pearson_r"]),
        "risk_holm_p": float(primary.loc["api_risk_mean", "p_holm_within_family"]),
        "figure_inches": dims,
        "palette": {
            "version_b_soft_blue": BLUE,
            "probe_heatmap": ["#F7F9FA", "#E5EFF2", "#C9DEE4", "#9FC1CC", "#6F9FAD"],
        },
        "sha256": {"pdf": sha256(pdf), "png": sha256(png)},
    }
    (FIG_DIR / "fig4_public_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
