"""Render Fig. 2: fresh-call repeatability and complete human-criterion baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REPEAT = ROOT / "experiments" / "api_repeat_stability"
BASELINES = ROOT / "experiments" / "api_aggregation_human_baselines"
FIG_DIR = ROOT / "output" / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

PROVIDER_FILE = REPEAT / "repeat_provider_dimension_summary.csv"
PANEL_FILE = REPEAT / "repeat_panel_dimension_summary.csv"
HUMAN_FILE = BASELINES / "aggregation_human_results.csv"

INK = "#17212B"
MUTED = "#667381"
GRID = "#D9E0E6"
# Restrained, colour-blind-safe family.  Aggregations also retain distinct
# marker shapes, so the figure remains interpretable in greyscale.
BLUE = "#1F4E79"       # mean: deep navy
GREEN = "#2A8C82"      # trimmed mean: muted teal
ORANGE_RED = "#C65D3B" # no-self mean: soft terracotta
GREY = "#6C7782"       # median: neutral slate
LIGHT_BLUE = "#8BAFC3" # individual judges: quiet blue-grey
REPEAT_CMAP = LinearSegmentedColormap.from_list(
    "repeatability_blue_teal",
    ["#F3F6F8", "#CADDE2", "#88B8BE", "#438B94", "#1B5D6C", "#17324D"],
)
METRIC_CMAP = LinearSegmentedColormap.from_list(
    "metric_blue_teal",
    ["#F3F6F8", "#C6DCE2", "#77ADB7", "#2F7C89", "#174F66"],
)

DIMS = ["fa", "cc", "lc", "tf", "mq", "risk"]
DIM_LABELS = ["Accuracy", "Complete.", "Clarity", "Task fit", "Miscon.", "Risk"]
PROVIDERS = ["anthropic", "openai", "gemini", "deepseek", "doubao", "qwen", "glm", "kimi", "mimo"]
PROVIDER_LABELS = ["Anthropic", "OpenAI", "Gemini", "DeepSeek", "Doubao", "Qwen", "GLM", "Kimi", "Mimo"]

METHOD_ORDER = [
    "ensemble::mean",
    "ensemble::trimmed_one_each_tail",
    "ensemble::median",
    "ensemble::no_self_mean",
    "single::gemini",
    "single::openai",
    "single::deepseek",
    "single::doubao",
    "single::qwen",
    "single::anthropic",
    "single::kimi",
    "single::mimo",
    "single::glm",
]
SHORT_LABELS = {
    "ensemble::mean": "Mean (9)",
    "ensemble::trimmed_one_each_tail": "Trimmed (7)",
    "ensemble::median": "Median (9)",
    "ensemble::no_self_mean": "No-self (8)",
    "single::gemini": "Gemini",
    "single::openai": "OpenAI",
    "single::deepseek": "DeepSeek",
    "single::doubao": "Doubao",
    "single::qwen": "Qwen",
    "single::anthropic": "Anthropic",
    "single::kimi": "Kimi",
    "single::mimo": "Mimo",
    "single::glm": "GLM",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 5.75,
            "axes.titlesize": 6.8,
            "axes.labelsize": 6.1,
            "xtick.labelsize": 5.25,
            "ytick.labelsize": 5.35,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    provider = pd.read_csv(PROVIDER_FILE)
    panel = pd.read_csv(PANEL_FILE)
    human = pd.read_csv(HUMAN_FILE)
    assert provider.shape[0] == 54 and provider["n_complete_stimuli"].eq(90).all()
    assert panel.shape[0] == 24 and panel["n_complete_stimuli"].eq(90).all()
    assert human.shape[0] == 13 and set(human["method_id"]) == set(METHOD_ORDER)
    assert human["n_stimuli"].eq(178).all() and human["n_rows"].eq(1068).all()
    mean = human.set_index("method_id").loc["ensemble::mean"]
    assert np.isclose(mean["standardized_beta"], 0.231563853, atol=1e-8)
    assert np.isclose(mean["bootstrap_ci_low"], 0.132891459, atol=1e-8)
    return provider, panel, human.set_index("method_id").loc[METHOD_ORDER].reset_index()


def draw_repeat_heatmap(ax: plt.Axes, provider: pd.DataFrame) -> None:
    pivot = provider.pivot(index="judge_provider", columns="dimension", values="icc_a1_absolute_single_measure")
    values = pivot.loc[PROVIDERS, DIMS].to_numpy()
    ax.imshow(values, cmap=REPEAT_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(6), DIM_LABELS, rotation=31, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(9), PROVIDER_LABELS)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0, pad=1.4)
    ax.set_title("A   Individual fresh-call repeatability", loc="left", fontweight="bold", pad=10)
    ax.text(0, 1.035, "ICC(A,1); 90 texts × 3 new calls", transform=ax.transAxes, color=MUTED, fontsize=5.15, va="bottom")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            rgba = REPEAT_CMAP(v)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.48 else INK
            ax.text(c, r, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=4.65, fontweight="semibold", color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_repeatability(ax: plt.Axes, panel: pd.DataFrame) -> None:
    y = np.arange(6)[::-1].astype(float)
    offsets = {"mean": 0.21, "trimmed_one_each_tail": 0.07, "no_self_mean": -0.07, "median": -0.21}
    styles = {
        "mean": ("Mean", BLUE, "o", True),
        "trimmed_one_each_tail": ("Trimmed", GREEN, "s", True),
        "no_self_mean": ("No-self", ORANGE_RED, "^", True),
        "median": ("Median", GREY, "D", False),
    }
    for agg in ["mean", "trimmed_one_each_tail", "no_self_mean", "median"]:
        frame = panel[panel["aggregation"] == agg].set_index("dimension").loc[DIMS]
        name, color, marker, filled = styles[agg]
        yy = y + offsets[agg]
        values = frame["icc_a1_absolute_single_measure"].to_numpy()
        if agg == "mean":
            low = frame["icc_concept_cluster_bootstrap_ci_low"].to_numpy()
            high = frame["icc_concept_cluster_bootstrap_ci_high"].to_numpy()
            ax.errorbar(values, yy, xerr=np.vstack([values - low, high - values]), fmt="none", ecolor=color, elinewidth=0.9, capsize=1.6, capthick=0.7, zorder=2)
        ax.scatter(values, yy, s=17, marker=marker, facecolor=color if filled else "white", edgecolor=color, linewidth=0.7, label=name, zorder=3)
    ax.set_yticks(y, DIM_LABELS)
    ax.set_xlim(0.70, 1.005)
    ax.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax.set_ylim(-0.55, 5.55)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Panel ICC(A,1)", labelpad=1.5)
    ax.set_title("B   Fixed-panel repeatability", loc="left", fontweight="bold", pad=10)
    ax.text(0, 1.035, "Mean bars: concept-bootstrap 95% CI", transform=ax.transAxes, color=MUTED, fontsize=5.15, va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=4, frameon=False, handletextpad=0.22, columnspacing=0.45, borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.2, pad=1)


def draw_human_slopes(ax: plt.Axes, human: pd.DataFrame) -> None:
    y = np.arange(len(human))[::-1].astype(float)
    beta = human["standardized_beta"].to_numpy()
    low = human["bootstrap_ci_low"].to_numpy()
    high = human["bootstrap_ci_high"].to_numpy()
    colors = []
    markers = []
    for method in human["method_id"]:
        if method == "ensemble::mean":
            colors.append(BLUE); markers.append("o")
        elif method == "ensemble::trimmed_one_each_tail":
            colors.append(GREEN); markers.append("s")
        elif method == "ensemble::median":
            colors.append(GREY); markers.append("D")
        elif method == "ensemble::no_self_mean":
            colors.append(ORANGE_RED); markers.append("^")
        else:
            colors.append(LIGHT_BLUE); markers.append("o")
    for i, (b, lo, hi, color, marker, p_holm) in enumerate(zip(beta, low, high, colors, markers, human["freedman_lane_p_holm_13"], strict=True)):
        ax.plot([lo, hi], [y[i], y[i]], color=color, lw=0.85, zorder=1)
        ax.plot([lo, lo], [y[i] - 0.12, y[i] + 0.12], color=color, lw=0.65)
        ax.plot([hi, hi], [y[i] - 0.12, y[i] + 0.12], color=color, lw=0.65)
        ax.scatter([b], [y[i]], s=18 if i < 4 else 13, marker=marker, facecolor=color if p_holm < 0.05 else "white", edgecolor=color, linewidth=0.7, zorder=2)
    ax.axvline(0, color=INK, lw=0.65)
    ax.axvline(0.20, color=MUTED, lw=0.65, linestyle=(0, (3, 2)))
    ax.text(0.202, len(human) - 0.1, ".20", ha="left", va="top", fontsize=4.7, color=MUTED)
    ax.axhline(len(human) - 4.5, color=GRID, lw=0.65)
    ax.set_yticks(y, [SHORT_LABELS[m] for m in human["method_id"]])
    for idx, tick in enumerate(ax.get_yticklabels()):
        if idx < 4:
            tick.set_fontweight("bold")
    ax.set_xlim(-0.03, 0.36)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_ylim(-0.6, len(human) - 0.4)
    ax.set_xlabel(r"Standardized slope, $\beta$", labelpad=1.5)
    ax.set_title("C   Broad human-criterion association", loc="left", fontweight="bold", pad=10)
    ax.text(0, 1.02, "Same 178 texts; 95% concept-cluster CI", transform=ax.transAxes, color=MUTED, fontsize=5.15, va="bottom")
    ax.text(0.99, 0.01, "filled: Holm $p<.05$", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.75, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.2, pad=1)


def draw_raw_metric_table(ax: plt.Axes, human: pd.DataFrame) -> None:
    metrics = [
        ("lin_ccc_quality_aligned_raw_scale", "CCC ↑"),
        ("spearman_rho_quality_aligned_raw_scale", "Spearman ↑"),
        ("mae_quality_aligned_raw_scale", "MAE ↓"),
    ]
    raw = np.column_stack([human[col].to_numpy() for col, _ in metrics])
    norm = np.zeros_like(raw)
    for c in range(raw.shape[1]):
        lo, hi = raw[:, c].min(), raw[:, c].max()
        norm[:, c] = (raw[:, c] - lo) / (hi - lo) if hi > lo else 0.5
    norm[:, 2] = 1.0 - norm[:, 2]
    ax.imshow(norm, cmap=METRIC_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(3), [name for _, name in metrics])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=2)
    ax.set_yticks(np.arange(len(human)), [SHORT_LABELS[m] for m in human["method_id"]])
    for idx, tick in enumerate(ax.get_yticklabels()):
        if idx < 4:
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            color = "white" if norm[r, c] > 0.62 else INK
            ax.text(c, r, f"{raw[r, c]:.3f}", ha="center", va="center", fontsize=4.8, color=color, fontweight="semibold")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(human), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.add_patch(Rectangle((-0.49, -0.49), 2.98, 3.98, fill=False, edgecolor=BLUE, linewidth=0.8, clip_on=False))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("D   Raw-score performance", loc="left", fontweight="bold", pad=19)


def main() -> None:
    provider, panel, human = load_and_validate()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    provider.to_csv(SOURCE_DIR / "fig2_fresh_call_provider_source.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(SOURCE_DIR / "fig2_fresh_call_panel_source.csv", index=False, encoding="utf-8-sig")
    human.to_csv(SOURCE_DIR / "fig2_human_baselines_source.csv", index=False, encoding="utf-8-sig")
    configure_style()

    width_in, height_in = 4.80, 4.34
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.90, 1.43],
        width_ratios=[1.17, 1.0],
        left=0.115,
        right=0.985,
        bottom=0.075,
        top=0.94,
        hspace=0.47,
        wspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    draw_repeat_heatmap(ax_a, provider)
    draw_panel_repeatability(ax_b, panel)
    draw_human_slopes(ax_c, human)
    draw_raw_metric_table(ax_d, human)

    pdf = FIG_DIR / "fig2_reliability.pdf"
    png = FIG_DIR / "fig2_reliability.png"
    fig.savefig(pdf, bbox_inches=None, pad_inches=0)
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    page = PdfReader(pdf).pages[0]
    dims = [float(page.mediabox.width) / 72, float(page.mediabox.height) / 72]
    mean = human.set_index("method_id").loc["ensemble::mean"]
    manifest = {
        "status": "PASS",
        "dominant_claim": "fresh-call aggregation is repeatable, while mean-versus-single human alignment must be judged with common uncertainty and raw-score metrics",
        "repeat_provider_dimension_cells": int(provider.shape[0]),
        "human_baselines": int(human.shape[0]),
        "human_common_rows": int(mean["n_rows"]),
        "mean_beta": float(mean["standardized_beta"]),
        "mean_beta_ci": [float(mean["bootstrap_ci_low"]), float(mean["bootstrap_ci_high"])],
        "figure_inches": dims,
        "palette": {
            "mean_deep_navy": BLUE,
            "trimmed_muted_teal": GREEN,
            "no_self_soft_terracotta": ORANGE_RED,
            "median_neutral_slate": GREY,
            "individual_judges_blue_grey": LIGHT_BLUE,
            "heatmaps": "monotonic light-to-dark blue-teal; no yellow endpoint",
        },
        "redundant_encoding": {
            "mean": "filled circle",
            "trimmed_mean": "filled square",
            "no_self_mean": "filled triangle",
            "median": "open diamond",
        },
        "sha256": {"pdf": sha256(pdf), "png": sha256(png)},
    }
    (FIG_DIR / "fig2_reliability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
