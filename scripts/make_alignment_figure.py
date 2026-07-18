"""Build the locked Full Paper Figure 3 from formal alignment outputs.

Single figure claim
-------------------
The overall human--API association is statistically reliable in both review
layers and stronger in the purposively selected dense re-review; its positive
direction is preserved across every prespecified, slope-comparable
specification.

Panel roles
-----------
A. Anchor evidence: forest plot of the overall and six dimension-specific
   standardized slopes for the first and selected reviews.  Intervals are the
   locked 5,000-draw domain-stratified concept-cluster bootstrap intervals.
   Overall significance uses the formal permutation p value; dimension rows
   use the six-test Benjamini--Hochberg q value.
B. Specification stability: all prespecified slope-comparable sensitivity
   points, grouped by family and review wave.  Bars are the observed min--max
   across scenarios in a family, not confidence intervals.  HC3 p values are
   descriptive and receive no significance encoding.  Model-form Spearman
   correlations and other differently scaled checks are deliberately absent.

The script validates the formal lock (quick=false, 5,000 bootstrap draws,
10,000 first-review permutations, 32,768 second-review exact permutations,
and QA PASS) before reading the result tables.  It never edits the manuscript.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader


PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = PROJECT / "analysis"
FIGURE_DIR = PROJECT / "output" / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

MANIFEST_PATH = ANALYSIS / "source_manifests" / "alignment_manifest.json"
VALIDATION_PATH = ANALYSIS / "alignment_validation.json"
EFFECTS_PATH = ANALYSIS / "alignment_effects.csv"
SENSITIVITY_PATH = ANALYSIS / "alignment_sensitivity.csv"

PDF_PATH = FIGURE_DIR / "fig3_alignment.pdf"
PNG_PATH = FIGURE_DIR / "fig3_alignment.png"
SOURCE_PATH = SOURCE_DIR / "fig3_alignment_source.csv"

WAVE_ORDER = ["首轮专家复核", "二次专家复核"]
WAVE_LABELS = {
    "首轮专家复核": "Broad review",
    "二次专家复核": "Selected re-annotation",
}
EFFECT_ORDER = ["all", "fa", "cc", "lc", "tf", "mq", "risk"]
EFFECT_LABELS = {
    "all": "Overall",
    "fa": "Accuracy",
    "cc": "Completeness",
    "lc": "Clarity",
    "tf": "Task fit",
    "mq": "Misconception handling",
    "risk": "Misleading risk",
}
FAMILY_ORDER = [
    "QC panel",
    "API aggregation",
    "leave-one-judge-out",
    "leave-one-generator-out",
    "leave-one-domain-out",
    "leave-one-concept-out",
    "fixed item exclusion",
    "bootstrap cluster unit",
    "risk direction",
]
FAMILY_LABELS = {
    "QC panel": "QC threshold",
    "API aggregation": "API score / UID",
    "leave-one-judge-out": "Drop judge",
    "leave-one-generator-out": "Drop generator",
    "leave-one-domain-out": "Drop domain",
    "leave-one-concept-out": "Drop concept",
    "fixed item exclusion": "Exclude P05",
    "bootstrap cluster unit": "Resample unit",
    "risk direction": "Risk coding",
}

# Muted, colorblind-safe conference palette. Wave is redundantly encoded by
# marker shape; significance in Panel A is encoded only by marker fill.
BLUE = "#245B8A"
ORANGE = "#C65A22"
BLACK = "#1B1F23"
GRAY = "#66717B"
MID_GRAY = "#AEB6BC"
LIGHT_GRAY = "#E7EAEC"
ROW_GRAY = "#F4F5F6"

# Final LNCS print size; never crop the canvas on export.
FIG_WIDTH_IN = 4.80
FIG_HEIGHT_IN = 3.20
MIN_FONT_PT = 7.0
PRACTICAL_THRESHOLD = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.handlelength": 1.25,
            "legend.handletextpad": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def validate_formal_lock() -> tuple[dict[str, object], dict[str, object]]:
    for path in (MANIFEST_PATH, VALIDATION_PATH, EFFECTS_PATH, SENSITIVITY_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))

    parameters = manifest.get("command_parameters", {})
    required = {
        "quick_false": parameters.get("quick") is False,
        "bootstrap_5000": parameters.get("bootstrap") == 5000,
        "permutations_10000": parameters.get("permutations") == 10000,
        "manifest_qa_pass": manifest.get("qa_status") == "PASS",
        "validation_qa_pass": validation.get("qa_status") == "PASS",
        "validation_quick_false": validation.get("quick_mode") is False,
        "second_exact_32768": validation.get("second_inference", {}).get(
            "n_exact_permutations"
        )
        == 32768,
        "validation_first_bootstrap_5000": validation.get("first_inference", {}).get(
            "bootstrap_draws"
        )
        == 5000,
        "validation_first_permutations_10000": validation.get(
            "first_inference", {}
        ).get("permutations")
        == 10000,
        "validation_second_bootstrap_5000": validation.get(
            "second_inference", {}
        ).get("concept_cluster_bootstrap_draws")
        == 5000,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise AssertionError(f"formal alignment lock failed: {failed}")

    for key, path in (
        ("effects", EFFECTS_PATH),
        ("sensitivity", SENSITIVITY_PATH),
        ("validation", VALIDATION_PATH),
    ):
        recorded = manifest["outputs"][key]
        if recorded["sha256"] != sha256_file(path) or recorded["bytes"] != path.stat().st_size:
            raise AssertionError(f"manifest hash/size mismatch for {key}")
    return manifest, validation


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = pd.read_csv(EFFECTS_PATH)
    sensitivity = pd.read_csv(SENSITIVITY_PATH)
    if effects.shape != (30, 32):
        raise ValueError(f"unexpected alignment_effects shape {effects.shape}")
    if sensitivity.shape != (116, 12):
        raise ValueError(f"unexpected alignment_sensitivity shape {sensitivity.shape}")
    if set(effects["wave"]) != set(WAVE_ORDER):
        raise ValueError("unexpected effect waves")
    if set(sensitivity["sensitivity_family"]) != set(FAMILY_ORDER) | {"model form"}:
        raise ValueError("unexpected sensitivity family set")
    return effects, sensitivity


def source_template(**values: object) -> dict[str, object]:
    row: dict[str, object] = {
        "panel": "",
        "plot_role": "",
        "wave": "",
        "wave_label": "",
        "effect_family": "",
        "effect_level": "",
        "effect_label": "",
        "sensitivity_family": "",
        "sensitivity_family_label": "",
        "scenario": "",
        "detail": "",
        "slope": np.nan,
        "lower": np.nan,
        "upper": np.nan,
        "interval_type": "",
        "formal_p_or_q": np.nan,
        "descriptive_hc3_p": np.nan,
        "multiplicity_method": "",
        "formal_significant": False,
        "practically_material_abs_ge_0_20": False,
        "original_inference_role": "",
        "is_primary_scenario": False,
        "n_rows": np.nan,
        "n_stimuli": np.nan,
        "scenarios_in_family": np.nan,
        "formal_quick": False,
        "formal_bootstrap": 5000,
        "formal_first_permutations": 10000,
        "formal_second_exact_permutations": 32768,
        "source_file": "",
    }
    row.update(values)
    return row


def panel_a_data(
    effects: pd.DataFrame,
    source_rows: list[dict[str, object]],
) -> pd.DataFrame:
    data = effects[effects["family"].isin(["overall", "dimension"])].copy()
    if len(data) != 14:
        raise AssertionError(f"Panel A requires 14 rows, found {len(data)}")
    if not (data["n_bootstrap_valid"] == 5000).all():
        raise AssertionError("Panel A contains fewer than 5,000 valid bootstrap draws")
    if not (data["bootstrap_invalid_proportion"] == 0).all():
        raise AssertionError("Panel A contains invalid bootstrap draws")
    if not data["bootstrap_routine_reportable"].all():
        raise AssertionError("Panel A contains suppressed bootstrap effects")

    data["effect_level"] = np.where(data["family"] == "overall", "all", data["level"])
    data["effect_label"] = data["effect_level"].map(EFFECT_LABELS)
    if data["effect_label"].isna().any():
        raise ValueError("unmapped Panel A effect")
    data["formal_p_or_q"] = np.where(
        data["family"] == "overall", data["permutation_p"], data["p_adjusted"]
    )
    data["formal_significant"] = data["formal_p_or_q"] < 0.05
    overall_sig = int(data.loc[data["family"] == "overall", "formal_significant"].sum())
    dimension_sig = int(data.loc[data["family"] == "dimension", "formal_significant"].sum())
    if overall_sig != 2 or dimension_sig != 5:
        raise AssertionError(
            f"expected two significant overall and five significant dimension rows, got {overall_sig}/{dimension_sig}"
        )

    for _, row in data.iterrows():
        source_rows.append(
            source_template(
                panel="A",
                plot_role="formal effect with concept-cluster bootstrap CI",
                wave=row["wave"],
                wave_label=WAVE_LABELS[row["wave"]],
                effect_family=row["family"],
                effect_level=row["effect_level"],
                effect_label=row["effect_label"],
                slope=row["slope"],
                lower=row["ci_low"],
                upper=row["ci_high"],
                interval_type=row["ci_method"],
                formal_p_or_q=row["formal_p_or_q"],
                multiplicity_method=(
                    "none; formal overall permutation p"
                    if row["family"] == "overall"
                    else row["multiplicity_method"]
                ),
                formal_significant=bool(row["formal_significant"]),
                practically_material_abs_ge_0_20=bool(
                    row["practically_material_abs_ge_0_20"]
                ),
                n_rows=row["n_rows"],
                n_stimuli=row["n_stimuli"],
                source_file=str(EFFECTS_PATH.relative_to(PROJECT)),
            )
        )
    return data


def is_primary_sensitivity(family: str, scenario: str) -> bool:
    return (family == "QC panel" and scenario == "final_primary") or (
        family == "API aggregation" and scenario == "ensemble_mean"
    )


def panel_b_data(
    sensitivity: pd.DataFrame,
    source_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    excluded = sensitivity[sensitivity["sensitivity_family"] == "model form"].copy()
    data = sensitivity[sensitivity["sensitivity_family"] != "model form"].copy()
    if len(excluded) != 16 or len(data) != 100:
        raise AssertionError(
            f"expected 16 non-comparable model-form rows and 100 slope rows; found {len(excluded)}, {len(data)}"
        )
    if not (data["slope"] > 0).all():
        raise AssertionError("a prespecified slope-comparable sensitivity reversed direction")
    data["family_label"] = data["sensitivity_family"].map(FAMILY_LABELS)
    if data["family_label"].isna().any():
        raise ValueError("unmapped sensitivity family")
    data["is_primary_scenario"] = [
        is_primary_sensitivity(family, scenario)
        for family, scenario in zip(data["sensitivity_family"], data["scenario"])
    ]

    for _, row in data.iterrows():
        interval_type = (
            "descriptive domain-stratified stimulus bootstrap interval; not significance encoded"
            if row["sensitivity_family"] == "bootstrap cluster unit"
            else "descriptive HC3 interval; not plotted and not significance encoded"
        )
        source_rows.append(
            source_template(
                panel="B",
                plot_role="prespecified sensitivity slope point",
                wave=row["wave"],
                wave_label=WAVE_LABELS[row["wave"]],
                sensitivity_family=row["sensitivity_family"],
                sensitivity_family_label=row["family_label"],
                scenario=row["scenario"],
                detail=row["detail"],
                slope=row["slope"],
                lower=row["ci_low_hc3"],
                upper=row["ci_high_hc3"],
                interval_type=interval_type,
                descriptive_hc3_p=row["p_hc3_unadjusted_descriptive"],
                multiplicity_method="none; descriptive only",
                formal_significant=False,
                practically_material_abs_ge_0_20=bool(abs(row["slope"]) >= 0.20),
                original_inference_role=row["inference_role"],
                is_primary_scenario=bool(row["is_primary_scenario"]),
                n_rows=row["n_rows"],
                n_stimuli=row["n_stimuli"],
                source_file=str(SENSITIVITY_PATH.relative_to(PROJECT)),
            )
        )

    summaries: list[dict[str, object]] = []
    for (family, wave), part in data.groupby(
        ["sensitivity_family", "wave"], sort=False
    ):
        summary = {
            "sensitivity_family": family,
            "family_label": FAMILY_LABELS[family],
            "wave": wave,
            "slope": float(part["slope"].median()),
            "lower": float(part["slope"].min()),
            "upper": float(part["slope"].max()),
            "n_scenarios": int(len(part)),
        }
        summaries.append(summary)
        source_rows.append(
            source_template(
                panel="B",
                plot_role="within-family slope median and min--max range",
                wave=wave,
                wave_label=WAVE_LABELS[wave],
                sensitivity_family=family,
                sensitivity_family_label=FAMILY_LABELS[family],
                scenario="family_summary",
                detail="median and min--max across all prespecified slope-comparable scenarios",
                slope=summary["slope"],
                lower=summary["lower"],
                upper=summary["upper"],
                interval_type="observed min--max across prespecified slopes; not a confidence interval",
                formal_significant=False,
                scenarios_in_family=summary["n_scenarios"],
                source_file=str(SENSITIVITY_PATH.relative_to(PROJECT)),
            )
        )
    summary_df = pd.DataFrame(summaries)
    if len(summary_df) != 15:
        raise AssertionError(f"expected 15 wave-by-family summaries, found {len(summary_df)}")
    return data, summary_df, len(excluded)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.tick_params(colors=BLACK)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def plot_panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Compact forest plot at final LNCS reproduction size."""
    ymap = {level: len(EFFECT_ORDER) - 1 - i for i, level in enumerate(EFFECT_ORDER)}
    specs = {
        WAVE_ORDER[0]: {"color": BLUE, "marker": "o", "offset": 0.14},
        WAVE_ORDER[1]: {"color": ORANGE, "marker": "s", "offset": -0.14},
    }

    ax.axhspan(5.53, 6.47, color=ROW_GRAY, zorder=-1)
    ax.axvline(0.0, color=MID_GRAY, linewidth=0.70, zorder=1)
    ax.axvline(
        PRACTICAL_THRESHOLD,
        color=GRAY,
        linestyle=(0, (2, 2)),
        linewidth=0.70,
        zorder=1,
    )
    for _, row in data.iterrows():
        spec = specs[row["wave"]]
        y = ymap[row["effect_level"]] + spec["offset"]
        slope = float(row["slope"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        significant = bool(row["formal_significant"])
        ax.errorbar(
            slope,
            y,
            xerr=np.array([[slope - low], [high - slope]]),
            fmt=spec["marker"],
            markersize=4.3 if row["effect_level"] == "all" else 3.8,
            markerfacecolor=spec["color"] if significant else "white",
            markeredgecolor=spec["color"],
            markeredgewidth=0.95,
            ecolor=spec["color"],
            elinewidth=0.95,
            capsize=1.7,
            capthick=0.75,
            zorder=3,
        )

    ax.axhline(5.5, color=MID_GRAY, linewidth=0.55)
    ax.set_xlim(-0.22, 1.14)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.6, 1.0])
    ax.set_ylim(-0.48, 6.58)
    ax.set_yticks([ymap[level] for level in EFFECT_ORDER])
    effect_tick_labels = [EFFECT_LABELS[level] for level in EFFECT_ORDER]
    effect_tick_labels[EFFECT_ORDER.index("mq")] = "Misconception\nhandling"
    ax.set_yticklabels(effect_tick_labels, linespacing=0.92)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.set_xlabel("Standardized slope, $\\beta$")
    ax.set_title(r"$\bf{A}$   Effect estimates", loc="left", pad=5)
    ax.text(
        PRACTICAL_THRESHOLD,
        6.55,
        "0.20 ref.",
        fontsize=7.0,
        color=GRAY,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.985,
        0.965,
        "95% cluster CI",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=GRAY,
    )
    style_axis(ax)


def plot_panel_b(
    ax: plt.Axes,
    data: pd.DataFrame,
    summaries: pd.DataFrame,
    primary_slopes: dict[str, float],
) -> None:
    """Range-and-point view of prespecified slope-comparable checks."""
    del primary_slopes  # Primary scenarios are encoded directly as diamonds.
    ymap = {family: len(FAMILY_ORDER) - 1 - i for i, family in enumerate(FAMILY_ORDER)}
    specs = {
        WAVE_ORDER[0]: {"color": BLUE, "marker": "o", "offset": 0.14},
        WAVE_ORDER[1]: {"color": ORANGE, "marker": "s", "offset": -0.14},
    }

    for i in range(len(FAMILY_ORDER)):
        if i % 2 == 0:
            ax.axhspan(i - 0.47, i + 0.47, color=ROW_GRAY, zorder=-2)
    ax.axvline(0.0, color=MID_GRAY, linewidth=0.70, zorder=1)
    ax.axvline(
        PRACTICAL_THRESHOLD,
        color=GRAY,
        linestyle=(0, (2, 2)),
        linewidth=0.70,
        zorder=1,
    )

    for family in FAMILY_ORDER:
        for wave in WAVE_ORDER:
            part = data[
                (data["sensitivity_family"] == family) & (data["wave"] == wave)
            ].sort_values("scenario")
            if part.empty:
                continue
            spec = specs[wave]
            base_y = ymap[family] + spec["offset"]
            summary = summaries[
                (summaries["sensitivity_family"] == family)
                & (summaries["wave"] == wave)
            ].iloc[0]

            ax.hlines(
                base_y,
                summary["lower"],
                summary["upper"],
                color=spec["color"],
                linewidth=1.25,
                alpha=0.92,
                zorder=2,
            )
            ax.vlines(
                [summary["lower"], summary["upper"]],
                base_y - 0.035,
                base_y + 0.035,
                color=spec["color"],
                linewidth=0.75,
                zorder=2,
            )
            jitters = (
                np.array([0.0])
                if len(part) == 1
                else np.linspace(-0.035, 0.035, len(part))
            )
            for jitter, (_, row) in zip(jitters, part.iterrows()):
                ax.scatter(
                    row["slope"],
                    base_y + jitter,
                    s=7,
                    marker=spec["marker"],
                    facecolor=spec["color"],
                    edgecolor=spec["color"],
                    linewidth=0.15,
                    alpha=0.34,
                    zorder=3,
                )
            ax.scatter(
                summary["slope"],
                base_y,
                s=18,
                marker=spec["marker"],
                facecolor="white",
                edgecolor=spec["color"],
                linewidth=0.85,
                zorder=5,
            )

    ax.set_xlim(-0.015, 0.82)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(-0.48, 8.58)
    ax.set_yticks([ymap[family] for family in FAMILY_ORDER])
    ax.set_yticklabels([FAMILY_LABELS[family] for family in FAMILY_ORDER])
    ax.set_xlabel("Standardized slope, $\\beta$")
    ax.set_title(r"$\bf{B}$   Sensitivity analysis", loc="left", pad=5)
    style_axis(ax)


def build_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_b_summaries: pd.DataFrame,
    primary_slopes: dict[str, float],
) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.22, 1.0],
        # Center the plot frames themselves; labels no longer determine alignment.
        left=0.170,
        right=0.830,
        bottom=0.155,
        top=0.825,
        wspace=0.62,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    plot_panel_a(axes[0], panel_a)
    plot_panel_b(axes[1], panel_b, panel_b_summaries, primary_slopes)

    shared_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=BLUE,
            markerfacecolor=BLUE,
            markersize=4.2,
            label="Broad review",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=ORANGE,
            markerfacecolor=ORANGE,
            markersize=4.2,
            label="Selected review",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            markerfacecolor="white",
            markeredgecolor=GRAY,
            color=GRAY,
            linewidth=0.8,
            markersize=3.9,
            label="B: median + range",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLACK,
            markeredgecolor=BLACK,
            markersize=3.9,
            label="A: filled = $p/q<.05$",
        ),
    ]
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.500, 0.985),
        ncol=4,
        handletextpad=0.30,
        columnspacing=0.80,
        borderaxespad=0,
    )
    return fig


def validate_outputs(
    source: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    excluded_model_form_rows: int,
) -> dict[str, object]:
    for path in (PDF_PATH, PNG_PATH, SOURCE_PATH):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"missing output: {path}")
    with Image.open(PNG_PATH) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0, 0))
        image.verify()
    with Image.open(PNG_PATH) as image:
        corner = image.convert("RGB").getpixel((0, 0))
    pdf = PdfReader(str(PDF_PATH))
    if len(pdf.pages) != 1:
        raise AssertionError(f"figure PDF has {len(pdf.pages)} pages")
    box = pdf.pages[0].mediabox
    pdf_width_pt = float(box.width)
    pdf_height_pt = float(box.height)
    expected_pdf_pt = (FIG_WIDTH_IN * 72.0, FIG_HEIGHT_IN * 72.0)
    if (
        abs(pdf_width_pt - expected_pdf_pt[0]) > 0.01
        or abs(pdf_height_pt - expected_pdf_pt[1]) > 0.01
    ):
        raise AssertionError(
            "PDF media box was cropped or resized: "
            f"{(pdf_width_pt, pdf_height_pt)} vs {expected_pdf_pt}"
        )
    expected = (round(FIG_WIDTH_IN * 600), round(FIG_HEIGHT_IN * 600))
    if abs(width - expected[0]) > 10 or abs(height - expected[1]) > 10:
        raise AssertionError(f"unexpected PNG dimensions {(width, height)}; expected {expected}")
    if dpi[0] < 590 or dpi[1] < 590:
        raise AssertionError(f"PNG is not 600 dpi: {dpi}")
    if corner != (255, 255, 255):
        raise AssertionError(f"PNG corner is not white: {corner}")
    if len(source) != 129:
        raise AssertionError(f"unexpected source row count {len(source)}")
    if excluded_model_form_rows != 16:
        raise AssertionError("model-form exclusion count changed")
    if not (panel_b["slope"] > 0).all():
        raise AssertionError("sensitivity direction is not uniformly positive")
    if int(panel_a["formal_significant"].sum()) != 7:
        raise AssertionError("formal significance count changed")
    return {
        "status": "PASS",
        "formal_lock": {
            "quick": False,
            "bootstrap": 5000,
            "first_permutations": 10000,
            "second_exact_permutations": 32768,
            "qa": "PASS",
        },
        "png_pixels": [width, height],
        "png_dpi": [round(float(dpi[0]), 1), round(float(dpi[1]), 1)],
        "pdf_media_box_pt": [pdf_width_pt, pdf_height_pt],
        "figure_inches": [FIG_WIDTH_IN, FIG_HEIGHT_IN],
        "minimum_configured_font_pt": MIN_FONT_PT,
        "source_rows": len(source),
        "panel_a_effects": len(panel_a),
        "panel_a_formal_significant": int(panel_a["formal_significant"].sum()),
        "panel_b_sensitivity_points": len(panel_b),
        "panel_b_min_slope": float(panel_b["slope"].min()),
        "panel_b_max_slope": float(panel_b["slope"].max()),
        "model_form_rows_excluded": excluded_model_form_rows,
        "redundant_encoding": {
            "first_review": "muted blue circle",
            "selected_review": "muted orange square",
            "primary_scenario": "retained in source table; not separately encoded",
            "formal_significance_panel_A": "filled marker; no asterisk clutter",
            "panel_B_family_summary": "open median marker plus min--max bar",
        },
        "sha256": {
            "pdf": sha256_file(PDF_PATH),
            "png": sha256_file(PNG_PATH),
            "source": sha256_file(SOURCE_PATH),
        },
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest, validation = validate_formal_lock()
    effects, sensitivity = read_inputs()

    source_rows: list[dict[str, object]] = []
    panel_a = panel_a_data(effects, source_rows)
    panel_b, panel_b_summaries, excluded = panel_b_data(sensitivity, source_rows)
    source = pd.DataFrame(source_rows)
    source.to_csv(SOURCE_PATH, index=False, encoding="utf-8-sig")

    primary_rows = panel_a[panel_a["family"] == "overall"].set_index("wave")
    primary_slopes = {
        wave: float(primary_rows.loc[wave, "slope"]) for wave in WAVE_ORDER
    }
    if not primary_slopes[WAVE_ORDER[1]] > primary_slopes[WAVE_ORDER[0]]:
        raise AssertionError("selected-review primary slope is not stronger")

    fig = build_figure(panel_a, panel_b, panel_b_summaries, primary_slopes)
    fig.savefig(
        PDF_PATH,
        format="pdf",
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        PNG_PATH,
        format="png",
        dpi=600,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    report = validate_outputs(source, panel_a, panel_b, excluded)
    report["primary"] = {
        WAVE_LABELS[wave]: {
            "slope": primary_slopes[wave],
            "ci": validation["headline"][
                "first_ci" if wave == WAVE_ORDER[0] else "second_descriptive_concept_cluster_bootstrap_ci"
            ],
            "formal_p": validation["headline"][
                "first_permutation_p" if wave == WAVE_ORDER[0] else "second_exact_p"
            ],
        }
        for wave in WAVE_ORDER
    }
    report["input_hashes"] = {
        "manifest": sha256_file(MANIFEST_PATH),
        "validation": sha256_file(VALIDATION_PATH),
        "effects": sha256_file(EFFECTS_PATH),
        "sensitivity": sha256_file(SENSITIVITY_PATH),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
