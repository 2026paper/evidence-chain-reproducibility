"""Render the five-layer evidence contract as a zero-arrow lane matrix.

Figure claim: all five analyses share controlled provenance, but each licenses a
different inference from its own post-QC material and analysis/resampling units.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIG_DIR = ROOT / "output" / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

INK = "#1B232C"
MUTED = "#637080"
RULE = "#D5DBE2"
ROW_FILL = "#F7F8FA"
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
PALE_BLUE = "#EAF4F9"
PALE_TEAL = "#EAF6F2"
PALE_ORANGE = "#FBEFEA"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_counts() -> dict[str, int]:
    cleaning = json.loads(
        (ANALYSIS / "final_cleaning_manifest.json").read_text(encoding="utf-8")
    )
    public = json.loads(
        (ANALYSIS / "source_manifests" / "public_abc_analysis_manifest.json").read_text(encoding="utf-8")
    )
    human = cleaning["counts"]["final_participants"]
    qc = public["quality_control"]
    counts = {
        "broad": human["首轮专家复核"],
        "selected": human["二次专家复核"],
        "reader": qc["final_responses"],
        "clusters": qc["final_participant_clusters"],
    }
    assert counts == {"broad": 18, "selected": 95, "reader": 178, "clusters": 167}
    return counts


def text_block(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7.0,
    color: str = INK,
    weight: str = "normal",
    ha: str = "left",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.08,
    )


def main() -> None:
    counts = load_counts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "layer": 1,
            "label": "API\nstability",
            "material": "9-judge score matrix\n810 items × 6 dimensions",
            "unit": "visible text ×\ndimension",
            "claim": "fixed-panel\nstability",
            "color": BLUE,
            "fill": PALE_BLUE,
            "material_type": "observed",
        },
        {
            "layer": 2,
            "label": "Broad\nreview",
            "material": f"{counts['broad']}/22 reviews retained\n178 linked visible texts",
            "unit": "concept cluster\nwithin domain",
            "claim": "broad human–API\ncriterion association",
            "color": TEAL,
            "fill": PALE_TEAL,
            "material_type": "observed",
        },
        {
            "layer": 3,
            "label": "Selected\nreview",
            "material": f"{counts['selected']}/162 reviews retained\n30 selected items",
            "unit": "domain × task blocks\nconcept-cluster CI",
            "claim": "selected-set\nhuman–API association",
            "color": TEAL,
            "fill": PALE_TEAL,
            "material_type": "observed",
        },
        {
            "layer": 4,
            "label": "Reader\nfielding",
            "material": f"{counts['reader']}/180 responses retained\n6 lay-adapted cases",
            "unit": f"participant cluster\n(n = {counts['clusters']})",
            "claim": "relative version ×\nphase change",
            "color": ORANGE,
            "fill": PALE_ORANGE,
            "material_type": "adapted",
        },
        {
            "layer": 5,
            "label": "Failure\nprobes",
            "material": "60 readers · 8 case pairs\nfixed 9-judge API panel",
            "unit": "respondent / case /\nfixed panel",
            "claim": "directional failure\nprofile",
            "color": ORANGE,
            "fill": PALE_ORANGE,
            "material_type": "engineered",
        },
    ]

    pd.DataFrame(
        [
            {
                "layer": row["layer"],
                "label": row["label"].replace("\n", " "),
                "post_qc_material": row["material"].replace("\n", " | "),
                "analysis_resampling_units": row["unit"].replace("\n", " | "),
                "licensed_inference": row["claim"].replace("\n", " "),
                "material_type": row["material_type"],
            }
            for row in rows
        ]
    ).to_csv(
        SOURCE_DIR / "fig1_five_layer_evidence_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "svg.hashsalt": "adma2026-fig1-v1",
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )

    width_in, height_in = 4.80, 2.50
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, 1)
    # Crop the now-redundant footer band while preserving the matrix geometry.
    ax.set_ylim(0.095, 1)
    ax.axis("off")

    text_block(
        ax,
        0.035,
        0.963,
        "Five parallel evidence contracts",
        size=8.5,
        weight="bold",
    )

    # Compact provenance strip. Each cell has a fixed width; no text can collide.
    bar_x, bar_y, bar_w, bar_h = 0.035, 0.835, 0.93, 0.095
    ax.add_patch(
        Rectangle(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            facecolor=PALE_BLUE,
            edgecolor=BLUE,
            linewidth=0.75,
        )
    )
    ax.add_patch(
        Rectangle(
            (bar_x, bar_y),
            0.150,
            bar_h,
            facecolor=BLUE,
            edgecolor=BLUE,
            linewidth=0,
        )
    )
    text_block(
        ax,
        0.110,
        0.883,
        "SHARED\nPROVENANCE",
        size=7.0,
        color="white",
        weight="bold",
        ha="center",
    )
    for x in (0.485, 0.715):
        ax.plot([x, x], [0.849, 0.916], color=RULE, lw=0.6)
    text_block(ax, 0.203, 0.883, "810 source records\n796 unique visible texts", size=7.0)
    text_block(ax, 0.505, 0.883, "fixed API panel\n9 judges", size=7.0)
    text_block(ax, 0.735, 0.883, "common rubric\n6 dimensions", size=7.0)

    header_y = 0.790
    text_block(ax, 0.043, header_y, "EVIDENCE\nLAYER", size=7.0, color=MUTED, weight="bold")
    text_block(ax, 0.202, header_y, "POST-QC\nMATERIAL", size=7.0, color=MUTED, weight="bold")
    text_block(
        ax,
        0.510,
        header_y,
        "ANALYSIS /\nRESAMPLING UNIT(S)",
        size=7.0,
        color=MUTED,
        weight="bold",
    )
    text_block(ax, 0.738, header_y, "LICENSED\nINFERENCE", size=7.0, color=MUTED, weight="bold")
    ax.plot([0.035, 0.965], [0.752, 0.752], color=INK, lw=0.8)

    centers = [0.681, 0.558, 0.435, 0.312, 0.189]
    row_h = 0.100
    for row, y in zip(rows, centers, strict=True):
        color = row["color"]
        ax.add_patch(
            Rectangle(
                (0.035, y - row_h / 2),
                0.93,
                row_h,
                facecolor=ROW_FILL,
                edgecolor="none",
            )
        )
        ax.add_patch(
            Rectangle(
                (0.035, y - row_h / 2),
                0.006,
                row_h,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            0.062,
            y,
            str(row["layer"]),
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "square,pad=0.18", "fc": color, "ec": color, "lw": 0},
        )
        text_block(ax, 0.090, y, row["label"], size=7.0, weight="bold")

        linestyle = "--" if row["material_type"] in {"adapted", "engineered"} else "-"
        ax.add_patch(
            Rectangle(
                (0.202, y - 0.040),
                0.293,
                0.080,
                facecolor=row["fill"],
                edgecolor=color,
                linewidth=0.65,
                linestyle=linestyle,
            )
        )
        text_block(ax, 0.216, y, row["material"], size=7.0)
        text_block(ax, 0.510, y, row["unit"], size=7.0, weight="semibold")

        # A vertical claim marker replaces all arrows and prevents process-flow readings.
        ax.add_patch(
            Rectangle(
                (0.718, y - 0.029),
                0.0035,
                0.058,
                facecolor=color,
                edgecolor="none",
            )
        )
        text_block(ax, 0.738, y, row["claim"], size=7.0, weight="bold")

    ax.plot([0.035, 0.965], [0.111, 0.111], color=RULE, lw=0.6)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    pdf_path = FIG_DIR / "fig1_five_layer_evidence.pdf"
    png_path = FIG_DIR / "fig1_five_layer_evidence.png"
    svg_path = FIG_DIR / "fig1_five_layer_evidence.svg"
    fig.savefig(pdf_path, bbox_inches=None, pad_inches=0)
    fig.savefig(png_path, dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(svg_path, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    page = PdfReader(pdf_path).pages[0]
    pdf_width = float(page.mediabox.width) / 72
    pdf_height = float(page.mediabox.height) / 72
    assert abs(pdf_width - width_in) < 0.01
    assert abs(pdf_height - height_in) < 0.01
    assert len(rows) == 5
    source_path = SOURCE_DIR / "fig1_five_layer_evidence_source.csv"
    report = {
        "status": "PASS",
        "dominant_claim": "shared provenance, five non-substitutable inferential contracts",
        "lanes": len(rows),
        "arrows": 0,
        "branching_arrows": 0,
        "crossing_arrows": 0,
        "figure_inches": [pdf_width, pdf_height],
        "minimum_font_pt": 7.0,
        "pdf_media_box_verified": True,
        "sha256": {
            "pdf": sha256(pdf_path),
            "png": sha256(png_path),
            "svg": sha256(svg_path),
            "source": sha256(source_path),
        },
    }
    (FIG_DIR / "fig1_evidence_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
