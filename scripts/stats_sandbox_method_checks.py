"""Independent method checks for the frozen ADMA reanalysis plan.

This sandbox does not read any legacy human-rating derivative and does not
write analysis results.  It verifies formulas against published examples and
prints the design supported by ``cleaned_human_ratings_long.csv``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests


SEED = 20260717
CATEGORIES = np.arange(1.0, 6.0)


def quadratic_agreement_weights(categories: np.ndarray = CATEGORIES) -> np.ndarray:
    """Return Gwet-style quadratic *agreement* weights (diagonal equals 1)."""
    categories = np.asarray(categories, dtype=float)
    scale = categories.max() - categories.min()
    if scale <= 0:
        raise ValueError("At least two distinct ordered categories are required")
    delta = categories[:, None] - categories[None, :]
    return 1.0 - (delta / scale) ** 2


def gwet_ac2(
    ratings: np.ndarray,
    categories: np.ndarray = CATEGORIES,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the Gwet AC1/AC2 point estimate for units x raters data.

    The implementation follows the raw-ratings equations used by irrCAC.  It
    supports missing cells, but units with no ratings must not be supplied.
    Confidence intervals are intentionally not supplied here; the production
    analysis should use the prespecified concept-cluster bootstrap.
    """
    x = np.asarray(ratings, dtype=float)
    categories = np.asarray(categories, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be a two-dimensional units x raters array")
    if weights is None:
        weights = quadratic_agreement_weights(categories)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(categories), len(categories)):
        raise ValueError("weights shape does not match the category domain")

    counts = np.column_stack([np.sum(x == category, axis=1) for category in categories])
    n_ratings = counts.sum(axis=1)
    if np.any(n_ratings == 0):
        raise ValueError("drop units with no ratings before computing AC2")
    valid = n_ratings >= 2
    if not np.any(valid):
        raise ValueError("at least one unit needs two ratings")

    weighted_counts = counts @ weights.T
    ordered_pair_agreement = np.sum(counts * (weighted_counts - 1.0), axis=1)
    observed = np.mean(
        ordered_pair_agreement[valid]
        / (n_ratings[valid] * (n_ratings[valid] - 1.0))
    )

    # Equal weight per unit is the raw-ratings irrCAC convention.
    marginal = np.mean(counts / n_ratings[:, None], axis=0)
    q = len(categories)
    expected = weights.sum() * np.sum(marginal * (1.0 - marginal)) / (q * (q - 1.0))
    coefficient = (observed - expected) / (1.0 - expected)
    return {"coefficient": float(coefficient), "pa": float(observed), "pe": float(expected)}


def krippendorff_alpha_ordinal(
    ratings: np.ndarray,
    categories: np.ndarray = CATEGORIES,
) -> float:
    """Compute customary ordinal Krippendorff alpha for units x raters data."""
    x = np.asarray(ratings, dtype=float)
    categories = np.asarray(categories, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be a two-dimensional units x raters array")

    counts = np.column_stack([np.sum(x == category, axis=1) for category in categories])
    n_per_unit = counts.sum(axis=1)
    if np.all(n_per_unit <= 1):
        raise ValueError("at least one unit needs two ratings")

    q = len(categories)
    coincidence = np.zeros((q, q), dtype=float)
    for row, n_i in zip(counts, n_per_unit):
        if n_i < 2:
            continue
        coincidence += (np.outer(row, row) - np.diag(row)) / (n_i - 1.0)

    marginal = coincidence.sum(axis=0)
    n_total = marginal.sum()
    if n_total <= 1:
        raise ValueError("insufficient pairable ratings")
    expected = (np.outer(marginal, marginal) - np.diag(marginal)) / (n_total - 1.0)

    distance = np.zeros((q, q), dtype=float)
    for i in range(q):
        for j in range(q):
            lo, hi = sorted((i, j))
            ordinal_span = marginal[lo : hi + 1].sum() - (marginal[i] + marginal[j]) / 2.0
            distance[i, j] = ordinal_span**2

    observed_disagreement = float(np.sum(coincidence * distance))
    expected_disagreement = float(np.sum(expected * distance))
    if expected_disagreement == 0:
        raise ValueError("alpha is undefined when expected disagreement is zero")
    return float(1.0 - observed_disagreement / expected_disagreement)


def icc_absolute_agreement(ratings: np.ndarray) -> dict[str, float]:
    """Return balanced two-way absolute-agreement ICC(A,1) and ICC(A,k)."""
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or np.isnan(x).any():
        raise ValueError("ICC(A,.) requires a complete two-dimensional matrix")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("ICC(A,.) requires at least two targets and two raters")

    grand = x.mean()
    target_means = x.mean(axis=1)
    rater_means = x.mean(axis=0)
    ss_target = k * np.sum((target_means - grand) ** 2)
    ss_rater = n * np.sum((rater_means - grand) ** 2)
    residual = x - target_means[:, None] - rater_means[None, :] + grand
    ss_error = np.sum(residual**2)
    ms_target = ss_target / (n - 1.0)
    ms_rater = ss_rater / (k - 1.0)
    ms_error = ss_error / ((n - 1.0) * (k - 1.0))

    numerator = ms_target - ms_error
    icc_a1 = numerator / (
        ms_target
        + (k - 1.0) * ms_error
        + k * (ms_rater - ms_error) / n
    )
    icc_ak = numerator / (ms_target + (ms_rater - ms_error) / n)
    return {
        "icc_a1": float(icc_a1),
        "icc_ak": float(icc_ak),
        "n_targets": int(n),
        "n_raters": int(k),
    }


def kendall_w(ratings: np.ndarray) -> float:
    """Tie-corrected Kendall W for complete items x judges score data."""
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or np.isnan(x).any():
        raise ValueError("Kendall W requires a complete items x judges matrix")
    n, m = x.shape
    if n < 2 or m < 2:
        raise ValueError("Kendall W requires at least two items and two judges")

    ranks = np.column_stack([rankdata(x[:, j], method="average") for j in range(m)])
    rank_sums = ranks.sum(axis=1)
    s = np.sum((rank_sums - m * (n + 1.0) / 2.0) ** 2)
    tie_term = 0.0
    for j in range(m):
        _, counts = np.unique(x[:, j], return_counts=True)
        tie_term += np.sum(counts**3 - counts)
    denominator = m**2 * (n**3 - n) - m * tie_term
    if denominator == 0:
        raise ValueError("Kendall W is undefined for this degenerate matrix")
    return float(12.0 * s / denominator)


def package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "krippendorff",
        "irrCAC",
        "pingouin",
        "pymer4",
        "linearmodels",
    ]
    versions: dict[str, str] = {}
    for name in names:
        if importlib.util.find_spec(name) is None:
            versions[name] = "MISSING"
            continue
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "installed"))
    return versions


def design_summary(data: pd.DataFrame) -> dict[str, object]:
    data = data.copy()
    data["item_placement"] = (
        data["wave"].astype(str)
        + "|"
        + data["domain"].astype(str)
        + "|"
        + data["item_local_id"].astype(str)
    )
    wave_rows: list[dict[str, object]] = []
    for wave, part in data.groupby("wave", sort=False):
        wave_rows.append(
            {
                "wave": wave,
                "rating_rows": int(len(part)),
                "participants": int(part["participant_id"].nunique()),
                "item_placements": int(part["item_placement"].nunique()),
                "unique_text_hashes": int(part["item_text_hash"].nunique()),
                "domains": sorted(part["domain"].unique().tolist()),
            }
        )

    panels: list[dict[str, object]] = []
    panel_base = data.drop_duplicates(
        ["wave", "domain", "participant_id", "item_placement", "dimension"]
    )
    for (wave, domain), part in panel_base.groupby(["wave", "domain"], sort=False):
        n_raters = int(part["participant_id"].nunique())
        n_items = int(part["item_placement"].nunique())
        panels.append(
            {
                "wave": wave,
                "domain": domain,
                "raters": n_raters,
                "item_placements": n_items,
                "gwet_alpha_icc_identifiable": bool(n_raters >= 2 and n_items >= 2),
                "warning": (
                    "one rater: between-rater reliability is not identifiable"
                    if n_raters < 2
                    else ("only two raters: point estimate identifiable, rater-population inference weak" if n_raters == 2 else "")
                ),
            }
        )

    duplicated_text = (
        data.drop_duplicates(["wave", "domain", "item_local_id", "item_text_hash"])
        .groupby(["wave", "domain", "item_text_hash"], as_index=False)
        .agg(local_ids=("item_local_id", lambda values: sorted(set(values))))
    )
    duplicated_text = duplicated_text[duplicated_text["local_ids"].map(len) > 1]
    duplicate_rows = duplicated_text.to_dict(orient="records")

    return {
        "waves": wave_rows,
        "panels": panels,
        "exact_text_duplicates_with_multiple_local_ids": duplicate_rows,
    }


def self_checks() -> dict[str, object]:
    # Example supplied in the Python irrCAC documentation.
    raw_four_raters = np.array(
        [
            [1, 1, np.nan, 1],
            [2, 2, 3, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 2, 3, 4],
            [4, 4, 4, 4],
            [1, 1, 2, 1],
            [2, 2, 2, 2],
            [np.nan, 5, 5, 5],
            [np.nan, np.nan, 1, 1],
            [np.nan, np.nan, 3, np.nan],
        ],
        dtype=float,
    )
    ac1 = gwet_ac2(raw_four_raters, weights=np.eye(5))
    ac2 = gwet_ac2(raw_four_raters, weights=quadratic_agreement_weights())
    assert round(ac1["coefficient"], 5) == 0.77544
    assert round(ac2["coefficient"], 3) == 0.914

    # Customary ordinal-alpha example used by fast-krippendorff.
    ordinal_example_raters_by_units = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ],
        dtype=float,
    )
    ordinal_alpha = krippendorff_alpha_ordinal(ordinal_example_raters_by_units.T)
    assert round(ordinal_alpha, 3) == 0.815

    perfect = np.tile(np.arange(1.0, 6.0)[:, None], (1, 3))
    icc = icc_absolute_agreement(perfect)
    assert np.isclose(icc["icc_a1"], 1.0)
    assert np.isclose(icc["icc_ak"], 1.0)
    assert np.isclose(kendall_w(perfect), 1.0)

    p_values = np.array([0.001, 0.01, 0.03, 0.2, 0.7, 0.9])
    bh = multipletests(p_values, method="fdr_bh")[1]
    holm = multipletests(p_values, method="holm")[1]
    assert np.all((0 <= bh) & (bh <= 1))
    assert np.all((0 <= holm) & (holm <= 1))

    return {
        "irrCAC_example_AC1": round(ac1["coefficient"], 5),
        "irrCAC_example_AC2_quadratic": round(ac2["coefficient"], 5),
        "ordinal_alpha_example": round(ordinal_alpha, 6),
        "perfect_ICC_A1": icc["icc_a1"],
        "perfect_ICC_Ak": icc["icc_ak"],
        "perfect_Kendall_W": kendall_w(perfect),
        "BH_adjusted_example": bh.round(6).tolist(),
        "Holm_adjusted_example": holm.round(6).tolist(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "analysis" / "cleaned_human_ratings_long.csv"
    data = pd.read_csv(data_path)
    report = {
        "seed_reserved_for_production_resampling": SEED,
        "packages": package_versions(),
        "self_checks": self_checks(),
        "cleaned_design": design_summary(data),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
