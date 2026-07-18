"""Reproducible reliability analysis for the ADMA 2026 full-paper rebuild.

This program intentionally reads only the freshly cleaned human-rating file,
the independently validated human--API crosswalk, and the current 7,290-row
API score matrix.  It does not read any legacy human-rating derivative or any
previous reliability result.

Outputs (all below ``analysis/``):

* reliability_human_estimates.csv
* reliability_human_panel_size_curve.csv
* reliability_api_estimates.csv
* reliability_api_ensemble_size_stability.csv
* reliability_qa.json
* reliability_manifest.json

The formulas for quadratic Gwet AC2, ordinal Krippendorff alpha, and the
absolute-agreement ICCs are the checked implementations used in
``stats_sandbox_method_checks.py``.  Concept-cluster bootstrap confidence
intervals are stratified by domain for multi-domain API scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


SCRIPT_VERSION = "4.0.0"
CLEANING_RULE_VERSION = "4.0.0"
SEED = 20260717
CATEGORIES = np.arange(1.0, 6.0)
DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("fa", "事实准确性"),
    ("cc", "概念完整性"),
    ("lc", "语言清晰度"),
    ("tf", "任务符合度"),
    ("mq", "误解处理质量"),
    ("risk", "误导风险"),
)
HUMAN_METRICS = ("gwet_ac2", "alpha_ordinal", "icc_a1", "icc_ak")
API_METRICS = ("alpha_interval", "icc_a1", "icc_ak", "kendall_w")


def quadratic_agreement_weights(categories: np.ndarray = CATEGORIES) -> np.ndarray:
    """Return quadratic agreement weights (one on the diagonal)."""
    categories = np.asarray(categories, dtype=float)
    scale = float(categories.max() - categories.min())
    if scale <= 0:
        raise ValueError("At least two ordered categories are required")
    delta = categories[:, None] - categories[None, :]
    return 1.0 - (delta / scale) ** 2


def gwet_ac2(
    ratings: np.ndarray,
    categories: np.ndarray = CATEGORIES,
    weights: np.ndarray | None = None,
) -> float:
    """Quadratic Gwet AC2 for a units-by-raters matrix with optional missingness."""
    x = np.asarray(ratings, dtype=float)
    categories = np.asarray(categories, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be two-dimensional")
    if x.shape[1] < 2:
        raise ValueError("Gwet AC2 is not identifiable with one rater")
    if weights is None:
        weights = quadratic_agreement_weights(categories)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(categories), len(categories)):
        raise ValueError("weights shape does not match categories")

    counts = np.column_stack([np.sum(x == category, axis=1) for category in categories])
    n_ratings = counts.sum(axis=1)
    if np.any(n_ratings == 0):
        raise ValueError("units with no ratings must be removed")
    valid = n_ratings >= 2
    if not np.any(valid):
        raise ValueError("at least one unit must have two ratings")

    weighted_counts = counts @ weights.T
    ordered_pair_agreement = np.sum(counts * (weighted_counts - 1.0), axis=1)
    observed = np.mean(
        ordered_pair_agreement[valid]
        / (n_ratings[valid] * (n_ratings[valid] - 1.0))
    )
    marginal = np.mean(counts / n_ratings[:, None], axis=0)
    q = len(categories)
    expected = weights.sum() * np.sum(marginal * (1.0 - marginal)) / (q * (q - 1.0))
    denominator = 1.0 - expected
    if np.isclose(denominator, 0.0):
        raise ValueError("Gwet AC2 is undefined for this degenerate matrix")
    return float((observed - expected) / denominator)


def krippendorff_alpha_ordinal(
    ratings: np.ndarray,
    categories: np.ndarray = CATEGORIES,
) -> float:
    """Customary ordinal Krippendorff alpha for units by raters."""
    x = np.asarray(ratings, dtype=float)
    categories = np.asarray(categories, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be two-dimensional")
    if x.shape[1] < 2:
        raise ValueError("alpha is not identifiable with one rater")

    counts = np.column_stack([np.sum(x == category, axis=1) for category in categories])
    n_per_unit = counts.sum(axis=1)
    if np.all(n_per_unit <= 1):
        raise ValueError("at least one unit must have two ratings")

    q = len(categories)
    coincidence = np.zeros((q, q), dtype=float)
    for row, n_i in zip(counts, n_per_unit):
        if n_i < 2:
            continue
        coincidence += (np.outer(row, row) - np.diag(row)) / (n_i - 1.0)

    marginal = coincidence.sum(axis=0)
    n_total = float(marginal.sum())
    if n_total <= 1:
        raise ValueError("insufficient pairable ratings")
    expected = (np.outer(marginal, marginal) - np.diag(marginal)) / (n_total - 1.0)

    distance = np.zeros((q, q), dtype=float)
    for i in range(q):
        for j in range(q):
            lo, hi = sorted((i, j))
            span = marginal[lo : hi + 1].sum() - (marginal[i] + marginal[j]) / 2.0
            distance[i, j] = span**2

    observed_disagreement = float(np.sum(coincidence * distance))
    expected_disagreement = float(np.sum(expected * distance))
    if np.isclose(expected_disagreement, 0.0):
        raise ValueError("alpha is undefined when expected disagreement is zero")
    return float(1.0 - observed_disagreement / expected_disagreement)


def krippendorff_alpha_interval(ratings: np.ndarray) -> float:
    """Interval Krippendorff alpha for a complete units-by-raters matrix.

    For complete panels this pairwise-distance form is algebraically identical
    to the coincidence-matrix definition and is much faster in the bootstrap.
    """
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or np.isnan(x).any():
        raise ValueError("interval alpha requires a complete two-dimensional matrix")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("interval alpha requires at least two units and two raters")
    total_n = n * k
    within_pair_distance = float(np.sum(k * np.sum(x * x, axis=1) - np.sum(x, axis=1) ** 2))
    flat = x.ravel()
    global_pair_distance = float(total_n * np.sum(flat * flat) - np.sum(flat) ** 2)
    if np.isclose(global_pair_distance, 0.0):
        raise ValueError("interval alpha is undefined when all ratings are constant")
    disagreement_ratio = (within_pair_distance / global_pair_distance) * (
        (total_n - 1.0) / (k - 1.0)
    )
    return float(1.0 - disagreement_ratio)


def icc_absolute_agreement(ratings: np.ndarray) -> tuple[float, float]:
    """Balanced two-way absolute-agreement ICC(A,1) and ICC(A,k)."""
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or np.isnan(x).any():
        raise ValueError("ICC(A,.) requires a complete two-dimensional matrix")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("ICC(A,.) requires at least two targets and two raters")

    grand = float(x.mean())
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
    denominator_a1 = ms_target + (k - 1.0) * ms_error + k * (ms_rater - ms_error) / n
    denominator_ak = ms_target + (ms_rater - ms_error) / n
    if np.isclose(denominator_a1, 0.0) or np.isclose(denominator_ak, 0.0):
        raise ValueError("ICC is undefined for this degenerate matrix")
    return float(numerator / denominator_a1), float(numerator / denominator_ak)


def kendall_w(ratings: np.ndarray) -> float:
    """Tie-corrected Kendall W for a complete items-by-judges matrix."""
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or np.isnan(x).any():
        raise ValueError("Kendall W requires a complete matrix")
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
    if np.isclose(denominator, 0.0):
        raise ValueError("Kendall W is undefined for this degenerate matrix")
    return float(12.0 * s / denominator)


def kendall_w_from_precomputed_ranks(
    ranks: np.ndarray,
    tie_terms: np.ndarray,
    subset: tuple[int, ...],
) -> float:
    """Kendall W for a judge subset whose per-judge ranks are precomputed."""
    selected = np.asarray(subset, dtype=int)
    n = ranks.shape[0]
    m = len(selected)
    if n < 2 or m < 2:
        return float("nan")
    rank_sums = ranks[:, selected].sum(axis=1)
    s = np.sum((rank_sums - m * (n + 1.0) / 2.0) ** 2)
    denominator = m**2 * (n**3 - n) - m * float(tie_terms[selected].sum())
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(12.0 * s / denominator)


def fast_spearman_against_precomputed_rank(
    values: np.ndarray,
    reference_rank: np.ndarray,
) -> float:
    """Spearman rho with one side's ranks already available."""
    value_rank = rankdata(values, method="average")
    value_centered = value_rank - value_rank.mean()
    reference_centered = reference_rank - reference_rank.mean()
    denominator = math.sqrt(
        float(np.sum(value_centered**2) * np.sum(reference_centered**2))
    )
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(np.sum(value_centered * reference_centered) / denominator)


def safe_metrics(
    matrix: np.ndarray,
    metric_names: Iterable[str],
) -> dict[str, float]:
    """Compute requested metrics; undefined metrics are returned as NaN."""
    requested = set(metric_names)
    result = {name: float("nan") for name in requested}
    try:
        if "gwet_ac2" in requested:
            result["gwet_ac2"] = gwet_ac2(matrix)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        pass
    try:
        if "alpha_ordinal" in requested:
            result["alpha_ordinal"] = krippendorff_alpha_ordinal(matrix)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        pass
    try:
        if "alpha_interval" in requested:
            result["alpha_interval"] = krippendorff_alpha_interval(matrix)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        pass
    if "icc_a1" in requested or "icc_ak" in requested:
        try:
            a1, ak = icc_absolute_agreement(matrix)
            if "icc_a1" in requested:
                result["icc_a1"] = a1
            if "icc_ak" in requested:
                result["icc_ak"] = ak
        except (ValueError, FloatingPointError, ZeroDivisionError):
            pass
    try:
        if "kendall_w" in requested:
            result["kendall_w"] = kendall_w(matrix)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        pass
    return result


def stable_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def quantiles(values: Iterable[float]) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.5)),
        float(np.quantile(array, 0.975)),
        int(array.size),
    )


def concept_cluster_bootstrap(
    matrix: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray,
    metric_names: tuple[str, ...],
    n_bootstrap: int,
    label: str,
) -> dict[str, tuple[float, float, int]]:
    """Domain-stratified concept-cluster percentile bootstrap intervals."""
    x = np.asarray(matrix, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    strata = np.asarray(strata, dtype=object)
    if not (len(x) == len(clusters) == len(strata)):
        raise ValueError("matrix, clusters, and strata must have equal row counts")
    rng = stable_rng(f"cluster_bootstrap|{label}|{n_bootstrap}")

    rows_by_key: dict[tuple[object, object], np.ndarray] = {}
    clusters_by_stratum: dict[object, np.ndarray] = {}
    for stratum in pd.unique(strata):
        stratum_clusters = pd.unique(clusters[strata == stratum])
        clusters_by_stratum[stratum] = np.asarray(stratum_clusters, dtype=object)
        for cluster in stratum_clusters:
            rows_by_key[(stratum, cluster)] = np.flatnonzero(
                (strata == stratum) & (clusters == cluster)
            )

    draws: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(n_bootstrap):
        sampled_rows: list[int] = []
        for stratum, available_clusters in clusters_by_stratum.items():
            sampled_clusters = rng.choice(
                available_clusters,
                size=len(available_clusters),
                replace=True,
            )
            for cluster in sampled_clusters:
                sampled_rows.extend(rows_by_key[(stratum, cluster)].tolist())
        estimates = safe_metrics(x[np.asarray(sampled_rows, dtype=int)], metric_names)
        for metric in metric_names:
            draws[metric].append(estimates[metric])

    intervals: dict[str, tuple[float, float, int]] = {}
    for metric, values in draws.items():
        low, _, high, valid = quantiles(values)
        intervals[metric] = (low, high, valid)
    return intervals


def matrix_from_panel(part: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return complete item-by-rater matrix plus item metadata."""
    duplicate_count = int(
        part.duplicated(["item_id", "participant_id"], keep=False).sum()
    )
    if duplicate_count:
        raise ValueError(f"panel contains {duplicate_count} duplicate item/rater cells")
    pivot = part.pivot(index="item_id", columns="participant_id", values="score_raw")
    item_meta = (
        part[["item_id", "concept_id", "domain"]]
        .drop_duplicates("item_id")
        .set_index("item_id")
        .loc[pivot.index]
        .reset_index()
    )
    return pivot.to_numpy(dtype=float), item_meta, pivot.columns.astype(str).tolist()


def human_estimates(
    merged_human: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = ["wave", "domain", "dimension"]
    for (wave, domain, dimension), part in merged_human.groupby(grouping, sort=True):
        matrix, item_meta, raters = matrix_from_panel(part)
        n_items, n_raters = matrix.shape
        complete = bool(np.isfinite(matrix).all())
        point = safe_metrics(matrix, HUMAN_METRICS)
        if n_raters < 2:
            intervals = {
                metric: (float("nan"), float("nan"), 0) for metric in HUMAN_METRICS
            }
            note = "not identifiable: one retained rater"
        elif not complete:
            # AC2 and alpha allow missing cells, but the balanced ICC does not.
            intervals = concept_cluster_bootstrap(
                matrix,
                item_meta["concept_id"].astype(str).to_numpy(),
                item_meta["domain"].astype(str).to_numpy(),
                HUMAN_METRICS,
                n_bootstrap,
                f"human|{wave}|{domain}|{dimension}",
            )
            note = "incomplete panel: ICC(A,.) undefined"
        else:
            intervals = concept_cluster_bootstrap(
                matrix,
                item_meta["concept_id"].astype(str).to_numpy(),
                item_meta["domain"].astype(str).to_numpy(),
                HUMAN_METRICS,
                n_bootstrap,
                f"human|{wave}|{domain}|{dimension}",
            )
            note = ""

        row: dict[str, object] = {
            "wave": wave,
            "domain": domain,
            "dimension": dimension,
            "n_items": n_items,
            "n_raters": n_raters,
            "n_concepts": int(item_meta["concept_id"].nunique()),
            "complete_panel": complete,
            "bootstrap_reps_requested": n_bootstrap if n_raters >= 2 else 0,
            "note": note,
        }
        for metric in HUMAN_METRICS:
            low, high, valid = intervals[metric]
            row[metric] = point[metric]
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_bootstrap_valid"] = valid
        rows.append(row)
    return pd.DataFrame(rows)


def enumerate_or_sample_subsets(
    n_raters: int,
    panel_size: int,
    max_subsets: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    count = math.comb(n_raters, panel_size)
    if count <= max_subsets:
        return list(itertools.combinations(range(n_raters), panel_size))
    subsets: set[tuple[int, ...]] = set()
    while len(subsets) < max_subsets:
        subset = tuple(sorted(rng.choice(n_raters, size=panel_size, replace=False).tolist()))
        subsets.add(subset)
    return sorted(subsets)


def human_panel_size_curve(
    merged_human: pd.DataFrame,
    second_wave: str,
    max_subsets: int,
) -> pd.DataFrame:
    """Reliability variability across observed-rater subpanels in wave two."""
    rows: list[dict[str, object]] = []
    second = merged_human[merged_human["wave"] == second_wave]
    for (domain, dimension), part in second.groupby(["domain", "dimension"], sort=True):
        matrix, _, raters = matrix_from_panel(part)
        n_items, n_raters = matrix.shape
        if not np.isfinite(matrix).all():
            raise ValueError("second-review panels must be complete for panel-size analysis")
        for panel_size in range(2, n_raters + 1):
            rng = stable_rng(f"panel_size|{second_wave}|{domain}|{dimension}|{panel_size}")
            subsets = enumerate_or_sample_subsets(
                n_raters, panel_size, max_subsets, rng
            )
            draws: dict[str, list[float]] = {metric: [] for metric in HUMAN_METRICS}
            for subset in subsets:
                estimates = safe_metrics(matrix[:, subset], HUMAN_METRICS)
                for metric in HUMAN_METRICS:
                    draws[metric].append(estimates[metric])
            row: dict[str, object] = {
                "wave": second_wave,
                "domain": domain,
                "dimension": dimension,
                "n_items": n_items,
                "available_raters": n_raters,
                "panel_size": panel_size,
                "total_possible_subpanels": math.comb(n_raters, panel_size),
                "subpanels_evaluated": len(subsets),
                "sampling": (
                    "all combinations"
                    if math.comb(n_raters, panel_size) <= max_subsets
                    else "deterministic random unique subpanels"
                ),
            }
            for metric in HUMAN_METRICS:
                low, median, high, valid = quantiles(draws[metric])
                row[f"{metric}_median"] = median
                row[f"{metric}_p025"] = low
                row[f"{metric}_p975"] = high
                row[f"{metric}_valid_subpanels"] = valid
            rows.append(row)
    return pd.DataFrame(rows)


def api_scope_ids(
    api: pd.DataFrame,
    crosswalk: pd.DataFrame,
    merged_human: pd.DataFrame,
    first_wave: str,
    second_wave: str,
) -> dict[str, set[str]]:
    scopes = {
        "all810": set(api["item_id"].astype(str)),
        "broad180": set(
            crosswalk.loc[crosswalk["wave"] == first_wave, "item_id"].astype(str)
        ),
        "final_first_human_covered": set(
            merged_human.loc[merged_human["wave"] == first_wave, "item_id"].astype(str)
        ),
        "selected30": set(
            crosswalk.loc[crosswalk["wave"] == second_wave, "item_id"].astype(str)
        ),
    }
    if any(not ids for ids in scopes.values()):
        raise ValueError(
            f"one or more API reliability scopes are empty: "
            f"{ {name: len(ids) for name, ids in scopes.items()} }"
        )
    all_ids = scopes["all810"]
    if any(not ids.issubset(all_ids) for name, ids in scopes.items() if name != "all810"):
        raise ValueError("an API reliability scope contains items absent from the current API matrix")
    if not scopes["final_first_human_covered"].issubset(scopes["broad180"]):
        raise ValueError("final first-review human-covered items exceed the broad first-review scope")
    if not scopes["selected30"].issubset(scopes["broad180"]):
        raise ValueError("selected second-review items are not a subset of the broad first-review scope")
    return scopes


def api_matrix(
    part: pd.DataFrame,
    score_column: str,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    duplicate_count = int(
        part.duplicated(["item_id", "paper_model_label"], keep=False).sum()
    )
    if duplicate_count:
        raise ValueError(f"API panel contains {duplicate_count} duplicate item/judge cells")
    pivot = part.pivot(index="item_id", columns="paper_model_label", values=score_column)
    metadata = (
        part[["item_id", "domain", "concept_id"]]
        .drop_duplicates("item_id")
        .set_index("item_id")
        .loc[pivot.index]
        .reset_index()
    )
    return pivot.to_numpy(dtype=float), metadata, pivot.columns.astype(str).tolist()


def api_estimates(
    api: pd.DataFrame,
    scopes: dict[str, set[str]],
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, ids in scopes.items():
        part_scope = api[api["item_id"].astype(str).isin(ids)]
        for score_column, dimension in DIMENSIONS:
            matrix, meta, judges = api_matrix(part_scope, score_column)
            if matrix.shape != (len(ids), 9) or not np.isfinite(matrix).all():
                raise ValueError(
                    f"{scope}/{dimension} is not a complete {len(ids)} x 9 API panel"
                )
            point = safe_metrics(matrix, API_METRICS)
            clusters = (
                meta["domain"].astype(str) + "|" + meta["concept_id"].astype(str)
            ).to_numpy()
            strata = meta["domain"].astype(str).to_numpy()
            intervals = concept_cluster_bootstrap(
                matrix,
                clusters,
                strata,
                API_METRICS,
                n_bootstrap,
                f"api|{scope}|{score_column}",
            )
            row: dict[str, object] = {
                "scope": scope,
                "dimension_code": score_column,
                "dimension": dimension,
                "n_items": matrix.shape[0],
                "n_judges": matrix.shape[1],
                "n_domains": int(meta["domain"].nunique()),
                "n_domain_concepts": int(pd.Series(clusters).nunique()),
                "bootstrap_reps_requested": n_bootstrap,
            }
            for metric in API_METRICS:
                low, high, valid = intervals[metric]
                row[metric] = point[metric]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                row[f"{metric}_bootstrap_valid"] = valid
            rows.append(row)
    return pd.DataFrame(rows)


def api_ensemble_size_stability(
    api: pd.DataFrame,
    scopes: dict[str, set[str]],
) -> pd.DataFrame:
    """Exact judge-subset stability curves for all 2^9-1 nonempty panels."""
    rows: list[dict[str, object]] = []
    for scope, ids in scopes.items():
        part_scope = api[api["item_id"].astype(str).isin(ids)]
        for score_column, dimension in DIMENSIONS:
            matrix, _, judges = api_matrix(part_scope, score_column)
            full_mean = matrix.mean(axis=1)
            full_mean_rank = rankdata(full_mean, method="average")
            n_judges = matrix.shape[1]
            judge_ranks = np.column_stack(
                [rankdata(matrix[:, j], method="average") for j in range(n_judges)]
            )
            judge_tie_terms = np.zeros(n_judges, dtype=float)
            for j in range(n_judges):
                _, counts = np.unique(matrix[:, j], return_counts=True)
                judge_tie_terms[j] = float(np.sum(counts**3 - counts))
            for panel_size in range(1, n_judges + 1):
                subsets = list(itertools.combinations(range(n_judges), panel_size))
                draws: dict[str, list[float]] = {
                    "mean_vs_full_spearman": [],
                    "mean_vs_full_mae": [],
                    "alpha_interval": [],
                    "icc_ak": [],
                    "kendall_w": [],
                }
                for subset in subsets:
                    submatrix = matrix[:, subset]
                    subset_mean = submatrix.mean(axis=1)
                    rho = fast_spearman_against_precomputed_rank(
                        subset_mean, full_mean_rank
                    )
                    draws["mean_vs_full_spearman"].append(float(rho))
                    draws["mean_vs_full_mae"].append(
                        float(np.mean(np.abs(subset_mean - full_mean)))
                    )
                    estimates = safe_metrics(submatrix, ("alpha_interval", "icc_ak"))
                    draws["alpha_interval"].append(estimates["alpha_interval"])
                    draws["icc_ak"].append(estimates["icc_ak"])
                    draws["kendall_w"].append(
                        kendall_w_from_precomputed_ranks(
                            judge_ranks, judge_tie_terms, subset
                        )
                    )

                row: dict[str, object] = {
                    "scope": scope,
                    "dimension_code": score_column,
                    "dimension": dimension,
                    "n_items": matrix.shape[0],
                    "available_judges": n_judges,
                    "panel_size": panel_size,
                    "judge_subsets_evaluated": len(subsets),
                }
                for metric, values in draws.items():
                    low, median, high, valid = quantiles(values)
                    row[f"{metric}_median"] = median
                    row[f"{metric}_p025"] = low
                    row[f"{metric}_p975"] = high
                    row[f"{metric}_valid_subsets"] = valid
                rows.append(row)
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_final_cleaning_manifest(
    project: Path,
    human_path: Path,
    human: pd.DataFrame,
    final_primary_path: Path,
    final_primary: pd.DataFrame,
) -> tuple[dict[str, Any], Path]:
    """Verify that reliability uses exactly the frozen final-cleaning data."""
    manifest_path = project / "analysis" / "final_cleaning_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("qa_status") != "PASS":
        raise ValueError("final cleaning manifest must have qa_status=PASS")
    if manifest.get("rule_version") != CLEANING_RULE_VERSION:
        raise ValueError(
            "final cleaning rule_version mismatch: "
            f"{manifest.get('rule_version')!r} != {CLEANING_RULE_VERSION!r}"
        )
    required_rules = {
        "first_repeat_similarity_pct": 75,
        "second_repeat_similarity_pct": 90,
        "absolute_seconds_per_unique_stimulus": 12,
        "first_duration_floor_seconds": 432,
        "second_duration_floor_seconds": 72,
        "missing_duration": "retain",
        "cross_domain_identity_exclusion": "none",
    }
    rules = manifest.get("rules")
    if not isinstance(rules, dict):
        raise ValueError("final cleaning manifest lacks a rules object")
    mismatched_rules = {
        key: {"expected": value, "observed": rules.get(key)}
        for key, value in required_rules.items()
        if rules.get(key) != value
    }
    if mismatched_rules:
        raise ValueError(f"final cleaning rules mismatch: {mismatched_rules}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("final cleaning manifest lacks an outputs object")

    def validate_output(
        key: str, expected_path: Path, expected_frame: pd.DataFrame
    ) -> dict[str, Any]:
        record = outputs.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"manifest output {key!r} is missing")
        missing = sorted({"path", "sha256", "bytes", "rows"} - set(record))
        if missing:
            raise ValueError(f"manifest output {key!r} lacks fields: {missing}")
        declared_path = Path(str(record["path"]))
        if not declared_path.is_absolute():
            declared_path = project / declared_path
        if declared_path.resolve() != expected_path.resolve():
            raise ValueError(
                f"manifest output {key!r} path mismatch: {declared_path} != {expected_path}"
            )
        observed = {
            "sha256": sha256_file(expected_path),
            "bytes": int(expected_path.stat().st_size),
            "rows": int(len(expected_frame)),
        }
        declared = {
            "sha256": str(record["sha256"]).lower(),
            "bytes": int(record["bytes"]),
            "rows": int(record["rows"]),
        }
        if declared != observed:
            raise ValueError(
                f"manifest output {key!r} does not match its file: "
                f"declared={declared}, observed={observed}"
            )
        return record

    human_record = validate_output(
        "cleaned_human_ratings_long", human_path, human
    )
    primary_record = validate_output("final_primary", final_primary_path, final_primary)
    for field in ("sha256", "bytes", "rows"):
        left = str(human_record[field]).lower() if field == "sha256" else int(human_record[field])
        right = str(primary_record[field]).lower() if field == "sha256" else int(primary_record[field])
        if left != right:
            raise ValueError(
                f"formal cleaned data and final_primary differ on {field}: {left} != {right}"
            )
    if list(human.columns) != list(final_primary.columns) or not human.equals(
        final_primary
    ):
        raise ValueError(
            "final_primary must exactly equal cleaned_human_ratings_long"
        )

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("final cleaning manifest lacks a counts object")
    if int(counts.get("final_long_rows", -1)) != len(human):
        raise ValueError("manifest final_long_rows does not match cleaned data")
    declared_participants = counts.get("final_participants")
    if not isinstance(declared_participants, dict):
        raise ValueError("manifest final_participants must be an object by wave")
    observed_participants = {
        str(wave): int(group["participant_id"].nunique())
        for wave, group in human.groupby("wave", sort=True)
    }
    normalized_declared = {
        str(wave): int(value) for wave, value in declared_participants.items()
    }
    if normalized_declared != observed_participants:
        raise ValueError(
            "manifest final_participants does not match cleaned data: "
            f"{normalized_declared} != {observed_participants}"
        )
    return manifest, manifest_path


def find_unique_api_input(workspace: Path) -> Path:
    candidates = sorted(workspace.rglob("api_test_scores_7290.csv"))
    candidates = [
        path
        for path in candidates
        if "40_GitHub" in str(path) and "rebuttal_update_20260714" in str(path)
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one current API matrix under 40_GitHub/rebuttal_update_20260714; found {candidates}"
        )
    return candidates[0]


def self_checks() -> dict[str, object]:
    """Formula checks inherited from the independent sandbox."""
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
    ac2 = gwet_ac2(raw_four_raters)
    if round(ac1, 5) != 0.77544 or round(ac2, 3) != 0.914:
        raise AssertionError("Gwet formula self-check failed")

    ordinal_example = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ],
        dtype=float,
    ).T
    ordinal_alpha = krippendorff_alpha_ordinal(ordinal_example)
    if round(ordinal_alpha, 3) != 0.815:
        raise AssertionError("ordinal alpha formula self-check failed")

    perfect = np.tile(np.arange(1.0, 6.0)[:, None], (1, 3))
    a1, ak = icc_absolute_agreement(perfect)
    if not (
        np.isclose(krippendorff_alpha_interval(perfect), 1.0)
        and np.isclose(a1, 1.0)
        and np.isclose(ak, 1.0)
        and np.isclose(kendall_w(perfect), 1.0)
    ):
        raise AssertionError("perfect-agreement self-check failed")
    return {
        "irrCAC_example_ac1": round(ac1, 5),
        "irrCAC_example_quadratic_ac2": round(ac2, 5),
        "ordinal_alpha_example": round(ordinal_alpha, 6),
        "perfect_interval_alpha": krippendorff_alpha_interval(perfect),
        "perfect_icc_a1": a1,
        "perfect_icc_ak": ak,
        "perfect_kendall_w": kendall_w(perfect),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
        help="Concept-cluster bootstrap replicates (default: 5000).",
    )
    parser.add_argument(
        "--panel-max-subsets",
        type=int,
        default=500,
        help="Maximum second-wave rater subpanels evaluated at each size.",
    )
    parser.add_argument(
        "--api-scores",
        help="Explicit path to the current api_test_scores_7290.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("bootstrap-reps must be at least 100")
    if args.panel_max_subsets < 50:
        raise ValueError("panel-max-subsets must be at least 50")

    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    analysis = project / "analysis"
    human_path = analysis / "cleaned_human_ratings_long.csv"
    final_primary_path = (
        analysis / "final_sensitivity_panels" / "final_primary.csv"
    )
    final_cleaning_manifest_path = analysis / "final_cleaning_manifest.json"
    crosswalk_path = analysis / "human_api_crosswalk.csv"
    api_path = (
        Path(args.api_scores).expanduser().resolve()
        if args.api_scores
        else find_unique_api_input(workspace)
    )
    for path in (
        human_path,
        final_primary_path,
        final_cleaning_manifest_path,
        crosswalk_path,
        api_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    human = pd.read_csv(human_path)
    final_primary = pd.read_csv(final_primary_path)
    crosswalk = pd.read_csv(crosswalk_path)
    api = pd.read_csv(api_path)
    final_cleaning_manifest, final_cleaning_manifest_path = (
        validate_final_cleaning_manifest(
            project,
            human_path,
            human,
            final_primary_path,
            final_primary,
        )
    )

    required_human = {
        "wave",
        "domain",
        "participant_id",
        "item_local_id",
        "dimension",
        "score_raw",
    }
    required_crosswalk = {
        "wave",
        "domain_cn",
        "item_local_id",
        "item_id",
        "concept_id",
    }
    required_api = {
        "item_id",
        "domain",
        "concept_id",
        "paper_model_label",
        *(code for code, _ in DIMENSIONS),
    }
    if not required_human.issubset(human.columns):
        raise ValueError(f"human file missing {required_human - set(human.columns)}")
    if not required_crosswalk.issubset(crosswalk.columns):
        raise ValueError(f"crosswalk missing {required_crosswalk - set(crosswalk.columns)}")
    if not required_api.issubset(api.columns):
        raise ValueError(f"API file missing {required_api - set(api.columns)}")

    merged_human = human.merge(
        crosswalk[["wave", "domain_cn", "item_local_id", "item_id", "concept_id"]],
        left_on=["wave", "domain", "item_local_id"],
        right_on=["wave", "domain_cn", "item_local_id"],
        how="left",
        validate="many_to_one",
    ).drop(columns="domain_cn")
    if merged_human[["item_id", "concept_id"]].isna().any().any():
        raise ValueError("some cleaned human ratings do not have a crosswalk match")

    wave_sizes = crosswalk.groupby("wave")["item_id"].nunique().sort_values()
    if wave_sizes.tolist() != [30, 180]:
        raise ValueError(f"expected crosswalk waves of 30 and 180 items; found {wave_sizes.to_dict()}")
    second_wave = str(wave_sizes.index[0])
    first_wave = str(wave_sizes.index[-1])

    checks = self_checks()
    human_results = human_estimates(merged_human, args.bootstrap_reps)
    panel_curve = human_panel_size_curve(
        merged_human, second_wave, args.panel_max_subsets
    )
    scopes = api_scope_ids(api, crosswalk, merged_human, first_wave, second_wave)
    api_results = api_estimates(api, scopes, args.bootstrap_reps)
    ensemble_curve = api_ensemble_size_stability(api, scopes)

    output_paths = {
        "human_estimates": analysis / "reliability_human_estimates.csv",
        "human_panel_size_curve": analysis / "reliability_human_panel_size_curve.csv",
        "api_estimates": analysis / "reliability_api_estimates.csv",
        "api_ensemble_size_stability": analysis
        / "reliability_api_ensemble_size_stability.csv",
    }
    human_results.to_csv(output_paths["human_estimates"], index=False, encoding="utf-8-sig")
    panel_curve.to_csv(
        output_paths["human_panel_size_curve"], index=False, encoding="utf-8-sig"
    )
    api_results.to_csv(output_paths["api_estimates"], index=False, encoding="utf-8-sig")
    ensemble_curve.to_csv(
        output_paths["api_ensemble_size_stability"], index=False, encoding="utf-8-sig"
    )

    one_rater = human_results[human_results["n_raters"] == 1]
    one_rater_metric_columns = [
        column
        for column in human_results.columns
        if any(column == metric or column.startswith(f"{metric}_ci_") for metric in HUMAN_METRICS)
    ]
    scope_counts = {name: len(ids) for name, ids in scopes.items()}
    expected_scope_counts = {
        "all810": int(api["item_id"].astype(str).nunique()),
        "broad180": int(
            crosswalk.loc[crosswalk["wave"] == first_wave, "item_id"]
            .astype(str)
            .nunique()
        ),
        "final_first_human_covered": int(
            merged_human.loc[merged_human["wave"] == first_wave, "item_id"]
            .astype(str)
            .nunique()
        ),
        "selected30": int(
            crosswalk.loc[crosswalk["wave"] == second_wave, "item_id"]
            .astype(str)
            .nunique()
        ),
    }
    expected_human_result_rows = int(
        merged_human.groupby(["wave", "domain", "dimension"]).ngroups
    )
    second_group_raters = merged_human.loc[
        merged_human["wave"] == second_wave
    ].groupby(["domain", "dimension"])["participant_id"].nunique()
    expected_panel_curve_rows = int(
        sum(max(int(n_raters) - 1, 0) for n_raters in second_group_raters)
    )
    expected_api_result_rows = len(scopes) * len(DIMENSIONS)
    expected_ensemble_rows = len(scopes) * len(DIMENSIONS) * int(
        api["paper_model_label"].nunique()
    )
    api_scope_rows_match = all(
        set(api_results.loc[api_results["scope"] == scope, "n_items"].astype(int))
        == {count}
        for scope, count in scope_counts.items()
    )
    qa_checks = {
        "human_input_rows_match_final_manifest": int(len(human))
        == int(final_cleaning_manifest["counts"]["final_long_rows"]),
        "final_primary_equals_formal_cleaned": bool(
            list(human.columns) == list(final_primary.columns)
            and human.equals(final_primary)
        ),
        "api_input_rows_7290": int(len(api)) == 7290,
        "api_items_810": int(api["item_id"].nunique()) == 810,
        "api_judges_9": int(api["paper_model_label"].nunique()) == 9,
        "crosswalk_rows_210": int(len(crosswalk)) == 210,
        "human_crosswalk_complete": not merged_human[["item_id", "concept_id"]].isna().any().any(),
        "scope_sizes_match_observed_inputs": scope_counts == expected_scope_counts,
        "api_scope_result_item_counts_match": api_scope_rows_match,
        "human_estimate_rows_match_observed_groups": int(len(human_results))
        == expected_human_result_rows,
        "one_rater_all_point_and_ci_na": bool(
            one_rater[one_rater_metric_columns].isna().all().all()
        ),
        "api_estimate_rows_match_scopes_dimensions": int(len(api_results))
        == expected_api_result_rows,
        "api_point_metrics_finite": bool(
            np.isfinite(api_results[list(API_METRICS)].to_numpy(dtype=float)).all()
        ),
        "ensemble_curve_rows_match_scopes_dimensions_panels": int(len(ensemble_curve))
        == expected_ensemble_rows,
        "ensemble_panel_sizes_1_to_9": sorted(ensemble_curve["panel_size"].unique().tolist())
        == list(range(1, 10)),
        "panel_curve_rows_match_observed_second_panels": int(len(panel_curve))
        == expected_panel_curve_rows,
        "formula_self_checks_passed": True,
    }
    if not all(qa_checks.values()):
        failed = [name for name, passed in qa_checks.items() if not passed]
        raise AssertionError(f"reliability QA failed: {failed}")

    qa_path = analysis / "reliability_qa.json"
    qa_payload: dict[str, object] = {
        "status": "PASS",
        "checks": qa_checks,
        "formula_self_checks": checks,
        "final_cleaning": {
            "rule_version": CLEANING_RULE_VERSION,
            "manifest_path": str(final_cleaning_manifest_path.relative_to(project)),
            "manifest_sha256": sha256_file(final_cleaning_manifest_path),
        },
        "observed": {
            "first_wave": first_wave,
            "second_wave": second_wave,
            "human_input_rows": int(len(human)),
            "one_rater_groups": int(len(one_rater)),
            "scope_item_counts": scope_counts,
            "expected_human_result_rows": expected_human_result_rows,
            "expected_panel_curve_rows": expected_panel_curve_rows,
            "human_result_rows": int(len(human_results)),
            "panel_curve_rows": int(len(panel_curve)),
            "api_result_rows": int(len(api_results)),
            "ensemble_curve_rows": int(len(ensemble_curve)),
        },
    }
    write_json(qa_path, qa_payload)

    output_hashes = {
        name: {
            "path": str(path.relative_to(project)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in output_paths.items()
    }
    output_hashes["qa"] = {
        "path": str(qa_path.relative_to(project)),
        "sha256": sha256_file(qa_path),
        "bytes": qa_path.stat().st_size,
    }
    manifest = {
        "analysis": "ADMA 2026 final-cleaning reliability engine",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "cleaning_rule_version": CLEANING_RULE_VERSION,
        "seed": SEED,
        "bootstrap": {
            "method": "domain-stratified concept-cluster percentile bootstrap",
            "repetitions": args.bootstrap_reps,
            "confidence_level": 0.95,
        },
        "second_review_panel_curve": {
            "method": "all rater combinations when feasible, otherwise deterministic unique subpanel sampling without replacement",
            "maximum_subpanels_per_size": args.panel_max_subsets,
        },
        "api_ensemble_curve": {
            "method": "exact enumeration of all nonempty subsets of nine current API judges",
            "subset_count_all_sizes": 511,
        },
        "input_policy": {
            "legacy_human_derivatives_read": False,
            "formal_human_input": "final cleaning rule version 4.0.0",
            "allowed_inputs_only": True,
        },
        "inputs": {
            "cleaned_human": {
                "path": str(human_path.relative_to(project)),
                "rows": int(len(human)),
                "sha256": sha256_file(human_path),
            },
            "final_primary": {
                "path": str(final_primary_path.relative_to(project)),
                "rows": int(len(final_primary)),
                "sha256": sha256_file(final_primary_path),
            },
            "final_cleaning_manifest": {
                "path": str(final_cleaning_manifest_path.relative_to(project)),
                "rule_version": final_cleaning_manifest["rule_version"],
                "sha256": sha256_file(final_cleaning_manifest_path),
            },
            "validated_crosswalk": {
                "path": str(crosswalk_path.relative_to(project)),
                "rows": int(len(crosswalk)),
                "sha256": sha256_file(crosswalk_path),
            },
            "current_api_scores": {
                "path": str(api_path.relative_to(workspace)),
                "rows": int(len(api)),
                "sha256": sha256_file(api_path),
            },
        },
        "scope_definitions": {
            "all810": "all current API-scored items",
            "broad180": "all 180 first-round questionnaire items, independent of retention",
            "final_first_human_covered": (
                "first-round items with at least one retained rating after final cleaning"
            ),
            "selected30": "the 30 purposively selected second-review items",
        },
        "scope_item_counts": {name: len(ids) for name, ids in scopes.items()},
        "metrics": {
            "human": list(HUMAN_METRICS),
            "api": list(API_METRICS),
            "human_one_rater_policy": "all reliability estimates and intervals are NA",
            "risk_direction": "raw high-is-risky scores; reliability is invariant to linear reversal",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "outputs": output_hashes,
        "qa_status": "PASS",
    }
    manifest_path = analysis / "reliability_manifest.json"
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "status": "PASS",
                "human_rows": len(human_results),
                "panel_curve_rows": len(panel_curve),
                "api_rows": len(api_results),
                "ensemble_curve_rows": len(ensemble_curve),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
