#!/usr/bin/env python3
"""Compare fixed API aggregation baselines against the broad human criterion.

All methods use the same 178 visible-text signatures, six quality-aligned
dimensions, fixed controls, domain-stratified concept-cluster bootstrap, and
shared Freedman--Lane stimulus maps.  Risk is transformed as 6 - raw risk for
both the API and human scores, so higher is better on every reported scale.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


SCRIPT_VERSION = "1.1.0"
SEED_BOOTSTRAP = 20260717
SEED_PERMUTATION = 20260717
N_BOOTSTRAP = 5_000
N_PERMUTATION = 10_000
DIMENSIONS = ["fa", "cc", "lc", "tf", "mq", "risk"]
CONTROLS = ["domain", "task", "dimension"]

BASE = Path(__file__).resolve().parent
PAPER_ROOT = BASE.parents[1]
ANALYSIS_ROOT = PAPER_ROOT / "analysis"
API_PATH = (
    PAPER_ROOT
    / "data"
    / "40_GitHub"
    / "rebuttal_update_20260714"
    / "api_test_scores_7290.csv"
)

GENERATOR_TO_JUDGE_PROVIDER = {
    "Alibaba Qwen": "qwen",
    "Anthropic Claude": "anthropic",
    "ByteDance Doubao": "doubao",
    "DeepSeek": "deepseek",
    "Google Gemini": "gemini",
    "Mimo": "mimo",
    "Moonshot Kimi": "kimi",
    "OpenAI GPT": "openai",
    "Zhipu GLM": "glm",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def control_matrix(data: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = [np.ones(len(data), dtype=float)]
    for control in CONTROLS:
        values = data[control].astype(str)
        levels = sorted(values.unique().tolist())
        for level in levels[1:]:
            columns.append((values == level).to_numpy(dtype=float))
    return np.column_stack(columns)


def safe_correlation(function: Any, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(function(x, y).statistic)


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx = float(np.var(x, ddof=0))
    vy = float(np.var(y, ddof=0))
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = vx + vy + float((x.mean() - y.mean()) ** 2)
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


def standardized_slope(data: pd.DataFrame, api_z_column: str = "api_z") -> float:
    z = control_matrix(data)
    x = data[api_z_column].to_numpy(float)
    y = data["human_z"].to_numpy(float)
    design = np.column_stack([z, x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(design, y, rcond=None)[0][-1])


def load_human_criterion() -> tuple[pd.DataFrame, dict[str, Any]]:
    item_path = ANALYSIS_ROOT / "alignment_human_api_item_dimension.csv"
    crosswalk_path = ANALYSIS_ROOT / "human_api_crosswalk.csv"
    cleaned_path = ANALYSIS_ROOT / "cleaned_human_ratings_long.csv"
    item = pd.read_csv(item_path)
    crosswalk = pd.read_csv(crosswalk_path)
    cleaned = pd.read_csv(cleaned_path)
    required = {
        "wave",
        "domain",
        "task",
        "concept_id",
        "concept",
        "selection_role",
        "stimulus_signature_sha256",
        "dimension",
        "human_score",
    }
    if not required.issubset(item.columns):
        raise AssertionError(f"Item-level human table lacks {sorted(required - set(item.columns))}")
    broad = item.loc[item["selection_role"] == "broad_scheduled_item", list(required)].copy()
    broad = broad.sort_values(["stimulus_signature_sha256", "dimension"]).reset_index(drop=True)
    if len(broad) != 1068 or broad["stimulus_signature_sha256"].nunique() != 178:
        raise AssertionError("Broad item-level fallback must be 178 stimuli x six dimensions")
    if set(broad["dimension"]) != set(DIMENSIONS):
        raise AssertionError("Broad human criterion does not contain the fixed six dimensions")
    per_stimulus = broad.groupby("stimulus_signature_sha256")["dimension"].agg(list)
    if not all(sorted(values) == sorted(DIMENSIONS) for values in per_stimulus):
        raise AssertionError("Each broad stimulus must have all six dimensions")
    if broad[["domain", "concept_id"]].drop_duplicates().shape[0] != 30:
        raise AssertionError("Broad human criterion must contain 30 domain-concept clusters")
    if broad["human_score"].isna().any() or not broad["human_score"].between(1, 5).all():
        raise AssertionError("Broad human scores must be complete and within 1--5")

    broad_crosswalk = crosswalk.loc[
        crosswalk["selection_role"] == "broad_scheduled_item"
    ].copy()
    if len(broad_crosswalk) != 180:
        raise AssertionError("Broad crosswalk must contain 180 questionnaire placements")
    if set(broad_crosswalk["stimulus_signature_sha256"]) != set(
        broad["stimulus_signature_sha256"]
    ):
        raise AssertionError("Broad crosswalk and item-level fallback signatures differ")
    if len(cleaned) != 7308:
        raise AssertionError("Cleaned human long table row count changed")
    return broad, {
        "alignment_human_api_item_dimension.csv": sha256_file(item_path),
        "human_api_crosswalk.csv": sha256_file(crosswalk_path),
        "cleaned_human_ratings_long.csv": sha256_file(cleaned_path),
        "broad_crosswalk_placements": int(len(broad_crosswalk)),
        "broad_visible_text_signatures": int(broad["stimulus_signature_sha256"].nunique()),
    }


def load_api_long() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    equivalence_path = ANALYSIS_ROOT / "api_stimulus_equivalence_810.csv"
    api = pd.read_csv(API_PATH)
    equivalence = pd.read_csv(equivalence_path)
    required = {
        "item_id",
        "source_item_uid",
        "generator",
        "judge_provider",
        "paper_model_label",
        "input_integrity_pass",
        *DIMENSIONS,
    }
    if not required.issubset(api.columns):
        raise AssertionError(f"API matrix lacks {sorted(required - set(api.columns))}")
    if len(api) != 7290 or api["source_item_uid"].nunique() != 810:
        raise AssertionError("Current API matrix must be 810 source UIDs x nine judges")
    integrity = api["input_integrity_pass"].astype(str).str.lower().isin(["true", "1"])
    if not integrity.all():
        raise AssertionError("API matrix contains input-integrity failures")
    for dimension in DIMENSIONS:
        values = pd.to_numeric(api[dimension], errors="coerce")
        if values.isna().any() or not values.between(1, 5).all():
            raise AssertionError(f"Invalid API values for {dimension}")
        api[dimension] = values
    coverage = api.groupby("source_item_uid")["judge_provider"].nunique()
    if not coverage.eq(9).all():
        raise AssertionError("Every source UID must have all nine judge providers")
    joined = api.merge(
        equivalence[
            ["source_item_uid", "item_id", "stimulus_signature_sha256", "source_equivalence_size"]
        ],
        on=["source_item_uid", "item_id"],
        how="left",
        validate="many_to_one",
    )
    if joined["stimulus_signature_sha256"].isna().any():
        raise AssertionError("API source UID lacks a visible-text signature")
    long = joined.melt(
        id_vars=[
            "source_item_uid",
            "stimulus_signature_sha256",
            "source_equivalence_size",
            "generator",
            "judge_provider",
            "paper_model_label",
        ],
        value_vars=DIMENSIONS,
        var_name="dimension",
        value_name="api_score_raw",
    )
    long["api_score_quality_aligned"] = np.where(
        long["dimension"] == "risk",
        6.0 - long["api_score_raw"].astype(float),
        long["api_score_raw"].astype(float),
    )
    long["generator_judge_provider"] = long["generator"].map(
        GENERATOR_TO_JUDGE_PROVIDER
    )
    if long["generator_judge_provider"].isna().any():
        missing = sorted(long.loc[long["generator_judge_provider"].isna(), "generator"].unique())
        raise AssertionError(f"Unmapped generator families: {missing}")
    judge_labels = (
        long[["judge_provider", "paper_model_label"]]
        .drop_duplicates()
        .sort_values("judge_provider")
    )
    if len(judge_labels) != 9 or judge_labels["judge_provider"].nunique() != 9:
        raise AssertionError("Judge provider-to-label mapping is not one-to-one")
    return long, judge_labels, {
        "api_test_scores_7290.csv": sha256_file(API_PATH),
        "api_stimulus_equivalence_810.csv": sha256_file(equivalence_path),
        "api_rows": int(len(api)),
        "source_uids": int(api["source_item_uid"].nunique()),
        "judge_providers": int(api["judge_provider"].nunique()),
    }


def build_method_scores(
    api_long: pd.DataFrame,
    judge_labels: pd.DataFrame,
    broad_signatures: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value = "api_score_quality_aligned"
    signature_judge = (
        api_long.groupby(
            ["stimulus_signature_sha256", "dimension", "judge_provider", "paper_model_label"],
            as_index=False,
            observed=True,
        )
        .agg(api_score=(value, "mean"), n_equivalent_source_uids=("source_item_uid", "nunique"))
    )
    counts = signature_judge.groupby(["stimulus_signature_sha256", "dimension"])[
        "judge_provider"
    ].nunique()
    if not counts.eq(9).all():
        raise AssertionError("Visible-text signature does not have nine judge-family values")

    method_frames: list[pd.DataFrame] = []
    method_rows: list[dict[str, Any]] = []

    def add_method(
        method_id: str,
        method_type: str,
        method_label: str,
        scores: pd.DataFrame,
        expected_judges: int,
        definition: str,
    ) -> None:
        frame = scores[["stimulus_signature_sha256", "dimension", "api_score"]].copy()
        frame = frame.loc[frame["stimulus_signature_sha256"].isin(broad_signatures)]
        if len(frame) != 1068 or frame.duplicated(
            ["stimulus_signature_sha256", "dimension"]
        ).any():
            raise AssertionError(f"{method_id} is not complete on the common 178x6 panel")
        frame["method_id"] = method_id
        frame["method_type"] = method_type
        frame["method_label"] = method_label
        method_frames.append(frame)
        method_rows.append(
            {
                "method_id": method_id,
                "method_type": method_type,
                "method_label": method_label,
                "expected_judges_per_source_uid": expected_judges,
                "definition": definition,
                "risk_direction": "quality_aligned_high_is_good; aligned risk = 6 - raw risk",
            }
        )

    for row in judge_labels.itertuples(index=False):
        single = signature_judge.loc[
            signature_judge["judge_provider"] == row.judge_provider,
            ["stimulus_signature_sha256", "dimension", "api_score"],
        ]
        add_method(
            f"single::{row.judge_provider}",
            "single_judge",
            str(row.paper_model_label),
            single,
            1,
            "one fixed judge family; equivalent source UIDs averaged within visible text",
        )

    mean_scores = (
        signature_judge.groupby(["stimulus_signature_sha256", "dimension"], as_index=False)
        .agg(api_score=("api_score", "mean"))
    )
    add_method(
        "ensemble::mean",
        "ensemble",
        "Nine-judge mean",
        mean_scores,
        9,
        "arithmetic mean of the nine judge-family values for each visible text",
    )
    median_scores = (
        signature_judge.groupby(["stimulus_signature_sha256", "dimension"], as_index=False)
        .agg(api_score=("api_score", "median"))
    )
    add_method(
        "ensemble::median",
        "ensemble",
        "Nine-judge median",
        median_scores,
        9,
        "median of the nine judge-family values for each visible text",
    )

    def trimmed(values: pd.Series) -> float:
        array = np.sort(values.to_numpy(float))
        if len(array) != 9:
            raise AssertionError("Trimmed mean requires nine judge-family values")
        return float(np.mean(array[1:-1]))

    trimmed_scores = (
        signature_judge.groupby(["stimulus_signature_sha256", "dimension"], as_index=False)
        .agg(api_score=("api_score", trimmed))
    )
    add_method(
        "ensemble::trimmed_one_each_tail",
        "ensemble",
        "Nine-judge trimmed mean",
        trimmed_scores,
        9,
        "drop one lowest and one highest judge-family value, then average the remaining seven",
    )

    no_self = api_long.loc[
        api_long["judge_provider"] != api_long["generator_judge_provider"]
    ].copy()
    no_self_coverage = no_self.groupby("source_item_uid")["judge_provider"].nunique()
    if not no_self_coverage.eq(8).all():
        raise AssertionError("No-self rule must retain eight judge families per generated source UID")
    no_self_scores = (
        no_self.groupby(["stimulus_signature_sha256", "dimension"], as_index=False)
        .agg(api_score=(value, "mean"))
    )
    add_method(
        "ensemble::no_self_mean",
        "ensemble",
        "No-self mean",
        no_self_scores,
        8,
        "for each generated source UID exclude its same-family judge, then pool the eight retained values by visible text",
    )
    methods = pd.concat(method_frames, ignore_index=True)
    metadata = pd.DataFrame(method_rows)
    if methods["method_id"].nunique() != 13 or len(methods) != 13 * 1068:
        raise AssertionError("Expected nine single judges plus four ensemble baselines")
    return methods, metadata


def merge_and_standardize(
    human: pd.DataFrame, methods: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = methods.merge(
        human,
        on=["stimulus_signature_sha256", "dimension"],
        how="inner",
        validate="many_to_one",
    )
    if len(data) != 13 * 1068:
        raise AssertionError("Methods do not share the exact common human sample")
    rows: list[dict[str, Any]] = []
    data["api_z"] = np.nan
    data["human_z"] = np.nan
    for (method_id, dimension), index in data.groupby(["method_id", "dimension"], sort=True).groups.items():
        api = data.loc[index, "api_score"].to_numpy(float)
        human_values = data.loc[index, "human_score"].to_numpy(float)
        api_mean = float(np.mean(api))
        api_sd = float(np.std(api, ddof=1))
        human_mean = float(np.mean(human_values))
        human_sd = float(np.std(human_values, ddof=1))
        if api_sd <= 0 or human_sd <= 0:
            raise AssertionError(f"Degenerate standardization: {(method_id, dimension)}")
        data.loc[index, "api_z"] = (api - api_mean) / api_sd
        data.loc[index, "human_z"] = (human_values - human_mean) / human_sd
        rows.append(
            {
                "method_id": method_id,
                "dimension": dimension,
                "n_rows": int(len(index)),
                "api_mean_quality_aligned": api_mean,
                "api_sample_sd": api_sd,
                "human_mean_quality_aligned": human_mean,
                "human_sample_sd": human_sd,
            }
        )
    constants = pd.DataFrame(rows)
    mean_constants = constants.loc[
        constants["method_id"] == "ensemble::mean",
        ["dimension", "api_mean_quality_aligned", "api_sample_sd"],
    ].rename(
        columns={
            "api_mean_quality_aligned": "mean_panel_api_mean",
            "api_sample_sd": "mean_panel_api_sd",
        }
    )
    data = data.merge(mean_constants, on="dimension", how="left", validate="many_to_one")
    data["api_z_on_primary_mean_scale"] = (
        data["api_score"] - data["mean_panel_api_mean"]
    ) / data["mean_panel_api_sd"]
    data["risk_direction"] = "quality_aligned_high_is_good; aligned risk = 6 - raw risk"
    return data, constants


def point_results(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method_id, group in data.groupby("method_id", sort=True):
        human = group["human_score"].to_numpy(float)
        api = group["api_score"].to_numpy(float)
        error = api - human
        rows.append(
            {
                "method_id": method_id,
                "n_rows": int(len(group)),
                "n_stimuli": int(group["stimulus_signature_sha256"].nunique()),
                "n_domain_concept_clusters": int(
                    group[["domain", "concept_id"]].drop_duplicates().shape[0]
                ),
                "controls": "+".join(CONTROLS),
                "standardized_beta": standardized_slope(group, "api_z"),
                "beta_on_primary_mean_frozen_scale": standardized_slope(
                    group, "api_z_on_primary_mean_scale"
                ),
                "lin_ccc_quality_aligned_raw_scale": concordance_correlation(human, api),
                "mae_quality_aligned_raw_scale": float(np.mean(np.abs(error))),
                "rmse_quality_aligned_raw_scale": float(np.sqrt(np.mean(error**2))),
                "spearman_rho_quality_aligned_raw_scale": safe_correlation(spearmanr, human, api),
                "pearson_r_quality_aligned_raw_scale": safe_correlation(pearsonr, human, api),
                "human_mean_quality_aligned": float(np.mean(human)),
                "api_mean_quality_aligned": float(np.mean(api)),
            }
        )
    return metadata.merge(pd.DataFrame(rows), on="method_id", validate="one_to_one")


def shared_cluster_bootstrap(
    data: pd.DataFrame, method_ids: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = data.loc[data["method_id"] == "ensemble::mean"].sort_values(
        ["stimulus_signature_sha256", "dimension"]
    )
    cluster_labels = (
        reference["domain"].astype(str) + "||" + reference["concept_id"].astype(str)
    ).to_numpy()
    clusters_by_domain: dict[str, list[str]] = {}
    for domain in sorted(reference["domain"].unique()):
        labels = sorted(set(cluster_labels[reference["domain"].to_numpy() == domain]))
        clusters_by_domain[str(domain)] = labels
    all_clusters = sorted(set(cluster_labels))
    cluster_index = {label: index for index, label in enumerate(all_clusters)}
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    counts = np.zeros((N_BOOTSTRAP, len(all_clusters)), dtype=float)
    for labels in clusters_by_domain.values():
        draw_counts = rng.multinomial(
            len(labels), np.repeat(1.0 / len(labels), len(labels)), size=N_BOOTSTRAP
        )
        counts[:, [cluster_index[label] for label in labels]] = draw_counts

    output: dict[str, np.ndarray] = {}
    invalid: dict[str, float] = {}
    for method_id in method_ids:
        subset = data.loc[data["method_id"] == method_id].sort_values(
            ["stimulus_signature_sha256", "dimension"]
        )
        labels = (
            subset["domain"].astype(str) + "||" + subset["concept_id"].astype(str)
        ).to_numpy()
        if not np.array_equal(labels, cluster_labels):
            raise AssertionError(f"Cluster order differs for {method_id}")
        z = control_matrix(subset)
        x = subset["api_z"].to_numpy(float)
        y = subset["human_z"].to_numpy(float)
        design = np.column_stack([z, x])
        p = design.shape[1]
        xtx_cluster = np.zeros((len(all_clusters), p, p), dtype=float)
        xty_cluster = np.zeros((len(all_clusters), p), dtype=float)
        for label in all_clusters:
            mask = labels == label
            position = cluster_index[label]
            local_x = design[mask]
            xtx_cluster[position] = local_x.T @ local_x
            xty_cluster[position] = local_x.T @ y[mask]
        xtx = np.einsum("bc,cij->bij", counts, xtx_cluster, optimize=True)
        xty = np.einsum("bc,ci->bi", counts, xty_cluster, optimize=True)
        ranks = np.linalg.matrix_rank(xtx)
        beta = np.einsum(
            "bij,bj->bi", np.linalg.pinv(xtx, rcond=1e-12), xty, optimize=True
        )[:, -1]
        beta[ranks < p] = np.nan
        output[method_id] = beta
        invalid[method_id] = float(np.mean(~np.isfinite(beta)))
    return output, {
        "method": "domain-stratified nonparametric domain-by-concept cluster bootstrap",
        "seed": SEED_BOOTSTRAP,
        "draws": N_BOOTSTRAP,
        "clusters_total": len(all_clusters),
        "clusters_by_domain": {key: len(value) for key, value in clusters_by_domain.items()},
        "invalid_proportion_by_method": invalid,
        "standardization": "method-by-dimension constants frozen from original common sample",
    }


def shared_freedman_lane(
    data: pd.DataFrame, method_ids: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = data.loc[data["method_id"] == "ensemble::mean"].sort_values(
        ["stimulus_signature_sha256", "dimension"]
    ).reset_index(drop=True)
    dimensions = sorted(DIMENSIONS)
    signatures = reference["stimulus_signature_sha256"].drop_duplicates().tolist()
    signature_index = {value: index for index, value in enumerate(signatures)}
    dimension_index = {value: index for index, value in enumerate(dimensions)}
    if len(reference) != len(signatures) * len(dimensions):
        raise AssertionError("Freedman--Lane canonical layout is incomplete")
    z = control_matrix(reference)
    y = reference["human_z"].to_numpy(float)
    fitted = z @ np.linalg.lstsq(z, y, rcond=None)[0]
    residual = y - fitted
    target_stimulus = reference["stimulus_signature_sha256"].map(signature_index).to_numpy(int)
    target_dimension = reference["dimension"].map(dimension_index).to_numpy(int)
    residual_lookup = np.full((len(signatures), len(dimensions)), np.nan)
    residual_lookup[target_stimulus, target_dimension] = residual
    if np.isnan(residual_lookup).any():
        raise AssertionError("Freedman--Lane residual lookup is incomplete")

    stimulus_meta = reference.drop_duplicates("stimulus_signature_sha256")
    strata = (
        stimulus_meta["domain"].astype(str) + "||" + stimulus_meta["task"].astype(str)
    ).to_numpy()
    stratum_levels = sorted(set(strata))
    stratum_positions = {
        level: np.flatnonzero(strata == level) for level in stratum_levels
    }
    rng = np.random.default_rng(SEED_PERMUTATION)
    source_maps = np.broadcast_to(
        np.arange(len(signatures), dtype=np.int16),
        (N_PERMUTATION, len(signatures)),
    ).copy()
    for draw in range(N_PERMUTATION):
        for positions in stratum_positions.values():
            source_maps[draw, positions] = rng.permutation(positions)

    nulls: dict[str, np.ndarray] = {}
    identity_differences: dict[str, float] = {}
    for method_id in method_ids:
        subset = data.loc[data["method_id"] == method_id].sort_values(
            ["stimulus_signature_sha256", "dimension"]
        ).reset_index(drop=True)
        if not np.array_equal(
            subset[["stimulus_signature_sha256", "dimension"]].to_numpy(),
            reference[["stimulus_signature_sha256", "dimension"]].to_numpy(),
        ):
            raise AssertionError(f"Freedman--Lane row layout differs for {method_id}")
        x = subset["api_z"].to_numpy(float)
        x_residual = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
        denominator = float(x_residual @ x_residual)
        observed_formula = float((y @ x_residual) / denominator)
        observed_ols = standardized_slope(subset)
        identity_differences[method_id] = observed_formula - observed_ols
        if not np.isclose(observed_formula, observed_ols, atol=1e-12, rtol=1e-12):
            raise AssertionError(f"Freedman--Lane identity mismatch for {method_id}")
        null = np.empty(N_PERMUTATION, dtype=float)
        for start in range(0, N_PERMUTATION, 250):
            stop = min(start + 250, N_PERMUTATION)
            maps = source_maps[start:stop]
            residual_permuted = residual_lookup[
                maps[:, target_stimulus], target_dimension[None, :]
            ]
            y_star = fitted[None, :] + residual_permuted
            null[start:stop] = (y_star @ x_residual) / denominator
        nulls[method_id] = null
    return nulls, {
        "method": "Freedman--Lane residual permutation",
        "seed": SEED_PERMUTATION,
        "draws": N_PERMUTATION,
        "permutation_strata": "domain-by-task",
        "n_strata": len(stratum_positions),
        "stratum_sizes": {
            level: int(len(positions)) for level, positions in stratum_positions.items()
        },
        "joint_six_dimension_movement": True,
        "shared_stimulus_maps_across_all_13_methods": True,
        "max_abs_identity_difference": float(
            max(abs(value) for value in identity_differences.values())
        ),
    }


def attach_inference(
    points: pd.DataFrame,
    bootstrap: dict[str, np.ndarray],
    permutation: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = points.copy()
    result["bootstrap_ci_low"] = np.nan
    result["bootstrap_ci_high"] = np.nan
    result["bootstrap_se"] = np.nan
    result["bootstrap_valid_draws"] = 0
    result["freedman_lane_p_two_sided"] = np.nan
    for index, row in result.iterrows():
        method_id = row["method_id"]
        draws = bootstrap[method_id]
        valid = draws[np.isfinite(draws)]
        result.loc[index, "bootstrap_ci_low"] = float(np.quantile(valid, 0.025))
        result.loc[index, "bootstrap_ci_high"] = float(np.quantile(valid, 0.975))
        result.loc[index, "bootstrap_se"] = float(np.std(valid, ddof=1))
        result.loc[index, "bootstrap_valid_draws"] = int(len(valid))
        null = permutation[method_id]
        observed = float(row["standardized_beta"])
        result.loc[index, "freedman_lane_p_two_sided"] = float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
        )
    result["freedman_lane_p_holm_13"] = multipletests(
        result["freedman_lane_p_two_sided"].to_numpy(float), method="holm"
    )[1]
    result["inference_scope"] = (
        "conditional association in the fixed broad-review panel; concept-cluster uncertainty"
    )
    return result


def exploration_validation(results: pd.DataFrame) -> dict[str, Any]:
    by_id = results.set_index("method_id")
    singles = results.loc[results["method_type"] == "single_judge", "standardized_beta"]
    observed = {
        "single_min": float(singles.min()),
        "single_max": float(singles.max()),
        "mean": float(by_id.loc["ensemble::mean", "standardized_beta"]),
        "median_method_standardized": float(
            by_id.loc["ensemble::median", "standardized_beta"]
        ),
        "median_primary_mean_frozen_scale": float(
            by_id.loc["ensemble::median", "beta_on_primary_mean_frozen_scale"]
        ),
        "trimmed": float(
            by_id.loc["ensemble::trimmed_one_each_tail", "standardized_beta"]
        ),
        "no_self": float(by_id.loc["ensemble::no_self_mean", "standardized_beta"]),
    }
    checks = {
        "single_range_matches_0.098_to_0.213": bool(
            abs(observed["single_min"] - 0.09843468598300924) < 1e-10
            and abs(observed["single_max"] - 0.2130349675568411) < 1e-10
        ),
        "mean_matches_0.232": bool(abs(observed["mean"] - 0.23156385314634806) < 1e-10),
        "historical_median_0.154_is_frozen_mean_scale": bool(
            abs(observed["median_primary_mean_frozen_scale"] - 0.1544609733965928)
            < 1e-10
        ),
        "trimmed_matches_0.233": bool(
            abs(observed["trimmed"] - 0.23339139640111323) < 1e-10
        ),
        "no_self_matches_0.210": bool(
            abs(observed["no_self"] - 0.2103304248947074) < 1e-10
        ),
    }
    return {
        "observed": observed,
        "checks": checks,
        "median_scale_note": (
            "The previously quoted median beta near .154 is exactly reproduced only when the "
            "median is placed on the primary mean panel's frozen API SD. Under the unified "
            "method-specific standardization used for formal comparison, median beta is about .197. "
            "Both values are reported; only the method-standardized beta receives the common CI/p workflow."
        ),
        "all_checks_pass": bool(all(checks.values())),
    }


def paired_beta_differences(
    results: pd.DataFrame,
    bootstrap: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Exploratory paired differences using the same cluster draws for both methods."""
    reference_id = "ensemble::mean"
    lookup = results.set_index("method_id")
    rows: list[dict[str, Any]] = []
    for comparator_id in results["method_id"]:
        if comparator_id == reference_id:
            continue
        paired = bootstrap[reference_id] - bootstrap[comparator_id]
        valid = paired[np.isfinite(paired)]
        ci_low = float(np.quantile(valid, 0.025))
        ci_high = float(np.quantile(valid, 0.975))
        difference = float(
            lookup.loc[reference_id, "standardized_beta"]
            - lookup.loc[comparator_id, "standardized_beta"]
        )
        rows.append(
            {
                "reference_method_id": reference_id,
                "reference_method_label": lookup.loc[reference_id, "method_label"],
                "comparator_method_id": comparator_id,
                "comparator_method_label": lookup.loc[comparator_id, "method_label"],
                "beta_difference_mean_minus_comparator": difference,
                "paired_cluster_bootstrap_ci_low": ci_low,
                "paired_cluster_bootstrap_ci_high": ci_high,
                "paired_cluster_bootstrap_se": float(np.std(valid, ddof=1)),
                "paired_cluster_bootstrap_valid_draws": int(len(valid)),
                "ci_crosses_zero": bool(ci_low <= 0.0 <= ci_high),
                "direction_at_point": (
                    "mean_higher" if difference > 0 else "mean_lower" if difference < 0 else "equal"
                ),
                "inference_role": (
                    "exploratory paired comparison; same domain-stratified concept-cluster draws "
                    "used for both beta estimates"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_report(
    path: Path,
    results: pd.DataFrame,
    paired: pd.DataFrame,
    validation: dict[str, Any],
    family_mapping: pd.DataFrame,
) -> None:
    ordered = pd.concat(
        [
            results.loc[results["method_type"] == "single_judge"].sort_values("method_label"),
            results.loc[results["method_type"] == "ensemble"].set_index("method_id").loc[
                [
                    "ensemble::mean",
                    "ensemble::median",
                    "ensemble::trimmed_one_each_tail",
                    "ensemble::no_self_mean",
                ]
            ].reset_index(),
        ],
        ignore_index=True,
    )
    lines = [
        "# Aggregation baselines against the broad human criterion",
        "",
        "## Common analysis contract",
        "",
        (
            "All 13 methods use the same 178 visible texts (1,068 text-by-dimension rows), "
            "30 domain-concept clusters, controls for domain, task, and dimension, 5,000 "
            "domain-stratified concept-cluster bootstrap draws, and 10,000 shared "
            "domain-by-task Freedman--Lane permutations. No method-specific row deletion occurs."
        ),
        "",
        (
            "Risk is quality-aligned for both humans and APIs as 6 - raw risk. Consequently, "
            "higher scores are better on all six dimensions. CCC and MAE are calculated on "
            "the original aligned 1--5 scale; beta uses method-by-dimension z scores whose "
            "constants are frozen before resampling."
        ),
        "",
        "## Complete results",
        "",
        "| Method | Beta | 95% cluster CI | FL p | Holm p | Lin CCC | MAE | Spearman | Beta on old mean scale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['method_label']} | {row['standardized_beta']:.3f} | "
            f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}] | "
            f"{row['freedman_lane_p_two_sided']:.4f} | "
            f"{row['freedman_lane_p_holm_13']:.4f} | "
            f"{row['lin_ccc_quality_aligned_raw_scale']:.3f} | "
            f"{row['mae_quality_aligned_raw_scale']:.3f} | "
            f"{row['spearman_rho_quality_aligned_raw_scale']:.3f} | "
            f"{row['beta_on_primary_mean_frozen_scale']:.3f} |"
        )
    single = results.loc[results["method_type"] == "single_judge"]
    mean = results.set_index("method_id").loc["ensemble::mean"]
    median = results.set_index("method_id").loc["ensemble::median"]
    trimmed = results.set_index("method_id").loc["ensemble::trimmed_one_each_tail"]
    no_self = results.set_index("method_id").loc["ensemble::no_self_mean"]
    lines.extend(
        [
            "",
            "## Audit of the earlier exploratory values",
            "",
            (
                f"The unified method-standardized analysis reproduces the single-judge range "
                f"({single['standardized_beta'].min():.3f}--{single['standardized_beta'].max():.3f}), "
                f"mean ({mean['standardized_beta']:.3f}), trimmed mean "
                f"({trimmed['standardized_beta']:.3f}), and no-self mean "
                f"({no_self['standardized_beta']:.3f})."
            ),
            "",
            (
                f"The earlier median value near .154 is also reproducible, but it is a different "
                f"scale: beta={median['beta_on_primary_mean_frozen_scale']:.3f} when the median "
                f"uses the mean panel's frozen API SD. With the same method-specific standardization "
                f"used for every formal baseline, the median beta is {median['standardized_beta']:.3f}. "
                "The latter is the comparable primary value; the former is retained only for provenance."
            ),
            "",
            "## Exploratory paired beta differences",
            "",
            (
                "Each interval below is calculated from the draw-by-draw difference between the mean "
                "panel beta and the comparator beta under the same 5,000 concept-cluster resamples. "
                "These comparisons are exploratory and were not multiplicity-adjusted."
            ),
            "",
            "| Comparator | Mean - comparator beta | Paired 95% CI | Crosses zero |",
            "|---|---:|---:|:---:|",
        ]
    )
    paired_ordered = paired.merge(
        ordered[["method_id"]].reset_index().rename(
            columns={"index": "display_order", "method_id": "comparator_method_id"}
        ),
        on="comparator_method_id",
        how="left",
        validate="one_to_one",
    ).sort_values("display_order")
    for _, row in paired_ordered.iterrows():
        lines.append(
            f"| {row['comparator_method_label']} | "
            f"{row['beta_difference_mean_minus_comparator']:.3f} | "
            f"[{row['paired_cluster_bootstrap_ci_low']:.3f}, "
            f"{row['paired_cluster_bootstrap_ci_high']:.3f}] | "
            f"{'yes' if row['ci_crosses_zero'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## No-self family mapping",
            "",
            "| Generator family | Excluded judge provider |",
            "|---|---|",
        ]
    )
    for _, row in family_mapping.sort_values("generator_family").iterrows():
        lines.append(f"| {row['generator_family']} | {row['judge_provider']} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "These are correlated baseline estimates on the same broad-review criterion. "
                "A larger point beta does not by itself prove one aggregation is superior to another; "
                "the paired intervals above are exploratory, unadjusted for multiplicity, and do not define "
                "a confirmatory superiority test. CCC remains modest because association and raw-score "
                "agreement answer different questions."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    human, human_meta = load_human_criterion()
    api_long, judge_labels, api_meta = load_api_long()
    methods, method_metadata = build_method_scores(
        api_long,
        judge_labels,
        set(human["stimulus_signature_sha256"]),
    )
    data, constants = merge_and_standardize(human, methods)
    method_ids = method_metadata["method_id"].tolist()
    points = point_results(data, method_metadata)
    bootstrap, bootstrap_meta = shared_cluster_bootstrap(data, method_ids)
    permutation, permutation_meta = shared_freedman_lane(data, method_ids)
    results = attach_inference(points, bootstrap, permutation)
    paired = paired_beta_differences(results, bootstrap)
    validation = exploration_validation(results)
    if not validation["all_checks_pass"]:
        raise AssertionError(f"Exploratory-value validation failed: {validation}")

    family_mapping = pd.DataFrame(
        [
            {"generator_family": generator, "judge_provider": provider}
            for generator, provider in GENERATOR_TO_JUDGE_PROVIDER.items()
        ]
    )
    release_data = data[
        [
            "method_id",
            "method_type",
            "method_label",
            "stimulus_signature_sha256",
            "domain",
            "task",
            "concept_id",
            "concept",
            "dimension",
            "human_score",
            "api_score",
            "human_z",
            "api_z",
            "api_z_on_primary_mean_scale",
            "risk_direction",
        ]
    ].sort_values(["method_id", "stimulus_signature_sha256", "dimension"])

    outputs = {
        "aggregation_human_common_data.csv": release_data,
        "aggregation_human_results.csv": results,
        "aggregation_paired_beta_differences.csv": paired,
        "aggregation_standardization_constants.csv": constants,
        "generator_judge_family_mapping.csv": family_mapping,
    }
    for filename, frame in outputs.items():
        write_csv(BASE / filename, frame)
    summary = {
        "analysis_version": SCRIPT_VERSION,
        "analysis_unit": "visible-text signature by dimension",
        "risk_direction": "quality-aligned high-is-good; both API and human risk are 6 - raw risk",
        "common_sample": {
            "stimuli": 178,
            "dimensions": 6,
            "rows_per_method": 1068,
            "domain_concept_clusters": 30,
            "methods": 13,
        },
        "methods": results.to_dict("records"),
        "exploratory_paired_beta_differences": paired.to_dict("records"),
        "exploratory_value_validation": validation,
        "bootstrap": bootstrap_meta,
        "permutation": permutation_meta,
    }
    write_json(BASE / "aggregation_human_results.json", summary)
    make_report(
        BASE / "AGGREGATION_HUMAN_BASELINES_REPORT.md",
        results,
        paired,
        validation,
        family_mapping,
    )

    output_hashes = {
        filename: sha256_file(BASE / filename)
        for filename in [
            *outputs.keys(),
            "aggregation_human_results.json",
            "AGGREGATION_HUMAN_BASELINES_REPORT.md",
        ]
    }
    manifest = {
        "analysis_version": SCRIPT_VERSION,
        "script": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        "inputs": {**human_meta, **api_meta},
        "methods": method_metadata.to_dict("records"),
        "bootstrap": bootstrap_meta,
        "permutation": permutation_meta,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "outputs": output_hashes,
        "qa": {
            "common_rows_per_method": bool(
                release_data.groupby("method_id").size().eq(1068).all()
            ),
            "common_signature_set": bool(
                release_data.groupby("method_id")["stimulus_signature_sha256"].nunique().eq(178).all()
            ),
            "all_bootstrap_draws_reportable": bool(
                (results["bootstrap_valid_draws"] >= 0.99 * N_BOOTSTRAP).all()
            ),
            "freedman_lane_identity_pass": bool(
                permutation_meta["max_abs_identity_difference"] < 1e-12
            ),
            "exploratory_values_validated": bool(validation["all_checks_pass"]),
            "no_secrets_or_credentials_read": True,
        },
        "secret_handling": "No API credentials are read or written by this analysis.",
    }
    write_json(BASE / "aggregation_human_manifest.json", manifest)

    forbidden = ["sk-", "aiza", "aq.", "api_key", "authorization: bearer"]
    scan_files = [
        *outputs.keys(),
        "aggregation_human_results.json",
        "AGGREGATION_HUMAN_BASELINES_REPORT.md",
        "aggregation_human_manifest.json",
    ]
    for filename in scan_files:
        text = (BASE / filename).read_text(encoding="utf-8-sig", errors="ignore").lower()
        if any(fragment in text for fragment in forbidden):
            raise AssertionError(f"Credential-like fragment detected in {filename}")

    print(
        json.dumps(
            {
                "qa": manifest["qa"],
                "headline": results[
                    [
                        "method_id",
                        "standardized_beta",
                        "bootstrap_ci_low",
                        "bootstrap_ci_high",
                        "freedman_lane_p_two_sided",
                        "lin_ccc_quality_aligned_raw_scale",
                        "mae_quality_aligned_raw_scale",
                        "spearman_rho_quality_aligned_raw_scale",
                    ]
                ].to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
