#!/usr/bin/env python3
"""Analyze three fresh API calls without importing historical scores.

The script is intentionally resumable.  Partial provider files yield an
explicit interim missingness audit; the same script becomes the final
nine-provider analysis as soon as every planned opaque call has one valid row.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


SCRIPT_VERSION = "1.0.3"
SEED = 2026071705
N_BOOTSTRAP = 2_000
DIMENSIONS = ["fa", "cc", "lc", "tf", "mq", "risk"]
REPLICATES = [1, 2, 3]
BASE = Path(__file__).resolve().parent

GENERATOR_TO_PROVIDER = {
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
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_snapshot(path: Path) -> pd.DataFrame:
    """Read one byte snapshot so an actively appended file cannot mix epochs."""
    payload = path.read_bytes()
    try:
        return pd.read_csv(io.BytesIO(payload))
    except pd.errors.ParserError:
        # An unfinished trailing row is never accepted as a completed call.
        last_newline = payload.rfind(b"\n")
        if last_newline <= 0:
            raise
        return pd.read_csv(io.BytesIO(payload[: last_newline + 1]))


def parse_success(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def score_rows_valid(frame: pd.DataFrame) -> pd.Series:
    parsed = parse_success(frame["parse_success"])
    numeric = frame[DIMENSIONS].apply(pd.to_numeric, errors="coerce")
    in_range = numeric.apply(lambda column: column.between(1, 5)).all(axis=1)
    if "http_status" in frame.columns:
        http_status = pd.to_numeric(frame["http_status"], errors="coerce")
        http_ok = http_status.between(200, 299)
    else:
        http_ok = pd.Series(True, index=frame.index)
    return parsed & in_range & http_ok


def load_planned_and_observed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mapping = pd.read_csv(BASE / "mapping_270.csv")
    grid = pd.read_csv(BASE / "judge_repetition_grid.csv")
    sample = pd.read_csv(BASE / "repeat_sample_90.csv")
    if len(mapping) != 270 or mapping["item_id"].nunique() != 270:
        raise AssertionError("mapping_270.csv must contain 270 unique opaque call IDs")
    if mapping["source_stimulus_id"].nunique() != 90:
        raise AssertionError("Expected 90 source stimuli")
    if not mapping.groupby("source_stimulus_id")["replicate_index"].apply(
        lambda values: set(values) == set(REPLICATES)
    ).all():
        raise AssertionError("Every stimulus must have fresh repetitions 1, 2, and 3")
    providers = sorted(grid["judge_provider"].unique().tolist())
    if len(providers) != 9:
        raise AssertionError("judge_repetition_grid.csv must specify nine judges")
    if len(sample) != 90 or sample["repeat_item_id"].nunique() != 90:
        raise AssertionError("repeat_sample_90.csv must contain 90 unique stimuli")
    if set(mapping["source_stimulus_id"]) != set(sample["repeat_item_id"]):
        raise AssertionError("Sample and repetition mapping identities differ")
    deviation_path = BASE / "provider_retry_deviations.json"
    if deviation_path.exists():
        deviation_payload = json.loads(deviation_path.read_text(encoding="utf-8"))
        deviations_by_provider = {
            row["provider"]: row for row in deviation_payload.get("deviations", [])
        }
    else:
        deviation_payload = {"deviations": []}
        deviations_by_provider = {}

    planned = pd.MultiIndex.from_product(
        [providers, mapping["item_id"].tolist()], names=["judge_provider", "item_id"]
    ).to_frame(index=False)
    planned = planned.merge(mapping, on="item_id", how="left", validate="many_to_one")
    planned = planned.merge(
        sample[["repeat_item_id", "concept", "generator_model"]],
        left_on="source_stimulus_id",
        right_on="repeat_item_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["repeat_item_id"])

    selected_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for provider in providers:
        path = BASE / "runs" / provider / "api_judge_scores_long.csv"
        run_manifest_path = BASE / "runs" / provider / "run_manifest.json"
        run_manifest = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.exists()
            else {}
        )
        deviation = deviations_by_provider.get(provider)
        deviation_note = (
            f"{deviation['scope']}: {deviation['trigger']}; "
            f"retry cap {deviation['changed_for_parse_retry_only']['provider_max_output_tokens']['from']}"
            f"->{deviation['changed_for_parse_retry_only']['provider_max_output_tokens']['to']}"
            if deviation
            else ""
        )
        if not path.exists():
            audit_rows.append(
                {
                    "judge_provider": provider,
                    "observed_file_rows": 0,
                    "observed_planned_rows": 0,
                    "unplanned_rows": 0,
                    "invalid_audit_rows": 0,
                    "transport_or_http_failure_rows": 0,
                    "schema_or_parse_failure_rows": 0,
                    "out_of_range_or_missing_score_rows": 0,
                    "duplicate_audit_rows": 0,
                    "run_manifest_status": run_manifest.get("status", "missing"),
                    "provider_max_tokens_override": run_manifest.get("provider_max_tokens_override"),
                    "provider_specific_retry_deviation": deviation_note,
                }
            )
            continue
        raw = read_csv_snapshot(path)
        required = {"item_id", "judge_provider", "parse_success", *DIMENSIONS}
        if not required.issubset(raw.columns):
            raise AssertionError(f"{provider}: missing columns {sorted(required - set(raw.columns))}")
        raw = raw.copy()
        raw["source_file_row"] = np.arange(1, len(raw) + 1)
        parsed_mask = parse_success(raw["parse_success"])
        numeric_scores = raw[DIMENSIONS].apply(pd.to_numeric, errors="coerce")
        range_mask = numeric_scores.apply(lambda column: column.between(1, 5)).all(axis=1)
        if "http_status" in raw.columns:
            http_mask = pd.to_numeric(raw["http_status"], errors="coerce").between(200, 299)
        else:
            http_mask = pd.Series(True, index=raw.index)
        raw["_valid"] = score_rows_valid(raw)
        raw["_planned"] = raw["item_id"].isin(set(mapping["item_id"]))
        raw["_provider_matches"] = raw["judge_provider"].eq(provider)
        eligible = raw.loc[raw["_planned"] & raw["_provider_matches"]].copy()
        # The runner may retain a failed audit row and then append one valid retry.
        # Select the first valid row per planned call, never an invalid predecessor.
        valid = eligible.loc[eligible["_valid"]].sort_values("source_file_row")
        chosen = valid.drop_duplicates("item_id", keep="first").copy()
        selected_rows.append(chosen)
        audit_rows.append(
            {
                "judge_provider": provider,
                "observed_file_rows": int(len(raw)),
                "observed_planned_rows": int(len(eligible)),
                "unplanned_rows": int((~raw["_planned"]).sum()),
                "provider_label_mismatch_rows": int((~raw["_provider_matches"]).sum()),
                "invalid_audit_rows": int((~eligible["_valid"]).sum()),
                "transport_or_http_failure_rows": int((~http_mask.loc[eligible.index]).sum()),
                "schema_or_parse_failure_rows": int(
                    (http_mask.loc[eligible.index] & ~parsed_mask.loc[eligible.index]).sum()
                ),
                "out_of_range_or_missing_score_rows": int(
                    (parsed_mask.loc[eligible.index] & ~range_mask.loc[eligible.index]).sum()
                ),
                "duplicate_audit_rows": int(len(eligible) - eligible["item_id"].nunique()),
                "valid_unique_rows_selected": int(len(chosen)),
                "run_manifest_status": run_manifest.get("status", "unknown"),
                "provider_max_tokens_override": run_manifest.get("provider_max_tokens_override"),
                "provider_specific_retry_deviation": deviation_note,
            }
        )
        source_records.append(
            {
                "judge_provider": provider,
                "path": str(path.relative_to(BASE)),
                "sha256": sha256_file(path),
                "snapshot_rows": int(len(raw)),
                "run_manifest_path": (
                    str(run_manifest_path.relative_to(BASE)) if run_manifest_path.exists() else None
                ),
                "run_manifest_sha256": (
                    sha256_file(run_manifest_path) if run_manifest_path.exists() else None
                ),
            }
        )
    observed = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if not observed.empty and observed.duplicated(["judge_provider", "item_id"]).any():
        raise AssertionError("Selected observed calls are not unique by provider and opaque item ID")
    return planned, observed, pd.DataFrame(audit_rows), {
        "input_sources": {
            "mapping_270.csv": sha256_file(BASE / "mapping_270.csv"),
            "judge_repetition_grid.csv": sha256_file(BASE / "judge_repetition_grid.csv"),
            "repeat_sample_90.csv": sha256_file(BASE / "repeat_sample_90.csv"),
        },
        "run_snapshots": source_records,
        "provider_retry_deviations": {
            "path": "provider_retry_deviations.json" if deviation_path.exists() else None,
            "sha256": sha256_file(deviation_path) if deviation_path.exists() else None,
            "records": deviation_payload.get("deviations", []),
        },
    }


def construct_call_status(
    planned: pd.DataFrame, observed: pd.DataFrame, file_audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        "judge_provider",
        "item_id",
        "paper_model_label",
        "judge_model_requested",
        "actual_api_model",
        "judge_model_returned",
        "judge_model",
        *DIMENSIONS,
        "parse_success",
        "retry_count",
        "http_status",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "system_fingerprint_or_backend_id",
    ]
    if observed.empty:
        observed_selected = pd.DataFrame(columns=metadata_columns)
    else:
        observed_selected = observed[[column for column in metadata_columns if column in observed.columns]].copy()
    status = planned.merge(
        observed_selected,
        on=["judge_provider", "item_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    status["call_status"] = np.where(status["_merge"].eq("both"), "valid", "missing")
    status = status.drop(columns=["_merge"])
    for dimension in DIMENSIONS:
        status[dimension] = pd.to_numeric(status[dimension], errors="coerce")
    valid = status.loc[status["call_status"] == "valid"].copy()

    rows: list[dict[str, Any]] = []
    for provider, group in status.groupby("judge_provider", observed=True, sort=True):
        valid_group = group.loc[group["call_status"] == "valid"]
        replicate_counts = valid_group.groupby("source_stimulus_id")["replicate_index"].nunique()
        audit = file_audit.loc[file_audit["judge_provider"] == provider]
        audit_record = audit.iloc[0].to_dict() if len(audit) else {}
        rows.append(
            {
                "judge_provider": provider,
                "planned_rows": int(len(group)),
                "valid_unique_rows": int(len(valid_group)),
                "missing_planned_rows": int((group["call_status"] == "missing").sum()),
                "complete_stimuli_three_reps": int((replicate_counts == 3).sum()),
                "provider_complete": bool(len(valid_group) == 270 and (replicate_counts == 3).sum() == 90),
                **{key: value for key, value in audit_record.items() if key != "judge_provider"},
                "retry_rows_selected": int(
                    (pd.to_numeric(valid_group.get("retry_count", 0), errors="coerce").fillna(0) > 0).sum()
                ),
                "requested_models": "|".join(
                    sorted(valid_group.get("judge_model_requested", pd.Series(dtype=str)).dropna().astype(str).unique())
                ),
                "returned_models": "|".join(
                    sorted(valid_group.get("judge_model_returned", pd.Series(dtype=str)).dropna().astype(str).unique())
                ),
            }
        )
    completion = pd.DataFrame(rows)
    return status, valid, completion


def icc_a1(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or np.isnan(values).any():
        return float("nan")
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator != 0 else float("nan")


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    covariance = np.cov(x, y, ddof=1)[0, 1]
    denominator = var_x + var_y + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


def quadratic_weighted_kappa(x: np.ndarray, y: np.ndarray, categories: int = 5) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    observed = np.zeros((categories, categories), dtype=float)
    for left, right in zip(x, y):
        if 1 <= left <= categories and 1 <= right <= categories:
            observed[left - 1, right - 1] += 1
    if observed.sum() == 0:
        return float("nan")
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    coordinates = np.arange(categories, dtype=float)
    weights = ((coordinates[:, None] - coordinates[None, :]) / (categories - 1)) ** 2
    expected_disagreement = float(np.sum(weights * expected))
    return (
        float(1.0 - np.sum(weights * observed) / expected_disagreement)
        if expected_disagreement > 0
        else float("nan")
    )


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def bootstrap_icc(
    matrix: np.ndarray,
    concepts: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(matrix)
    stimulus_values = np.empty(N_BOOTSTRAP, dtype=float)
    for draw in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        stimulus_values[draw] = icc_a1(matrix[indices])
    unique_concepts = np.unique(concepts)
    concept_rows = {concept: np.flatnonzero(concepts == concept) for concept in unique_concepts}
    concept_values = np.empty(N_BOOTSTRAP, dtype=float)
    for draw in range(N_BOOTSTRAP):
        sampled = rng.choice(unique_concepts, size=len(unique_concepts), replace=True)
        indices = np.concatenate([concept_rows[concept] for concept in sampled])
        concept_values[draw] = icc_a1(matrix[indices])
    stimulus_valid = stimulus_values[np.isfinite(stimulus_values)]
    concept_valid = concept_values[np.isfinite(concept_values)]
    return {
        "icc_stimulus_bootstrap_ci_low": float(np.quantile(stimulus_valid, 0.025)),
        "icc_stimulus_bootstrap_ci_high": float(np.quantile(stimulus_valid, 0.975)),
        "icc_stimulus_bootstrap_valid_draws": int(len(stimulus_valid)),
        "icc_concept_cluster_bootstrap_ci_low": float(np.quantile(concept_valid, 0.025)),
        "icc_concept_cluster_bootstrap_ci_high": float(np.quantile(concept_valid, 0.975)),
        "icc_concept_cluster_bootstrap_valid_draws": int(len(concept_valid)),
    }


def pairwise_rows(
    matrix: np.ndarray,
    identity: dict[str, Any],
    include_kappa: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        x, y = matrix[:, left], matrix[:, right]
        difference = x - y
        row = {
            **identity,
            "replicate_pair": f"{left + 1}-{right + 1}",
            "n_stimuli": int(len(x)),
            "exact_agreement": float(np.mean(x == y)),
            "within_one_category": float(np.mean(np.abs(difference) <= 1)),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "spearman_rho": safe_spearman(x, y),
            "lin_ccc": lin_ccc(x, y),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(x, y) if include_kappa else np.nan,
        }
        rows.append(row)
    return rows


def summarize_matrix(
    matrix: np.ndarray,
    concepts: np.ndarray,
    identity: dict[str, Any],
    rng: np.random.Generator,
    include_kappa: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = pairwise_rows(matrix, identity, include_kappa)
    pair_frame = pd.DataFrame(pairs)
    item_means = matrix.mean(axis=1, keepdims=True)
    row = {
        **identity,
        "n_complete_stimuli": int(len(matrix)),
        "all_three_exact_agreement": float(np.mean(np.ptp(matrix, axis=1) == 0)),
        "mean_within_item_sd": float(np.mean(np.std(matrix, axis=1, ddof=1))),
        "median_within_item_sd": float(np.median(np.std(matrix, axis=1, ddof=1))),
        "mean_within_item_mad_from_item_mean": float(np.mean(np.abs(matrix - item_means))),
        "mean_pairwise_exact_agreement": float(pair_frame["exact_agreement"].mean()),
        "mean_pairwise_within_one_category": float(pair_frame["within_one_category"].mean()),
        "mean_absolute_pairwise_difference": float(pair_frame["mean_absolute_difference"].mean()),
        "mean_pairwise_rmse": float(pair_frame["rmse"].mean()),
        "mean_pairwise_spearman": float(pair_frame["spearman_rho"].mean()),
        "min_pairwise_spearman": float(pair_frame["spearman_rho"].min()),
        "mean_pairwise_lin_ccc": float(pair_frame["lin_ccc"].mean()),
        "min_pairwise_lin_ccc": float(pair_frame["lin_ccc"].min()),
        "mean_pairwise_quadratic_weighted_kappa": (
            float(pair_frame["quadratic_weighted_kappa"].mean()) if include_kappa else np.nan
        ),
        "icc_a1_absolute_single_measure": icc_a1(matrix),
        **bootstrap_icc(matrix, concepts, rng),
    }
    return row, pairs


def provider_repeatability(
    valid: pd.DataFrame, providers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    pair_rows_all: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for provider in providers:
        provider_frame = valid.loc[valid["judge_provider"] == provider]
        for dimension in DIMENSIONS:
            pivot = provider_frame.pivot_table(
                index=["source_stimulus_id", "concept_id"],
                columns="replicate_index",
                values=dimension,
                aggfunc="first",
            ).reindex(columns=REPLICATES)
            complete = pivot.dropna()
            identity = {"judge_provider": provider, "dimension": dimension}
            if len(complete) < 2:
                summary_rows.append(
                    {
                        **identity,
                        "status": "insufficient_complete_stimuli",
                        "n_complete_stimuli": int(len(complete)),
                    }
                )
                continue
            matrix = complete.to_numpy(float)
            concepts = complete.index.get_level_values("concept_id").to_numpy()
            row, pairs = summarize_matrix(matrix, concepts, identity, rng, include_kappa=True)
            row["status"] = "complete_provider" if len(complete) == 90 else "interim_partial_provider"
            summary_rows.append(row)
            pair_rows_all.extend(pairs)
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows_all)


def build_panel_cells(valid: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    frame = valid.copy()
    frame["self_provider"] = frame["generator"].map(GENERATOR_TO_PROVIDER)
    if frame["self_provider"].isna().any():
        raise AssertionError("Unmapped generator family in no-self panel")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["source_stimulus_id", "concept_id", "domain", "task_type", "generator", "replicate_index"],
        observed=True,
        sort=True,
    ):
        base = dict(
            zip(
                ["source_stimulus_id", "concept_id", "domain", "task_type", "generator", "replicate_index"],
                keys,
            )
        )
        for dimension in DIMENSIONS:
            values = group[["judge_provider", dimension, "self_provider"]].dropna()
            scores = values[dimension].to_numpy(float)
            fixed_complete = len(values) == len(providers) and values["judge_provider"].nunique() == len(providers)
            nonself = values.loc[values["judge_provider"] != values["self_provider"], dimension].to_numpy(float)
            no_self_complete = len(nonself) == len(providers) - 1
            sorted_scores = np.sort(scores)
            rows.append(
                {
                    **base,
                    "dimension": dimension,
                    "n_judges_valid": int(len(scores)),
                    "fixed_9_complete": bool(fixed_complete),
                    "panel_mean": float(np.mean(scores)) if fixed_complete else np.nan,
                    "panel_median": float(np.median(scores)) if fixed_complete else np.nan,
                    "panel_trimmed_one_each_tail": (
                        float(np.mean(sorted_scores[1:-1])) if fixed_complete else np.nan
                    ),
                    "no_self_n_judges_valid": int(len(nonself)),
                    "no_self_complete": bool(no_self_complete),
                    "panel_no_self_mean": float(np.mean(nonself)) if no_self_complete else np.nan,
                }
            )
    return pd.DataFrame(rows)


def panel_repeatability(panel_cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    configurations = [
        ("mean", "panel_mean", "fixed_9_complete"),
        ("median", "panel_median", "fixed_9_complete"),
        ("trimmed_one_each_tail", "panel_trimmed_one_each_tail", "fixed_9_complete"),
        ("no_self_mean", "panel_no_self_mean", "no_self_complete"),
    ]
    summary_rows: list[dict[str, Any]] = []
    pair_rows_all: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 1)
    for dimension in DIMENSIONS:
        dimension_frame = panel_cells.loc[panel_cells["dimension"] == dimension]
        for aggregation, value_column, completeness_column in configurations:
            eligible = dimension_frame.loc[dimension_frame[completeness_column]].copy()
            pivot = eligible.pivot_table(
                index=["source_stimulus_id", "concept_id"],
                columns="replicate_index",
                values=value_column,
                aggfunc="first",
            ).reindex(columns=REPLICATES)
            complete = pivot.dropna()
            identity = {"aggregation": aggregation, "dimension": dimension}
            if len(complete) < 2:
                summary_rows.append(
                    {
                        **identity,
                        "status": "insufficient_complete_stimuli",
                        "n_complete_stimuli": int(len(complete)),
                    }
                )
                continue
            matrix = complete.to_numpy(float)
            concepts = complete.index.get_level_values("concept_id").to_numpy()
            row, pairs = summarize_matrix(matrix, concepts, identity, rng, include_kappa=False)
            expected_judges = 8 if aggregation == "no_self_mean" else 9
            row["status"] = (
                "complete_fixed_panel" if len(complete) == 90 else "interim_partial_fixed_panel"
            )
            row["expected_judges_per_stimulus_replicate"] = expected_judges
            summary_rows.append(row)
            pair_rows_all.extend(pairs)
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows_all)


def residualize_domain_task(frame: pd.DataFrame, dimension: str) -> np.ndarray:
    design = pd.get_dummies(
        frame[["domain", "task_type"]].astype(str), drop_first=True, dtype=float
    )
    design.insert(0, "intercept", 1.0)
    y = frame[dimension].to_numpy(float)
    beta = np.linalg.lstsq(design.to_numpy(float), y, rcond=None)[0]
    fitted = design.to_numpy(float) @ beta
    return y - fitted + np.mean(y)


def balanced_variance_decomposition(
    valid: pd.DataFrame, all_complete: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not all_complete:
        return pd.DataFrame(
            [
                {
                    "dimension": dimension,
                    "status": "not_run_until_all_9_providers_complete",
                }
                for dimension in DIMENSIONS
            ]
        )
    providers = sorted(valid["judge_provider"].unique())
    stimuli = sorted(valid["source_stimulus_id"].unique())
    for dimension in DIMENSIONS:
        frame = valid.sort_values(["source_stimulus_id", "judge_provider", "replicate_index"]).copy()
        frame["adjusted_score"] = residualize_domain_task(frame, dimension)
        pivot = frame.pivot_table(
            index="source_stimulus_id",
            columns=["judge_provider", "replicate_index"],
            values="adjusted_score",
            aggfunc="first",
        ).reindex(index=stimuli, columns=pd.MultiIndex.from_product([providers, REPLICATES]))
        if pivot.isna().any().any() or pivot.shape != (90, 27):
            raise AssertionError("Variance decomposition requires a balanced 90x9x3 array")
        y = pivot.to_numpy(float).reshape(90, 9, 3)
        i_count, j_count, k_count = y.shape
        grand = float(y.mean())
        cell_mean = y.mean(axis=2)
        rep_mean = y.mean(axis=(0, 1))
        residual = y - cell_mean[:, :, None] - rep_mean[None, None, :] + grand
        df_error = (i_count * j_count - 1) * (k_count - 1)
        sigma_error = float(np.sum(residual**2) / df_error)
        ms_rep = float(i_count * j_count * np.sum((rep_mean - grand) ** 2) / (k_count - 1))
        sigma_rep = max((ms_rep - sigma_error) / (i_count * j_count), 0.0)

        stim_mean = cell_mean.mean(axis=1)
        judge_mean = cell_mean.mean(axis=0)
        interaction = cell_mean - stim_mean[:, None] - judge_mean[None, :] + grand
        ms_stimulus = float(j_count * np.sum((stim_mean - grand) ** 2) / (i_count - 1))
        ms_judge = float(i_count * np.sum((judge_mean - grand) ** 2) / (j_count - 1))
        ms_interaction = float(
            np.sum(interaction**2) / ((i_count - 1) * (j_count - 1))
        )
        sigma_interaction = max(ms_interaction - sigma_error / k_count, 0.0)
        sigma_stimulus = max((ms_stimulus - ms_interaction) / j_count, 0.0)
        sigma_judge = max((ms_judge - ms_interaction) / i_count, 0.0)
        components = {
            "variance_stimulus": sigma_stimulus,
            "variance_judge": sigma_judge,
            "variance_stimulus_by_judge": sigma_interaction,
            "variance_replicate_occasion": sigma_rep,
            "variance_residual_fresh_call": sigma_error,
        }
        total = sum(components.values())
        judge_disagreement = sigma_judge + sigma_interaction
        fresh_call_instability = sigma_rep + sigma_error
        rows.append(
            {
                "dimension": dimension,
                "status": "balanced_method_of_moments_after_domain_task_residualization",
                **components,
                **{key.replace("variance_", "proportion_"): value / total for key, value in components.items()},
                "judge_disagreement_variance": judge_disagreement,
                "fresh_call_instability_variance": fresh_call_instability,
                "judge_disagreement_share_of_judge_plus_fresh": (
                    judge_disagreement / (judge_disagreement + fresh_call_instability)
                ),
                "fresh_call_instability_share_of_judge_plus_fresh": (
                    fresh_call_instability / (judge_disagreement + fresh_call_instability)
                ),
                "note": (
                    "Fast balanced linear method-of-moments decomposition; negative component estimates "
                    "are truncated at zero. The residual fresh-call component also absorbs omitted "
                    "replicate interactions; this is not an ordinal cumulative-link model."
                ),
            }
        )
    return pd.DataFrame(rows)


def make_report(
    path: Path,
    completion: pd.DataFrame,
    provider_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    variance: pd.DataFrame,
    final_complete: bool,
) -> None:
    status_label = "FINAL COMPLETE" if final_complete else "INTERIM / RUNS STILL IN PROGRESS"
    lines = [
        "# Three-fresh-call API repeat-stability report",
        "",
        f"**Analysis status: {status_label}.**",
        "",
        (
            "Only the three calls in this experiment are analyzed. No score from the historical "
            "7,290-row matrix is used as a repetition. Each provider-by-dimension estimate uses only "
            "stimuli with all three fresh responses; missing calls are not imputed."
        ),
        "",
        "## Completion audit",
        "",
        "| Provider | Valid planned calls | Missing | Complete 3-call stimuli | Complete |",
        "|---|---:|---:|---:|:---:|",
    ]
    for _, row in completion.sort_values("judge_provider").iterrows():
        lines.append(
            f"| {row['judge_provider']} | {int(row['valid_unique_rows'])}/270 | "
            f"{int(row['missing_planned_rows'])} | {int(row['complete_stimuli_three_reps'])}/90 | "
            f"{'yes' if row['provider_complete'] else 'no'} |"
        )
    invalid_total = int(completion["invalid_audit_rows"].fillna(0).sum())
    duplicate_total = int(completion["duplicate_audit_rows"].fillna(0).sum())
    observed_total = int(completion["observed_file_rows"].fillna(0).sum())
    if invalid_total or duplicate_total:
        lines.extend(
            [
                "",
                (
                    f"The provider files contained {observed_total} physical rows. Exactly 2,430 unique "
                    f"planned calls were selected; {invalid_total} invalid audit row(s), including "
                    f"{duplicate_total} duplicate planned-ID row(s), were counted in the audit but were "
                    "not analyzed as repetitions."
                ),
            ]
        )
    deviations = completion.loc[
        completion["provider_specific_retry_deviation"].fillna("").astype(str).str.len() > 0,
        ["judge_provider", "provider_specific_retry_deviation"],
    ]
    if len(deviations):
        lines.extend(["", "Provider-specific retry deviations (failed attempts remain retries, not repetitions):", ""])
        for _, row in deviations.iterrows():
            lines.append(f"- {row['judge_provider']}: {row['provider_specific_retry_deviation']}.")
    lines.extend(["", "## Repeatability results", ""])
    if not final_complete:
        complete_count = int(completion["provider_complete"].sum())
        lines.append(
            f"{complete_count}/9 providers are complete. Provider estimates from partial files are "
            "labelled interim, and fixed nine-judge panel results are not treated as final. Re-run "
            "this script after completion; it will automatically use all 2,430 planned calls."
        )
    else:
        ps = provider_summary.loc[provider_summary["status"] == "complete_provider"]
        mean_panel = panel_summary.loc[
            (panel_summary["aggregation"] == "mean")
            & (panel_summary["status"] == "complete_fixed_panel")
        ]
        lines.extend(
            [
                (
                    f"Across the {len(ps)} provider-by-dimension cells, single-call absolute-agreement "
                    f"ICC(A,1) ranged from {ps['icc_a1_absolute_single_measure'].min():.3f} to "
                    f"{ps['icc_a1_absolute_single_measure'].max():.3f} (median "
                    f"{ps['icc_a1_absolute_single_measure'].median():.3f}). Mean pairwise exact "
                    f"agreement ranged from {ps['mean_pairwise_exact_agreement'].min():.3f} to "
                    f"{ps['mean_pairwise_exact_agreement'].max():.3f}."
                ),
                "",
                (
                    f"For the fixed nine-judge arithmetic mean, replicate ICC(A,1) across the six "
                    f"dimensions ranged from {mean_panel['icc_a1_absolute_single_measure'].min():.3f} "
                    f"to {mean_panel['icc_a1_absolute_single_measure'].max():.3f}; the median was "
                    f"{mean_panel['icc_a1_absolute_single_measure'].median():.3f}. Dimension-specific "
                    "mean, median, trimmed-mean, and no-self estimates are all retained in the tables."
                ),
                "",
                (
                    "The variance table separates fixed-snapshot judge disagreement from fresh-call "
                    "instability using the balanced 90x9x3 design after domain/task residualization."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "These statistics estimate repeat-call stability for the frozen prompt, model identifiers, "
                "and 90-stimulus sample. They do not establish human validity, stability after provider "
                "model updates, or calibration to reader outcomes."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all nine providers have 270 valid planned calls",
    )
    args = parser.parse_args()

    planned, observed, file_audit, source_meta = load_planned_and_observed()
    call_status, valid, completion = construct_call_status(planned, observed, file_audit)
    providers = sorted(planned["judge_provider"].unique().tolist())
    final_complete = bool(completion["provider_complete"].all())
    if args.require_complete and not final_complete:
        missing = completion.loc[~completion["provider_complete"], ["judge_provider", "missing_planned_rows"]]
        raise SystemExit(f"Incomplete providers: {missing.to_dict('records')}")

    provider_summary, provider_pairs = provider_repeatability(valid, providers)
    panel_cells = build_panel_cells(valid, providers)
    panel_summary, panel_pairs = panel_repeatability(panel_cells)
    variance = balanced_variance_decomposition(valid, final_complete)

    safe_status_columns = [
        "judge_provider",
        "item_id",
        "source_stimulus_id",
        "source_item_uid",
        "api_item_id",
        "replicate_index",
        "runner_order",
        "domain",
        "concept_id",
        "concept",
        "task_type",
        "generator",
        "text_sha256",
        "call_status",
        "paper_model_label",
        "judge_model_requested",
        "actual_api_model",
        "judge_model_returned",
        "judge_model",
        *DIMENSIONS,
        "retry_count",
        "http_status",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "system_fingerprint_or_backend_id",
    ]
    safe_status_columns = [column for column in safe_status_columns if column in call_status.columns]
    outputs = {
        "repeat_call_status_2430.csv": call_status[safe_status_columns],
        "repeat_provider_completion.csv": completion,
        "repeat_provider_dimension_summary.csv": provider_summary,
        "repeat_provider_pairwise_metrics.csv": provider_pairs,
        "repeat_panel_scores_by_stimulus_replicate.csv": panel_cells,
        "repeat_panel_dimension_summary.csv": panel_summary,
        "repeat_panel_pairwise_metrics.csv": panel_pairs,
        "repeat_variance_decomposition.csv": variance,
    }
    for filename, frame in outputs.items():
        write_csv(BASE / filename, frame)
    make_report(
        BASE / "REPEAT_STABILITY_REPORT.md",
        completion,
        provider_summary,
        panel_summary,
        variance,
        final_complete,
    )

    results = {
        "analysis_version": SCRIPT_VERSION,
        "status": "final_complete" if final_complete else "interim_incomplete_providers",
        "historical_scores_used_as_repetitions": False,
        "expected_design": {
            "stimuli": 90,
            "providers": 9,
            "fresh_repetitions": 3,
            "planned_calls": 2430,
            "dimensions": DIMENSIONS,
        },
        "completion": completion.to_dict("records"),
        "provider_dimension_summary": provider_summary.to_dict("records"),
        "panel_dimension_summary": panel_summary.to_dict("records"),
        "variance_decomposition": variance.to_dict("records"),
        "method_notes": {
            "icc": "ICC(A,1), two-way absolute-agreement single-measure formula across fresh repetitions",
            "weighted_kappa": "quadratic-weighted Cohen kappa, averaged descriptively over the three repetition pairs",
            "bootstrap": (
                f"{N_BOOTSTRAP} fixed-seed draws; both individual-stimulus and 30-concept-cluster intervals reported"
            ),
            "complete_cells": "all provider metrics require all three fresh calls; no imputation",
            "panel": "fixed nine-judge aggregations require all nine valid judges within every stimulus-repetition cell",
        },
    }
    write_json(BASE / "repeat_stability_analysis_results.json", results)

    output_hashes = {
        filename: sha256_file(BASE / filename)
        for filename in [*outputs.keys(), "REPEAT_STABILITY_REPORT.md", "repeat_stability_analysis_results.json"]
    }
    manifest = {
        "analysis_version": SCRIPT_VERSION,
        "status": results["status"],
        "script": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        "sources": source_meta,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "bootstrap": {"seed": SEED, "draws": N_BOOTSTRAP},
        "outputs": output_hashes,
        "qa": {
            "planned_rows": int(len(call_status)),
            "valid_rows": int((call_status["call_status"] == "valid").sum()),
            "missing_rows": int((call_status["call_status"] == "missing").sum()),
            "providers_complete": int(completion["provider_complete"].sum()),
            "all_nine_complete": final_complete,
            "score_range_valid_for_selected_rows": bool(
                valid[DIMENSIONS].apply(lambda column: pd.to_numeric(column).between(1, 5).all()).all()
            ),
            "historical_scores_imported": False,
        },
        "secret_handling": "No credential values are read or written by this analysis script.",
    }
    write_json(BASE / "repeat_stability_analysis_manifest.json", manifest)

    forbidden_fragments = ["sk-", "aiza", "aq.", "api_key", "authorization: bearer"]
    scan_files = [
        *outputs.keys(),
        "REPEAT_STABILITY_REPORT.md",
        "repeat_stability_analysis_results.json",
        "repeat_stability_analysis_manifest.json",
    ]
    for filename in scan_files:
        text = (BASE / filename).read_text(encoding="utf-8-sig", errors="ignore").lower()
        if any(fragment in text for fragment in forbidden_fragments):
            raise AssertionError(f"Credential-like fragment detected in {filename}")

    print(
        json.dumps(
            {
                "status": results["status"],
                "valid_calls": int(len(valid)),
                "providers_complete": int(completion["provider_complete"].sum()),
                "provider_completion": completion[
                    [
                        "judge_provider",
                        "valid_unique_rows",
                        "missing_planned_rows",
                        "complete_stimuli_three_reps",
                        "provider_complete",
                    ]
                ].to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
