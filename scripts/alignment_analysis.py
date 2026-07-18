#!/usr/bin/env python3
"""Reproducible human--API alignment inference for the ADMA full paper.

This program is intentionally independent of every legacy human-rating derivative.
It reads only the current questionnaire cleaning outputs, the independently rebuilt
human--API crosswalk, the stimulus-equivalence map, and the current 7,290-row API
score matrix.  The inferential unit is the text stimulus signature.  Consequently,
API ratings from all current source UIDs that render to the same stimulus are pooled
before any human--API analysis.

Primary first-review inference uses a domain-stratified concept-cluster bootstrap
and a joint-six-dimension Freedman--Lane permutation.  The selected second review
uses the complete 2^15 set of within-domain-by-task pair swaps and adjusts for the
pre-existing high-disagreement versus low-disagreement-control selection role.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests


SCRIPT_VERSION = "4.0.0"
CLEANING_RULE_VERSION = "4.0.0"
SEED_BOOTSTRAP = 20260717
SEED_PERMUTATION = 20260717
SEED_SECOND_BOOTSTRAP = 20260717
SEED_SECOND_ITEM_BOOTSTRAP = 20260717
SEED_DISPERSION_BOOTSTRAP = 20260717
N_BOOTSTRAP = 5_000
N_PERMUTATION = 10_000
DIMENSIONS = ("fa", "cc", "lc", "tf", "mq", "risk")
DIMENSION_CN_TO_CODE = {
    "事实准确性": "fa",
    "概念完整性": "cc",
    "语言清晰度": "lc",
    "任务符合度": "tf",
    "误解处理质量": "mq",
    "误导风险": "risk",
}
WAVE_FIRST = "首轮专家复核"
WAVE_SECOND = "二次专家复核"
DOMAIN_CN_TO_EN = {
    "物理": "Physics",
    "化学": "Chemistry",
    "生物": "Biology",
    "地理": "Geoscience",
    "大气科学": "Climate/Environment",
}
SELECTION_HIGH = "high_human_disagreement"
SELECTION_LOW = "low_human_disagreement_control"
MAIN_CONTROLS = ("domain", "task", "dimension")
SECOND_EXTRA_CONTROL = "selection_role"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def validate_final_cleaning_manifest(
    project_root: Path,
    cleaned_path: Path,
    cleaned: pd.DataFrame,
    final_primary_path: Path,
    final_primary: pd.DataFrame,
) -> tuple[dict[str, Any], Path]:
    """Bind formal inference to the frozen final-cleaning release.

    The manifest is treated as an input contract, not merely provenance.  Both
    declared output files must match their on-disk bytes and row counts, and the
    final-primary panel must be byte-identical to the formal cleaned input.
    """
    manifest_path = project_root / "analysis" / "final_cleaning_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("qa_status") != "PASS":
        raise AssertionError("Final cleaning manifest must have qa_status=PASS")
    if manifest.get("rule_version") != CLEANING_RULE_VERSION:
        raise AssertionError(
            "Final cleaning rule_version mismatch: "
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
        raise AssertionError("Final cleaning manifest lacks a rules object")
    mismatched_rules = {
        key: {"expected": value, "observed": rules.get(key)}
        for key, value in required_rules.items()
        if rules.get(key) != value
    }
    if mismatched_rules:
        raise AssertionError(f"Final cleaning rules mismatch: {mismatched_rules}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise AssertionError("Final cleaning manifest lacks an outputs object")

    def validate_output(
        key: str, expected_path: Path, expected_frame: pd.DataFrame
    ) -> dict[str, Any]:
        record = outputs.get(key)
        if not isinstance(record, dict):
            raise AssertionError(f"Manifest output {key!r} is missing")
        missing = sorted({"path", "sha256", "bytes", "rows"} - set(record))
        if missing:
            raise AssertionError(f"Manifest output {key!r} lacks fields: {missing}")
        declared_path = Path(str(record["path"]))
        if not declared_path.is_absolute():
            declared_path = project_root / declared_path
        if declared_path.resolve() != expected_path.resolve():
            raise AssertionError(
                f"Manifest output {key!r} path mismatch: {declared_path} != {expected_path}"
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
            raise AssertionError(
                f"Manifest output {key!r} does not match its file: "
                f"declared={declared}, observed={observed}"
            )
        return record

    cleaned_record = validate_output(
        "cleaned_human_ratings_long", cleaned_path, cleaned
    )
    primary_record = validate_output("final_primary", final_primary_path, final_primary)
    for field in ("sha256", "bytes", "rows"):
        left = str(cleaned_record[field]).lower() if field == "sha256" else int(cleaned_record[field])
        right = str(primary_record[field]).lower() if field == "sha256" else int(primary_record[field])
        if left != right:
            raise AssertionError(
                f"Formal cleaned data and final_primary differ on {field}: {left} != {right}"
            )
    if list(cleaned.columns) != list(final_primary.columns) or not cleaned.equals(
        final_primary
    ):
        raise AssertionError(
            "final_primary must exactly equal cleaned_human_ratings_long"
        )

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise AssertionError("Final cleaning manifest lacks a counts object")
    if int(counts.get("final_long_rows", -1)) != len(cleaned):
        raise AssertionError("Manifest final_long_rows does not match cleaned data")
    declared_participants = counts.get("final_participants")
    if not isinstance(declared_participants, dict):
        raise AssertionError("Manifest final_participants must be an object by wave")
    observed_participants = {
        str(wave): int(group["participant_id"].nunique())
        for wave, group in cleaned.groupby("wave", sort=True)
    }
    normalized_declared = {
        str(wave): int(value) for wave, value in declared_participants.items()
    }
    if normalized_declared != observed_participants:
        raise AssertionError(
            "Manifest final_participants does not match cleaned data: "
            f"{normalized_declared} != {observed_participants}"
        )
    return manifest, manifest_path


def find_current_api_scores(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(project_root.parent.rglob("api_test_scores_7290.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one current api_test_scores_7290.csv below the project "
            f"parent, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"{label} is missing columns: {missing}")


def canonical_task(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("Task "):
        text = text[5:].strip()
    if text not in {"A", "B", "C"}:
        raise AssertionError(f"Unexpected task label: {value!r}")
    return text


def finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(value)
    return result


def control_matrix(data: pd.DataFrame, controls: Sequence[str]) -> np.ndarray:
    """Return intercept plus treatment-coded fixed effects, dropping constants."""
    columns: list[np.ndarray] = [np.ones(len(data), dtype=float)]
    for control in controls:
        values = data[control].astype(str)
        levels = sorted(values.unique().tolist())
        if len(levels) <= 1:
            continue
        for level in levels[1:]:
            columns.append((values == level).to_numpy(dtype=float))
    return np.column_stack(columns)


def standardized_fixed_slope(
    data: pd.DataFrame,
    controls: Sequence[str],
    weights: np.ndarray | None = None,
    analytic: bool = True,
) -> dict[str, float]:
    """OLS slope on pre-frozen wave-by-dimension human/API z scores.

    Standardization constants are estimated once on the original matched sample
    using sample SDs (ddof=1).  They are never re-estimated inside a bootstrap or
    permutation draw.
    """
    require_columns(data, ["api_z", "human_z"], "standardized analysis data")
    x = data["api_z"].to_numpy(dtype=float)
    y = data["human_z"].to_numpy(dtype=float)
    if len(data) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "slope": float("nan"),
            "slope_raw": float("nan"),
            "se_hc3": float("nan"),
            "ci_low_hc3": float("nan"),
            "ci_high_hc3": float("nan"),
            "p_hc3": float("nan"),
            "rank": float("nan"),
        }
    z = control_matrix(data, controls)
    design = np.column_stack([z, x])
    if weights is None:
        sqrt_w = np.ones(len(data), dtype=float)
    else:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=float))
    design_w = design * sqrt_w[:, None]
    y_w = y * sqrt_w
    beta = np.linalg.lstsq(design_w, y_w, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(design_w))
    raw = float(beta[-1])
    slope = raw
    result = {
        "slope": float(slope),
        "slope_raw": raw,
        "se_hc3": float("nan"),
        "ci_low_hc3": float("nan"),
        "ci_high_hc3": float("nan"),
        "p_hc3": float("nan"),
        "rank": rank,
    }
    if not analytic or weights is not None or rank < design.shape[1]:
        return result
    fitted = design @ beta
    residual = y - fitted
    xtx_inv = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    scale = residual / np.clip(1.0 - leverage, 1e-10, None)
    meat = design.T @ ((scale**2)[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv
    se_raw = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    se_std = se_raw
    z_value = raw / se_raw if se_raw > 0 else float("nan")
    result.update(
        {
            "se_hc3": se_std,
            "ci_low_hc3": slope - 1.959963984540054 * se_std,
            "ci_high_hc3": slope + 1.959963984540054 * se_std,
            "p_hc3": float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan"),
        }
    )
    return result


@dataclass(frozen=True)
class EffectSpec:
    family: str
    level: str
    mask: np.ndarray
    controls: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.family}::{self.level}"


def make_effect_specs(data: pd.DataFrame, wave: str) -> list[EffectSpec]:
    extras = (SECOND_EXTRA_CONTROL,) if wave == WAVE_SECOND else ()
    specs = [
        EffectSpec(
            family="overall",
            level="all",
            mask=np.ones(len(data), dtype=bool),
            controls=MAIN_CONTROLS + extras,
        )
    ]
    for family in ("dimension", "task", "domain"):
        levels = sorted(data[family].astype(str).unique().tolist())
        controls = tuple(c for c in MAIN_CONTROLS if c != family) + extras
        for level in levels:
            specs.append(
                EffectSpec(
                    family=family,
                    level=level,
                    mask=(data[family].astype(str) == level).to_numpy(),
                    controls=controls,
                )
            )
    return specs


def effect_point_table(data: pd.DataFrame, wave: str) -> tuple[pd.DataFrame, list[EffectSpec]]:
    specs = make_effect_specs(data, wave)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        subset = data.loc[spec.mask].reset_index(drop=True)
        estimate = standardized_fixed_slope(subset, spec.controls)
        rows.append(
            {
                "wave": wave,
                "family": spec.family,
                "level": spec.level,
                "effect_key": spec.key,
                "n_rows": int(len(subset)),
                "n_stimuli": int(subset["stimulus_signature_sha256"].nunique()),
                "n_domain_concepts": int(
                    subset[["domain", "concept_id"]].drop_duplicates().shape[0]
                ),
                "controls": "+".join(spec.controls),
                **estimate,
            }
        )
    return pd.DataFrame(rows), specs


def residualized_x_estimator(data: pd.DataFrame, spec: EffectSpec) -> dict[str, Any]:
    subset = data.loc[spec.mask]
    x = subset["api_z"].to_numpy(dtype=float)
    z = control_matrix(subset, spec.controls)
    beta = np.linalg.lstsq(z, x, rcond=None)[0]
    xhat = z @ beta
    xres = x - xhat
    denom = float(xres @ xres)
    if denom <= 0:
        raise AssertionError(f"Degenerate residualized API score for {spec.key}")
    return {
        "mask": spec.mask,
        "xres": xres,
        "xhat": xhat,
        "denom": denom,
    }


def slopes_for_y_matrix(
    y_matrix: np.ndarray,
    estimators: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for key, estimator in estimators.items():
        y = y_matrix[:, estimator["mask"]]
        raw = (y @ estimator["xres"]) / estimator["denom"]
        results[key] = raw
    return results


def load_and_validate_inputs(
    project_root: Path, api_path: Path
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, Any],
    Path,
]:
    analysis = project_root / "analysis"
    cleaned_path = analysis / "cleaned_human_ratings_long.csv"
    crosswalk_path = analysis / "human_api_crosswalk.csv"
    equivalence_path = analysis / "api_stimulus_equivalence_810.csv"
    cleaned = pd.read_csv(cleaned_path)
    crosswalk = pd.read_csv(crosswalk_path)
    equivalence = pd.read_csv(equivalence_path)
    api = pd.read_csv(api_path)

    require_columns(
        cleaned,
        [
            "wave",
            "domain",
            "task",
            "participant_id",
            "item_local_id",
            "dimension",
            "score_raw",
            "score_quality_aligned",
        ],
        "cleaned human ratings",
    )
    require_columns(
        crosswalk,
        [
            "wave",
            "domain_cn",
            "domain",
            "task",
            "item_local_id",
            "source_item_uid",
            "concept_id",
            "stimulus_signature_sha256",
            "selection_role",
        ],
        "human--API crosswalk",
    )
    require_columns(
        equivalence,
        [
            "source_item_uid",
            "item_id",
            "stimulus_signature_sha256",
            "source_equivalence_size",
        ],
        "stimulus equivalence",
    )
    require_columns(
        api,
        [
            "item_id",
            "source_item_uid",
            "domain",
            "concept_id",
            "task_type",
            "generator",
            "paper_model_label",
            "input_integrity_pass",
            *DIMENSIONS,
        ],
        "current API scores",
    )

    if cleaned.empty:
        raise AssertionError("Final cleaned human-rating file is empty")
    if len(crosswalk) != 210 or crosswalk.duplicated(
        ["wave", "domain_cn", "item_local_id"]
    ).any():
        raise AssertionError("Crosswalk must contain 210 unique questionnaire placements")
    if set(crosswalk["selection_role"].dropna().unique()) != {
        "broad_scheduled_item",
        SELECTION_HIGH,
        SELECTION_LOW,
    }:
        raise AssertionError("Unexpected or missing selection_role values")
    second_roles = crosswalk[crosswalk["wave"] == WAVE_SECOND]
    role_blocks = second_roles.groupby(["domain", "task"])["selection_role"].agg(list)
    if len(role_blocks) != 15 or any(
        sorted(values) != sorted([SELECTION_HIGH, SELECTION_LOW])
        for values in role_blocks
    ):
        raise AssertionError("Each second-review domain-by-task block must have one item per selection role")
    if len(equivalence) != 810 or equivalence["source_item_uid"].nunique() != 810:
        raise AssertionError("Stimulus equivalence must map exactly 810 unique source UIDs")
    if len(api) != 7_290 or api["source_item_uid"].nunique() != 810:
        raise AssertionError("Current API matrix must contain 7,290 rows over 810 source UIDs")
    api_counts = api.groupby("source_item_uid")["paper_model_label"].nunique()
    if not (api_counts == 9).all():
        raise AssertionError(f"Current API matrix is not complete 9/9: {api_counts.value_counts().to_dict()}")
    integrity = api["input_integrity_pass"].astype(str).str.lower().isin(["true", "1"])
    if not integrity.all():
        raise AssertionError(f"Found {(~integrity).sum()} API rows failing current input-integrity flag")
    for column in DIMENSIONS:
        values = pd.to_numeric(api[column], errors="coerce")
        if values.isna().any() or not values.between(1, 5).all():
            raise AssertionError(f"API column {column} must be complete and within 1--5")
    human_values = pd.to_numeric(cleaned["score_raw"], errors="coerce")
    if human_values.isna().any() or not human_values.between(1, 5).all():
        raise AssertionError("Cleaned human scores must be complete and within 1--5")
    if set(cleaned["dimension"].unique()) != set(DIMENSION_CN_TO_CODE):
        raise AssertionError("Human dimension labels do not match the six frozen dimensions")

    sensitivity_dir = analysis / "final_sensitivity_panels"
    expected = [
        "final_primary",
        "threshold_70",
        "threshold_80",
        "threshold_90",
        "threshold_95",
        "attention_only",
        "strict_first_both_pairs",
        "second_case05_case07_mean",
    ]
    panel_paths = {name: sensitivity_dir / f"{name}.csv" for name in expected}
    missing_panels = [str(path) for path in panel_paths.values() if not path.exists()]
    if missing_panels:
        raise FileNotFoundError(f"Missing final sensitivity panels: {missing_panels}")
    panels = {name: pd.read_csv(path) for name, path in panel_paths.items()}
    canonical_keys = [
        "wave",
        "domain",
        "participant_id",
        "item_local_id",
        "dimension",
    ]
    cleaned_cmp = cleaned.sort_values(canonical_keys).reset_index(drop=True)
    primary_cmp = panels["final_primary"].sort_values(canonical_keys).reset_index(drop=True)
    if list(cleaned_cmp.columns) != list(primary_cmp.columns) or not cleaned_cmp.equals(primary_cmp):
        raise AssertionError(
            "final_primary panel must exactly equal the cleaned human long file"
        )
    cleaning_manifest, cleaning_manifest_path = validate_final_cleaning_manifest(
        project_root,
        cleaned_path,
        cleaned,
        panel_paths["final_primary"],
        panels["final_primary"],
    )
    return (
        cleaned,
        crosswalk,
        equivalence,
        panels,
        api,
        cleaning_manifest,
        cleaning_manifest_path,
    )


def make_api_long(api: pd.DataFrame, equivalence: pd.DataFrame) -> pd.DataFrame:
    eq = equivalence[
        ["source_item_uid", "item_id", "stimulus_signature_sha256", "source_equivalence_size"]
    ].copy()
    joined = api.merge(
        eq,
        on=["source_item_uid", "item_id"],
        how="left",
        validate="many_to_one",
    )
    if joined["stimulus_signature_sha256"].isna().any():
        raise AssertionError("Some current API rows do not map to a stimulus signature")
    long = joined.melt(
        id_vars=[
            "item_id",
            "source_item_uid",
            "domain",
            "concept_id",
            "concept",
            "task_type",
            "generator",
            "paper_model_label",
            "stimulus_signature_sha256",
            "source_equivalence_size",
        ],
        value_vars=list(DIMENSIONS),
        var_name="dimension",
        value_name="api_score_raw",
    )
    long["api_score_raw"] = long["api_score_raw"].astype(float)
    long["api_score_aligned"] = np.where(
        long["dimension"] == "risk", 6.0 - long["api_score_raw"], long["api_score_raw"]
    )
    return long


def aggregate_api(
    api_long: pd.DataFrame,
    method: str = "mean",
    minimum_judges_per_uid: int = 9,
    excluded_judge: str | None = None,
    excluded_generator: str | None = None,
    raw_risk: bool = False,
    include_provenance: bool = True,
) -> pd.DataFrame:
    work = api_long.copy()
    if excluded_judge is not None:
        work = work[work["paper_model_label"] != excluded_judge]
    if excluded_generator is not None:
        work = work[work["generator"] != excluded_generator]
    coverage = (
        work.groupby("source_item_uid")["paper_model_label"].nunique().rename("judge_coverage")
    )
    keep_uids = coverage[coverage >= minimum_judges_per_uid].index
    work = work[work["source_item_uid"].isin(keep_uids)].copy()
    work["uid_judge_coverage"] = work["source_item_uid"].map(coverage).astype(int)
    value = "api_score_raw" if raw_risk else "api_score_aligned"
    if method not in {"mean", "median"}:
        raise ValueError(method)
    aggregate_function = "mean" if method == "mean" else "median"
    grouped = work.groupby(["stimulus_signature_sha256", "dimension"], as_index=False)
    result = grouped.agg(
        api_score=(value, aggregate_function),
        api_score_raw=("api_score_raw", aggregate_function),
        n_api_ratings=(value, "size"),
        n_source_uids=("source_item_uid", "nunique"),
        n_judges=("paper_model_label", "nunique"),
        min_uid_judge_coverage=("uid_judge_coverage", "min"),
        max_uid_judge_coverage=("uid_judge_coverage", "max"),
    )
    if include_provenance:
        keys = ["stimulus_signature_sha256", "dimension"]
        provenance = (
            work.groupby(keys, as_index=False)
            .agg(
                source_uids=("source_item_uid", lambda values: "|".join(sorted(set(values)))),
                generators=("generator", lambda values: "|".join(sorted(set(values)))),
            )
        )
        result = result.merge(provenance, on=keys, how="left", validate="one_to_one")
    result["aggregation"] = method
    result["risk_direction"] = "raw_high_is_risk" if raw_risk else "aligned_high_is_good"
    return result


def fold_human_panel(panel: pd.DataFrame, crosswalk: pd.DataFrame, raw_risk: bool = False) -> pd.DataFrame:
    work = panel.copy()
    work["task"] = work["task"].map(canonical_task)
    work["dimension"] = work["dimension"].map(DIMENSION_CN_TO_CODE)
    if work["dimension"].isna().any():
        raise AssertionError("Unmapped human dimension")
    mapping = crosswalk[
        [
            "wave",
            "domain_cn",
            "domain",
            "task",
            "item_local_id",
            "stimulus_signature_sha256",
            "concept_id",
            "concept",
            "selection_role",
        ]
    ].rename(columns={"domain": "domain_en", "task": "crosswalk_task"})
    merged = work.merge(
        mapping,
        left_on=["wave", "domain", "item_local_id"],
        right_on=["wave", "domain_cn", "item_local_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["stimulus_signature_sha256"].isna().any():
        missing = merged.loc[
            merged["stimulus_signature_sha256"].isna(),
            ["wave", "domain", "item_local_id"],
        ].drop_duplicates()
        raise AssertionError(f"Human rows lack a crosswalk: {missing.to_dict('records')}")
    if not (merged["task"] == merged["crosswalk_task"]).all():
        raise AssertionError("Task mismatch between cleaned ratings and crosswalk")
    score_column = "score_raw" if raw_risk else "score_quality_aligned"
    merged["human_value"] = pd.to_numeric(merged[score_column], errors="raise")
    if raw_risk:
        # For raw-direction sensitivity, only risk rows are requested downstream.
        merged = merged[merged["dimension"] == "risk"].copy()

    # A stimulus can occur at more than one questionnaire placement.  Average those
    # placements within participant first so a duplicate rendering does not double
    # weight that participant, then average participants at the stimulus level.
    participant = (
        merged.groupby(
            [
                "wave",
                "domain_en",
                "task",
                "concept_id",
                "concept",
                "selection_role",
                "participant_id",
                "stimulus_signature_sha256",
                "dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            participant_stimulus_score=("human_value", "mean"),
            n_questionnaire_placements=("item_local_id", "nunique"),
        )
    )
    folded = (
        participant.groupby(
            [
                "wave",
                "domain_en",
                "task",
                "concept_id",
                "concept",
                "selection_role",
                "stimulus_signature_sha256",
                "dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            human_score=("participant_stimulus_score", "mean"),
            human_sd=("participant_stimulus_score", "std"),
            n_human_raters=("participant_id", "nunique"),
            n_questionnaire_placements=("n_questionnaire_placements", "max"),
        )
        .rename(columns={"domain_en": "domain"})
    )
    folded["human_sd"] = folded["human_sd"].fillna(0.0)
    return folded


def merge_panel_api(panel: pd.DataFrame, crosswalk: pd.DataFrame, api_agg: pd.DataFrame, raw_risk: bool = False) -> pd.DataFrame:
    human = fold_human_panel(panel, crosswalk, raw_risk=raw_risk)
    merged = human.merge(
        api_agg,
        on=["stimulus_signature_sha256", "dimension"],
        how="left",
        validate="many_to_one",
    )
    missing = merged["api_score"].isna()
    if missing.any():
        merged = merged.loc[~missing].copy()
    merged["human_score"] = merged["human_score"].astype(float)
    merged["api_score"] = merged["api_score"].astype(float)
    return merged.sort_values(
        ["wave", "domain", "task", "concept_id", "stimulus_signature_sha256", "dimension"]
    ).reset_index(drop=True)


def freeze_within_dimension_standardization(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create wave-by-dimension z scores once using original-sample sample SDs."""
    work = data.copy()
    rows: list[dict[str, Any]] = []
    work["api_z"] = np.nan
    work["human_z"] = np.nan
    for (wave, dimension), index in work.groupby(["wave", "dimension"], sort=True).groups.items():
        api_values = work.loc[index, "api_score"].to_numpy(dtype=float)
        human_values = work.loc[index, "human_score"].to_numpy(dtype=float)
        api_mean = float(np.mean(api_values))
        human_mean = float(np.mean(human_values))
        api_sd = float(np.std(api_values, ddof=1))
        human_sd = float(np.std(human_values, ddof=1))
        if api_sd <= 0 or human_sd <= 0:
            raise AssertionError(f"Degenerate wave-by-dimension SD: {(wave, dimension)}")
        work.loc[index, "api_z"] = (api_values - api_mean) / api_sd
        work.loc[index, "human_z"] = (human_values - human_mean) / human_sd
        rows.append(
            {
                "wave": wave,
                "dimension": dimension,
                "n_rows": int(len(index)),
                "api_mean_aligned": api_mean,
                "api_sample_sd": api_sd,
                "human_mean_aligned": human_mean,
                "human_sample_sd": human_sd,
                "api_mean_raw_risk": 6.0 - api_mean if dimension == "risk" else api_mean,
                "human_mean_raw_risk": 6.0 - human_mean if dimension == "risk" else human_mean,
            }
        )
    if work[["api_z", "human_z"]].isna().any().any():
        raise AssertionError("Failed to apply frozen standardization")
    return work, pd.DataFrame(rows)


def apply_frozen_standardization(
    data: pd.DataFrame,
    constants: pd.DataFrame,
    raw_risk: bool = False,
) -> pd.DataFrame:
    """Apply primary matched-sample constants to a fixed sensitivity sample."""
    work = data.copy()
    suffix = "_raw_risk" if raw_risk else "_aligned"
    mean_api_column = "api_mean_raw_risk" if raw_risk else "api_mean_aligned"
    mean_human_column = "human_mean_raw_risk" if raw_risk else "human_mean_aligned"
    lookup = constants[
        [
            "wave",
            "dimension",
            mean_api_column,
            "api_sample_sd",
            mean_human_column,
            "human_sample_sd",
        ]
    ].copy()
    work = work.merge(
        lookup,
        on=["wave", "dimension"],
        how="left",
        validate="many_to_one",
    )
    if work["api_sample_sd"].isna().any():
        raise AssertionError(f"Missing frozen standardization constant{suffix}")
    work["api_z"] = (work["api_score"] - work[mean_api_column]) / work["api_sample_sd"]
    work["human_z"] = (work["human_score"] - work[mean_human_column]) / work["human_sample_sd"]
    return work.drop(
        columns=[
            mean_api_column,
            "api_sample_sd",
            mean_human_column,
            "human_sample_sd",
        ]
    )


def merge_panel_api_scheduled_uid(
    panel: pd.DataFrame,
    crosswalk: pd.DataFrame,
    api_long: pd.DataFrame,
) -> pd.DataFrame:
    """Sensitivity dataset using only each questionnaire placement's scheduled UID."""
    work = panel.copy()
    work["task"] = work["task"].map(canonical_task)
    work["dimension"] = work["dimension"].map(DIMENSION_CN_TO_CODE)
    mapping = crosswalk[
        [
            "wave",
            "domain_cn",
            "domain",
            "task",
            "item_local_id",
            "source_item_uid",
            "stimulus_signature_sha256",
            "concept_id",
            "concept",
            "selection_role",
        ]
    ].rename(
        columns={
            "domain": "domain_en",
            "task": "crosswalk_task",
            "stimulus_signature_sha256": "equivalence_signature_sha256",
        }
    )
    merged = work.merge(
        mapping,
        left_on=["wave", "domain", "item_local_id"],
        right_on=["wave", "domain_cn", "item_local_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["source_item_uid"].isna().any() or not (
        merged["task"] == merged["crosswalk_task"]
    ).all():
        raise AssertionError("Scheduled-UID sensitivity mapping failed")
    participant = (
        merged.groupby(
            [
                "wave",
                "domain_en",
                "task",
                "concept_id",
                "concept",
                "selection_role",
                "participant_id",
                "source_item_uid",
                "equivalence_signature_sha256",
                "dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(participant_stimulus_score=("score_quality_aligned", "mean"))
    )
    human = (
        participant.groupby(
            [
                "wave",
                "domain_en",
                "task",
                "concept_id",
                "concept",
                "selection_role",
                "source_item_uid",
                "equivalence_signature_sha256",
                "dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            human_score=("participant_stimulus_score", "mean"),
            human_sd=("participant_stimulus_score", "std"),
            n_human_raters=("participant_id", "nunique"),
        )
        .rename(columns={"domain_en": "domain"})
    )
    api_uid = (
        api_long.groupby(["source_item_uid", "dimension"], as_index=False)
        .agg(
            api_score=("api_score_aligned", "mean"),
            api_score_raw=("api_score_raw", "mean"),
            n_api_ratings=("api_score_aligned", "size"),
            n_judges=("paper_model_label", "nunique"),
        )
    )
    result = human.merge(
        api_uid,
        on=["source_item_uid", "dimension"],
        how="left",
        validate="many_to_one",
    )
    if result["api_score"].isna().any() or not (result["n_judges"] == 9).all():
        raise AssertionError("Scheduled UID lacks complete current API ratings")
    result["stimulus_signature_sha256"] = "scheduled_uid::" + result["source_item_uid"].astype(str)
    result["human_sd"] = result["human_sd"].fillna(0.0)
    return result.sort_values(
        ["wave", "domain", "task", "source_item_uid", "dimension"]
    ).reset_index(drop=True)


def respondent_alignment_data(
    panel: pd.DataFrame,
    crosswalk: pd.DataFrame,
    api_agg: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    """Build participant-by-stimulus-by-dimension data without legacy derivatives."""
    work = panel.copy()
    work["task"] = work["task"].map(canonical_task)
    work["dimension"] = work["dimension"].map(DIMENSION_CN_TO_CODE)
    mapping = crosswalk[
        [
            "wave",
            "domain_cn",
            "domain",
            "task",
            "item_local_id",
            "stimulus_signature_sha256",
            "concept_id",
            "selection_role",
        ]
    ].rename(columns={"domain": "domain_en", "task": "crosswalk_task"})
    merged = work.merge(
        mapping,
        left_on=["wave", "domain", "item_local_id"],
        right_on=["wave", "domain_cn", "item_local_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["stimulus_signature_sha256"].isna().any() or not (
        merged["task"] == merged["crosswalk_task"]
    ).all():
        raise AssertionError("Respondent-level crosswalk failed")
    participant = (
        merged.groupby(
            [
                "wave",
                "domain_en",
                "task",
                "concept_id",
                "selection_role",
                "participant_id",
                "stimulus_signature_sha256",
                "dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(human_score=("score_quality_aligned", "mean"))
        .rename(columns={"domain_en": "domain"})
    )
    api_values = api_agg[
        ["stimulus_signature_sha256", "dimension", "api_score"]
    ].copy()
    participant = participant.merge(
        api_values,
        on=["stimulus_signature_sha256", "dimension"],
        how="left",
        validate="many_to_one",
    )
    participant = apply_frozen_standardization(participant, constants)
    participant["human_score_ordinal"] = participant["human_score"].round().astype(int)
    participant["domain_concept"] = (
        participant["domain"].astype(str) + "||" + participant["concept_id"].astype(str)
    )
    return participant


def mixed_model_attempt(data: pd.DataFrame) -> dict[str, Any]:
    """Actually attempt the prespecified respondent-level crossed-VC mixed model."""
    import statsmodels.formula.api as smf

    first = data[data["wave"] == WAVE_FIRST].copy()
    first["top_group"] = "all"
    formula = "human_z ~ api_z + C(domain) + C(task) + C(dimension)"
    vc_formula = {
        "rater": "0 + C(participant_id)",
        "stimulus": "0 + C(stimulus_signature_sha256)",
        "domain_concept": "0 + C(domain_concept)",
    }
    model = smf.mixedlm(
        formula,
        first,
        groups=first["top_group"],
        re_formula="0",
        vc_formula=vc_formula,
        use_sparse=True,
    )
    fixed_rank = int(np.linalg.matrix_rank(model.exog))
    fixed_columns = int(model.exog.shape[1])
    attempts: list[dict[str, Any]] = []
    for optimizer in ("lbfgs", "powell"):
        caught: list[str] = []
        record: dict[str, Any] = {"optimizer": optimizer}
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=True,
                    method=optimizer,
                    maxiter=500,
                    full_output=True,
                    disp=False,
                )
            caught = [str(item.message) for item in warning_records]
            variances = [float(value) for value in np.atleast_1d(result.vcomp)]
            fixed_coefficients = {
                str(key): float(value) for key, value in result.fe_params.items()
            }
            fixed_standard_errors = {
                str(key): float(value)
                for key, value in zip(result.fe_params.index, np.asarray(result.bse_fe))
            }
            all_fixed_finite = bool(
                np.isfinite(list(fixed_coefficients.values())).all()
                and np.isfinite(list(fixed_standard_errors.values())).all()
            )
            record.update(
                {
                    "success": True,
                    "converged": bool(result.converged),
                    "api_z_slope": float(result.params.get("api_z", np.nan)),
                    "api_z_se": float(result.bse.get("api_z", np.nan)),
                    "api_z_p": float(result.pvalues.get("api_z", np.nan)),
                    "log_likelihood": float(result.llf),
                    "fixed_coefficients": fixed_coefficients,
                    "fixed_standard_errors": fixed_standard_errors,
                    "all_fixed_coefficients_and_se_finite": all_fixed_finite,
                    "variance_components": variances,
                    "residual_scale": float(result.scale),
                    "boundary_variance_component_lt_1e_8": bool(
                        any(value < 1e-8 for value in variances)
                    ),
                    "warnings": caught,
                }
            )
            try:
                hessian, hessian_singular = model.hessian(
                    np.asarray(result.params, dtype=float)
                )
                eigenvalues = np.linalg.eigvalsh(np.asarray(hessian, dtype=float))
                absolute = np.abs(eigenvalues)
                condition = float(absolute.max() / max(absolute.min(), 1e-300))
                record.update(
                    {
                        "hessian_available": True,
                        "hessian_singular_flag": bool(hessian_singular),
                        "hessian_eigenvalue_min": float(eigenvalues.min()),
                        "hessian_eigenvalue_max": float(eigenvalues.max()),
                        "hessian_abs_condition_number": condition,
                    }
                )
            except Exception as diagnostic_exc:
                record.update(
                    {
                        "hessian_available": False,
                        "hessian_exception_type": type(diagnostic_exc).__name__,
                        "hessian_exception": str(diagnostic_exc),
                    }
                )
        except Exception as exc:  # objective audit must retain failure details
            record.update(
                {
                    "success": False,
                    "converged": False,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "warnings": caught,
                }
            )
        attempts.append(record)

    successful = [item for item in attempts if item.get("success")]
    slopes = [item["api_z_slope"] for item in successful if np.isfinite(item["api_z_slope"])]
    slope_range = float(max(slopes) - min(slopes)) if len(slopes) >= 2 else float("nan")
    log_likelihoods = [
        item["log_likelihood"]
        for item in successful
        if np.isfinite(item.get("log_likelihood", np.nan))
    ]
    log_likelihood_range = (
        float(max(log_likelihoods) - min(log_likelihoods))
        if len(log_likelihoods) >= 2
        else float("nan")
    )
    reasons: list[str] = []
    if fixed_rank < fixed_columns:
        reasons.append("fixed-effect design is rank deficient")
    if len(successful) < 2 or not all(item.get("converged") for item in successful):
        reasons.append("both prespecified optimizers did not converge successfully")
    if any(not item.get("all_fixed_coefficients_and_se_finite", False) for item in successful):
        reasons.append("at least one optimizer returned a non-finite fixed coefficient or SE")
    if any(item.get("boundary_variance_component_lt_1e_8") for item in successful):
        reasons.append("at least one variance component is on the <1e-8 boundary")
    if any(not item.get("hessian_available", False) for item in successful):
        reasons.append("Hessian diagnostics are unavailable for at least one optimizer")
    if any(item.get("hessian_singular_flag") for item in successful):
        reasons.append("statsmodels flagged a singular Hessian")
    if any(item.get("hessian_abs_condition_number", math.inf) >= 1e12 for item in successful):
        reasons.append("Hessian absolute condition number is >=1e12")
    if not np.isfinite(slope_range) or slope_range > 0.02:
        reasons.append("optimizer slope range is unavailable or exceeds 0.02 SD")
    if not np.isfinite(log_likelihood_range) or log_likelihood_range > 1e-4:
        reasons.append("optimizer log-likelihood range is unavailable or exceeds 1e-4")
    stable = len(reasons) == 0
    return {
        "model": "respondent-level crossed variance-component linear mixed model",
        "formula": formula,
        "top_level_group": "single constant group",
        "variance_components": vc_formula,
        "n_rows": int(len(first)),
        "n_raters": int(first["participant_id"].nunique()),
        "n_stimuli": int(first["stimulus_signature_sha256"].nunique()),
        "n_domain_concept_clusters": int(first["domain_concept"].nunique()),
        "fixed_design_rank": fixed_rank,
        "fixed_design_columns": fixed_columns,
        "optimizers": attempts,
        "optimizer_slope_range": slope_range if np.isfinite(slope_range) else None,
        "optimizer_log_likelihood_range": (
            log_likelihood_range if np.isfinite(log_likelihood_range) else None
        ),
        "stable_for_primary_inference": stable,
        "fallback_used": not stable,
        "objective_fallback_reasons": reasons,
    }


def ordinal_gee_attempt(data: pd.DataFrame) -> dict[str, Any]:
    """Attempt the prespecified cumulative-logit ordinal GEE model-form check."""
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE

    output: dict[str, Any] = {
        "model": "cumulative-logit OrdinalGEE with exchangeable within-rater correlation",
        "role": "model-form sensitivity only; not primary inference",
        "outcome_construction": (
            "risk was quality-aligned; text-equivalent questionnaire placements were "
            "averaged within participant and then rounded to the nearest integer category"
        ),
        "maximum_iterations": 200,
        "report_effect_in_manuscript": False,
        "reporting_reason": (
            "retain the attempt audit only; no coefficient is reportable when the "
            "iteration-limit/finite-inference stability criteria fail"
        ),
        "waves": [],
    }
    for wave in (WAVE_FIRST, WAVE_SECOND):
        subset = data[data["wave"] == wave].copy()
        extras = " + C(selection_role)" if wave == WAVE_SECOND else ""
        formula = (
            "human_score_ordinal ~ api_z + C(domain) + C(task) + C(dimension)" + extras
        )
        record: dict[str, Any] = {
            "wave": wave,
            "formula": formula,
            "n_rows": int(len(subset)),
            "n_raters": int(subset["participant_id"].nunique()),
        }
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                model = OrdinalGEE.from_formula(
                    formula,
                    groups="participant_id",
                    data=subset,
                    cov_struct=Exchangeable(),
                )
                result = model.fit(maxiter=200)
            history = getattr(result, "fit_history", {}) or {}
            warning_text = [str(item.message) for item in warning_records]
            api_parameter = float(result.params.get("api_z", np.nan))
            api_se = float(result.bse.get("api_z", np.nan))
            api_p = float(result.pvalues.get("api_z", np.nan))
            finite = bool(np.isfinite([api_parameter, api_se, api_p]).all())
            iteration_limit = any(
                "Iteration limit reached" in message for message in warning_text
            )
            usable = finite and not iteration_limit
            record.update(
                {
                    "fit_returned": True,
                    "success": usable,
                    "converged_without_iteration_limit": not iteration_limit,
                    "finite_api_inference": finite,
                    "api_z_log_odds": api_parameter,
                    "api_z_odds_ratio": float(np.exp(api_parameter)),
                    "api_z_se_robust": api_se,
                    "api_z_p_robust": api_p,
                    "iterations": int(len(history.get("params", []))),
                    "warnings": warning_text,
                    "failure_interpretation": (
                        "not usable as a stable model-form check because convergence or finite-inference criteria failed"
                        if not usable
                        else ""
                    ),
                    "cluster_count_warning": (
                        "robust GEE covariance is fragile with fewer than 30 rater clusters"
                        if subset["participant_id"].nunique() < 30
                        else ""
                    ),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "success": False,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "failure_interpretation": (
                        "Ordinal GEE was attempted but is not reported as affirmative evidence."
                    ),
                }
            )
        output["waves"].append(record)
    return output


def stratified_cluster_bootstrap(
    data: pd.DataFrame,
    specs: list[EffectSpec],
    n_bootstrap: int,
    seed: int,
    cluster_column: str,
) -> pd.DataFrame:
    """Sample fixed clusters within domain using batched OLS sufficient statistics."""
    rng = np.random.default_rng(seed)
    cluster_labels = data["domain"].astype(str) + "||" + data[cluster_column].astype(str)
    clusters_by_domain: dict[str, list[str]] = {}
    for domain in sorted(data["domain"].unique()):
        labels = sorted(cluster_labels[data["domain"] == domain].unique().tolist())
        clusters_by_domain[str(domain)] = labels
    all_clusters = sorted(cluster_labels.unique().tolist())
    cluster_index = {label: idx for idx, label in enumerate(all_clusters)}
    counts = np.zeros((n_bootstrap, len(all_clusters)), dtype=float)
    for labels in clusters_by_domain.values():
        probabilities = np.repeat(1.0 / len(labels), len(labels))
        domain_counts = rng.multinomial(len(labels), probabilities, size=n_bootstrap)
        positions = [cluster_index[label] for label in labels]
        counts[:, positions] = domain_counts

    result = pd.DataFrame({"replicate": np.arange(1, n_bootstrap + 1, dtype=int)})
    global_labels = cluster_labels.to_numpy()
    for spec in specs:
        subset = data.loc[spec.mask].reset_index(drop=True)
        subset_labels = global_labels[spec.mask]
        z = control_matrix(subset, spec.controls)
        x = subset["api_z"].to_numpy(dtype=float)
        y = subset["human_z"].to_numpy(dtype=float)
        design = np.column_stack([z, x])
        p = design.shape[1]
        xtx_cluster = np.zeros((len(all_clusters), p, p), dtype=float)
        xty_cluster = np.zeros((len(all_clusters), p), dtype=float)
        for label in sorted(set(subset_labels)):
            rows = subset_labels == label
            position = cluster_index[str(label)]
            local_design = design[rows]
            local_y = y[rows]
            xtx_cluster[position] = local_design.T @ local_design
            xty_cluster[position] = local_design.T @ local_y
        xtx_draws = np.einsum("bc,cij->bij", counts, xtx_cluster, optimize=True)
        xty_draws = np.einsum("bc,ci->bi", counts, xty_cluster, optimize=True)
        draw_rank = np.linalg.matrix_rank(xtx_draws)
        invalid = draw_rank < p
        beta = np.einsum(
            "bij,bj->bi", np.linalg.pinv(xtx_draws, rcond=1e-12), xty_draws, optimize=True
        )
        slopes = beta[:, -1]
        slopes[invalid] = np.nan
        result[spec.key] = slopes
    return result


def canonical_permutation_layout(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, int]]:
    canonical = data.sort_values(["stimulus_signature_sha256", "dimension"]).reset_index(drop=True)
    dimensions = sorted(DIMENSIONS)
    per_stim_dims = canonical.groupby("stimulus_signature_sha256")["dimension"].agg(list)
    if any(sorted(values) != dimensions for values in per_stim_dims):
        raise AssertionError("Every stimulus must have exactly the frozen six dimensions")
    signatures = canonical["stimulus_signature_sha256"].drop_duplicates().tolist()
    signature_index = {value: index for index, value in enumerate(signatures)}
    dimension_index = {value: index for index, value in enumerate(dimensions)}
    return canonical, signatures, signature_index, dimension_index


def spec_specific_freedman_lane_null(
    canonical: pd.DataFrame,
    spec: EffectSpec,
    source_maps: np.ndarray,
    signature_index: dict[str, int],
    dimension_index: dict[str, int],
    compute_item_spearman: bool = False,
    batch_size: int = 250,
) -> tuple[np.ndarray, float, np.ndarray | None, float | None]:
    """FL null for one effect using its own subset and reduced fixed effects.

    All effects receive the same stimulus-level source maps.  Therefore dimensions
    move jointly even though every effect fits its own valid reduced model.
    """
    subset = canonical.loc[spec.mask].reset_index(drop=True)
    y = subset["human_z"].to_numpy(dtype=float)
    z = control_matrix(subset, spec.controls)
    fitted = z @ np.linalg.lstsq(z, y, rcond=None)[0]
    residual = y - fitted
    x = subset["api_z"].to_numpy(dtype=float)
    x_residual = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    denominator = float(x_residual @ x_residual)
    if denominator <= 0:
        raise AssertionError(f"Degenerate API residual for {spec.key}")
    target_stimulus = subset["stimulus_signature_sha256"].map(signature_index).to_numpy(dtype=int)
    target_dimension = subset["dimension"].map(dimension_index).to_numpy(dtype=int)
    residual_lookup = np.full(
        (len(signature_index), len(dimension_index)), np.nan, dtype=float
    )
    residual_lookup[target_stimulus, target_dimension] = residual

    observed = float((y @ x_residual) / denominator)
    point = standardized_fixed_slope(subset, spec.controls, analytic=False)["slope"]
    identity_difference = observed - point
    if not np.isclose(identity_difference, 0.0, atol=1e-12, rtol=1e-12):
        raise AssertionError(f"Identity FL mismatch for {spec.key}: {identity_difference}")

    n_draws = len(source_maps)
    slopes = np.empty(n_draws, dtype=float)
    spearman_values = np.empty(n_draws, dtype=float) if compute_item_spearman else None
    observed_spearman: float | None = None
    if compute_item_spearman:
        if len(subset) != len(canonical):
            raise AssertionError("Item Spearman is defined only for the overall effect")
        n_stimuli = len(signature_index)
        api_item = x.reshape(n_stimuli, len(DIMENSIONS)).mean(axis=1)
        api_rank = rankdata(api_item, method="average")
        api_rank_centered = api_rank - api_rank.mean()
        api_rank_norm = float(np.sqrt(api_rank_centered @ api_rank_centered))
        observed_spearman = safe_correlation(
            spearmanr,
            api_item,
            y.reshape(n_stimuli, len(DIMENSIONS)).mean(axis=1),
        )

    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        maps = source_maps[start:stop]
        source_stimulus = maps[:, target_stimulus]
        residual_permuted = residual_lookup[
            source_stimulus, target_dimension[None, :]
        ]
        if np.isnan(residual_permuted).any():
            raise AssertionError(
                f"Restricted map moved {spec.key} outside its subset/stratum"
            )
        y_star = fitted[None, :] + residual_permuted
        slopes[start:stop] = (y_star @ x_residual) / denominator
        if compute_item_spearman and spearman_values is not None:
            human_item = y_star.reshape(
                stop - start, len(signature_index), len(DIMENSIONS)
            ).mean(axis=2)
            human_rank = rankdata(human_item, method="average", axis=1)
            human_rank -= human_rank.mean(axis=1, keepdims=True)
            spearman_values[start:stop] = (
                human_rank @ api_rank_centered
            ) / (api_rank_norm * np.sqrt(np.sum(human_rank**2, axis=1)))
    return slopes, float(identity_difference), spearman_values, observed_spearman


def first_freedman_lane(
    data: pd.DataFrame,
    specs: list[EffectSpec],
    n_permutation: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Spec-specific FL nulls driven by one shared set of domain-by-task maps."""
    canonical, signatures, signature_index, dimension_index = canonical_permutation_layout(data)
    stimulus_meta = canonical.drop_duplicates("stimulus_signature_sha256")
    strata = (
        stimulus_meta["domain"].astype(str) + "||" + stimulus_meta["task"].astype(str)
    ).to_numpy()
    stratum_levels = sorted(set(strata))
    stratum_positions = {
        level: np.flatnonzero(strata == level) for level in stratum_levels
    }
    stratum_sizes = {level: int(len(value)) for level, value in stratum_positions.items()}
    if not stratum_sizes or sum(stratum_sizes.values()) != len(signatures):
        raise AssertionError("First FL strata do not partition the observed stimuli")
    if min(stratum_sizes.values()) < 2:
        raise AssertionError(
            f"First FL requires at least two observed stimuli per domain-task stratum: {stratum_sizes}"
        )
    observed_domain_tasks = int(
        stimulus_meta[["domain", "task"]].drop_duplicates().shape[0]
    )
    if len(stratum_positions) != observed_domain_tasks:
        raise AssertionError("First FL stratum count does not match observed domain-task cells")
    minimum_size = min(stratum_sizes.values())
    minimum_strata = sorted(
        level for level, size in stratum_sizes.items() if size == minimum_size
    )

    rng = np.random.default_rng(seed)
    source_maps = np.broadcast_to(
        np.arange(len(signatures), dtype=np.int16),
        (n_permutation, len(signatures)),
    ).copy()
    for draw in range(n_permutation):
        for positions in stratum_positions.values():
            source_maps[draw, positions] = rng.permutation(positions)

    canonical_specs = make_effect_specs(canonical, WAVE_FIRST)
    if [spec.key for spec in canonical_specs] != [spec.key for spec in specs]:
        raise AssertionError("Canonical first effect specification mismatch")
    result = pd.DataFrame({"permutation": np.arange(1, n_permutation + 1)})
    identity_differences: dict[str, float] = {}
    observed_spearman: float | None = None
    for spec in canonical_specs:
        null, identity, spearman_null, spearman_observed = spec_specific_freedman_lane_null(
            canonical,
            spec,
            source_maps,
            signature_index,
            dimension_index,
            compute_item_spearman=spec.family == "overall",
        )
        result[spec.key] = null
        identity_differences[spec.key] = identity
        if spearman_null is not None:
            result["model_form::item_spearman"] = spearman_null
            observed_spearman = spearman_observed
    qa = {
        "identity_permutation_matches_observed_max_abs_difference": float(
            max(abs(value) for value in identity_differences.values())
        ),
        "simple_effect_reduced_models": "each subset uses its own controls-only reduced model",
        "shared_stimulus_maps_across_all_effects": True,
        "joint_six_dimension_movement": True,
        "movement_implementation": "one source stimulus index selects all included dimension residuals",
        "permutation_strata": "domain-by-task",
        "n_strata": int(len(stratum_positions)),
        "stratum_size_by_label": stratum_sizes,
        "stratum_sizes": sorted(stratum_sizes.values()),
        "minimum_stratum_size": int(minimum_size),
        "minimum_size_strata": minimum_strata,
        "observed_item_level_spearman": float(observed_spearman),
    }
    return result, qa, source_maps


def second_exact_pair_permutation(
    data: pd.DataFrame,
    specs: list[EffectSpec],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Enumerate spec-specific FL residual swaps using shared 2^15 pair maps."""
    canonical, signatures, signature_index, dimension_index = canonical_permutation_layout(data)
    stimulus_meta = canonical.drop_duplicates("stimulus_signature_sha256").reset_index(drop=True)
    pairs: list[tuple[int, int, str]] = []
    for (domain, task), group in stimulus_meta.groupby(["domain", "task"], sort=True):
        if len(group) != 2 or set(group["selection_role"]) != {SELECTION_HIGH, SELECTION_LOW}:
            raise AssertionError(f"Invalid exact-permutation pair: {(domain, task)}")
        ordered = group.sort_values("selection_role")
        left, right = [signature_index[value] for value in ordered["stimulus_signature_sha256"]]
        pairs.append((left, right, f"{domain}|{task}"))
    if len(pairs) != 15:
        raise AssertionError(f"Expected 15 second-review pairs, found {len(pairs)}")
    n_exact = 2 ** len(pairs)
    source_maps = np.broadcast_to(
        np.arange(len(signatures), dtype=np.int16), (n_exact, len(signatures))
    ).copy()
    codes = np.arange(n_exact, dtype=np.uint32)
    for bit, (left, right, _label) in enumerate(pairs):
        swap = ((codes >> bit) & 1).astype(bool)
        source_maps[swap, left] = right
        source_maps[swap, right] = left

    canonical_specs = make_effect_specs(canonical, WAVE_SECOND)
    if [spec.key for spec in canonical_specs] != [spec.key for spec in specs]:
        raise AssertionError("Canonical second effect specification mismatch")
    result = pd.DataFrame({"permutation_code": np.arange(n_exact, dtype=int)})
    identity_differences: dict[str, float] = {}
    observed_spearman: float | None = None
    for spec in canonical_specs:
        null, identity, spearman_null, spearman_observed = spec_specific_freedman_lane_null(
            canonical,
            spec,
            source_maps,
            signature_index,
            dimension_index,
            compute_item_spearman=spec.family == "overall",
            batch_size=512,
        )
        result[spec.key] = null
        identity_differences[spec.key] = identity
        if spearman_null is not None:
            result["model_form::item_spearman"] = spearman_null
            observed_spearman = spearman_observed
    spearman_identity_difference = float(
        result.loc[0, "model_form::item_spearman"] - float(observed_spearman)
    )
    if not np.isclose(spearman_identity_difference, 0.0, atol=1e-12, rtol=1e-12):
        raise AssertionError(f"Second identity Spearman mismatch: {spearman_identity_difference}")
    pair_manifest = {
        "n_pairs": len(pairs),
        "n_exact_permutations": n_exact,
        "pairs": [label for _left, _right, label in pairs],
        "exchangeability_condition": (
            "Within each domain-by-task block, the high-human-disagreement and "
            "low-human-disagreement-control stimuli are conditionally exchangeable under "
            "the null; each effect moves its own reduced-model residuals with one shared map."
        ),
        "simple_effect_reduced_models": "each subset includes selection_role and its own remaining controls",
        "shared_stimulus_maps_across_all_effects": True,
        "joint_six_dimension_movement": True,
        "identity_permutation_code": 0,
        "identity_permutation_matches_observed_max_abs_difference": float(
            max(abs(value) for value in identity_differences.values())
        ),
        "observed_item_level_spearman": float(observed_spearman),
        "identity_item_spearman_difference": spearman_identity_difference,
        "limitation": (
            "Because selection_role was defined from prior human disagreement, exact "
            "block permutation supports conditional association only and cannot justify "
            "population-wide generalization from the purposively selected 30 stimuli."
        ),
    }
    return result, pair_manifest, source_maps


def add_inference_to_effects(
    points: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    permutation: pd.DataFrame,
    permutation_method: str,
    bootstrap_method: str | None = None,
) -> pd.DataFrame:
    result = points.copy()
    result["ci_low"] = result["ci_low_hc3"]
    result["ci_high"] = result["ci_high_hc3"]
    result["ci_method"] = "HC3 normal approximation"
    result["bootstrap_se"] = np.nan
    result["n_bootstrap_valid"] = 0
    result["bootstrap_invalid_proportion"] = np.nan
    result["bootstrap_routine_reportable"] = False
    result["permutation_p"] = np.nan
    result["permutation_method"] = permutation_method
    result["p_adjusted"] = np.nan
    result["multiplicity_method"] = "none"
    for index, row in result.iterrows():
        key = row["effect_key"]
        observed = float(row["slope"])
        null = permutation[key].dropna().to_numpy(dtype=float)
        if permutation_method.startswith("exact"):
            p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
        else:
            p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
        result.loc[index, "permutation_p"] = p_value
        if bootstrap is not None:
            all_draws = bootstrap[key]
            draws = all_draws.dropna().to_numpy(dtype=float)
            invalid_proportion = float(all_draws.isna().mean())
            result.loc[index, "bootstrap_invalid_proportion"] = invalid_proportion
            result.loc[index, "n_bootstrap_valid"] = int(len(draws))
            if invalid_proportion <= 0.01:
                result.loc[index, "ci_low"] = float(np.quantile(draws, 0.025))
                result.loc[index, "ci_high"] = float(np.quantile(draws, 0.975))
                result.loc[index, "ci_method"] = bootstrap_method or "percentile cluster bootstrap"
                result.loc[index, "bootstrap_se"] = float(np.std(draws, ddof=1))
                result.loc[index, "bootstrap_routine_reportable"] = True
            else:
                result.loc[index, "ci_low"] = np.nan
                result.loc[index, "ci_high"] = np.nan
                result.loc[index, "ci_method"] = "bootstrap_not_routinely_estimable"

    for family, method, label in [
        ("dimension", "fdr_bh", "Benjamini-Hochberg"),
        ("task", "holm", "Holm"),
        ("domain", "holm", "Holm"),
    ]:
        mask = result["family"] == family
        if mask.any():
            adjusted = multipletests(result.loc[mask, "permutation_p"], method=method)[1]
            result.loc[mask, "p_adjusted"] = adjusted
            result.loc[mask, "multiplicity_method"] = label
    overall = result["family"] == "overall"
    result.loc[overall, "p_adjusted"] = result.loc[overall, "permutation_p"]
    result["practically_material_abs_ge_0_20"] = result["slope"].abs() >= 0.20
    result["statistically_reliable_positive"] = (
        (result["family"] == "overall")
        & (result["slope"] > 0)
        & (result["ci_low"] > 0)
        & (result["permutation_p"] < 0.05)
    )
    result["adjusted_reliable_positive"] = (
        (result["slope"] > 0)
        & (result["ci_low"] > 0)
        & (result["p_adjusted"] < 0.05)
    )
    return result


def heterogeneity_tests(
    data: pd.DataFrame,
    source_maps: np.ndarray,
    wave: str,
    exact: bool,
) -> pd.DataFrame:
    """FL interaction tests allowing a nonzero common API slope under the null."""
    canonical, signatures, _signature_index, _dimension_index = canonical_permutation_layout(data)
    y = canonical["human_z"].to_numpy(dtype=float)
    x = canonical["api_z"].to_numpy(dtype=float)
    controls = MAIN_CONTROLS + ((SECOND_EXTRA_CONTROL,) if wave == WAVE_SECOND else ())
    z = control_matrix(canonical, controls)
    reduced = np.column_stack([z, x])
    reduced_rank = int(np.linalg.matrix_rank(reduced))
    reduced_beta = np.linalg.lstsq(reduced, y, rcond=None)[0]
    fitted = reduced @ reduced_beta
    residual_matrix = (y - fitted).reshape(len(signatures), len(DIMENSIONS))
    fitted_matrix = fitted.reshape(len(signatures), len(DIMENSIONS))

    def sse_matrix(values: np.ndarray, design: np.ndarray) -> np.ndarray:
        inverse = np.linalg.pinv(design.T @ design, rcond=1e-12)
        cross = values @ design
        explained = np.einsum("bi,ij,bj->b", cross, inverse, cross, optimize=True)
        return np.maximum(np.sum(values**2, axis=1) - explained, 0.0)

    rows: list[dict[str, Any]] = []
    for family in ("task", "domain"):
        values = canonical[family].astype(str)
        levels = sorted(values.unique().tolist())
        interactions = np.column_stack(
            [x * (values == level).to_numpy(dtype=float) for level in levels[1:]]
        )
        full = np.column_stack([reduced, interactions])
        full_rank = int(np.linalg.matrix_rank(full))
        q = full_rank - reduced_rank
        denominator_df = len(canonical) - full_rank
        if q != len(levels) - 1 or denominator_df <= 0:
            raise AssertionError(
                f"Invalid {wave} {family} interaction rank: q={q}, levels={levels}"
            )
        observed_y = y.reshape(1, -1)
        observed_sse_reduced = sse_matrix(observed_y, reduced)[0]
        observed_sse_full = sse_matrix(observed_y, full)[0]
        observed = float(
            ((observed_sse_reduced - observed_sse_full) / q)
            / (observed_sse_full / denominator_df)
        )
        null_statistic = np.empty(len(source_maps), dtype=float)
        for start in range(0, len(source_maps), 250):
            stop = min(start + 250, len(source_maps))
            y_star = (
                fitted_matrix[None, :, :] + residual_matrix[source_maps[start:stop]]
            ).reshape(stop - start, -1)
            sse_reduced = sse_matrix(y_star, reduced)
            sse_full = sse_matrix(y_star, full)
            null_statistic[start:stop] = (
                ((sse_reduced - sse_full) / q)
                / np.maximum(sse_full / denominator_df, 1e-300)
            )
        if exact:
            identity_difference = float(null_statistic[0] - observed)
            if not np.isclose(identity_difference, 0.0, atol=1e-10, rtol=1e-10):
                raise AssertionError(
                    f"Exact heterogeneity identity mismatch for {wave} {family}: "
                    f"{identity_difference}"
                )
            p_value = float(np.mean(null_statistic >= observed - 1e-12))
        else:
            identity_difference = np.nan
            p_value = float((1 + np.sum(null_statistic >= observed)) / (len(null_statistic) + 1))
        rows.append(
            {
                "wave": wave,
                "family": family,
                "n_levels": int(len(levels)),
                "statistic": "incremental_F_for_api_z_by_family_interactions",
                "observed": observed,
                "permutation_p": p_value,
                "n_permutations": int(len(null_statistic)),
                "exact": bool(exact),
                "null_model": "common nonzero api_z slope plus all fixed main effects",
                "full_model": f"null model plus api_z-by-{family} interactions",
                "interaction_df": int(q),
                "denominator_df": int(denominator_df),
                "exact_identity_difference": identity_difference,
            }
        )
    result = pd.DataFrame(rows)
    result["p_adjusted_holm_within_wave"] = multipletests(
        result["permutation_p"], method="holm"
    )[1]
    result["significant_at_0_05"] = result["p_adjusted_holm_within_wave"] < 0.05
    return result


def model_form_checks(
    data: pd.DataFrame,
    source_maps: np.ndarray,
    permutation_output: pd.DataFrame,
    wave: str,
    exact: bool,
) -> pd.DataFrame:
    """Dimension rank checks plus an explicitly stratified item-level rank check."""
    canonical, signatures, signature_index, dimension_index = canonical_permutation_layout(data)
    rows: list[dict[str, Any]] = []

    def p_from_null(observed: float, null: np.ndarray) -> float:
        if exact:
            return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
        return float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))

    specs = {
        spec.level: spec
        for spec in make_effect_specs(canonical, wave)
        if spec.family == "dimension"
    }
    dimension_row_indices: list[int] = []
    for dimension in sorted(DIMENSIONS):
        spec = specs[dimension]
        subset = canonical.loc[spec.mask].reset_index(drop=True)
        x = subset["api_z"].to_numpy(dtype=float)
        y = subset["human_z"].to_numpy(dtype=float)
        observed = safe_correlation(spearmanr, x, y)
        z = control_matrix(subset, spec.controls)
        fitted = z @ np.linalg.lstsq(z, y, rcond=None)[0]
        residual = y - fitted
        target = subset["stimulus_signature_sha256"].map(signature_index).to_numpy(dtype=int)
        residual_lookup = np.full(len(signatures), np.nan, dtype=float)
        residual_lookup[target] = residual
        null = np.empty(len(source_maps), dtype=float)
        x_rank = rankdata(x, method="average")
        x_rank -= x_rank.mean()
        x_norm = float(np.sqrt(x_rank @ x_rank))
        for start in range(0, len(source_maps), 250):
            stop = min(start + 250, len(source_maps))
            residual_permuted = residual_lookup[source_maps[start:stop][:, target]]
            if np.isnan(residual_permuted).any():
                raise AssertionError(f"Dimension Spearman map escaped subset: {dimension}")
            y_star = fitted[None, :] + residual_permuted
            y_rank = rankdata(y_star, method="average", axis=1)
            y_rank -= y_rank.mean(axis=1, keepdims=True)
            null[start:stop] = (y_rank @ x_rank) / (
                x_norm * np.sqrt(np.sum(y_rank**2, axis=1))
            )
        permutation_output[f"model_form::dimension_spearman::{dimension}"] = null
        dimension_row_indices.append(len(rows))
        rows.append(
            {
                "wave": wave,
                "scope": "dimension",
                "level": dimension,
                "model_form": "dimension-specific item-level Spearman",
                "definition": (
                    "Spearman correlation across stimuli within one dimension; p uses "
                    "that dimension's controls-only FL residuals and the shared joint-stimulus maps"
                ),
                "effect": observed,
                "two_sided_permutation_p": p_from_null(observed, null),
                "p_adjusted_bh": np.nan,
                "n_permutations": int(len(null)),
                "exact": bool(exact),
                "inference_role": "dimension_model_form_BH_family",
                "practically_material_abs_ge_0_20": bool(abs(observed) >= 0.20),
            }
        )
    adjusted = multipletests(
        [rows[index]["two_sided_permutation_p"] for index in dimension_row_indices],
        method="fdr_bh",
    )[1]
    for index, value in zip(dimension_row_indices, adjusted):
        rows[index]["p_adjusted_bh"] = float(value)

    n_stimuli = len(signatures)
    api_item = canonical["api_z"].to_numpy(dtype=float).reshape(
        n_stimuli, len(DIMENSIONS)
    ).mean(axis=1)
    human_item = canonical["human_z"].to_numpy(dtype=float).reshape(
        n_stimuli, len(DIMENSIONS)
    ).mean(axis=1)
    item_observed = safe_correlation(spearmanr, api_item, human_item)
    item_null = permutation_output["model_form::item_spearman"].to_numpy(dtype=float)
    rows.append(
        {
            "wave": wave,
            "scope": "six_dimension_mean",
            "level": "all",
            "model_form": "unadjusted descriptive Spearman of six-dimension item means",
            "definition": (
                "Spearman correlation of per-stimulus means of the six fixed z scores; "
                "this descriptive composite is not the stacked fixed-effect estimand"
            ),
            "effect": item_observed,
            "two_sided_permutation_p": p_from_null(item_observed, item_null),
            "p_adjusted_bh": np.nan,
            "n_permutations": int(len(item_null)),
            "exact": bool(exact),
            "inference_role": "unadjusted_descriptive_not_comparable_to_primary_beta",
            "practically_material_abs_ge_0_20": bool(abs(item_observed) >= 0.20),
        }
    )

    stimulus_meta = canonical.drop_duplicates("stimulus_signature_sha256").reset_index(drop=True)
    strata = (
        stimulus_meta["domain"].astype(str) + "||" + stimulus_meta["task"].astype(str)
    ).to_numpy()

    def within_stratum_centered_ranks(values: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=float)
        for level in sorted(set(strata)):
            positions = np.flatnonzero(strata == level)
            local = rankdata(values[positions], method="average")
            result[positions] = local - local.mean()
        return result

    api_within_rank = within_stratum_centered_ranks(api_item)
    human_within_rank = within_stratum_centered_ranks(human_item)
    denominator = float(
        np.sqrt(api_within_rank @ api_within_rank)
        * np.sqrt(human_within_rank @ human_within_rank)
    )
    within_observed = float((api_within_rank @ human_within_rank) / denominator)
    within_null = (human_within_rank[source_maps] @ api_within_rank) / denominator
    permutation_output["model_form::within_domain_task_rank"] = within_null
    rows.append(
        {
            "wave": wave,
            "scope": "within_domain_task_item",
            "level": "all",
            "model_form": "within-domain-by-task stratified item-level rank association",
            "definition": (
                "Pearson correlation of ranks centered separately inside each domain-by-task "
                "stratum for the six-dimension item means; human ranks are restricted-permuted"
            ),
            "effect": within_observed,
            "two_sided_permutation_p": p_from_null(within_observed, within_null),
            "p_adjusted_bh": np.nan,
            "n_permutations": int(len(within_null)),
            "exact": bool(exact),
            "inference_role": "stratified_partial_item_model_form",
            "practically_material_abs_ge_0_20": bool(abs(within_observed) >= 0.20),
        }
    )
    return pd.DataFrame(rows)


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    vx = float(np.var(x, ddof=0))
    vy = float(np.var(y, ddof=0))
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = vx + vy + float((x.mean() - y.mean()) ** 2)
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


def safe_correlation(function: Any, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(function(x, y).statistic)


def agreement_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", data)]
    for family in ("dimension", "task", "domain"):
        groups.extend(
            (family, str(level), group)
            for level, group in data.groupby(family, sort=True)
        )
    for family, level, group in groups:
        human = group["human_score"].to_numpy(dtype=float)
        api = group["api_score"].to_numpy(dtype=float)
        error = api - human
        rows.append(
            {
                "wave": str(group["wave"].iloc[0]),
                "family": family,
                "level": level,
                "n_rows": int(len(group)),
                "n_stimuli": int(group["stimulus_signature_sha256"].nunique()),
                "human_mean": float(human.mean()),
                "api_mean": float(api.mean()),
                "signed_error_api_minus_human": float(error.mean()),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "pearson_r": safe_correlation(pearsonr, human, api),
                "spearman_rho": safe_correlation(spearmanr, human, api),
                "lin_ccc": concordance_correlation(human, api),
                "within_0_20_raw": float(np.mean(np.abs(error) <= 0.20)),
                "within_0_50_raw": float(np.mean(np.abs(error) <= 0.50)),
            }
        )
    return rows


def dispersion_disagreement_analysis(
    data: pd.DataFrame,
    source_maps: np.ndarray,
    wave: str,
    n_bootstrap: int,
    seed: int,
    exact: bool,
) -> pd.DataFrame:
    """Relate human within-stimulus dispersion to absolute human--API error."""
    canonical, signatures, signature_index, _dimension_index = canonical_permutation_layout(data)
    min_raters = canonical.groupby("stimulus_signature_sha256")["n_human_raters"].min()
    eligible_signatures = sorted(min_raters[min_raters >= 2].index.tolist())
    if not eligible_signatures:
        raise AssertionError(f"No dispersion-identifiable stimuli in {wave}")
    eligible = canonical[
        canonical["stimulus_signature_sha256"].isin(eligible_signatures)
    ].sort_values(["stimulus_signature_sha256", "dimension"]).reset_index(drop=True)
    n_stimuli = len(eligible_signatures)
    if len(eligible) != n_stimuli * len(DIMENSIONS):
        raise AssertionError("Dispersion panel is not stimulus by six dimensions")
    global_positions = np.array(
        [signature_index[value] for value in eligible_signatures], dtype=int
    )
    local_lookup = np.full(len(signatures), -1, dtype=int)
    local_lookup[global_positions] = np.arange(n_stimuli)
    restricted_maps = local_lookup[source_maps[:, global_positions]]
    if (restricted_maps < 0).any():
        raise AssertionError(
            "A restricted dispersion map moved an eligible stimulus to an ineligible stratum"
        )

    human_dispersion = eligible["human_sd"].to_numpy(dtype=float).reshape(
        n_stimuli, len(DIMENSIONS)
    )
    absolute_disagreement = np.abs(
        eligible["human_score"].to_numpy(dtype=float)
        - eligible["api_score"].to_numpy(dtype=float)
    ).reshape(n_stimuli, len(DIMENSIONS))
    stimulus_meta = eligible.drop_duplicates("stimulus_signature_sha256").reset_index(drop=True)
    cluster_labels = (
        stimulus_meta["domain"].astype(str)
        + "||"
        + stimulus_meta["concept_id"].astype(str)
    ).to_numpy()
    clusters_by_domain: dict[str, list[str]] = {}
    for domain in sorted(stimulus_meta["domain"].unique()):
        clusters_by_domain[str(domain)] = sorted(
            set(cluster_labels[stimulus_meta["domain"].to_numpy() == domain])
        )
    cluster_positions = {
        label: np.flatnonzero(cluster_labels == label)
        for labels in clusters_by_domain.values()
        for label in labels
    }
    rng = np.random.default_rng(seed)
    bootstrap_indices: list[np.ndarray] = []
    for _draw in range(n_bootstrap):
        positions: list[np.ndarray] = []
        for labels in clusters_by_domain.values():
            sampled = rng.choice(labels, size=len(labels), replace=True)
            positions.extend(cluster_positions[str(label)] for label in sampled)
        bootstrap_indices.append(np.concatenate(positions))

    def rowwise_spearman(fixed: np.ndarray, permuted: np.ndarray) -> np.ndarray:
        fixed_rank = rankdata(fixed, method="average")
        fixed_rank -= fixed_rank.mean()
        fixed_norm = float(np.sqrt(fixed_rank @ fixed_rank))
        permuted_rank = rankdata(permuted, method="average", axis=1)
        permuted_rank -= permuted_rank.mean(axis=1, keepdims=True)
        return (permuted_rank @ fixed_rank) / (
            fixed_norm * np.sqrt(np.sum(permuted_rank**2, axis=1))
        )

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, np.ndarray, np.ndarray, str, str]] = []
    dimensions = sorted(DIMENSIONS)
    for position, dimension in enumerate(dimensions):
        scopes.append(
            (
                "dimension",
                dimension,
                human_dispersion[:, position],
                absolute_disagreement[:, position],
                "sample SD across retained participant ratings for this dimension",
                "absolute difference between human consensus and equivalent-UID API mean",
            )
        )
    scopes.append(
        (
            "overall",
            "all",
            np.sqrt(np.mean(human_dispersion**2, axis=1)),
            np.mean(absolute_disagreement, axis=1),
            "root-mean-square of the six dimension-specific participant sample SDs",
            "mean absolute human-consensus minus API disagreement across six dimensions",
        )
    )
    for scope, level, dispersion, disagreement, dispersion_definition, disagreement_definition in scopes:
        observed = safe_correlation(spearmanr, dispersion, disagreement)
        if scope == "dimension":
            dimension_position = dimensions.index(level)
            permuted_disagreement = absolute_disagreement[
                restricted_maps, dimension_position
            ]
        else:
            permuted_disagreement = np.mean(
                absolute_disagreement[restricted_maps], axis=2
            )
        null = rowwise_spearman(dispersion, permuted_disagreement)
        if exact:
            p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
        else:
            p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
        bootstrap_values = np.empty(n_bootstrap, dtype=float)
        for draw, indices in enumerate(bootstrap_indices):
            bootstrap_values[draw] = safe_correlation(
                spearmanr, dispersion[indices], disagreement[indices]
            )
        valid = bootstrap_values[np.isfinite(bootstrap_values)]
        rows.append(
            {
                "wave": wave,
                "scope": scope,
                "level": level,
                "n_stimuli": int(n_stimuli),
                "excluded_stimuli_with_fewer_than_two_raters": int(
                    len(signatures) - n_stimuli
                ),
                "human_dispersion_definition": dispersion_definition,
                "absolute_disagreement_definition": disagreement_definition,
                "spearman_rho": observed,
                "bootstrap_ci_low": float(np.quantile(valid, 0.025)),
                "bootstrap_ci_high": float(np.quantile(valid, 0.975)),
                "bootstrap_valid_draws": int(len(valid)),
                "bootstrap_invalid_proportion": float(1.0 - len(valid) / n_bootstrap),
                "two_sided_restricted_permutation_p": p_value,
                "p_adjusted_bh": np.nan,
                "n_permutations": int(len(null)),
                "exact_permutation": bool(exact),
                "inference_role": (
                    "prespecified_secondary_first_review"
                    if wave == WAVE_FIRST
                    else "exploratory_selected_second_review"
                ),
            }
        )
    result = pd.DataFrame(rows)
    dimension_mask = result["scope"] == "dimension"
    result.loc[dimension_mask, "p_adjusted_bh"] = multipletests(
        result.loc[dimension_mask, "two_sided_restricted_permutation_p"],
        method="fdr_bh",
    )[1]
    return result


def cross_wave_overlap(item_dimension: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = item_dimension[item_dimension["wave"] == WAVE_FIRST].copy()
    second = item_dimension[item_dimension["wave"] == WAVE_SECOND].copy()
    expected_signatures = set(first["stimulus_signature_sha256"]) & set(
        second["stimulus_signature_sha256"]
    )
    if not expected_signatures:
        raise AssertionError("Final panels contain no cross-wave stimulus overlap")
    first_cols = [
        "stimulus_signature_sha256",
        "dimension",
        "domain",
        "task",
        "concept_id",
        "human_score",
        "n_human_raters",
    ]
    paired = first[first_cols].merge(
        second[first_cols + ["selection_role"]],
        on=["stimulus_signature_sha256", "dimension"],
        how="inner",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    observed_signatures = set(paired["stimulus_signature_sha256"])
    expected_rows = len(expected_signatures) * len(DIMENSIONS)
    if observed_signatures != expected_signatures or len(paired) != expected_rows:
        raise AssertionError(
            "Cross-wave overlap must contain every observed shared stimulus exactly once "
            f"per dimension: expected {len(expected_signatures)} x {len(DIMENSIONS)}, "
            f"got {paired['stimulus_signature_sha256'].nunique()} signatures and {len(paired)} rows"
        )
    api_lookup = item_dimension[
        ["stimulus_signature_sha256", "dimension", "api_score"]
    ].drop_duplicates()
    paired = paired.merge(
        api_lookup,
        on=["stimulus_signature_sha256", "dimension"],
        how="left",
        validate="one_to_one",
    )
    paired["second_minus_first"] = paired["human_score_second"] - paired["human_score_first"]
    summaries: list[dict[str, Any]] = []
    for level, group in [("all", paired)] + list(paired.groupby("dimension", sort=True)):
        x = group["human_score_first"].to_numpy(dtype=float)
        y = group["human_score_second"].to_numpy(dtype=float)
        summaries.append(
            {
                "dimension": str(level),
                "n_rows": int(len(group)),
                "n_stimuli": int(group["stimulus_signature_sha256"].nunique()),
                "first_mean": float(x.mean()),
                "second_mean": float(y.mean()),
                "mean_second_minus_first": float((y - x).mean()),
                "mae_between_waves": float(np.abs(y - x).mean()),
                "pearson_r": safe_correlation(pearsonr, x, y),
                "spearman_rho": safe_correlation(spearmanr, x, y),
                "lin_ccc": concordance_correlation(x, y),
            }
        )
    return paired, pd.DataFrame(summaries)


def one_sensitivity_row(
    data: pd.DataFrame,
    wave: str,
    family: str,
    scenario: str,
    detail: str,
) -> dict[str, Any]:
    controls = MAIN_CONTROLS + ((SECOND_EXTRA_CONTROL,) if wave == WAVE_SECOND else ())
    estimate = standardized_fixed_slope(data, controls)
    return {
        "sensitivity_family": family,
        "scenario": scenario,
        "detail": detail,
        "inference_role": "descriptive_HC3_only",
        "wave": wave,
        "n_rows": int(len(data)),
        "n_stimuli": int(data["stimulus_signature_sha256"].nunique()),
        "controls": "+".join(controls),
        "slope": estimate["slope"],
        "ci_low_hc3": estimate["ci_low_hc3"],
        "ci_high_hc3": estimate["ci_high_hc3"],
        "p_hc3_unadjusted_descriptive": estimate["p_hc3"],
    }


def build_threshold_refit_sensitivity(
    panels: dict[str, pd.DataFrame],
    crosswalk: pd.DataFrame,
    api_long: pd.DataFrame,
) -> pd.DataFrame:
    """Refit first-review standardization at each repeat-similarity threshold.

    The main sensitivity table keeps primary standardization constants frozen so
    one-factor changes stay on one scale.  This companion table answers a separate
    interpretive question: whether the standardized point estimate crosses the
    .20 materiality reference when each threshold defines its own human panel.
    """
    api_mean = aggregate_api(
        api_long, method="mean", minimum_judges_per_uid=9, include_provenance=False
    )
    scenarios = (
        ("threshold_70", 70),
        ("final_primary", 75),
        ("threshold_80", 80),
        ("threshold_90", 90),
        ("threshold_95", 95),
    )
    rows: list[dict[str, Any]] = []
    for scenario, threshold in scenarios:
        merged_raw = merge_panel_api(panels[scenario], crosswalk, api_mean)
        refit, _constants = freeze_within_dimension_standardization(merged_raw)
        first = refit[refit["wave"] == WAVE_FIRST].reset_index(drop=True)
        row = one_sensitivity_row(
            first,
            WAVE_FIRST,
            "QC threshold refit",
            scenario,
            "scenario-specific wave-by-dimension standardization; descriptive HC3 only",
        )
        row["threshold_pct"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def build_sensitivities(
    panels: dict[str, pd.DataFrame],
    crosswalk: pd.DataFrame,
    api_long: pd.DataFrame,
    standardization_constants: pd.DataFrame,
    second_item_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    api_mean = aggregate_api(
        api_long, method="mean", minimum_judges_per_uid=9, include_provenance=False
    )
    qc_applicability = {
        "final_primary": {WAVE_FIRST, WAVE_SECOND},
        "threshold_70": {WAVE_FIRST},
        "threshold_80": {WAVE_FIRST},
        "threshold_90": {WAVE_FIRST},
        "threshold_95": {WAVE_FIRST},
        "attention_only": {WAVE_FIRST, WAVE_SECOND},
        "strict_first_both_pairs": {WAVE_FIRST},
        "second_case05_case07_mean": {WAVE_SECOND},
    }
    for scenario, panel in panels.items():
        merged = apply_frozen_standardization(
            merge_panel_api(panel, crosswalk, api_mean), standardization_constants
        )
        for wave in sorted(qc_applicability[scenario]):
            subset = merged[merged["wave"] == wave].reset_index(drop=True)
            if not subset.empty:
                rows.append(one_sensitivity_row(subset, wave, "QC panel", scenario, "frozen one-factor QC change"))

    primary_panel = panels["final_primary"]
    api_variants = [
        ("ensemble_mean", aggregate_api(api_long, "mean", 9, include_provenance=False), "all current ratings; complete 9/9 per UID"),
        ("equivalence_class_median", aggregate_api(api_long, "median", 9, include_provenance=False), "median across all current ratings from every text-equivalent UID"),
        ("coverage_complete_9_of_9", aggregate_api(api_long, "mean", 9, include_provenance=False), "retain UIDs with 9/9 current judges"),
        ("coverage_at_least_7_of_9", aggregate_api(api_long, "mean", 7, include_provenance=False), "retain UIDs with >=7/9 current judges"),
    ]
    for scenario, api_agg, detail in api_variants:
        merged = apply_frozen_standardization(
            merge_panel_api(primary_panel, crosswalk, api_agg), standardization_constants
        )
        for wave in (WAVE_FIRST, WAVE_SECOND):
            subset = merged[merged["wave"] == wave].reset_index(drop=True)
            rows.append(one_sensitivity_row(subset, wave, "API aggregation", scenario, detail))

    raw_api = aggregate_api(
        api_long, "mean", 9, raw_risk=True, include_provenance=False
    )
    raw_merged = apply_frozen_standardization(
        merge_panel_api(primary_panel, crosswalk, raw_api, raw_risk=True),
        standardization_constants,
        raw_risk=True,
    )
    for wave in (WAVE_FIRST, WAVE_SECOND):
        subset = raw_merged[raw_merged["wave"] == wave].reset_index(drop=True)
        controls = tuple(c for c in MAIN_CONTROLS if c != "dimension") + (
            (SECOND_EXTRA_CONTROL,) if wave == WAVE_SECOND else ()
        )
        estimate = standardized_fixed_slope(subset, controls)
        rows.append(
            {
                "sensitivity_family": "risk direction",
                "scenario": "risk_both_raw_high_is_risk",
                "detail": "both human and API risk left in raw high-is-risk direction",
                "inference_role": "descriptive_HC3_only",
                "wave": wave,
                "n_rows": int(len(subset)),
                "n_stimuli": int(subset["stimulus_signature_sha256"].nunique()),
                "controls": "+".join(controls),
                "slope": estimate["slope"],
                "ci_low_hc3": estimate["ci_low_hc3"],
                "ci_high_hc3": estimate["ci_high_hc3"],
                "p_hc3_unadjusted_descriptive": estimate["p_hc3"],
            }
        )

    judges = sorted(api_long["paper_model_label"].unique().tolist())
    for judge in judges:
        api_agg = aggregate_api(
            api_long, "mean", 8, excluded_judge=judge, include_provenance=False
        )
        merged = apply_frozen_standardization(
            merge_panel_api(primary_panel, crosswalk, api_agg), standardization_constants
        )
        for wave in (WAVE_FIRST, WAVE_SECOND):
            subset = merged[merged["wave"] == wave].reset_index(drop=True)
            rows.append(one_sensitivity_row(subset, wave, "leave-one-judge-out", f"exclude::{judge}", "eight-judge equivalent-stimulus mean"))

    generators = sorted(api_long["generator"].unique().tolist())
    for generator in generators:
        api_agg = aggregate_api(
            api_long, "mean", 9, excluded_generator=generator, include_provenance=False
        )
        merged = apply_frozen_standardization(
            merge_panel_api(primary_panel, crosswalk, api_agg), standardization_constants
        )
        for wave in (WAVE_FIRST, WAVE_SECOND):
            subset = merged[merged["wave"] == wave].reset_index(drop=True)
            if not subset.empty:
                rows.append(one_sensitivity_row(subset, wave, "leave-one-generator-out", f"exclude::{generator}", "generator omitted before stimulus-signature pooling"))

    main = apply_frozen_standardization(
        merge_panel_api(primary_panel, crosswalk, api_mean), standardization_constants
    )
    scheduled = apply_frozen_standardization(
        merge_panel_api_scheduled_uid(primary_panel, crosswalk, api_long),
        standardization_constants,
    )
    for wave in (WAVE_FIRST, WAVE_SECOND):
        subset = scheduled[scheduled["wave"] == wave].reset_index(drop=True)
        rows.append(
            one_sensitivity_row(
                subset,
                wave,
                "API aggregation",
                "scheduled_source_uid_mean",
                "nine current judges for the crosswalk-scheduled source UID only",
            )
        )
    for wave in (WAVE_FIRST, WAVE_SECOND):
        wave_data = main[main["wave"] == wave].reset_index(drop=True)
        for domain in sorted(wave_data["domain"].unique()):
            subset = wave_data[wave_data["domain"] != domain].reset_index(drop=True)
            rows.append(one_sensitivity_row(subset, wave, "leave-one-domain-out", f"exclude::{domain}", "analysis sample omits one domain"))
    first_main = main[main["wave"] == WAVE_FIRST].reset_index(drop=True)
    concept_cluster = (
        first_main["domain"].astype(str) + "||" + first_main["concept_id"].astype(str)
    )
    for label in sorted(concept_cluster.unique()):
        subset = first_main[concept_cluster != label].reset_index(drop=True)
        rows.append(
            one_sensitivity_row(
                subset,
                WAVE_FIRST,
                "leave-one-concept-out",
                f"exclude::{label}",
                "one domain-by-concept cluster omitted",
            )
        )
    p05 = crosswalk[
        (crosswalk["wave"] == WAVE_SECOND) & (crosswalk["item_local_id"] == "P05")
    ]
    if len(p05) != 1:
        raise AssertionError(f"Expected exactly one second-review P05 mapping, found {len(p05)}")
    p05_signature = str(p05.iloc[0]["stimulus_signature_sha256"])
    second_without_p05 = main[
        (main["wave"] == WAVE_SECOND)
        & (main["stimulus_signature_sha256"] != p05_signature)
    ].reset_index(drop=True)
    rows.append(
        one_sensitivity_row(
            second_without_p05,
            WAVE_SECOND,
            "fixed item exclusion",
            "exclude_P05",
            "remove the prespecified second-review P05 stimulus",
        )
    )
    item_draws = second_item_bootstrap["overall::all"].dropna().to_numpy(dtype=float)
    second_main = main[main["wave"] == WAVE_SECOND].reset_index(drop=True)
    item_row = one_sensitivity_row(
        second_main,
        WAVE_SECOND,
        "bootstrap cluster unit",
        "second_domain_stratified_stimulus_bootstrap",
        "six stimuli resampled with replacement within each domain; fixed original z constants",
    )
    item_row["ci_low_hc3"] = float(np.quantile(item_draws, 0.025))
    item_row["ci_high_hc3"] = float(np.quantile(item_draws, 0.975))
    item_row["p_hc3_unadjusted_descriptive"] = np.nan
    item_row["inference_role"] = "descriptive_domain_stratified_item_bootstrap"
    rows.append(item_row)
    return pd.DataFrame(rows)


def leave_one_cluster_influence(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full = standardized_fixed_slope(data, MAIN_CONTROLS)["slope"]
    cluster = data["domain"].astype(str) + "||" + data["concept_id"].astype(str)
    for label in sorted(cluster.unique()):
        subset = data[cluster != label].reset_index(drop=True)
        estimate = standardized_fixed_slope(subset, MAIN_CONTROLS)
        domain, concept = label.split("||", 1)
        rows.append(
            {
                "excluded_domain": domain,
                "excluded_concept_id": concept,
                "n_rows": int(len(subset)),
                "n_stimuli": int(subset["stimulus_signature_sha256"].nunique()),
                "full_slope": full,
                "leave_one_cluster_slope": estimate["slope"],
                "delta_from_full": estimate["slope"] - full,
                "sign_reversal": bool(np.sign(estimate["slope"]) != np.sign(full)),
            }
        )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    if compressed:
        frame.to_csv(path, index=False, encoding="utf-8", compression="gzip")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "scipy", "statsmodels"):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-scores", help="Explicit current api_test_scores_7290.csv")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATION)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Development-only run with 100 bootstrap and 200 random permutations; exact 2^15 remains complete.",
    )
    args = parser.parse_args()
    n_bootstrap = 100 if args.quick else args.bootstrap
    n_permutation = 200 if args.quick else args.permutations
    if n_bootstrap <= 0 or n_permutation <= 0:
        raise ValueError("Resample counts must be positive")

    project_root = Path(__file__).resolve().parents[1]
    analysis = project_root / "analysis"
    api_path = find_current_api_scores(project_root, args.api_scores)
    (
        cleaned,
        crosswalk,
        equivalence,
        panels,
        api,
        final_cleaning_manifest,
        final_cleaning_manifest_path,
    ) = load_and_validate_inputs(project_root, api_path)
    api_long = make_api_long(api, equivalence)
    api_main = aggregate_api(api_long, method="mean", minimum_judges_per_uid=9)
    item_dimension_raw = merge_panel_api(cleaned, crosswalk, api_main)
    item_dimension, standardization_constants = freeze_within_dimension_standardization(
        item_dimension_raw
    )
    first = item_dimension[item_dimension["wave"] == WAVE_FIRST].reset_index(drop=True)
    second = item_dimension[item_dimension["wave"] == WAVE_SECOND].reset_index(drop=True)
    for wave, frame in ((WAVE_FIRST, first), (WAVE_SECOND, second)):
        n_stimuli = int(frame["stimulus_signature_sha256"].nunique())
        if n_stimuli == 0:
            raise AssertionError(f"Final {wave} panel has no matched stimuli")
        if len(frame) != n_stimuli * len(DIMENSIONS):
            raise AssertionError(
                f"Final {wave} panel is not a complete stimulus-by-six-dimension matrix: "
                f"{n_stimuli} stimuli, {len(frame)} rows"
            )
        dimension_counts = frame.groupby("stimulus_signature_sha256")[
            "dimension"
        ].nunique()
        if not (dimension_counts == len(DIMENSIONS)).all():
            raise AssertionError(f"Final {wave} panel has incomplete dimension coverage")

    first_points, first_specs = effect_point_table(first, WAVE_FIRST)
    second_points, second_specs = effect_point_table(second, WAVE_SECOND)
    bootstrap = stratified_cluster_bootstrap(
        first, first_specs, n_bootstrap, SEED_BOOTSTRAP, "concept_id"
    )
    second_bootstrap = stratified_cluster_bootstrap(
        second, second_specs, n_bootstrap, SEED_SECOND_BOOTSTRAP, "concept_id"
    )
    second_item_bootstrap = stratified_cluster_bootstrap(
        second,
        second_specs,
        n_bootstrap,
        SEED_SECOND_ITEM_BOOTSTRAP,
        "stimulus_signature_sha256",
    )
    first_permutation, first_permutation_qa, first_source_maps = first_freedman_lane(
        first, first_specs, n_permutation, SEED_PERMUTATION
    )
    second_permutation, pair_manifest, second_source_maps = second_exact_pair_permutation(
        second, second_specs
    )
    first_effects = add_inference_to_effects(
        first_points,
        bootstrap,
        first_permutation,
        "spec-specific Freedman-Lane reduced-model residual permutation with shared joint-six-dimension domain-by-task maps",
        "domain-stratified domain-by-concept cluster percentile bootstrap",
    )
    second_effects = add_inference_to_effects(
        second_points,
        second_bootstrap,
        second_permutation,
        "exact 2^15 spec-specific Freedman-Lane residual swaps with shared joint-six-dimension domain-by-task maps",
        "descriptive domain-stratified domain-by-concept cluster percentile bootstrap",
    )
    effects = pd.concat([first_effects, second_effects], ignore_index=True)

    first_heterogeneity = heterogeneity_tests(
        first, first_source_maps, WAVE_FIRST, exact=False
    )
    second_heterogeneity = heterogeneity_tests(
        second, second_source_maps, WAVE_SECOND, exact=True
    )
    heterogeneity = pd.concat(
        [first_heterogeneity, second_heterogeneity], ignore_index=True
    )
    effects = effects.merge(
        heterogeneity[[
            "wave",
            "family",
            "permutation_p",
            "p_adjusted_holm_within_wave",
            "significant_at_0_05",
        ]].rename(
            columns={
                "permutation_p": "omnibus_heterogeneity_p",
                "p_adjusted_holm_within_wave": "omnibus_heterogeneity_p_holm",
                "significant_at_0_05": "omnibus_heterogeneity_significant",
            }
        ),
        on=["wave", "family"],
        how="left",
        validate="many_to_one",
    )

    model_form = pd.concat(
        [
            model_form_checks(
                first,
                first_source_maps,
                first_permutation,
                WAVE_FIRST,
                exact=False,
            ),
            model_form_checks(
                second,
                second_source_maps,
                second_permutation,
                WAVE_SECOND,
                exact=True,
            ),
        ],
        ignore_index=True,
    )

    agreement = pd.DataFrame(agreement_rows(first) + agreement_rows(second))
    dispersion_disagreement = pd.concat(
        [
            dispersion_disagreement_analysis(
                first,
                first_source_maps,
                WAVE_FIRST,
                n_bootstrap,
                SEED_DISPERSION_BOOTSTRAP,
                exact=False,
            ),
            dispersion_disagreement_analysis(
                second,
                second_source_maps,
                WAVE_SECOND,
                n_bootstrap,
                SEED_DISPERSION_BOOTSTRAP,
                exact=True,
            ),
        ],
        ignore_index=True,
    )
    overlap, overlap_summary = cross_wave_overlap(item_dimension)
    sensitivity = build_sensitivities(
        panels,
        crosswalk,
        api_long,
        standardization_constants,
        second_item_bootstrap,
    )
    threshold_refit = build_threshold_refit_sensitivity(
        panels,
        crosswalk,
        api_long,
    )
    model_sensitivity_rows = []
    for row in model_form.to_dict(orient="records"):
        model_sensitivity_rows.append(
            {
                "sensitivity_family": "model form",
                "scenario": f"{row['scope']}::{row['level']}",
                "detail": row["definition"],
                "inference_role": row["inference_role"],
                "wave": row["wave"],
                "n_rows": np.nan,
                "n_stimuli": (
                    int(first["stimulus_signature_sha256"].nunique())
                    if row["wave"] == WAVE_FIRST
                    else int(second["stimulus_signature_sha256"].nunique())
                ),
                "controls": "rank-based stratified model-form check",
                "slope": row["effect"],
                "ci_low_hc3": np.nan,
                "ci_high_hc3": np.nan,
                "p_hc3_unadjusted_descriptive": row["two_sided_permutation_p"],
            }
        )
    sensitivity = pd.concat(
        [sensitivity, pd.DataFrame(model_sensitivity_rows)], ignore_index=True
    )
    influence = leave_one_cluster_influence(first)
    respondent_data = respondent_alignment_data(
        cleaned, crosswalk, api_main, standardization_constants
    )
    mixed_attempt = mixed_model_attempt(respondent_data)
    ordinal_attempt = ordinal_gee_attempt(respondent_data)

    paths = {
        "api_by_stimulus_dimension": analysis / "alignment_api_by_stimulus_dimension.csv",
        "human_api_item_dimension": analysis / "alignment_human_api_item_dimension.csv",
        "standardization_constants": analysis / "alignment_standardization_constants.csv",
        "effects": analysis / "alignment_effects.csv",
        "heterogeneity": analysis / "alignment_heterogeneity.csv",
        "model_form_checks": analysis / "alignment_model_form_checks.csv",
        "absolute_agreement": analysis / "alignment_absolute_agreement.csv",
        "dispersion_disagreement": analysis / "alignment_dispersion_disagreement.csv",
        "cross_wave_overlap": analysis / "alignment_cross_wave_overlap.csv",
        "cross_wave_overlap_summary": analysis / "alignment_cross_wave_overlap_summary.csv",
        "sensitivity": analysis / "alignment_sensitivity.csv",
        "threshold_refit_sensitivity": analysis / "alignment_threshold_refit.csv",
        "leave_one_cluster": analysis / "alignment_leave_one_cluster.csv",
        "first_bootstrap_draws": analysis / "alignment_first_bootstrap_draws.csv.gz",
        "second_concept_bootstrap_draws": analysis / "alignment_second_concept_bootstrap_draws.csv.gz",
        "second_item_bootstrap_draws": analysis / "alignment_second_item_bootstrap_draws.csv.gz",
        "first_permutation_draws": analysis / "alignment_first_permutation_draws.csv.gz",
        "second_exact_draws": analysis / "alignment_second_exact_draws.csv.gz",
        "validation": analysis / "alignment_validation.json",
        "mixed_model_attempt": analysis / "alignment_mixed_model_attempt.json",
        "ordinal_gee_attempt": analysis / "alignment_ordinal_gee_attempt.json",
        "manifest": analysis / "alignment_manifest.json",
    }
    write_csv(api_main, paths["api_by_stimulus_dimension"])
    write_csv(item_dimension, paths["human_api_item_dimension"])
    write_csv(standardization_constants, paths["standardization_constants"])
    write_csv(effects, paths["effects"])
    write_csv(heterogeneity, paths["heterogeneity"])
    write_csv(model_form, paths["model_form_checks"])
    write_csv(agreement, paths["absolute_agreement"])
    write_csv(dispersion_disagreement, paths["dispersion_disagreement"])
    write_csv(overlap, paths["cross_wave_overlap"])
    write_csv(overlap_summary, paths["cross_wave_overlap_summary"])
    write_csv(sensitivity, paths["sensitivity"])
    write_csv(threshold_refit, paths["threshold_refit_sensitivity"])
    write_csv(influence, paths["leave_one_cluster"])
    write_csv(bootstrap, paths["first_bootstrap_draws"], compressed=True)
    write_csv(second_bootstrap, paths["second_concept_bootstrap_draws"], compressed=True)
    write_csv(second_item_bootstrap, paths["second_item_bootstrap_draws"], compressed=True)
    write_csv(first_permutation, paths["first_permutation_draws"], compressed=True)
    write_csv(second_permutation, paths["second_exact_draws"], compressed=True)
    paths["mixed_model_attempt"].write_text(
        stable_json(mixed_attempt) + "\n", encoding="utf-8"
    )
    paths["ordinal_gee_attempt"].write_text(
        stable_json(ordinal_attempt) + "\n", encoding="utf-8"
    )

    first_overall = effects[(effects["wave"] == WAVE_FIRST) & (effects["family"] == "overall")].iloc[0]
    second_overall = effects[(effects["wave"] == WAVE_SECOND) & (effects["family"] == "overall")].iloc[0]
    first_family_sizes = {
        family: int(
            effects[(effects["wave"] == WAVE_FIRST) & (effects["family"] == family)][
                "level"
            ].nunique()
        )
        for family in ("dimension", "task", "domain")
    }
    second_family_sizes = {
        family: int(
            effects[(effects["wave"] == WAVE_SECOND) & (effects["family"] == family)][
                "level"
            ].nunique()
        )
        for family in ("dimension", "task", "domain")
    }
    expected_first_family_sizes = {
        family: int(first[family].nunique())
        for family in ("dimension", "task", "domain")
    }
    expected_second_family_sizes = {
        family: int(second[family].nunique())
        for family in ("dimension", "task", "domain")
    }
    if expected_first_family_sizes["dimension"] != len(DIMENSIONS):
        raise AssertionError("First final panel does not contain all six dimensions")
    if expected_second_family_sizes["dimension"] != len(DIMENSIONS):
        raise AssertionError("Second final panel does not contain all six dimensions")
    if first_family_sizes != expected_first_family_sizes:
        raise AssertionError(f"Unexpected first effect family sizes: {first_family_sizes}")
    if second_family_sizes != expected_second_family_sizes:
        raise AssertionError(f"Unexpected second effect family sizes: {second_family_sizes}")
    if (second_effects["permutation_p"] <= 0).any():
        raise AssertionError("A complete exact permutation p-value cannot be zero")
    first_clusters = int(first[["domain", "concept_id"]].drop_duplicates().shape[0])
    second_clusters = int(second[["domain", "concept_id"]].drop_duplicates().shape[0])
    if (
        first[["domain", "concept_id"]].isna().any().any()
        or second[["domain", "concept_id"]].isna().any().any()
        or first_clusters < first["domain"].nunique()
        or second_clusters < second["domain"].nunique()
    ):
        raise AssertionError(
            "Final panels have missing or degenerate domain-concept clusters: "
            f"first={first_clusters}, second={second_clusters}"
        )
    bootstrap_frames = {
        "first_concept_cluster": bootstrap,
        "second_concept_cluster": second_bootstrap,
        "second_item": second_item_bootstrap,
    }
    bootstrap_invalid_by_effect = {
        label: {
            column: float(frame[column].isna().mean())
            for column in frame.columns
            if column != "replicate"
        }
        for label, frame in bootstrap_frames.items()
    }
    bootstrap_invalid = {
        label: float(max(values.values(), default=0.0))
        for label, values in bootstrap_invalid_by_effect.items()
    }
    bootstrap_rank_warnings = {
        label: {
            effect: proportion
            for effect, proportion in values.items()
            if proportion > 0.01
        }
        for label, values in bootstrap_invalid_by_effect.items()
    }
    bootstrap_rank_warnings = {
        label: values for label, values in bootstrap_rank_warnings.items() if values
    }
    for label, effect_frame in [
        ("first_concept_cluster", first_effects),
        ("second_concept_cluster", second_effects),
    ]:
        warned = set(bootstrap_rank_warnings.get(label, {}))
        for _index, effect_row in effect_frame.iterrows():
            key = str(effect_row["effect_key"])
            should_report = key not in warned
            if bool(effect_row["bootstrap_routine_reportable"]) != should_report:
                raise AssertionError(
                    f"Bootstrap suppression mismatch for {label} {key}"
                )
    expected_heterogeneity = {
        (wave, family): int(frame[family].nunique())
        for wave, frame in ((WAVE_FIRST, first), (WAVE_SECOND, second))
        for family in ("task", "domain")
    }
    observed_heterogeneity = {
        (str(row["wave"]), str(row["family"])): int(row["n_levels"])
        for row in heterogeneity.to_dict(orient="records")
    }
    if observed_heterogeneity != expected_heterogeneity:
        raise AssertionError(
            "Omnibus heterogeneity output does not match observed final-panel levels: "
            f"{observed_heterogeneity} != {expected_heterogeneity}"
        )
    first_stratum_count = int(first[["domain", "task"]].drop_duplicates().shape[0])
    if (
        first_permutation_qa["n_strata"] != first_stratum_count
        or sum(first_permutation_qa["stratum_sizes"])
        != first["stimulus_signature_sha256"].nunique()
        or first_permutation_qa["minimum_stratum_size"] < 2
        or first_permutation_qa[
            "identity_permutation_matches_observed_max_abs_difference"
        ]
        > 1e-10
    ):
        raise AssertionError("First permutation stratum QA failed")
    expected_dimension_model_rows = len(DIMENSIONS) * 2
    expected_model_form_rows = expected_dimension_model_rows + 4
    if (
        len(model_form) != expected_model_form_rows
        or int((model_form["scope"] == "dimension").sum())
        != expected_dimension_model_rows
        or model_form.loc[
            model_form["scope"] == "dimension", "p_adjusted_bh"
        ].isna().any()
    ):
        raise AssertionError("Model-form dimension/stratified Spearman QA failed")
    expected_dispersion_rows = (len(DIMENSIONS) + 1) * 2
    if len(dispersion_disagreement) != expected_dispersion_rows or (
        dispersion_disagreement["bootstrap_invalid_proportion"] > 0.01
    ).any():
        raise AssertionError("Dispersion--disagreement QA failed")
    dispersion_counts = {
        str(wave): int(group["n_stimuli"].iloc[0])
        for wave, group in dispersion_disagreement.groupby("wave")
    }
    expected_dispersion_counts = {
        wave: int(
            (frame.groupby("stimulus_signature_sha256")["n_human_raters"].min() >= 2).sum()
        )
        for wave, frame in ((WAVE_FIRST, first), (WAVE_SECOND, second))
    }
    if dispersion_counts != expected_dispersion_counts:
        raise AssertionError(
            "Dispersion-identifiable stimulus counts do not match final panels: "
            f"{dispersion_counts} != {expected_dispersion_counts}"
        )
    if sensitivity["inference_role"].isna().any():
        raise AssertionError("Every sensitivity row must state its descriptive inference role")
    if len(threshold_refit) != 5 or not (
        threshold_refit["inference_role"] == "descriptive_HC3_only"
    ).all():
        raise AssertionError("Threshold-refit sensitivity must contain five descriptive rows")
    qa_status = "PASS"
    validation = {
        "qa_status": qa_status,
        "script_version": SCRIPT_VERSION,
        "quick_mode": bool(args.quick),
        "legacy_human_derivatives_read": False,
        "final_cleaning": {
            "rule_version": CLEANING_RULE_VERSION,
            "manifest_path": str(final_cleaning_manifest_path),
            "manifest_sha256": sha256_file(final_cleaning_manifest_path),
            "rules": final_cleaning_manifest["rules"],
            "formal_primary_is_final_primary": True,
        },
        "risk_coding": "human and API risk transformed as 6 - raw for all main unified-direction models",
        "standardization": {
            "rule": "within each wave-by-dimension on the original matched sample",
            "sd": "sample SD (ddof=1)",
            "frozen_during_all_bootstrap_and_permutation_draws": True,
            "constants_rows": int(len(standardization_constants)),
        },
        "stimulus_folding": {
            "unit": "stimulus_signature_sha256",
            "api_rule": "mean of all current judge ratings over every text-equivalent source UID",
            "human_rule": "mean placements within participant, then mean retained participants",
            "api_signature_count": int(api_main["stimulus_signature_sha256"].nunique()),
            "first_review_stimuli": int(first["stimulus_signature_sha256"].nunique()),
            "second_review_stimuli": int(second["stimulus_signature_sha256"].nunique()),
        },
        "generator_fixed_effect": {
            "included": False,
            "reason": "generator is not uniquely identifiable after pooling all text-equivalent UIDs by stimulus signature",
        },
        "first_inference": {
            "bootstrap_draws": n_bootstrap,
            "bootstrap_seed": SEED_BOOTSTRAP,
            "bootstrap_unit": "domain-by-concept cluster, resampled within domain",
            "permutations": n_permutation,
            "permutation_seed": SEED_PERMUTATION,
            "permutation_unit": (
                "shared joint-stimulus domain-by-task maps; every simple effect uses "
                "its own subset-specific controls-only Freedman-Lane reduced model"
            ),
            "qa": first_permutation_qa,
        },
        "second_inference": {
            **pair_manifest,
            "concept_cluster_bootstrap_draws": n_bootstrap,
            "concept_cluster_bootstrap_seed": SEED_SECOND_BOOTSTRAP,
            "concept_cluster_bootstrap_role": "descriptive CI for purposively selected second review",
            "item_bootstrap_sensitivity_draws": n_bootstrap,
            "item_bootstrap_sensitivity_seed": SEED_SECOND_ITEM_BOOTSTRAP,
        },
        "cross_wave_overlap": {
            "stimuli": int(overlap["stimulus_signature_sha256"].nunique()),
            "rows": int(len(overlap)),
        },
        "mixed_model_estimability": {
            "stable_for_primary_inference": bool(
                mixed_attempt["stable_for_primary_inference"]
            ),
            "fallback_used": bool(mixed_attempt["fallback_used"]),
            "objective_fallback_reasons": mixed_attempt["objective_fallback_reasons"],
        },
        "model_form_checks": {
            "model_form_rows": int(len(model_form)),
            "dimension_spearman_rows": int(
                (model_form["scope"] == "dimension").sum()
            ),
            "six_dimension_mean_rows_are_unadjusted_descriptive": bool(
                (
                    model_form.loc[
                        model_form["scope"] == "six_dimension_mean",
                        "inference_role",
                    ]
                    == "unadjusted_descriptive_not_comparable_to_primary_beta"
                ).all()
            ),
            "ordinal_gee_attempt_success_by_wave": {
                str(row["wave"]): bool(row.get("success"))
                for row in ordinal_attempt["waves"]
            },
        },
        "headline": {
            "first_standardized_slope": finite_float(first_overall["slope"]),
            "first_ci": [finite_float(first_overall["ci_low"]), finite_float(first_overall["ci_high"])],
            "first_permutation_p": finite_float(first_overall["permutation_p"]),
            "first_positive_reliable_rule": bool(first_overall["statistically_reliable_positive"]),
            "second_standardized_slope": finite_float(second_overall["slope"]),
            "second_descriptive_concept_cluster_bootstrap_ci": [finite_float(second_overall["ci_low"]), finite_float(second_overall["ci_high"])],
            "second_exact_p": finite_float(second_overall["permutation_p"]),
        },
        "checks": {
            "cleaned_rows": int(len(cleaned)),
            "crosswalk_rows": int(len(crosswalk)),
            "equivalence_rows": int(len(equivalence)),
            "api_rows": int(len(api)),
            "api_complete_9_of_9": True,
            "second_pairs_one_high_one_low": True,
            "second_exact_permutations": int(len(second_permutation)),
            "final_primary_equals_cleaned": True,
            "first_effect_family_sizes": first_family_sizes,
            "second_effect_family_sizes": second_family_sizes,
            "first_domain_concept_clusters": first_clusters,
            "second_domain_concept_clusters": second_clusters,
            "bootstrap_invalid_proportions": bootstrap_invalid,
            "bootstrap_invalid_proportions_by_effect": bootstrap_invalid_by_effect,
            "bootstrap_routine_reporting_threshold": "each effect must have <=1% invalid draws",
            "bootstrap_rank_deficiency_warnings": bootstrap_rank_warnings,
            "bootstrap_effects_above_threshold_are_suppressed": True,
            "heterogeneity_omnibus_rows": int(len(heterogeneity)),
            "heterogeneity_null_allows_common_nonzero_slope": True,
            "dispersion_disagreement_rows": int(len(dispersion_disagreement)),
            "dispersion_identifiable_stimuli_by_wave": dispersion_counts,
            "mixed_model_attempt_recorded": True,
            "ordinal_gee_attempt_recorded": True,
            "threshold_refit_rows": int(len(threshold_refit)),
        },
    }
    paths["validation"].write_text(stable_json(validation) + "\n", encoding="utf-8")

    input_paths = {
        "cleaned_human_ratings_long": analysis / "cleaned_human_ratings_long.csv",
        "human_api_crosswalk": analysis / "human_api_crosswalk.csv",
        "api_stimulus_equivalence_810": analysis / "api_stimulus_equivalence_810.csv",
        "current_api_test_scores_7290": api_path,
        "final_cleaning_manifest": final_cleaning_manifest_path,
        **{
            f"sensitivity_panel_{name}": analysis
            / "final_sensitivity_panels"
            / f"{name}.csv"
            for name in panels
        },
    }
    output_hashes = {
        key: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for key, path in paths.items()
        if key != "manifest"
    }
    manifest = {
        "qa_status": qa_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "cleaning_rule_version": CLEANING_RULE_VERSION,
        "final_cleaning_manifest_sha256": sha256_file(
            final_cleaning_manifest_path
        ),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "command_parameters": {
            "quick": bool(args.quick),
            "bootstrap": n_bootstrap,
            "permutations": n_permutation,
        },
        "seeds": {
            "first_cluster_bootstrap": SEED_BOOTSTRAP,
            "first_freedman_lane_permutation": SEED_PERMUTATION,
            "second_concept_cluster_bootstrap": SEED_SECOND_BOOTSTRAP,
            "second_item_bootstrap_sensitivity": SEED_SECOND_ITEM_BOOTSTRAP,
            "dispersion_disagreement_bootstrap_each_wave": SEED_DISPERSION_BOOTSTRAP,
            "second_exact_permutation": "none (complete enumeration)",
        },
        "packages": package_versions(),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for key, path in input_paths.items()
        },
        "outputs": output_hashes,
        "legacy_human_derivatives_read": False,
        "old_human_result_files_read": [],
        "validation_path": str(paths["validation"]),
    }
    paths["manifest"].write_text(stable_json(manifest) + "\n", encoding="utf-8")
    print(stable_json(validation))


if __name__ == "__main__":
    main()
