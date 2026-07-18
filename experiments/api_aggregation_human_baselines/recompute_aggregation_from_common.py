#!/usr/bin/env python3
"""Recompute all aggregation baselines from the frozen public common table.

This reviewer-facing entry point deliberately starts from the de-identified
``aggregation_human_common_data.csv``.  It re-runs point estimates, the shared
cluster bootstrap, Freedman--Lane permutations, paired beta differences, and
standardization constants, then checks both frozen CSV and JSON results.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Dynamic loading below must remain read-only in a reviewer checkout.
sys.dont_write_bytecode = True

BASE = Path(__file__).resolve().parent
REPOSITORY_ROOT = BASE.parents[1]
OUTPUT = REPOSITORY_ROOT / "verification" / "aggregation_from_common_recalculation.json"
TOLERANCE = 1e-12

FORMAL_PATH = BASE / "analyze_aggregation_human_baselines.py"
SPEC = importlib.util.spec_from_file_location("aggregation_formal", FORMAL_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load formal aggregation analysis: {FORMAL_PATH}")
formal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = formal
SPEC.loader.exec_module(formal)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def compare_frames(
    observed: pd.DataFrame,
    frozen: pd.DataFrame,
    keys: list[str],
    label: str,
) -> dict[str, Any]:
    observed = observed.sort_values(keys).reset_index(drop=True)
    frozen = frozen.sort_values(keys).reset_index(drop=True)
    if list(observed.columns) != list(frozen.columns):
        raise AssertionError(
            f"{label}: column mismatch: observed={list(observed.columns)}, frozen={list(frozen.columns)}"
        )
    if len(observed) != len(frozen):
        raise AssertionError(f"{label}: row mismatch {len(observed)} != {len(frozen)}")
    maximum = 0.0
    numeric_columns: list[str] = []
    text_columns: list[str] = []
    for column in observed.columns:
        left_numeric = pd.to_numeric(observed[column], errors="coerce")
        right_numeric = pd.to_numeric(frozen[column], errors="coerce")
        numeric_mask = left_numeric.notna() | right_numeric.notna()
        if numeric_mask.all():
            numeric_columns.append(column)
            difference = np.abs(left_numeric.to_numpy(float) - right_numeric.to_numpy(float))
            finite = difference[np.isfinite(difference)]
            if finite.size:
                maximum = max(maximum, float(np.max(finite)))
            if not np.allclose(
                left_numeric.to_numpy(float),
                right_numeric.to_numpy(float),
                atol=TOLERANCE,
                rtol=TOLERANCE,
                equal_nan=True,
            ):
                raise AssertionError(f"{label}: numeric mismatch in {column}")
        else:
            text_columns.append(column)
            left = observed[column].fillna("").astype(str).tolist()
            right = frozen[column].fillna("").astype(str).tolist()
            if left != right:
                raise AssertionError(f"{label}: text mismatch in {column}")
    return {
        "rows": int(len(observed)),
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "maximum_absolute_numeric_difference": maximum,
    }


def constants_from_common(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method_id, dimension), group in data.groupby(
        ["method_id", "dimension"], sort=True
    ):
        rows.append(
            {
                "method_id": method_id,
                "dimension": dimension,
                "n_rows": int(len(group)),
                "api_mean_quality_aligned": float(group["api_score"].mean()),
                "api_sample_sd": float(group["api_score"].std(ddof=1)),
                "human_mean_quality_aligned": float(group["human_score"].mean()),
                "human_sample_sd": float(group["human_score"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    common_path = BASE / "aggregation_human_common_data.csv"
    results_csv_path = BASE / "aggregation_human_results.csv"
    results_json_path = BASE / "aggregation_human_results.json"
    paired_path = BASE / "aggregation_paired_beta_differences.csv"
    constants_path = BASE / "aggregation_standardization_constants.csv"

    data = pd.read_csv(common_path)
    required = {
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
    }
    if not required.issubset(data.columns):
        raise AssertionError(f"Common table lacks {sorted(required - set(data.columns))}")
    if len(data) != 13 * 1068 or data["method_id"].nunique() != 13:
        raise AssertionError("Expected 13 methods x 1,068 common rows")
    if not data.groupby("method_id").size().eq(1068).all():
        raise AssertionError("Methods do not share the fixed 1,068-row analysis set")
    if not data.groupby("method_id")["stimulus_signature_sha256"].nunique().eq(178).all():
        raise AssertionError("Methods do not share 178 visible texts")

    frozen_results = pd.read_csv(results_csv_path)
    metadata_columns = [
        "method_id",
        "method_type",
        "method_label",
        "expected_judges_per_source_uid",
        "definition",
        "risk_direction",
    ]
    metadata = frozen_results[metadata_columns].drop_duplicates("method_id")
    method_ids = metadata["method_id"].tolist()

    points = formal.point_results(data, metadata)
    bootstrap, bootstrap_meta = formal.shared_cluster_bootstrap(data, method_ids)
    permutation, permutation_meta = formal.shared_freedman_lane(data, method_ids)
    observed_results = formal.attach_inference(points, bootstrap, permutation)
    observed_paired = formal.paired_beta_differences(observed_results, bootstrap)
    observed_constants = constants_from_common(data)
    validation = formal.exploration_validation(observed_results)
    if not validation["all_checks_pass"]:
        raise AssertionError(f"Exploratory value checks failed: {validation}")

    csv_check = compare_frames(
        observed_results,
        frozen_results,
        ["method_id"],
        "results CSV",
    )
    paired_check = compare_frames(
        observed_paired,
        pd.read_csv(paired_path),
        ["reference_method_id", "comparator_method_id"],
        "paired-difference CSV",
    )
    constants_check = compare_frames(
        observed_constants,
        pd.read_csv(constants_path),
        ["method_id", "dimension"],
        "standardization constants CSV",
    )

    frozen_json = json.loads(results_json_path.read_text(encoding="utf-8"))
    json_methods = pd.DataFrame(frozen_json["methods"])[list(observed_results.columns)]
    json_paired = pd.DataFrame(frozen_json["exploratory_paired_beta_differences"])[
        list(observed_paired.columns)
    ]
    json_method_check = compare_frames(
        observed_results,
        json_methods,
        ["method_id"],
        "results JSON methods",
    )
    json_paired_check = compare_frames(
        observed_paired,
        json_paired,
        ["reference_method_id", "comparator_method_id"],
        "results JSON paired differences",
    )

    payload = {
        "qa_status": "PASS",
        "entry_point": "experiments/api_aggregation_human_baselines/recompute_aggregation_from_common.py",
        "analysis_ready_input": {
            "path": "experiments/api_aggregation_human_baselines/aggregation_human_common_data.csv",
            "sha256": sha256_file(common_path),
            "rows": int(len(data)),
            "methods": int(data["method_id"].nunique()),
            "rows_per_method": 1068,
            "visible_texts_per_method": 178,
            "domain_concept_clusters": 30,
        },
        "recomputed": {
            "point_and_raw_scale_metrics": True,
            "cluster_bootstrap_draws": formal.N_BOOTSTRAP,
            "freedman_lane_permutations": formal.N_PERMUTATION,
            "paired_beta_differences": True,
            "standardization_constants": True,
        },
        "frozen_result_checks": {
            "results_csv": csv_check,
            "paired_csv": paired_check,
            "constants_csv": constants_check,
            "results_json_methods": json_method_check,
            "results_json_paired": json_paired_check,
        },
        "bootstrap_metadata": bootstrap_meta,
        "permutation_metadata": permutation_meta,
        "exploratory_value_validation": validation,
        "tolerance": TOLERANCE,
        "privacy_boundary": (
            "The public entry point starts from a frozen, de-identified analysis-ready common table; "
            "it does not reconstruct that table from raw participant exports or identifiers."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qa_status": payload["qa_status"],
                "output": str(OUTPUT.relative_to(REPOSITORY_ROOT)),
                "maximum_absolute_numeric_difference": max(
                    check["maximum_absolute_numeric_difference"]
                    for check in payload["frozen_result_checks"].values()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
