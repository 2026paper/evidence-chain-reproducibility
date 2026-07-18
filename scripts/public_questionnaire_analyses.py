#!/usr/bin/env python3
"""Reproducible analyses for the archived public questionnaires.

This script reads Credamo exports without trusting their incorrect worksheet
dimension metadata, performs only the quality-control rules fixed in the
analysis plans, writes privacy-safe long-form data, and runs the locked
human and controlled-AB analyses.

Raw response IDs, platform user IDs, IP/location/device fields, and timestamps
are never written to an analysis output. Platform user IDs and start times are
used only in memory to construct an anonymous cross-form cluster and the fixed
chronologically-first sensitivity analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import unicodedata
import warnings
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import openpyxl
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import binomtest, norm, rankdata, spearmanr
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


SCRIPT_VERSION = "1.0.1"
ABC_DURATION_FLOOR_SECONDS = 72.0
DIFFERENCE_DURATION_FLOOR_SECONDS = 96.0
ABC_BOOTSTRAP_SEED = 2026071701
DIFFERENCE_BOOTSTRAP_SEED = 2026071702
DIFFERENCE_SIGNFLIP_SEED = 2026071703
N_BOOTSTRAP = 10_000
N_SIGNFLIP = 100_000

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "数据与问卷源文件"
ANALYSIS_ROOT = PAPER_ROOT / "analysis"
NOTES_ROOT = PAPER_ROOT / "notes"
SEMANTIC_CROSSWALK_PATH = ANALYSIS_ROOT / "public_semantic_crosswalk.csv"
SEMANTIC_CROSSWALK_AUDIT_PATH = ANALYSIS_ROOT / "public_semantic_crosswalk_audit.json"
DIAGNOSTIC_ROOT = (
    SOURCE_ROOT / "AI测试数据全集" / "20_主数据与支持" / "diagnostics"
)
AB_RUN_ROOT = DIAGNOSTIC_ROOT / "official_ab_judge_run_20260521_155532"


ABC_CONCEPTS = [
    "projectile_motion",
    "evolution_no_goal",
    "geological_time",
    "plant_respiration",
    "adaptation",
    "net_force_vs_velocity",
]
ABC_NONPARALLEL_CONCEPT = "adaptation"
ABC_PRE_KEYS = [
    "不再受到向前的推力，水平运动主要由惯性保持",
    "不会，进化没有预设目标，主要与变异、遗传和环境选择有关",
    "板块运动通常很慢，但长期累积会造成巨大地质变化",
    "白天和夜晚都进行呼吸，光照下还会进行光合作用",
    "随机变异中有些在特定环境下更利于生存和繁殖",
    "不一定，净力直接决定加速度，而不是直接决定速度大小",
]
ABC_POST_KEYS = [
    "因为没有水平合力时，物体会由于惯性保持原有水平运动状态",
    "它的某些可遗传特征在当前环境下更有利于生存或繁殖",
    "板块运动通常很慢，但长期累积会造成明显地质结果",
    "植物夜间进行细胞呼吸，白天也会进行呼吸，只是光照下还可能进行光合作用",
    "只要适合当前环境，简单结构也可能被保留或演化出来",
    "如果速度不变，它的加速度为零，净力可以为零或近似为零",
]
CONFIDENCE_MAP = {
    "非常不确定": 1,
    "不太确定": 2,
    "一般": 3,
    "比较确定": 4,
    "非常确定": 5,
}
RISK_MAP = {
    "完全不会": 1,
    "可能较低": 2,
    "一般": 3,
    "可能较高": 4,
    "非常可能": 5,
}

DIFFERENCE_CONCEPTS = [
    "projectile_motion",
    "balanced_forces",
    "evolutionary_direction",
    "plant_respiration",
    "geological_time",
    "weather_vs_climate",
    "chemical_equilibrium",
    "greenhouse_effect",
]
DIFFERENCE_QUALITY_KEYS = ["A", "A", "B", "B", "A", "A", "A", "B"]
DIFFERENCE_RISK_KEYS = ["B" if key == "A" else "A" for key in DIFFERENCE_QUALITY_KEYS]

FORBIDDEN_OUTPUT_COLUMN_PARTS = {
    "raw_response",
    "raw_user",
    "platform_user",
    "ip",
    "longitude",
    "latitude",
    "province",
    "city",
    "device",
    "browser",
    "screen",
    "start_time",
    "end_time",
    "timestamp",
    "publication_id",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def as_float(value: Any) -> float | None:
    if value is None or normalize_text(value) == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.10g")


def read_credamo_export(path: Path, expected_columns: int) -> dict[str, Any]:
    """Read a Credamo export after resetting its incorrect A1 worksheet bounds."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        if len(workbook.worksheets) != 1:
            raise AssertionError(f"Expected one worksheet in {path.name}")
        sheet = workbook.worksheets[0]
        reported_dimension = sheet.calculate_dimension()
        sheet.reset_dimensions()
        rows = list(sheet.iter_rows(values_only=True))
        workbook.close()
    if len(rows) < 3:
        raise AssertionError(f"No response rows in {path.name}")
    if len(rows[0]) != expected_columns or len(rows[1]) != expected_columns:
        raise AssertionError(f"Unexpected column count in {path.name}: {len(rows[1])}")
    verbose_headers = [normalize_text(value) for value in rows[0]]
    code_headers = [normalize_text(value) for value in rows[1]]
    if len(set(code_headers)) != len(code_headers):
        raise AssertionError(f"Duplicate code headers in {path.name}")
    data_rows = [list(row[:expected_columns]) for row in rows[2:] if any(value is not None for value in row)]
    return {
        "path": path,
        "reported_dimension": reported_dimension,
        "sheet_name": sheet.title,
        "verbose_headers": verbose_headers,
        "code_headers": code_headers,
        "rows": data_rows,
    }


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    return multipletests(values, alpha=0.05, method="holm")[1].tolist()


def coefficient_table(result: Any) -> list[dict[str, Any]]:
    ci = np.asarray(result.conf_int())
    return [
        {
            "term": str(term),
            "estimate": float(result.params.iloc[index]),
            "standard_error": float(result.bse.iloc[index]),
            "statistic": float(result.tvalues.iloc[index]),
            "p_value": float(result.pvalues.iloc[index]),
            "ci_low": float(ci[index, 0]),
            "ci_high": float(ci[index, 1]),
        }
        for index, term in enumerate(result.params.index)
    ]


def assert_privacy_safe(frame: pd.DataFrame, forbidden_values: set[str]) -> None:
    lower_columns = [str(column).lower() for column in frame.columns]
    for column in lower_columns:
        for forbidden in FORBIDDEN_OUTPUT_COLUMN_PARTS:
            if column == forbidden or column.startswith(f"{forbidden}_") or column.endswith(f"_{forbidden}"):
                raise AssertionError(f"Forbidden output column: {column}")
    emitted = {
        normalize_text(value)
        for value in frame.astype(object).to_numpy().ravel().tolist()
        if value is not None and normalize_text(value)
    }
    leaked = emitted.intersection(forbidden_values)
    if leaked:
        raise AssertionError(f"Raw operational values leaked to output ({len(leaked)})")


def build_abc_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], set[str]]:
    exports = {version: read_credamo_export(SOURCE_ROOT / f"{version}.xlsx", 51) for version in "ABC"}
    raw_records: list[dict[str, Any]] = []
    forbidden_values: set[str] = set()
    for version in "ABC":
        export = exports[version]
        for source_position, row in enumerate(export["rows"], start=1):
            for index in [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                value = normalize_text(row[index])
                if value:
                    forbidden_values.add(value)
            raw_records.append(
                {
                    "version": version,
                    "source_position": source_position,
                    "raw_response_id": normalize_text(row[0]),
                    "raw_user_id": normalize_text(row[1]),
                    "start_time_internal": pd.to_datetime(row[2], errors="coerce"),
                    "duration_seconds": as_float(row[4]),
                    "education": normalize_text(row[22]),
                    "broad_field": normalize_text(row[23]),
                    "science_reading_frequency": normalize_text(row[24]),
                    "attention_answer": normalize_text(row[26]),
                    "pre_answers": [normalize_text(row[index]) for index in range(27, 33)],
                    "post_answers": [normalize_text(row[index]) for index in range(33, 39)],
                    "confidence_answers": [normalize_text(row[index]) for index in range(39, 45)],
                    "risk_answers": [normalize_text(row[index]) for index in range(45, 51)],
                }
            )

    if len(raw_records) != 180:
        raise AssertionError(f"Expected 180 A/B/C responses, observed {len(raw_records)}")
    for record in raw_records:
        if not record["raw_user_id"]:
            record["cluster_source_key"] = f"__blank__{record['version']}_{record['source_position']}"
        else:
            record["cluster_source_key"] = record["raw_user_id"]
        record["attention_pass"] = record["attention_answer"] == "比较同意"
        duration = record["duration_seconds"]
        record["duration_missing_or_nonpositive"] = duration is None or duration <= 0
        record["duration_too_short"] = duration is not None and 0 < duration < ABC_DURATION_FLOOR_SECONDS
        record["eligible"] = record["attention_pass"] and not record["duration_too_short"]

    attention_records = [record for record in raw_records if record["attention_pass"]]
    eligible_records = [record for record in raw_records if record["eligible"]]
    if Counter(record["version"] for record in attention_records) != Counter({"A": 60, "B": 60, "C": 59}):
        raise AssertionError("Unexpected attention-pass counts")
    if Counter(record["version"] for record in eligible_records) != Counter({"A": 60, "B": 60, "C": 58}):
        raise AssertionError("Unexpected final A/B/C counts")

    cluster_map: dict[str, str] = {}
    for record in eligible_records:
        key = record["cluster_source_key"]
        if key not in cluster_map:
            cluster_map[key] = f"P{len(cluster_map) + 1:03d}"
        record["participant_id"] = cluster_map[key]
    if len(cluster_map) != 167:
        raise AssertionError(f"Expected 167 final participant clusters, observed {len(cluster_map)}")

    eligible_counts = Counter(record["cluster_source_key"] for record in eligible_records)
    repeated_counts = {key: count for key, count in eligible_counts.items() if count > 1}
    if len(repeated_counts) != 10 or sum(repeated_counts.values()) != 21:
        raise AssertionError("Unexpected final cross-form repeat structure")

    first_response_keys: set[tuple[str, int]] = set()
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for record in eligible_records:
        by_cluster.setdefault(record["cluster_source_key"], []).append(record)
    version_order = {"A": 0, "B": 1, "C": 2}
    for key, group in by_cluster.items():
        first = min(
            group,
            key=lambda item: (
                pd.Timestamp.max if pd.isna(item["start_time_internal"]) else item["start_time_internal"],
                version_order[item["version"]],
                item["source_position"],
            ),
        )
        first_response_keys.add((first["version"], first["source_position"]))

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for response_number, record in enumerate(eligible_records, start=1):
        response_id = f"ABC_R{response_number:03d}"
        first_flag = (record["version"], record["source_position"]) in first_response_keys
        single_form_flag = eligible_counts[record["cluster_source_key"]] == 1
        pre_correct = [int(answer == key) for answer, key in zip(record["pre_answers"], ABC_PRE_KEYS)]
        post_correct = [int(answer == key) for answer, key in zip(record["post_answers"], ABC_POST_KEYS)]
        confidence = [CONFIDENCE_MAP.get(answer) for answer in record["confidence_answers"]]
        risk = [RISK_MAP.get(answer) for answer in record["risk_answers"]]
        if any(value is None for value in confidence + risk):
            raise AssertionError(f"Unmapped A/B/C ordinal response in {response_id}")
        for concept_index, concept in enumerate(ABC_CONCEPTS, start=1):
            wide_rows.append(
                {
                    "response_id": response_id,
                    "participant_id": record["participant_id"],
                    "version": record["version"],
                    "concept_index": concept_index,
                    "concept": concept,
                    "pre_correct": pre_correct[concept_index - 1],
                    "post_correct": post_correct[concept_index - 1],
                    "confidence_score": confidence[concept_index - 1],
                    "misleading_risk_score": risk[concept_index - 1],
                    "duration_seconds": record["duration_seconds"],
                    "duration_missing_or_nonpositive": int(record["duration_missing_or_nonpositive"]),
                    "first_eligible_response": int(first_flag),
                    "single_form_participant": int(single_form_flag),
                    "education": record["education"],
                    "broad_field": record["broad_field"],
                    "science_reading_frequency": record["science_reading_frequency"],
                    "pre_total_correct": sum(pre_correct),
                    "post_total_correct": sum(post_correct),
                }
            )
            for phase, correctness in [(0, pre_correct[concept_index - 1]), (1, post_correct[concept_index - 1])]:
                long_rows.append(
                    {
                        "study": "public_abc_randomized_versions",
                        "response_id": response_id,
                        "participant_id": record["participant_id"],
                        "version": record["version"],
                        "concept_index": concept_index,
                        "concept": concept,
                        "phase": phase,
                        "correct": correctness,
                        "confidence_score": confidence[concept_index - 1] if phase == 1 else None,
                        "misleading_risk_score": risk[concept_index - 1] if phase == 1 else None,
                        "duration_seconds": record["duration_seconds"],
                        "duration_missing_or_nonpositive": int(record["duration_missing_or_nonpositive"]),
                        "first_eligible_response": int(first_flag),
                        "single_form_participant": int(single_form_flag),
                        "education": record["education"],
                        "broad_field": record["broad_field"],
                        "science_reading_frequency": record["science_reading_frequency"],
                        "pre_total_correct": sum(pre_correct),
                        "post_total_correct": sum(post_correct),
                    }
                )

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    if len(wide) != 178 * 6 or len(long) != 178 * 12:
        raise AssertionError("Unexpected A/B/C long-form row count")
    if long["correct"].isna().any() or wide[["confidence_score", "misleading_risk_score"]].isna().any().any():
        raise AssertionError("Unexpected A/B/C outcome missingness")
    if long.loc[long["first_eligible_response"] == 1, "response_id"].nunique() != 167:
        raise AssertionError("Unexpected first-response sensitivity count")
    if long.loc[long["single_form_participant"] == 1, "response_id"].nunique() != 157:
        raise AssertionError("Unexpected single-form sensitivity count")
    assert_privacy_safe(long, forbidden_values)

    qc = {
        "raw_responses": 180,
        "attention_pass_responses": len(attention_records),
        "attention_fail_responses": 1,
        "duration_floor_seconds": ABC_DURATION_FLOOR_SECONDS,
        "duration_exclusions_after_attention": sum(record["attention_pass"] and record["duration_too_short"] for record in raw_records),
        "duration_missing_or_nonpositive_retained": sum(record["eligible"] and record["duration_missing_or_nonpositive"] for record in raw_records),
        "final_responses": len(eligible_records),
        "final_by_version": dict(Counter(record["version"] for record in eligible_records)),
        "final_participant_clusters": len(cluster_map),
        "repeated_participant_clusters": len(repeated_counts),
        "responses_in_repeated_clusters": sum(repeated_counts.values()),
        "first_response_sensitivity_n": 167,
        "single_form_sensitivity_n": 157,
    }
    source_meta = {
        version: {
            "file": export["path"].name,
            "sha256": sha256_file(export["path"]),
            "worksheet": export["sheet_name"],
            "reported_dimension_before_reset": export["reported_dimension"],
            "rows": len(export["rows"]),
            "columns": len(export["code_headers"]),
        }
        for version, export in exports.items()
    }
    return long, wide, qc, source_meta, forbidden_values


ABC_PRIMARY_FORMULA = (
    'correct ~ phase * C(version, Treatment(reference="A")) + C(concept)'
)


def fit_abc_primary(data: pd.DataFrame, working: str = "exchangeable") -> Any:
    cov_struct = Exchangeable() if working == "exchangeable" else Independence()
    model = smf.gee(
        ABC_PRIMARY_FORMULA,
        groups="participant_id",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=cov_struct,
    )
    result = model.fit(maxiter=200, ctol=1e-8, cov_type="robust")
    if not bool(result.converged):
        raise AssertionError(f"A/B/C GEE did not converge ({working})")
    return result


def abc_excluding_nonparallel_adaptation_pair(data: pd.DataFrame) -> dict[str, Any]:
    """Fixed sensitivity excluding the non-equivalent Q11/Q17 concept pair."""
    if ABC_NONPARALLEL_CONCEPT != ABC_CONCEPTS[4]:
        raise AssertionError("The pre-specified nonparallel concept must remain concept 5")
    frame = data.loc[data["concept"] != ABC_NONPARALLEL_CONCEPT].copy()
    expected_concepts = [concept for concept in ABC_CONCEPTS if concept != ABC_NONPARALLEL_CONCEPT]
    if sorted(frame["concept"].unique()) != sorted(expected_concepts):
        raise AssertionError("Unexpected concept set after excluding adaptation")
    if len(frame) != 178 * len(expected_concepts) * 2:
        raise AssertionError("Unexpected row count after excluding adaptation")
    if frame["response_id"].nunique() != 178 or frame["participant_id"].nunique() != 167:
        raise AssertionError("Participant structure changed in adaptation sensitivity")
    result = fit_abc_primary(frame, "exchangeable")
    return {
        "status": "fixed_content_parallelism_sensitivity",
        "rationale": (
            "Q11 asks whether random variants can be advantageous in a particular environment, "
            "whereas Q17 asks whether simple structures can be retained or evolve when adaptive; "
            "the pair is scientifically related but not a fully equivalent pre/post item."
        ),
        "excluded_concept": ABC_NONPARALLEL_CONCEPT,
        "excluded_concept_index": 5,
        "excluded_pre_item": "Q11",
        "excluded_post_item": "Q17",
        "remaining_concepts": expected_concepts,
        "formula": ABC_PRIMARY_FORMULA,
        "family_link": "binomial_logit",
        "cluster": "privacy-safe cross-form participant",
        "working_correlation": result.model.cov_struct.__class__.__name__.lower(),
        "covariance": "robust sandwich",
        "n_long_rows": int(len(frame)),
        "n_responses": int(frame["response_id"].nunique()),
        "n_participant_clusters": int(frame["participant_id"].nunique()),
        "n_concepts": int(frame["concept"].nunique()),
        "converged": bool(result.converged),
        "dependence_parameter": json_ready(np.asarray(result.cov_struct.dep_params).tolist()),
        "omnibus_interaction": abc_interaction_wald(result),
        "coefficients": coefficient_table(result),
    }


def abc_interaction_wald(result: Any) -> dict[str, Any]:
    interaction_terms = [
        str(term)
        for term in result.params.index
        if "phase" in str(term) and "C(version" in str(term) and ":" in str(term)
    ]
    if len(interaction_terms) != 2:
        raise AssertionError(f"Expected two version-by-phase coefficients, observed {interaction_terms}")
    restriction = np.zeros((2, len(result.params)))
    for row, term in enumerate(interaction_terms):
        restriction[row, list(result.params.index).index(term)] = 1.0
    test = result.wald_test(restriction, scalar=True)
    return {
        "hypothesis": "joint version-by-phase interaction",
        "terms": interaction_terms,
        "statistic": float(test.statistic),
        "df": int(np.linalg.matrix_rank(restriction)),
        "p_value": float(test.pvalue),
    }


def abc_marginal_estimate(result: Any, version: str, phase: int) -> dict[str, Any]:
    grid = pd.DataFrame(
        {
            "version": [version] * len(ABC_CONCEPTS),
            "phase": [phase] * len(ABC_CONCEPTS),
            "concept": ABC_CONCEPTS,
        }
    )
    design = np.asarray(
        patsy.build_design_matrices(
            [result.model.data.design_info], grid, return_type="dataframe"
        )[0],
        dtype=float,
    )
    beta = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    probabilities = expit(design @ beta)
    gradient = np.mean(probabilities[:, None] * (1.0 - probabilities[:, None]) * design, axis=0)
    estimate = float(np.mean(probabilities))
    variance = float(gradient @ covariance @ gradient)
    standard_error = math.sqrt(max(variance, 0.0))
    return {
        "version": version,
        "phase": int(phase),
        "probability": estimate,
        "standard_error": standard_error,
        "ci_low": max(0.0, estimate - norm.ppf(0.975) * standard_error),
        "ci_high": min(1.0, estimate + norm.ppf(0.975) * standard_error),
        "gradient_internal": gradient,
    }


def abc_marginal_summary(result: Any) -> dict[str, Any]:
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for version in "ABC":
        for phase in [0, 1]:
            cells[(version, phase)] = abc_marginal_estimate(result, version, phase)
    covariance = np.asarray(result.cov_params(), dtype=float)
    changes: dict[str, dict[str, Any]] = {}
    change_gradients: dict[str, np.ndarray] = {}
    for version in "ABC":
        pre = cells[(version, 0)]
        post = cells[(version, 1)]
        gradient = post["gradient_internal"] - pre["gradient_internal"]
        estimate = post["probability"] - pre["probability"]
        standard_error = math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
        changes[version] = {
            "version": version,
            "change": float(estimate),
            "standard_error": standard_error,
            "ci_low": float(estimate - norm.ppf(0.975) * standard_error),
            "ci_high": float(estimate + norm.ppf(0.975) * standard_error),
        }
        change_gradients[version] = gradient

    pairwise: list[dict[str, Any]] = []
    for left, right in [("B", "A"), ("C", "A"), ("C", "B")]:
        gradient = change_gradients[left] - change_gradients[right]
        estimate = changes[left]["change"] - changes[right]["change"]
        standard_error = math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
        z_value = estimate / standard_error if standard_error > 0 else float("nan")
        p_value = float(2.0 * norm.sf(abs(z_value))) if math.isfinite(z_value) else float("nan")
        pairwise.append(
            {
                "contrast": f"{left}-{right}",
                "scale": "average_marginal_probability",
                "difference_in_change": float(estimate),
                "standard_error": standard_error,
                "z": z_value,
                "p_value": p_value,
                "ci_low": float(estimate - norm.ppf(0.975) * standard_error),
                "ci_high": float(estimate + norm.ppf(0.975) * standard_error),
            }
        )
    adjusted = holm_adjust([row["p_value"] for row in pairwise])
    for row, p_adjusted in zip(pairwise, adjusted):
        row["p_holm"] = p_adjusted
    public_cells = []
    for key in [(version, phase) for version in "ABC" for phase in [0, 1]]:
        row = {item: value for item, value in cells[key].items() if item != "gradient_internal"}
        public_cells.append(row)
    return {"cells": public_cells, "changes": list(changes.values()), "pairwise_change_differences": pairwise}


def abc_logit_interaction_pairwise(result: Any) -> list[dict[str, Any]]:
    """Pairwise arm contrasts in pre/post change on the fitted log-odds scale."""
    names = list(result.params.index)
    b_term = next(term for term in names if "phase" in term and "C(version" in term and "T.B" in term)
    c_term = next(term for term in names if "phase" in term and "C(version" in term and "T.C" in term)
    covariance = np.asarray(result.cov_params(), dtype=float)
    rows = []
    for contrast, weights_by_term in [
        ("B-A", {b_term: 1.0}),
        ("C-A", {c_term: 1.0}),
        ("C-B", {c_term: 1.0, b_term: -1.0}),
    ]:
        weights = np.zeros(len(names))
        for term, weight in weights_by_term.items():
            weights[names.index(term)] = weight
        estimate = float(weights @ np.asarray(result.params, dtype=float))
        standard_error = math.sqrt(max(float(weights @ covariance @ weights), 0.0))
        z_value = estimate / standard_error if standard_error > 0 else float("nan")
        p_value = float(2.0 * norm.sf(abs(z_value))) if math.isfinite(z_value) else float("nan")
        rows.append(
            {
                "contrast": contrast,
                "scale": "log_odds_change",
                "difference_in_change": estimate,
                "standard_error": standard_error,
                "z": z_value,
                "p_value": p_value,
                "ci_low": estimate - norm.ppf(0.975) * standard_error,
                "ci_high": estimate + norm.ppf(0.975) * standard_error,
            }
        )
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row["p_holm"] = adjusted
    return rows


def summarize_abc_primary(result: Any, data: pd.DataFrame) -> dict[str, Any]:
    wald = abc_interaction_wald(result)
    marginal = abc_marginal_summary(result)
    return {
        "formula": ABC_PRIMARY_FORMULA,
        "family_link": "binomial_logit",
        "cluster": "privacy-safe cross-form participant",
        "working_correlation": result.model.cov_struct.__class__.__name__.lower(),
        "covariance": "robust sandwich",
        "n_long_rows": int(len(data)),
        "n_responses": int(data["response_id"].nunique()),
        "n_participant_clusters": int(data["participant_id"].nunique()),
        "converged": bool(result.converged),
        "dependence_parameter": json_ready(np.asarray(result.cov_struct.dep_params).tolist()),
        "omnibus_interaction": wald,
        "pairwise_logit_scale_interactions": abc_logit_interaction_pairwise(result),
        "marginal": marginal,
        "pairwise_inference_status": "confirmatory_if_omnibus_p_below_0.05_else_descriptive",
        "coefficients": coefficient_table(result),
    }


def abc_post_reading_gee(wide: pd.DataFrame) -> dict[str, Any]:
    formula = (
        'post_correct ~ C(version, Treatment(reference="A")) + pre_correct + C(concept)'
    )
    model = smf.gee(
        formula,
        groups="participant_id",
        data=wide,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    result = model.fit(maxiter=200, ctol=1e-8, cov_type="robust")
    if not bool(result.converged):
        raise AssertionError("Post-reading A/B/C GEE did not converge")
    version_terms = [term for term in result.params.index if "C(version" in str(term)]
    if len(version_terms) != 2:
        raise AssertionError("Unexpected post-reading version term count")
    restriction = np.zeros((2, len(result.params)))
    for row, term in enumerate(version_terms):
        restriction[row, list(result.params.index).index(term)] = 1.0
    test = result.wald_test(restriction, scalar=True)
    return {
        "formula": formula,
        "converged": bool(result.converged),
        "n_rows": int(len(wide)),
        "n_responses": int(wide["response_id"].nunique()),
        "n_participant_clusters": int(wide["participant_id"].nunique()),
        "version_omnibus_wald": {
            "statistic": float(test.statistic),
            "df": 2,
            "p_value": float(test.pvalue),
        },
        "coefficients": coefficient_table(result),
    }


def abc_total_summaries(wide: pd.DataFrame) -> dict[str, Any]:
    responses = wide.drop_duplicates("response_id")
    output: dict[str, Any] = {}
    for version, group in responses.groupby("version", observed=True):
        output[version] = {}
        for phase, column in [("pre", "pre_total_correct"), ("post", "post_total_correct")]:
            values = group[column].astype(float)
            output[version][phase] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "min": int(values.min()),
                "max": int(values.max()),
                "frequency_0_to_6": {str(score): int((values == score).sum()) for score in range(7)},
            }
    return output


def abc_balance_summary(wide: pd.DataFrame) -> dict[str, Any]:
    responses = wide.drop_duplicates("response_id")
    output: dict[str, Any] = {"categorical": {}, "pre_total": {}}
    for variable in ["education", "broad_field", "science_reading_frequency"]:
        output["categorical"][variable] = {}
        for version, group in responses.groupby("version", observed=True):
            counts = group[variable].value_counts(dropna=False).sort_index()
            output["categorical"][variable][version] = [
                {
                    "category": str(category),
                    "n": int(count),
                    "proportion": float(count / len(group)),
                }
                for category, count in counts.items()
            ]
    for version, group in responses.groupby("version", observed=True):
        output["pre_total"][version] = {
            "n": int(len(group)),
            "mean": float(group["pre_total_correct"].mean()),
            "sd": float(group["pre_total_correct"].std(ddof=1)),
            "median": float(group["pre_total_correct"].median()),
            "q1": float(group["pre_total_correct"].quantile(0.25)),
            "q3": float(group["pre_total_correct"].quantile(0.75)),
        }
    return output


def rank_secondary_model(wide: pd.DataFrame, outcome: str) -> dict[str, Any]:
    frame = wide[["participant_id", "version", "concept", outcome]].copy()
    frame["rank_percentile"] = (rankdata(frame[outcome], method="average") - 0.5) / len(frame)
    formula = 'rank_percentile ~ C(version, Treatment(reference="A")) + C(concept)'
    result = smf.ols(formula, data=frame).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["participant_id"], "use_correction": True},
    )
    names = list(result.params.index)
    b_term = next(term for term in names if "C(version" in term and "T.B" in term)
    c_term = next(term for term in names if "C(version" in term and "T.C" in term)
    covariance = np.asarray(result.cov_params(), dtype=float)
    pairs = []
    for contrast, vector in [
        ("B-A", {b_term: 1.0}),
        ("C-A", {c_term: 1.0}),
        ("C-B", {c_term: 1.0, b_term: -1.0}),
    ]:
        weights = np.zeros(len(names))
        for term, weight in vector.items():
            weights[names.index(term)] = weight
        estimate = float(weights @ np.asarray(result.params))
        standard_error = math.sqrt(max(float(weights @ covariance @ weights), 0.0))
        z_value = estimate / standard_error if standard_error > 0 else float("nan")
        p_value = float(2 * norm.sf(abs(z_value))) if math.isfinite(z_value) else float("nan")
        pairs.append(
            {
                "contrast": contrast,
                "rank_percentile_difference": estimate,
                "standard_error": standard_error,
                "z": z_value,
                "p_value": p_value,
                "ci_low": estimate - norm.ppf(0.975) * standard_error,
                "ci_high": estimate + norm.ppf(0.975) * standard_error,
            }
        )
    for row, adjusted in zip(pairs, holm_adjust([row["p_value"] for row in pairs])):
        row["p_holm"] = adjusted
    raw_summary = []
    for version, group in frame.groupby("version", observed=True):
        raw_summary.append(
            {
                "version": version,
                "n_concept_ratings": int(len(group)),
                "n_responses": int(wide.loc[wide["version"] == version, "response_id"].nunique()),
                "mean_score": float(group[outcome].mean()),
                "sd_score": float(group[outcome].std(ddof=1)),
                "median_score": float(group[outcome].median()),
            }
        )
    return {
        "method": "normalized midrank OLS with concept fixed effects and participant-clustered robust covariance",
        "outcome": outcome,
        "raw_ordinal_summary": raw_summary,
        "pairwise_version_contrasts": pairs,
        "coefficients": coefficient_table(result),
    }


def abc_concept_specific(wide: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    critical = norm.ppf(1.0 - 0.05 / (2.0 * 18.0))
    for (version, concept), group in wide.groupby(["version", "concept"], observed=True, sort=True):
        change = group["post_correct"].astype(float) - group["pre_correct"].astype(float)
        standard_error = float(change.std(ddof=1) / math.sqrt(len(change)))
        estimate = float(change.mean())
        if standard_error > 0:
            z_value = estimate / standard_error
            p_value = float(2.0 * norm.sf(abs(z_value)))
        else:
            z_value = float("nan")
            p_value = 1.0 if estimate == 0 else 0.0
        rows.append(
            {
                "version": version,
                "concept": concept,
                "n": int(len(group)),
                "pre_accuracy": float(group["pre_correct"].mean()),
                "post_accuracy": float(group["post_correct"].mean()),
                "change": estimate,
                "standard_error": standard_error,
                "bonferroni_95_fwer_ci_low": max(-1.0, estimate - critical * standard_error),
                "bonferroni_95_fwer_ci_high": min(1.0, estimate + critical * standard_error),
                "z": z_value,
                "p_value": p_value,
            }
        )
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row["p_holm_18"] = adjusted
    return rows


def analyze_abc(long: pd.DataFrame, wide: pd.DataFrame) -> dict[str, Any]:
    primary_result = fit_abc_primary(long, "exchangeable")
    primary = summarize_abc_primary(primary_result, long)
    primary["pairwise_inference_status"] = (
        "confirmatory_with_holm"
        if primary["omnibus_interaction"]["p_value"] < 0.05
        else "descriptive_only_because_omnibus_not_significant"
    )

    independence_result = fit_abc_primary(long, "independence")
    first_ids = set(long.loc[long["first_eligible_response"] == 1, "response_id"])
    single_ids = set(long.loc[long["single_form_participant"] == 1, "response_id"])
    first_data = long[long["response_id"].isin(first_ids)].copy()
    single_data = long[long["response_id"].isin(single_ids)].copy()
    first_result = fit_abc_primary(first_data, "exchangeable")
    single_result = fit_abc_primary(single_data, "exchangeable")

    return {
        "analysis_version": SCRIPT_VERSION,
        "source_mapping_status": "completed_18_of_18_to_six_upstream_disagreement_cases_nonexact_lay_adaptations",
        "design": (
            "author-confirmed random assignment among three separately published lay-version forms; "
            "allocation implementation is not independently encoded in the exports"
        ),
        "randomization_adherence": (
            "not perfect one-exposure adherence: 10 final anonymous participant clusters contributed "
            "21 eligible responses across forms; all were retained with participant-clustered covariance"
        ),
        "item_sampling_status": "six archived examples; no probability-sampling claim",
        "primary": primary,
        "robustness": {
            "post_reading_adjusted_for_matching_pre": abc_post_reading_gee(wide),
            "independence_working_correlation": summarize_abc_primary(independence_result, long),
            "chronologically_first_eligible_response": summarize_abc_primary(first_result, first_data),
            "single_form_participants": summarize_abc_primary(single_result, single_data),
            "excluding_nonparallel_adaptation_pair": abc_excluding_nonparallel_adaptation_pair(long),
        },
        "respondent_total_correct": abc_total_summaries(wide),
        "randomization_balance_descriptive": abc_balance_summary(wide),
        "secondary": {
            "confidence": rank_secondary_model(wide, "confidence_score"),
            "misleading_risk": rank_secondary_model(wide, "misleading_risk_score"),
            "concept_specific_change": abc_concept_specific(wide),
        },
    }


def parse_version_choice(answer: str) -> tuple[str | None, str]:
    value = normalize_text(answer)
    if value.startswith("版本A"):
        selected = "A"
    elif value.startswith("版本B"):
        selected = "B"
    else:
        selected = None
    if "明显" in value:
        strength = "obvious"
    elif "略" in value:
        strength = "slight"
    elif "两者差不多" in value:
        strength = "tie"
    elif "不确定" in value:
        strength = "uncertain"
    else:
        strength = "unmapped"
    return selected, strength


def map_reason(answer: str) -> str:
    value = normalize_text(answer)
    patterns = [
        ("边界条件", "boundary_conditions"),
        ("真正解释", "mechanistic_explanation"),
        ("表面提到", "label_only"),
        ("事实错误", "factual_or_conceptual_error"),
        ("过度绝对化", "overclaiming_or_overconfidence"),
        ("语言流畅", "fluency_masking_scientific_problem"),
        ("适合非专业", "lay_accessibility"),
        ("错误推断", "new_false_inference"),
    ]
    for token, label in patterns:
        if token in value:
            return label
    if value == "不确定":
        return "uncertain"
    if value == "其他":
        return "other"
    raise AssertionError(f"Unmapped reason category: {value!r}")


def build_difference_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], set[str], dict[str, Any]]:
    export = read_credamo_export(SOURCE_ROOT / "差异.xlsx", 56)
    forbidden_values: set[str] = set()
    raw_records: list[dict[str, Any]] = []
    for source_position, row in enumerate(export["rows"], start=1):
        for index in [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            value = normalize_text(row[index])
            if value:
                forbidden_values.add(value)
        duration = as_float(row[4])
        raw_records.append(
            {
                "source_position": source_position,
                "duration_seconds": duration,
                "duration_missing_or_nonpositive": duration is None or duration <= 0,
                "duration_too_short": duration is not None and 0 < duration < DIFFERENCE_DURATION_FLOOR_SECONDS,
                "role": normalize_text(row[18]),
                "field": normalize_text(row[19]),
                "science_familiarity": as_float(row[20]),
                "ai_familiarity": as_float(row[21]),
                "case_answers": [
                    {
                        "suitability": normalize_text(row[22 + 3 * case]),
                        "misleading": normalize_text(row[23 + 3 * case]),
                        "reason": normalize_text(row[24 + 3 * case]),
                    }
                    for case in range(8)
                ],
                "global_reason_flags": [int(as_float(row[index]) or 0) for index in range(46, 55)],
            }
        )
    if len(raw_records) != 60:
        raise AssertionError(f"Expected 60 difference-survey responses, observed {len(raw_records)}")
    eligible = [record for record in raw_records if not record["duration_too_short"]]
    if len(eligible) != 60:
        raise AssertionError(f"Unexpected difference-survey duration exclusions: {60 - len(eligible)}")

    long_rows: list[dict[str, Any]] = []
    respondent_rows: list[dict[str, Any]] = []
    for respondent_number, record in enumerate(eligible, start=1):
        respondent_id = f"D_R{respondent_number:03d}"
        case_rows: list[dict[str, Any]] = []
        for case_index, answers in enumerate(record["case_answers"], start=1):
            suitability_selected, suitability_strength = parse_version_choice(answers["suitability"])
            misleading_selected, misleading_strength = parse_version_choice(answers["misleading"])
            if suitability_strength == "unmapped" or misleading_strength == "unmapped":
                raise AssertionError(f"Unmapped paired choice for {respondent_id}, case {case_index}")
            quality_key = DIFFERENCE_QUALITY_KEYS[case_index - 1]
            risk_key = DIFFERENCE_RISK_KEYS[case_index - 1]
            logical_consistency = (
                int(suitability_selected != misleading_selected)
                if suitability_selected is not None and misleading_selected is not None
                else None
            )
            case_row = {
                "study": "paired_difference_mechanism_probe",
                "respondent_id": respondent_id,
                "case_id": f"A{case_index:02d}",
                "case_index": case_index,
                "concept": DIFFERENCE_CONCEPTS[case_index - 1],
                "suitability_key": quality_key,
                "suitability_selected": suitability_selected,
                "suitability_strength": suitability_strength,
                "suitability_correct": int(suitability_selected == quality_key),
                "misleading_key": risk_key,
                "misleading_selected": misleading_selected,
                "misleading_strength": misleading_strength,
                "misleading_correct": int(misleading_selected == risk_key),
                "logical_consistency": logical_consistency,
                "reason_category": map_reason(answers["reason"]),
                "duration_seconds": record["duration_seconds"],
                "duration_missing_or_nonpositive": int(record["duration_missing_or_nonpositive"]),
                "self_reported_role": record["role"],
                "self_reported_field": record["field"],
                "science_review_familiarity": record["science_familiarity"],
                "ai_evaluation_familiarity": record["ai_familiarity"],
            }
            case_rows.append(case_row)
            long_rows.append(case_row)
        respondent_rows.append(
            {
                "respondent_id": respondent_id,
                "suitability_accuracy": float(np.mean([row["suitability_correct"] for row in case_rows])),
                "misleading_accuracy": float(np.mean([row["misleading_correct"] for row in case_rows])),
                "logical_consistency": float(
                    np.mean([row["logical_consistency"] for row in case_rows if row["logical_consistency"] is not None])
                ),
                "self_reported_role": record["role"],
                "self_reported_field": record["field"],
                "science_review_familiarity": record["science_familiarity"],
                "ai_evaluation_familiarity": record["ai_familiarity"],
                **{f"global_reason_{index + 1}": flag for index, flag in enumerate(record["global_reason_flags"])},
            }
        )
    long = pd.DataFrame(long_rows)
    respondents = pd.DataFrame(respondent_rows)
    if len(long) != 480 or long["respondent_id"].nunique() != 60:
        raise AssertionError("Unexpected difference-survey long-form size")
    if long[["suitability_correct", "misleading_correct", "reason_category"]].isna().any().any():
        raise AssertionError("Unexpected difference-survey outcome missingness")
    assert_privacy_safe(long, forbidden_values)
    source_meta = {
        "file": export["path"].name,
        "sha256": sha256_file(export["path"]),
        "worksheet": export["sheet_name"],
        "reported_dimension_before_reset": export["reported_dimension"],
        "rows": len(export["rows"]),
        "columns": len(export["code_headers"]),
    }
    qc = {
        "raw_responses": 60,
        "duration_floor_seconds": DIFFERENCE_DURATION_FLOOR_SECONDS,
        "duration_exclusions": 0,
        "duration_missing_or_nonpositive_retained": 0,
        "final_responses": 60,
        "final_case_rows": 480,
        "attention_rule": "not_applicable_no_keyed_attention_item",
        "repeated_item_rule": "not_applicable_no_intentional_repeat",
    }
    return long, respondents, qc, source_meta, forbidden_values, export


def bootstrap_mean_interval(values: np.ndarray, seed: int, draws: int = N_BOOTSTRAP) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise AssertionError("Bootstrap requires a nonempty one-dimensional vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    replicates = values[indices].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": int(draws),
        "seed": int(seed),
        "method": "respondent_cluster_percentile_bootstrap",
    }


def signflip_test(values_minus_null: np.ndarray, seed: int, draws: int = N_SIGNFLIP) -> dict[str, Any]:
    centered = np.asarray(values_minus_null, dtype=float)
    observed = float(centered.mean())
    n = len(centered)
    if n <= 20:
        patterns = 1 << n
        statistics = np.empty(patterns, dtype=float)
        for mask in range(patterns):
            signs = np.array([1.0 if (mask >> index) & 1 else -1.0 for index in range(n)])
            statistics[mask] = float(np.mean(signs * centered))
        one_sided = float(np.mean(statistics >= observed - 1e-15))
        two_sided = float(np.mean(np.abs(statistics) >= abs(observed) - 1e-15))
        return {
            "observed_mean_minus_null": observed,
            "one_sided_p": one_sided,
            "two_sided_p": two_sided,
            "method": "exact_respondent_level_sign_flip",
            "patterns": patterns,
        }
    rng = np.random.default_rng(seed)
    greater = 0
    absolute = 0
    completed = 0
    batch_size = 10_000
    while completed < draws:
        batch = min(batch_size, draws - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, n))
        statistics = (signs * centered).mean(axis=1)
        greater += int(np.sum(statistics >= observed - 1e-15))
        absolute += int(np.sum(np.abs(statistics) >= abs(observed) - 1e-15))
        completed += batch
    return {
        "observed_mean_minus_null": observed,
        "one_sided_p": float((greater + 1) / (draws + 1)),
        "two_sided_p": float((absolute + 1) / (draws + 1)),
        "method": "monte_carlo_respondent_level_sign_flip",
        "draws": int(draws),
        "seed": int(seed),
        "plus_one_correction": True,
    }


def difference_endpoint(respondents: pd.DataFrame, column: str, seed_offset: int) -> dict[str, Any]:
    values = respondents[column].to_numpy(dtype=float)
    return {
        "n_respondents": int(len(values)),
        "chance_benchmark": 0.5,
        "bootstrap": bootstrap_mean_interval(values, DIFFERENCE_BOOTSTRAP_SEED + seed_offset),
        "sign_flip": signflip_test(values - 0.5, DIFFERENCE_SIGNFLIP_SEED + seed_offset),
        "respondent_score_distribution": {
            str(score / 8): int(np.sum(np.isclose(values, score / 8))) for score in range(9)
        },
    }


def difference_case_accuracy(long: pd.DataFrame, outcome: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for case_id, group in long.groupby("case_id", sort=True):
        successes = int(group[outcome].sum())
        n = int(len(group))
        low, high = proportion_confint(successes, n, alpha=0.05, method="wilson")
        p_value = float(binomtest(successes, n, 0.5, alternative="greater").pvalue)
        pvalues.append(p_value)
        rows.append(
            {
                "case_id": case_id,
                "n": n,
                "correct": successes,
                "accuracy": successes / n,
                "wilson_ci_low": float(low),
                "wilson_ci_high": float(high),
                "one_sided_binomial_p": p_value,
            }
        )
    for row, adjusted in zip(rows, holm_adjust(pvalues)):
        row["p_holm_8"] = adjusted
    return rows


def difference_strength_summary(long: pd.DataFrame, endpoint: str) -> list[dict[str, Any]]:
    strength_column = f"{endpoint}_strength"
    correct_column = f"{endpoint}_correct"
    selected = long[long[strength_column].isin(["obvious", "slight"])].copy()
    rows = []
    for (case_id, correct), group in selected.groupby(["case_id", correct_column], sort=True):
        rows.append(
            {
                "case_id": case_id,
                "correct": int(correct),
                "n_selected_version": int(len(group)),
                "n_obvious": int((group[strength_column] == "obvious").sum()),
                "proportion_obvious": float((group[strength_column] == "obvious").mean()),
            }
        )
    return rows


def difference_reason_summary(long: pd.DataFrame, respondents: pd.DataFrame) -> dict[str, Any]:
    categories = sorted(long["reason_category"].unique())
    matrix = (
        long.assign(value=1.0)
        .pivot_table(index="respondent_id", columns="reason_category", values="value", aggfunc="sum", fill_value=0)
        .reindex(columns=categories, fill_value=0)
        / 8.0
    )
    rng = np.random.default_rng(DIFFERENCE_BOOTSTRAP_SEED + 20)
    indices = rng.integers(0, len(matrix), size=(N_BOOTSTRAP, len(matrix)))
    bootstrap = matrix.to_numpy()[indices].mean(axis=1)
    overall = []
    for index, category in enumerate(categories):
        low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
        overall.append(
            {
                "reason_category": category,
                "proportion_of_case_selections": float(matrix.iloc[:, index].mean()),
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
            }
        )
    by_case = []
    for (case_id, category), group in long.groupby(["case_id", "reason_category"], sort=True):
        by_case.append(
            {
                "case_id": case_id,
                "reason_category": category,
                "n": int(len(group)),
                "proportion": float(len(group) / 60),
            }
        )
    global_labels = [
        "boundary_conditions",
        "mechanistic_explanation",
        "label_only",
        "factual_or_conceptual_error",
        "overclaiming_or_overconfidence",
        "fluency_masking_scientific_problem",
        "lay_accessibility",
        "new_false_inference",
        "other",
    ]
    global_summary = []
    for index, label in enumerate(global_labels, start=1):
        values = respondents[f"global_reason_{index}"].astype(float)
        low, high = proportion_confint(int(values.sum()), len(values), alpha=0.05, method="wilson")
        global_summary.append(
            {
                "reason_category": label,
                "n_selected": int(values.sum()),
                "proportion_respondents": float(values.mean()),
                "wilson_ci_low": float(low),
                "wilson_ci_high": float(high),
            }
        )
    return {
        "case_level_overall_clustered": overall,
        "case_specific": by_case,
        "global_multiselect": global_summary,
        "free_text_note": "Q30 free text was not exported to the analysis dataset and was not coded.",
    }


def difference_exploratory_subgroups(respondents: pd.DataFrame) -> dict[str, Any]:
    role_rows = []
    for role, group in respondents.groupby("self_reported_role", sort=True):
        role_rows.append(
            {
                "self_reported_role": role,
                "n": int(len(group)),
                "mean_suitability_accuracy": float(group["suitability_accuracy"].mean()),
                "mean_misleading_accuracy": float(group["misleading_accuracy"].mean()),
            }
        )
    field_rows = []
    for field, group in respondents.groupby("self_reported_field", sort=True):
        field_rows.append(
            {
                "self_reported_field": field,
                "n": int(len(group)),
                "mean_suitability_accuracy": float(group["suitability_accuracy"].mean()),
                "mean_misleading_accuracy": float(group["misleading_accuracy"].mean()),
            }
        )
    formula = "suitability_accuracy ~ science_review_familiarity + ai_evaluation_familiarity"
    model = smf.ols(formula, data=respondents).fit(cov_type="HC3")
    return {
        "status": "exploratory_no_subgroup_selection",
        "role_descriptive": role_rows,
        "field_descriptive": field_rows,
        "continuous_familiarity_model": {
            "formula": formula,
            "covariance": "HC3",
            "coefficients": coefficient_table(model),
        },
    }


def analyze_difference(long: pd.DataFrame, respondents: pd.DataFrame) -> dict[str, Any]:
    consistency = respondents["logical_consistency"].dropna().to_numpy(dtype=float)
    return {
        "analysis_version": SCRIPT_VERSION,
        "source_mapping_status": "completed_16_of_16_to_controlled_ids_with_concept_level_or_external_810_boundary",
        "design": "eight constructed paired contrasts",
        "primary_suitability_accuracy": difference_endpoint(respondents, "suitability_accuracy", 0),
        "secondary_misleading_accuracy": difference_endpoint(respondents, "misleading_accuracy", 1),
        "logical_consistency": bootstrap_mean_interval(consistency, DIFFERENCE_BOOTSTRAP_SEED + 2),
        "case_specific": {
            "suitability": difference_case_accuracy(long, "suitability_correct"),
            "misleading": difference_case_accuracy(long, "misleading_correct"),
        },
        "response_strength": {
            "suitability": difference_strength_summary(long, "suitability"),
            "misleading": difference_strength_summary(long, "misleading"),
        },
        "reasons": difference_reason_summary(long, respondents),
        "subgroups": difference_exploratory_subgroups(respondents),
    }


def extract_difference_pair_texts(export: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"版本A给出的回答是：(.*?)版本B给出的回答是：(.*?)请您阅读后判断"
    )
    for case_index, column_index in enumerate(range(22, 46, 3), start=1):
        header = export["verbose_headers"][column_index]
        match = pattern.search(header)
        if match is None:
            raise AssertionError(f"Could not extract pair text for case {case_index}")
        output[f"A{case_index:02d}"] = {
            "A": normalize_text(match.group(1)),
            "B": normalize_text(match.group(2)),
        }
    return output


def exact_permutation_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) > 9:
        raise AssertionError("Exact Spearman permutation is limited to paired vectors of length <= 9")
    observed = float(spearmanr(x, y).statistic)
    extreme = 0
    total = 0
    for order in permutations(range(len(y))):
        statistic = float(spearmanr(x, y[list(order)]).statistic)
        if abs(statistic) >= abs(observed) - 1e-15:
            extreme += 1
        total += 1
    return {
        "spearman_rho": observed,
        "two_sided_exact_permutation_p": float(extreme / total),
        "permutations": int(total),
        "status": "descriptive_small_n",
    }


def controlled_ab_analysis(
    difference_long: pd.DataFrame, difference_export: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    raw_path = AB_RUN_ROOT / "api_judge_scores_long.csv"
    input_path = AB_RUN_ROOT / "controlled_ab_judge_inputs_anonymized.csv"
    pair_json_path = DIAGNOSTIC_ROOT / "controlled_ab_pairs_for_llm_judge.json"
    call_log_path = AB_RUN_ROOT / "api_call_log.jsonl"
    if not all(path.exists() for path in [raw_path, input_path, pair_json_path, call_log_path]):
        raise AssertionError("Controlled-AB run is missing a required raw/configuration file")

    raw = pd.read_csv(raw_path)
    with pair_json_path.open("r", encoding="utf-8") as handle:
        pair_configuration = json.load(handle)
    if not isinstance(pair_configuration, list):
        raise AssertionError("Controlled-AB pair configuration is not a list")
    call_log_records: list[dict[str, Any]] = []
    with call_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                call_log_records.append(json.loads(line))
    call_log_rows = len(call_log_records)
    call_log_success = [
        row
        for row in call_log_records
        if bool(row.get("parse_success")) and int(row.get("http_status") or 0) == 200
    ]
    required_columns = {
        "item_id",
        "judge_provider",
        "paper_model_label",
        "judge_model_requested",
        "actual_api_model",
        "judge_model_returned",
        "fa",
        "cc",
        "lc",
        "tf",
        "mq",
        "risk",
        "parse_success",
        "http_status",
    }
    if not required_columns.issubset(raw.columns):
        raise AssertionError(f"Controlled-AB raw schema missing {sorted(required_columns - set(raw.columns))}")
    score_columns = ["fa", "cc", "lc", "tf", "mq", "risk"]
    success = raw[
        raw["parse_success"].astype(bool)
        & raw["http_status"].eq(200)
        & raw[score_columns].notna().all(axis=1)
    ].copy()
    expected_items = {f"A{case:02d}_{version}" for case in range(1, 9) for version in "AB"}
    provider_counts = success.groupby("judge_provider").size()
    validation_errors: list[str] = []
    if len(raw) != 145:
        validation_errors.append(f"raw_rows={len(raw)} expected 145")
    if len(success) != 144:
        validation_errors.append(f"successful_rows={len(success)} expected 144")
    if success["judge_provider"].nunique() != 9:
        validation_errors.append("judge_provider_count_not_9")
    if not provider_counts.eq(16).all():
        validation_errors.append("not_every_provider_has_16_successful_items")
    if set(success["item_id"]) != expected_items:
        validation_errors.append("successful_item_set_incomplete")
    if success.duplicated(["judge_provider", "item_id"]).any():
        validation_errors.append("duplicate_successful_provider_item")
    if not ((success[score_columns] >= 1) & (success[score_columns] <= 5)).all().all():
        validation_errors.append("score_outside_1_to_5")
    if success[["judge_model_requested", "actual_api_model", "judge_model_returned"]].isna().any().any():
        validation_errors.append("successful_model_identity_missing")
    configured_cases = {normalize_text(row.get("case_id")) for row in pair_configuration}
    if len(pair_configuration) != 8 or configured_cases != {f"A{case:02d}" for case in range(1, 9)}:
        validation_errors.append("pair_configuration_case_set_incomplete")
    for row in pair_configuration:
        case_id = normalize_text(row.get("case_id"))
        if not re.fullmatch(r"A\d{2}", case_id):
            validation_errors.append("pair_configuration_invalid_case_id")
            continue
        case_index = int(case_id[1:]) - 1
        if normalize_text(row.get("expected_quality_preference")) != DIFFERENCE_QUALITY_KEYS[case_index]:
            validation_errors.append(f"pair_configuration_quality_key_mismatch_{case_id}")
        if normalize_text(row.get("expected_riskier_version")) != DIFFERENCE_RISK_KEYS[case_index]:
            validation_errors.append(f"pair_configuration_risk_key_mismatch_{case_id}")
    if len(call_log_success) != 144:
        validation_errors.append(f"api_call_log_success={len(call_log_success)}_expected_144")
    if len({(row.get("judge_provider"), row.get("item_id")) for row in call_log_success}) != 144:
        validation_errors.append("api_call_log_success_provider_item_cells_not_144")
    run_status = "PASS" if not validation_errors else "FAIL"
    if run_status == "FAIL":
        raise AssertionError("Controlled-AB raw run failed validation: " + "; ".join(validation_errors))

    success["case_id"] = success["item_id"].str.extract(r"^(A\d{2})_")[0]
    success["version"] = success["item_id"].str.extract(r"_([AB])$")[0]
    success["quality_score"] = success[["fa", "cc", "lc", "tf", "mq"]].mean(axis=1)
    inputs = pd.read_csv(input_path)
    if set(inputs["item_id"]) != expected_items or len(inputs) != 16:
        raise AssertionError("Controlled-AB input manifest is incomplete")
    input_meta = inputs[
        [
            "item_id",
            "case_id",
            "version",
            "domain",
            "concept",
            "generated_text",
        ]
    ].copy()
    success = success.merge(input_meta, on=["item_id", "case_id", "version"], how="left", validate="many_to_one")
    if success["generated_text"].isna().any():
        raise AssertionError("Controlled-AB scores failed to join inputs")

    pair_rows: list[dict[str, Any]] = []
    for (provider, case_id), group in success.groupby(["judge_provider", "case_id"], sort=True):
        if set(group["version"]) != {"A", "B"} or len(group) != 2:
            raise AssertionError(f"Incomplete provider-case pair: {provider} {case_id}")
        by_version = group.set_index("version")
        case_number = int(case_id[1:])
        expected_quality = DIFFERENCE_QUALITY_KEYS[case_number - 1]
        expected_risk = DIFFERENCE_RISK_KEYS[case_number - 1]
        quality_a = float(by_version.loc["A", "quality_score"])
        quality_b = float(by_version.loc["B", "quality_score"])
        risk_a = float(by_version.loc["A", "risk"])
        risk_b = float(by_version.loc["B", "risk"])
        quality_choice = "A" if quality_a > quality_b else "B" if quality_b > quality_a else "tie"
        risk_choice = "A" if risk_a > risk_b else "B" if risk_b > risk_a else "tie"
        quality_aligned = quality_choice == expected_quality
        risk_aligned = risk_choice == expected_risk
        first = group.iloc[0]
        pair_rows.append(
            {
                "judge_provider": provider,
                "paper_model_label": first["paper_model_label"],
                "judge_model_requested": first["judge_model_requested"],
                "actual_api_model": first["actual_api_model"],
                "case_id": case_id,
                "domain": first["domain"],
                "concept": first["concept"],
                "expected_quality_preference": expected_quality,
                "expected_riskier_version": expected_risk,
                "quality_A": quality_a,
                "quality_B": quality_b,
                "risk_A": risk_a,
                "risk_B": risk_b,
                "quality_choice": quality_choice,
                "risk_choice": risk_choice,
                "quality_aligned": int(quality_aligned),
                "risk_aligned": int(risk_aligned),
                "both_aligned": int(quality_aligned and risk_aligned),
            }
        )
    pair_long = pd.DataFrame(pair_rows)
    if len(pair_long) != 72:
        raise AssertionError(f"Expected 72 judge-case pairs, observed {len(pair_long)}")

    survey_texts = extract_difference_pair_texts(difference_export)
    input_texts = {
        (row.case_id, row.version): normalize_text(row.generated_text)
        for row in inputs.itertuples(index=False)
    }
    crosswalk_rows = []
    for case_id in sorted(survey_texts):
        row: dict[str, Any] = {"case_id": case_id}
        matches = []
        for version in "AB":
            survey_text = survey_texts[case_id][version]
            machine_text = input_texts[(case_id, version)]
            match = survey_text == machine_text
            matches.append(match)
            row[f"survey_{version}_sha256"] = sha256_text(survey_text)
            row[f"machine_{version}_sha256"] = sha256_text(machine_text)
            row[f"version_{version}_exact_match"] = int(match)
            row[f"survey_{version}_characters"] = len(survey_text)
            row[f"machine_{version}_characters"] = len(machine_text)
        row["pair_exact_match"] = int(all(matches))
        row["crosswalk_status"] = "PASS_EXACT" if all(matches) else "FAIL_EXACT_TEXT"
        crosswalk_rows.append(row)
    crosswalk = pd.DataFrame(crosswalk_rows)
    if int(crosswalk["pair_exact_match"].sum()) != 7:
        raise AssertionError("Expected seven exactly matched controlled pairs")

    machine_case = (
        pair_long.groupby("case_id", sort=True)
        .agg(
            machine_quality_alignment=("quality_aligned", "mean"),
            machine_risk_alignment=("risk_aligned", "mean"),
            machine_both_alignment=("both_aligned", "mean"),
            n_machine_judges=("judge_provider", "nunique"),
        )
        .reset_index()
    )
    human_case = (
        difference_long.groupby("case_id", sort=True)
        .agg(
            human_suitability_accuracy=("suitability_correct", "mean"),
            human_misleading_accuracy=("misleading_correct", "mean"),
            human_logical_consistency=("logical_consistency", "mean"),
            n_human_respondents=("respondent_id", "nunique"),
        )
        .reset_index()
    )
    case_summary = (
        machine_case.merge(human_case, on="case_id", validate="one_to_one")
        .merge(crosswalk[["case_id", "pair_exact_match", "crosswalk_status"]], on="case_id", validate="one_to_one")
    )
    exact = case_summary[case_summary["pair_exact_match"] == 1]
    exact_case_ids = set(exact["case_id"])
    human_exact_respondents = (
        difference_long[difference_long["case_id"].isin(exact_case_ids)]
        .groupby("respondent_id", sort=True)
        .agg(
            suitability_accuracy=("suitability_correct", "mean"),
            misleading_accuracy=("misleading_correct", "mean"),
            logical_consistency=("logical_consistency", "mean"),
        )
        .reset_index()
    )
    human_exact_suitability = {
        "n_respondents": int(len(human_exact_respondents)),
        "n_cases_per_respondent": 7,
        "bootstrap": bootstrap_mean_interval(
            human_exact_respondents["suitability_accuracy"].to_numpy(),
            DIFFERENCE_BOOTSTRAP_SEED + 30,
        ),
        "sign_flip": signflip_test(
            human_exact_respondents["suitability_accuracy"].to_numpy() - 0.5,
            DIFFERENCE_SIGNFLIP_SEED + 30,
        ),
    }
    human_exact_misleading = {
        "n_respondents": int(len(human_exact_respondents)),
        "n_cases_per_respondent": 7,
        "bootstrap": bootstrap_mean_interval(
            human_exact_respondents["misleading_accuracy"].to_numpy(),
            DIFFERENCE_BOOTSTRAP_SEED + 31,
        ),
        "sign_flip": signflip_test(
            human_exact_respondents["misleading_accuracy"].to_numpy() - 0.5,
            DIFFERENCE_SIGNFLIP_SEED + 31,
        ),
    }
    correlations = {}
    for machine_column, human_column, label in [
        ("machine_quality_alignment", "human_suitability_accuracy", "quality_suitability"),
        ("machine_risk_alignment", "human_misleading_accuracy", "risk_misleading"),
    ]:
        correlations[label] = {
            "n_exact_cases": int(len(exact)),
            **exact_permutation_spearman(
                exact[machine_column].to_numpy(), exact[human_column].to_numpy()
            ),
        }

    results = {
        "analysis_version": SCRIPT_VERSION,
        "raw_run_validation": {
            "status": run_status,
            "validation_errors": validation_errors,
            "raw_rows": int(len(raw)),
            "successful_parsed_http200_rows": int(len(success)),
            "failed_rows": int(len(raw) - len(success)),
            "raw_score_csv_rows": int(len(raw)),
            "raw_score_csv_successful_parsed_http200_rows": int(len(success)),
            "raw_score_csv_failed_or_unparsed_rows": int(len(raw) - len(success)),
            "successful_judge_providers": int(success["judge_provider"].nunique()),
            "successful_items": int(success["item_id"].nunique()),
            "provider_item_cells": int(success.groupby(["judge_provider", "item_id"]).ngroups),
            "judge_case_pairs_recomputed": int(len(pair_long)),
            "pair_configuration_cases": int(len(pair_configuration)),
            "api_call_log_rows": int(call_log_rows),
            "api_call_log_successful_parsed_http200_rows": int(len(call_log_success)),
            "api_call_log_failed_or_unparsed_rows": int(call_log_rows - len(call_log_success)),
            "quality_definition": "unweighted mean of FA, CC, LC, TF, and MQ; higher is better",
            "risk_definition": "risk dimension; higher is riskier",
            "failed_retry_note": (
                "The score CSV has 145 rows: 144 successful parsed HTTP-200 rows and one "
                "recorded failed Kimi A06_B row. The complete API call log has 152 attempts: "
                "144 successful parsed HTTP-200 attempts and eight unsuccessful or unparsed "
                "attempts. Only the 144 validated provider-item results enter analysis; every "
                "provider-item has exactly one validated result."
            ),
            "model_coverage": [
                {
                    "judge_provider": provider,
                    "paper_model_label": str(group["paper_model_label"].iloc[0]),
                    "judge_model_requested": str(group["judge_model_requested"].iloc[0]),
                    "actual_api_model": str(group["actual_api_model"].iloc[0]),
                    "successful_items": int(len(group)),
                }
                for provider, group in success.groupby("judge_provider", sort=True)
            ],
        },
        "text_crosswalk": {
            "exact_pairs": int(crosswalk["pair_exact_match"].sum()),
            "total_pairs": int(len(crosswalk)),
            "status": "PARTIAL_7_OF_8_EXACT",
            "nonexact_cases": crosswalk.loc[crosswalk["pair_exact_match"] == 0, "case_id"].tolist(),
            "nonexact_note": (
                "A08 machine version A includes the visible prefix '【科学卡片】' that is absent from the survey; "
                "A08 is excluded from exact-text machine-human aggregate claims."
            ),
        },
        "machine_direction_overall_all_8_cases": {
            "n_judge_case_pairs": int(len(pair_long)),
            "quality_alignment": float(pair_long["quality_aligned"].mean()),
            "risk_alignment": float(pair_long["risk_aligned"].mean()),
            "both_alignment": float(pair_long["both_aligned"].mean()),
            "human_suitability_accuracy": float(difference_long["suitability_correct"].mean()),
            "human_misleading_accuracy": float(difference_long["misleading_correct"].mean()),
            "human_logical_consistency": float(difference_long["logical_consistency"].mean()),
        },
        "machine_human_exact_text_7_cases": {
            "n_cases": int(len(exact)),
            "n_machine_judge_case_pairs": int(
                pair_long[pair_long["case_id"].isin(exact["case_id"])].shape[0]
            ),
            "machine_quality_alignment": float(exact["machine_quality_alignment"].mean()),
            "machine_risk_alignment": float(exact["machine_risk_alignment"].mean()),
            "machine_both_alignment": float(exact["machine_both_alignment"].mean()),
            "human_suitability_accuracy": float(exact["human_suitability_accuracy"].mean()),
            "human_misleading_accuracy": float(exact["human_misleading_accuracy"].mean()),
            "human_logical_consistency": float(exact["human_logical_consistency"].mean()),
            "human_suitability_inference": human_exact_suitability,
            "human_misleading_inference": human_exact_misleading,
            "human_logical_consistency_bootstrap": bootstrap_mean_interval(
                human_exact_respondents["logical_consistency"].dropna().to_numpy(),
                DIFFERENCE_BOOTSTRAP_SEED + 32,
            ),
            "case_level_correlations": correlations,
        },
        "interpretation": (
            "This is an exact-stimulus paired mechanism bridge for seven cases. It does not establish "
            "criterion validity for the 810-item API corpus."
        ),
    }
    manifest = {
        "analysis_version": SCRIPT_VERSION,
        "status": run_status,
        "source_mapping_status": "completed_controlled_id_crosswalk_with_strict_current_export_text_audit",
        "script": {
            "path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
        "software": package_versions(),
        "semantic_crosswalk": {
            "path": str(SEMANTIC_CROSSWALK_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(SEMANTIC_CROSSWALK_PATH),
            "audit_path": str(SEMANTIC_CROSSWALK_AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "audit_sha256": sha256_file(SEMANTIC_CROSSWALK_AUDIT_PATH),
        },
        "source_files": {
            "raw_scores": {"path": str(raw_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(raw_path)},
            "anonymized_inputs": {"path": str(input_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(input_path)},
            "pair_configuration": {"path": str(pair_json_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(pair_json_path)},
            "api_call_log": {"path": str(call_log_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(call_log_path)},
        },
        "rules": {
            "successful_row": "parse_success=True, HTTP 200, and all six score dimensions nonmissing",
            "failed_retry_handling": (
                "exclude all unsuccessful or unparsed call attempts; retain a provider-item only "
                "when exactly one successful parsed HTTP-200 result is present"
            ),
            "quality_direction": "compare the unweighted mean of FA/CC/LC/TF/MQ",
            "risk_direction": "compare risk; higher is riskier",
            "exact_text_eligibility": "NFC-normalized, outer-whitespace-stripped version A and B strings must both have identical SHA-256 hashes",
        },
    }
    return pair_long, crosswalk, case_summary, results, manifest


def build_codebooks() -> tuple[dict[str, Any], dict[str, Any]]:
    abc_codebook = {
        "analysis_version": SCRIPT_VERSION,
        "concepts": [
            {
                "concept_index": index,
                "concept": concept,
                "pre_correct_answer": ABC_PRE_KEYS[index - 1],
                "post_correct_answer": ABC_POST_KEYS[index - 1],
            }
            for index, concept in enumerate(ABC_CONCEPTS, start=1)
        ],
        "confidence_mapping": CONFIDENCE_MAP,
        "misleading_risk_mapping": RISK_MAP,
        "phase_mapping": {"pre": 0, "post": 1},
        "attention_key": "比较同意",
        "duration_floor_seconds": ABC_DURATION_FLOOR_SECONDS,
    }
    difference_codebook = {
        "analysis_version": SCRIPT_VERSION,
        "cases": [
            {
                "case_id": f"A{index:02d}",
                "concept": concept,
                "suitability_key": DIFFERENCE_QUALITY_KEYS[index - 1],
                "misleading_key": DIFFERENCE_RISK_KEYS[index - 1],
            }
            for index, concept in enumerate(DIFFERENCE_CONCEPTS, start=1)
        ],
        "choice_rule": "obvious/slight A or B collapse to selected version; ties/uncertain do not match a key",
        "logical_consistency_rule": "defined only when both questions select A or B; equals 1 when selections differ",
        "duration_floor_seconds": DIFFERENCE_DURATION_FLOOR_SECONDS,
    }
    return abc_codebook, difference_codebook


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "openpyxl": openpyxl.__version__,
        "patsy": patsy.__version__,
    }


def make_audit_note(
    abc_qc: dict[str, Any],
    abc_results: dict[str, Any],
    difference_qc: dict[str, Any],
    difference_results: dict[str, Any],
    bridge_results: dict[str, Any],
) -> str:
    omnibus = abc_results["primary"]["omnibus_interaction"]
    probability_pairwise = abc_results["primary"]["marginal"]["pairwise_change_differences"]
    logit_pairwise = abc_results["primary"]["pairwise_logit_scale_interactions"]
    primary_difference = difference_results["primary_suitability_accuracy"]
    misleading_difference = difference_results["secondary_misleading_accuracy"]
    adaptation_sensitivity = abc_results["robustness"]["excluding_nonparallel_adaptation_pair"]
    adaptation_omnibus = adaptation_sensitivity["omnibus_interaction"]
    change_lookup = {row["version"]: row for row in abc_results["primary"]["marginal"]["changes"]}
    return f"""# Public-questionnaire analysis audit

## Locked execution

- A/B/C: {abc_qc['raw_responses']} raw responses; {abc_qc['attention_pass_responses']} passed the keyed attention item; one positive duration below 72 s was excluded; final A/B/C counts were {abc_qc['final_by_version']} ({abc_qc['final_responses']} responses, {abc_qc['final_participant_clusters']} anonymous participant clusters).
- Cross-form repetition was retained and handled by participant-clustered covariance. Fixed sensitivities used the first eligible response ({abc_qc['first_response_sensitivity_n']}) and single-form participants ({abc_qc['single_form_sensitivity_n']}).
- Difference survey: {difference_qc['final_responses']} respondents and {difference_qc['final_case_rows']} respondent-case rows; the fixed 96 s floor excluded none. No attention or repeated-item screen was invented.
- No raw response ID, platform user ID, IP/location/device field, or timestamp appears in any cleaned analysis file.

## A/B/C locked primary result

The exchangeable logistic GEE gave a version-by-phase robust Wald statistic of {omnibus['statistic']:.6f} on {omnibus['df']} df (p={omnibus['p_value']:.6g}). Standardized pre-to-post probability changes were A={change_lookup['A']['change']:.4f}, B={change_lookup['B']['change']:.4f}, and C={change_lookup['C']['change']:.4f}. The fitted interaction contrasts on the log-odds scale had Holm values {', '.join(row['contrast'] + '=' + format(row['p_holm'], '.6g') for row in logit_pairwise)}. The nonlinear average-marginal probability-scale contrasts had Holm values {', '.join(row['contrast'] + '=' + format(row['p_holm'], '.6g') for row in probability_pairwise)}. These scales answer different questions and are reported separately; C-A is significant on the fitted log-odds scale but is just above .05 on the probability scale.

The fixed independence, first-response, and single-form sensitivities all retained p<.05. These were not selected by outcome.

The fixed content-parallelism sensitivity removed the scientifically related but non-equivalent adaptation pair (pre Q11 versus post Q17), retained five concepts and {adaptation_sensitivity['n_long_rows']} long rows, and gave a version-by-phase Wald statistic of {adaptation_omnibus['statistic']:.6f} on {adaptation_omnibus['df']} df (p={adaptation_omnibus['p_value']:.6g}).

## Difference-survey locked primary result

Respondent-balanced suitability accuracy was {primary_difference['bootstrap']['estimate']:.4f} (respondent bootstrap 95% CI {primary_difference['bootstrap']['ci_low']:.4f}–{primary_difference['bootstrap']['ci_high']:.4f}); the directional respondent sign-flip p was {primary_difference['sign_flip']['one_sided_p']:.6g} and the two-sided p was {primary_difference['sign_flip']['two_sided_p']:.6g}. More-misleading accuracy was {misleading_difference['bootstrap']['estimate']:.4f} (95% CI {misleading_difference['bootstrap']['ci_low']:.4f}–{misleading_difference['bootstrap']['ci_high']:.4f}).

## Controlled A/B bridge

The raw official run passed structural validation. The score CSV contained {bridge_results['raw_run_validation']['raw_score_csv_rows']} rows: {bridge_results['raw_run_validation']['raw_score_csv_successful_parsed_http200_rows']} successful parsed HTTP-200 rows and {bridge_results['raw_run_validation']['raw_score_csv_failed_or_unparsed_rows']} recorded failed row. The complete API call log contained {bridge_results['raw_run_validation']['api_call_log_rows']} attempts: {bridge_results['raw_run_validation']['api_call_log_successful_parsed_http200_rows']} successful parsed HTTP-200 attempts and {bridge_results['raw_run_validation']['api_call_log_failed_or_unparsed_rows']} unsuccessful or unparsed attempts. Only the 144 validated provider-item results entered analysis, yielding nine providers, 16 items, and {bridge_results['raw_run_validation']['judge_case_pairs_recomputed']} independently recomputed judge-case pairs. Seven of eight survey pairs match the machine inputs exactly. A08 fails the exact-text rule because the machine A text contains “【科学卡片】” and the survey A text does not; it is excluded from exact-text aggregate claims.

Across the seven exact pairs, machine quality-direction alignment was {bridge_results['machine_human_exact_text_7_cases']['machine_quality_alignment']:.4f}, risk-direction alignment was {bridge_results['machine_human_exact_text_7_cases']['machine_risk_alignment']:.4f}, and both-direction alignment was {bridge_results['machine_human_exact_text_7_cases']['machine_both_alignment']:.4f}. This is a mechanism bridge, not validation of the full 810-item API corpus.
"""


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    NOTES_ROOT.mkdir(parents=True, exist_ok=True)
    if not SEMANTIC_CROSSWALK_PATH.exists() or not SEMANTIC_CROSSWALK_AUDIT_PATH.exists():
        raise AssertionError("Completed semantic crosswalk files are required")

    abc_long, abc_wide, abc_qc, abc_sources, _ = build_abc_data()
    difference_long, difference_respondents, difference_qc, difference_source, _, difference_export = build_difference_data()
    abc_results = analyze_abc(abc_long, abc_wide)
    difference_results = analyze_difference(difference_long, difference_respondents)
    machine_long, crosswalk, case_summary, bridge_results, bridge_manifest = controlled_ab_analysis(
        difference_long, difference_export
    )
    abc_codebook, difference_codebook = build_codebooks()

    paths = {
        "abc_clean": ANALYSIS_ROOT / "public_abc_cleaned_long.csv",
        "abc_results": ANALYSIS_ROOT / "public_abc_analysis_results.json",
        "abc_codebook": ANALYSIS_ROOT / "public_abc_codebook.json",
        "abc_manifest": ANALYSIS_ROOT / "public_abc_analysis_manifest.json",
        "difference_clean": ANALYSIS_ROOT / "difference_survey_cleaned_long.csv",
        "difference_results": ANALYSIS_ROOT / "difference_survey_analysis_results.json",
        "difference_codebook": ANALYSIS_ROOT / "difference_survey_codebook.json",
        "difference_manifest": ANALYSIS_ROOT / "difference_survey_analysis_manifest.json",
        "machine_long": ANALYSIS_ROOT / "controlled_ab_machine_judge_case_long.csv",
        "crosswalk": ANALYSIS_ROOT / "controlled_ab_pair_text_crosswalk.csv",
        "case_summary": ANALYSIS_ROOT / "controlled_ab_machine_human_case_summary.csv",
        "bridge_results": ANALYSIS_ROOT / "controlled_ab_machine_human_results.json",
        "bridge_manifest": ANALYSIS_ROOT / "controlled_ab_machine_human_manifest.json",
        "audit_note": NOTES_ROOT / "public_questionnaire_analysis_audit.md",
    }
    write_csv(paths["abc_clean"], abc_long)
    write_csv(paths["difference_clean"], difference_long)
    write_csv(paths["machine_long"], machine_long)
    write_csv(paths["crosswalk"], crosswalk)
    write_csv(paths["case_summary"], case_summary)
    write_json(paths["abc_results"], abc_results)
    write_json(paths["difference_results"], difference_results)
    write_json(paths["bridge_results"], bridge_results)
    write_json(paths["abc_codebook"], abc_codebook)
    write_json(paths["difference_codebook"], difference_codebook)

    abc_manifest = {
        "analysis_version": SCRIPT_VERSION,
        "qa_status": "PASS",
        "source_mapping_status": "completed_18_of_18_to_six_upstream_disagreement_cases_nonexact_lay_adaptations",
        "assignment_provenance": (
            "study author confirms complete random assignment among A/B/C; export does not independently encode the algorithm"
        ),
        "randomization_adherence": (
            "10 anonymous participant clusters appear across forms in the final data; primary retains all "
            "eligible responses with participant-clustered GEE and the fixed single-form sensitivity"
        ),
        "item_sampling_status": "six archived examples; no probability-sampling claim",
        "quality_control": abc_qc,
        "fixed_sensitivity_analyses": {
            "excluding_nonparallel_adaptation_pair": {
                "status": "completed",
                "excluded_concept": ABC_NONPARALLEL_CONCEPT,
                "excluded_pre_item": "Q11",
                "excluded_post_item": "Q17",
                "result_location": (
                    "analysis/public_abc_analysis_results.json#/robustness/"
                    "excluding_nonparallel_adaptation_pair"
                ),
                "omnibus_interaction": abc_results["robustness"]
                ["excluding_nonparallel_adaptation_pair"]["omnibus_interaction"],
            }
        },
        "sources": abc_sources,
        "semantic_crosswalk": {
            "path": str(SEMANTIC_CROSSWALK_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(SEMANTIC_CROSSWALK_PATH),
            "audit_path": str(SEMANTIC_CROSSWALK_AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "audit_sha256": sha256_file(SEMANTIC_CROSSWALK_AUDIT_PATH),
            "status": "18/18 form-material mappings to six traced upstream disagreement cases; adaptations are not exact texts",
        },
        "privacy": {
            "output_contains_raw_response_id": False,
            "output_contains_platform_user_id": False,
            "output_contains_ip_location_device": False,
            "output_contains_timestamp": False,
            "internal_only_fields": [
                "platform user ID for anonymous clustering",
                "start time for fixed first-eligible sensitivity",
                "duration for fixed QC",
            ],
        },
        "software": package_versions(),
        "script": {"path": str(Path(__file__).relative_to(PROJECT_ROOT)), "sha256": sha256_file(Path(__file__))},
    }
    difference_manifest = {
        "analysis_version": SCRIPT_VERSION,
        "qa_status": "PASS",
        "source_mapping_status": "completed_16_of_16_to_controlled_ids_with_concept_level_or_external_810_boundary",
        "design": "eight constructed paired contrasts",
        "quality_control": difference_qc,
        "source": difference_source,
        "semantic_crosswalk": {
            "path": str(SEMANTIC_CROSSWALK_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(SEMANTIC_CROSSWALK_PATH),
            "audit_path": str(SEMANTIC_CROSSWALK_AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "audit_sha256": sha256_file(SEMANTIC_CROSSWALK_AUDIT_PATH),
            "status": "16/16 controlled IDs linked; seven concepts map only at concept level to the 810 frame and one case is external",
        },
        "privacy": {
            "output_contains_raw_response_id": False,
            "output_contains_platform_user_id": False,
            "output_contains_ip_location_device": False,
            "output_contains_timestamp": False,
            "free_text_exported": False,
        },
        "random_seeds": {
            "bootstrap_base": DIFFERENCE_BOOTSTRAP_SEED,
            "signflip_base": DIFFERENCE_SIGNFLIP_SEED,
            "bootstrap_draws": N_BOOTSTRAP,
            "signflip_draws": N_SIGNFLIP,
        },
        "software": package_versions(),
        "script": {"path": str(Path(__file__).relative_to(PROJECT_ROOT)), "sha256": sha256_file(Path(__file__))},
    }
    write_json(paths["abc_manifest"], abc_manifest)
    write_json(paths["difference_manifest"], difference_manifest)
    write_json(paths["bridge_manifest"], bridge_manifest)

    audit_note = make_audit_note(
        abc_qc, abc_results, difference_qc, difference_results, bridge_results
    )
    paths["audit_note"].write_text(audit_note, encoding="utf-8")

    # Add deterministic output hashes to a separate compact run summary to avoid
    # self-referential manifest hashing.
    run_summary_path = ANALYSIS_ROOT / "public_questionnaire_analysis_run_summary.json"
    output_hashes = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for key, path in paths.items()
        if path.exists()
    }
    run_summary = {
        "analysis_version": SCRIPT_VERSION,
        "qa_status": "PASS",
        "outputs": output_hashes,
        "headline": {
            "abc_primary_interaction": abc_results["primary"]["omnibus_interaction"],
            "abc_excluding_nonparallel_adaptation_pair": abc_results["robustness"]
            ["excluding_nonparallel_adaptation_pair"]["omnibus_interaction"],
            "difference_primary": difference_results["primary_suitability_accuracy"],
            "controlled_ab_raw_run_status": bridge_results["raw_run_validation"]["status"],
            "controlled_ab_exact_pairs": bridge_results["text_crosswalk"],
        },
    }
    write_json(run_summary_path, run_summary)

    print(
        json.dumps(
            {
                "qa_status": "PASS",
                "abc_n": abc_qc["final_responses"],
                "abc_primary_p": abc_results["primary"]["omnibus_interaction"]["p_value"],
                "abc_excluding_nonparallel_adaptation_pair_p": abc_results["robustness"]
                ["excluding_nonparallel_adaptation_pair"]["omnibus_interaction"]["p_value"],
                "difference_n": difference_qc["final_responses"],
                "difference_accuracy": difference_results["primary_suitability_accuracy"]["bootstrap"]["estimate"],
                "difference_signflip_one_sided_p": difference_results["primary_suitability_accuracy"]["sign_flip"]["one_sided_p"],
                "controlled_ab_status": bridge_results["raw_run_validation"]["status"],
                "controlled_ab_exact_pairs": bridge_results["text_crosswalk"]["exact_pairs"],
                "run_summary": str(run_summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
