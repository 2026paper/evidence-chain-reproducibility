#!/usr/bin/env python3
"""Recompute public-study headline checks from the privacy-minimized release.

This entry point never reads raw workbooks, exact durations, background fields,
API call logs, complete prompts, or re-identification maps.  It recomputes the
A/B/C primary GEE, the fixed Q11/Q17 (adaptation) exclusion sensitivity, and
the controlled-case machine/human aggregate bridge from released tables only.
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

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUTPUT = ROOT / "verification" / "cleaned_only_public_recalculation.json"
FLOAT_TOLERANCE = 1e-10

FORMAL_SCRIPT = Path(__file__).resolve().with_name("public_questionnaire_analyses.py")
SPEC = importlib.util.spec_from_file_location("release_frozen_public_analysis", FORMAL_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot import frozen formal script: {FORMAL_SCRIPT}")
formal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = formal
SPEC.loader.exec_module(formal)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"Non-finite result: {value!r}")
    return number


def close(observed: Any, frozen: Any, label: str, tolerance: float = FLOAT_TOLERANCE) -> float:
    observed_number = finite_float(observed)
    frozen_number = finite_float(frozen)
    difference = observed_number - frozen_number
    if abs(difference) > tolerance:
        raise AssertionError(
            f"{label} differs from frozen result: observed={observed_number}, "
            f"frozen={frozen_number}, difference={difference}"
        )
    return difference


def as_binary(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    result = normalized.map(mapping)
    if result.isna().any():
        raise AssertionError(f"Unmapped binary value in {label}: {sorted(normalized[result.isna()].unique())}")
    return result.astype(int)


def recompute_abc() -> dict[str, Any]:
    path = ANALYSIS / "public_abc_cleaned_long.csv"
    frame = pd.read_csv(path)
    required = {
        "response_id", "participant_id", "version", "concept_index", "concept",
        "phase", "correct", "first_eligible_response", "single_form_participant",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"Released A/B/C table lacks {missing}")
    forbidden = {
        "duration_seconds", "education", "broad_field", "science_reading_frequency",
        "platform_user_id", "raw_response_id", "timestamp",
    }
    present = sorted(forbidden & set(frame.columns))
    if present:
        raise AssertionError(f"Privacy-forbidden A/B/C columns present: {present}")
    for column in ["concept_index", "phase", "correct", "first_eligible_response", "single_form_participant"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    if len(frame) != 2136 or frame["response_id"].nunique() != 178 or frame["participant_id"].nunique() != 167:
        raise AssertionError("Unexpected released A/B/C sample structure")

    primary_fit = formal.fit_abc_primary(frame, "exchangeable")
    primary = formal.summarize_abc_primary(primary_fit, frame)
    exclusion = formal.abc_excluding_nonparallel_adaptation_pair(frame)
    frozen = json.loads((ANALYSIS / "public_abc_analysis_results.json").read_text(encoding="utf-8"))
    primary_frozen = frozen["primary"]["omnibus_interaction"]
    exclusion_frozen = frozen["robustness"]["excluding_nonparallel_adaptation_pair"]["omnibus_interaction"]

    comparisons = {
        "primary_statistic_difference": close(
            primary["omnibus_interaction"]["statistic"],
            primary_frozen["statistic"],
            "A/B/C primary Wald statistic",
        ),
        "primary_p_difference": close(
            primary["omnibus_interaction"]["p_value"],
            primary_frozen["p_value"],
            "A/B/C primary p value",
        ),
        "q11_q17_exclusion_statistic_difference": close(
            exclusion["omnibus_interaction"]["statistic"],
            exclusion_frozen["statistic"],
            "Q11/Q17 exclusion Wald statistic",
        ),
        "q11_q17_exclusion_p_difference": close(
            exclusion["omnibus_interaction"]["p_value"],
            exclusion_frozen["p_value"],
            "Q11/Q17 exclusion p value",
        ),
    }
    return {
        "input": {
            "path": "analysis/public_abc_cleaned_long.csv",
            "sha256": sha256_file(path),
            "rows": len(frame),
            "response_instances": int(frame["response_id"].nunique()),
            "participant_clusters": int(frame["participant_id"].nunique()),
        },
        "primary": primary,
        "q11_q17_exclusion": exclusion,
        "frozen_result_comparison": comparisons,
    }


def case_aggregate(machine: pd.DataFrame, human: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    for column in ["quality_aligned", "risk_aligned", "both_aligned"]:
        machine[column] = as_binary(machine[column], column)
    for column in ["suitability_correct", "misleading_correct", "logical_consistency"]:
        human[column] = pd.to_numeric(human[column], errors="raise").astype(float)
    crosswalk["pair_exact_match"] = pd.to_numeric(crosswalk["pair_exact_match"], errors="raise").astype(int)
    machine_case = (
        machine.groupby("case_id", sort=True)
        .agg(
            machine_quality_alignment=("quality_aligned", "mean"),
            machine_risk_alignment=("risk_aligned", "mean"),
            machine_both_alignment=("both_aligned", "mean"),
            n_machine_judges=("judge_provider", "nunique"),
        )
        .reset_index()
    )
    human_case = (
        human.groupby("case_id", sort=True)
        .agg(
            human_suitability_accuracy=("suitability_correct", "mean"),
            human_misleading_accuracy=("misleading_correct", "mean"),
            human_logical_consistency=("logical_consistency", "mean"),
            n_human_respondents=("respondent_id", "nunique"),
        )
        .reset_index()
    )
    return (
        machine_case.merge(human_case, on="case_id", validate="one_to_one")
        .merge(
            crosswalk[["case_id", "pair_exact_match", "crosswalk_status"]],
            on="case_id",
            validate="one_to_one",
        )
        .sort_values("case_id")
        .reset_index(drop=True)
    )


def compare_case_summary(observed: pd.DataFrame, frozen: pd.DataFrame) -> dict[str, Any]:
    observed = observed.sort_values("case_id").reset_index(drop=True)
    frozen = frozen.sort_values("case_id").reset_index(drop=True)
    if list(observed["case_id"]) != list(frozen["case_id"]):
        raise AssertionError("Controlled case IDs differ from frozen summary")
    numeric = [
        "machine_quality_alignment", "machine_risk_alignment", "machine_both_alignment",
        "n_machine_judges", "human_suitability_accuracy", "human_misleading_accuracy",
        "human_logical_consistency", "n_human_respondents", "pair_exact_match",
    ]
    max_difference = 0.0
    for column in numeric:
        difference = np.max(np.abs(observed[column].astype(float) - frozen[column].astype(float)))
        max_difference = max(max_difference, float(difference))
    if max_difference > FLOAT_TOLERANCE:
        raise AssertionError(f"Controlled case summary differs by {max_difference}")
    if list(observed["crosswalk_status"].astype(str)) != list(frozen["crosswalk_status"].astype(str)):
        raise AssertionError("Controlled crosswalk status differs")
    return {"rows": len(observed), "maximum_absolute_numeric_difference": max_difference}


def recompute_controlled() -> dict[str, Any]:
    machine_path = ANALYSIS / "controlled_ab_machine_judge_case_long.csv"
    human_path = ANALYSIS / "difference_survey_cleaned_long.csv"
    crosswalk_path = ANALYSIS / "controlled_ab_pair_text_crosswalk.csv"
    machine = pd.read_csv(machine_path)
    human = pd.read_csv(human_path)
    crosswalk = pd.read_csv(crosswalk_path)
    if len(machine) != 72 or machine["judge_provider"].nunique() != 9 or machine["case_id"].nunique() != 8:
        raise AssertionError("Unexpected machine controlled-case structure")
    if len(human) != 480 or human["respondent_id"].nunique() != 60 or human["case_id"].nunique() != 8:
        raise AssertionError("Unexpected human controlled-case structure")
    if len(crosswalk) != 8 or int(pd.to_numeric(crosswalk["pair_exact_match"]).sum()) != 7:
        raise AssertionError("Unexpected exact-text crosswalk structure")

    cases = case_aggregate(machine.copy(), human.copy(), crosswalk.copy())
    frozen_cases = pd.read_csv(ANALYSIS / "controlled_ab_machine_human_case_summary.csv")
    case_check = compare_case_summary(cases, frozen_cases)
    frozen = json.loads((ANALYSIS / "controlled_ab_machine_human_results.json").read_text(encoding="utf-8"))

    for column in ["quality_aligned", "risk_aligned", "both_aligned"]:
        machine[column] = as_binary(machine[column], column)
    for column in ["suitability_correct", "misleading_correct", "logical_consistency"]:
        human[column] = pd.to_numeric(human[column], errors="raise").astype(float)

    overall = {
        "n_judge_case_pairs": len(machine),
        "quality_alignment": float(machine["quality_aligned"].mean()),
        "risk_alignment": float(machine["risk_aligned"].mean()),
        "both_alignment": float(machine["both_aligned"].mean()),
        "human_suitability_accuracy": float(human["suitability_correct"].mean()),
        "human_misleading_accuracy": float(human["misleading_correct"].mean()),
        "human_logical_consistency": float(human["logical_consistency"].mean()),
    }
    exact = cases[cases["pair_exact_match"] == 1]
    exact_summary = {
        "n_cases": len(exact),
        "n_machine_judge_case_pairs": int(machine[machine["case_id"].isin(exact["case_id"])].shape[0]),
        "machine_quality_alignment": float(exact["machine_quality_alignment"].mean()),
        "machine_risk_alignment": float(exact["machine_risk_alignment"].mean()),
        "machine_both_alignment": float(exact["machine_both_alignment"].mean()),
        "human_suitability_accuracy": float(exact["human_suitability_accuracy"].mean()),
        "human_misleading_accuracy": float(exact["human_misleading_accuracy"].mean()),
        "human_logical_consistency": float(exact["human_logical_consistency"].mean()),
    }
    comparison: dict[str, float] = {}
    for section_name, observed in [
        ("machine_direction_overall_all_8_cases", overall),
        ("machine_human_exact_text_7_cases", exact_summary),
    ]:
        frozen_section = frozen[section_name]
        for key, value in observed.items():
            comparison[f"{section_name}.{key}"] = close(
                value, frozen_section[key], f"{section_name}.{key}"
            )

    return {
        "inputs": {
            "machine_case_long": {"sha256": sha256_file(machine_path), "rows": len(machine)},
            "human_difference_long": {"sha256": sha256_file(human_path), "rows": len(human)},
            "pair_text_crosswalk": {"sha256": sha256_file(crosswalk_path), "rows": len(crosswalk)},
        },
        "case_summary": cases.to_dict(orient="records"),
        "case_summary_frozen_comparison": case_check,
        "overall_all_8_cases": overall,
        "exact_text_7_cases": exact_summary,
        "frozen_result_differences": comparison,
    }


def main() -> None:
    abc = recompute_abc()
    controlled = recompute_controlled()
    payload = {
        "qa_status": "PASS",
        "analysis_scope": "cleaned-only public headline recalculation",
        "privacy": {
            "raw_workbooks_read": False,
            "exact_duration_read": False,
            "background_fields_read": False,
            "api_call_log_read": False,
            "participant_rows_written": False,
            "output_is_aggregate_only": True,
        },
        "tolerance": FLOAT_TOLERANCE,
        "abc": abc,
        "controlled": controlled,
        "boundary": (
            "A/B/C participants were randomly allocated among three separate 见数平台 forms; "
            "the six popularized materials were purposively selected and are not a random corpus sample."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "qa_status": payload["qa_status"],
        "output": str(OUTPUT),
        "abc_primary_p": abc["primary"]["omnibus_interaction"]["p_value"],
        "abc_q11_q17_exclusion_p": abc["q11_q17_exclusion"]["omnibus_interaction"]["p_value"],
        "controlled_cases": len(controlled["case_summary"]),
        "exact_text_cases": controlled["exact_text_7_cases"]["n_cases"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
