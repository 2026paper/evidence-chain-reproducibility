#!/usr/bin/env python3
"""Generate outcome-independent QC sensitivity panels from the raw expert surveys.

The primary rule uses 75% repeat similarity for the long first review and 90% for
the short selected review, together with the shared attention and duration gates.
This program emits pseudonymized long-form panels for fixed one-factor sensitivity
checks.  No legacy human-rating derivative is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_surveys as cs


OUTPUT_DIR = cs.ANALYSIS_ROOT / "sensitivity_panels"
SCENARIOS = (
    "first_threshold_70",
    "primary_75_90",
    "first_threshold_80",
    "first_threshold_90",
    "first_threshold_95",
    "no_repeat_attention_only",
    "strict_first_both_pairs_75",
    "second_case05_case07_mean",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_or_none(left: Any, right: Any) -> float | None:
    a = cs.as_float(left)
    b = cs.as_float(right)
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def second_rows_with_case05_case07_mean(
    data: cs.WorkbookData,
    retained_rows: list[list[Any]],
    retained_ids: list[str],
    domain: str,
) -> list[dict[str, Any]]:
    ratings, _dictionary = cs.second_expert_long_rows(
        data, retained_rows, retained_ids, domain
    )
    prefix = cs.second_review_prefix(data.headers)
    case05 = cs.second_case_score_columns(data, f"{prefix}05")
    case07 = cs.second_case_score_columns(data, f"{prefix}07")
    if len(case05) != 6 or len(case07) != 6:
        raise AssertionError(
            f"Expected six Case05 and six Case07 columns in {data.path.name}"
        )
    replacements: dict[tuple[str, str], float] = {}
    for raw_row, participant_id in zip(retained_rows, retained_ids):
        for idx, dimension in enumerate(cs.DIMENSIONS):
            value = mean_or_none(raw_row[case05[idx]], raw_row[case07[idx]])
            if value is not None:
                replacements[(participant_id, dimension)] = value
    changed = 0
    for rating in ratings:
        if rating["item_local_id"] != f"{prefix}05":
            continue
        key = (rating["participant_id"], rating["dimension"])
        if key not in replacements:
            continue
        value = replacements[key]
        rating["score_raw"] = value
        rating["score_quality_aligned"] = (
            6.0 - value if rating["dimension"] == "误导风险" else value
        )
        changed += 1
    expected = len(retained_rows) * 6
    if changed != expected:
        raise AssertionError(
            f"Expected {expected} Case05 replacements in {data.path.name}, got {changed}"
        )
    return ratings


def selection_flags(
    attention_pass: bool,
    duration_pass: bool,
    primary_similarity: float | None,
    diagnostic_similarity: float | None,
    wave: str,
) -> dict[str, bool]:
    primary_value = float(primary_similarity) if primary_similarity is not None else None
    diagnostic_value = (
        float(diagnostic_similarity) if diagnostic_similarity is not None else None
    )
    eligible = attention_pass and duration_pass
    flags = {
        "first_threshold_70": False,
        "primary_75_90": False,
        "first_threshold_80": False,
        "first_threshold_90": False,
        "first_threshold_95": False,
        "no_repeat_attention_only": eligible,
        "strict_first_both_pairs_75": False,
        "second_case05_case07_mean": False,
    }
    if wave == "首轮专家复核":
        for scenario, threshold in (
            ("first_threshold_70", 70.0),
            ("primary_75_90", 75.0),
            ("first_threshold_80", 80.0),
            ("first_threshold_90", 90.0),
            ("first_threshold_95", 95.0),
        ):
            flags[scenario] = (
                eligible
                and primary_value is not None
                and primary_value >= threshold
            )
        flags["strict_first_both_pairs_75"] = (
            eligible
            and primary_value is not None
            and primary_value >= 75.0
            and diagnostic_value is not None
            and diagnostic_value >= 75.0
        )
    elif wave == "二次专家复核":
        selected_primary = (
            eligible
            and primary_value is not None
            and primary_value >= 90.0
        )
        flags["primary_75_90"] = selected_primary
        flags["second_case05_case07_mean"] = selected_primary
    return flags


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(
        (cs.SURVEY_ROOT / filename for filename in cs.FIRST_EXPERT_FILES),
        key=lambda path: path.name,
    )
    source_paths += sorted(cs.SECOND_REVIEW_ROOT.glob("*.xlsx"), key=lambda path: path.name)
    if len(source_paths) != 10 or any(not path.exists() for path in source_paths):
        raise AssertionError(f"Expected ten expert-review workbooks, found {source_paths}")

    scenario_ratings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inclusion_rows: list[dict[str, Any]] = []
    input_hashes = {str(path): sha256_file(path) for path in source_paths}

    for path in source_paths:
        data = cs.read_credamo_workbook(path)
        wave, domain, _sheet_name = cs.source_kind(path, data.headers)
        relative_source = str(path.relative_to(cs.SURVEY_ROOT))
        attention_columns, attention_expected, _attention_rule = cs.attention_spec(
            data, wave
        )
        primary_specs, diagnostic_specs = cs.repeat_specs(data, wave)

        scenario_raw_rows: dict[str, list[list[Any]]] = defaultdict(list)
        scenario_ids: dict[str, list[str]] = defaultdict(list)

        for raw_row, excel_row in zip(data.rows, data.excel_row_numbers):
            participant_id = cs.pseudonym(
                relative_source, raw_row[0] if raw_row else excel_row
            )
            if attention_columns:
                attention_pass, _failures = cs.exact_attention_result(
                    [raw_row[index] for index in attention_columns], attention_expected
                )
            else:
                attention_pass = True
            primary_metrics = [
                cs.row_repeat_metrics(raw_row, spec) for spec in primary_specs
            ]
            diagnostic_metrics = [
                cs.row_repeat_metrics(raw_row, spec) for spec in diagnostic_specs
            ]
            primary_similarity = (
                primary_metrics[0]["similarity"] if primary_metrics else None
            )
            diagnostic_similarity = (
                diagnostic_metrics[0]["similarity"] if diagnostic_metrics else None
            )
            duration = cs.positive_duration_seconds(data.headers, raw_row)
            duration_floor = cs.duration_floor_for_wave(wave)
            duration_pass = (
                duration_floor is None
                or duration is None
                or duration >= duration_floor
            )
            flags = selection_flags(
                attention_pass,
                duration_pass,
                primary_similarity,
                diagnostic_similarity,
                wave,
            )
            for scenario, included in flags.items():
                inclusion_rows.append(
                    {
                        "scenario": scenario,
                        "wave": wave,
                        "domain": domain,
                        "source_file": relative_source,
                        "participant_id": participant_id,
                        "included": included,
                        "attention_pass": attention_pass,
                        "duration_pass": duration_pass,
                        "primary_repeat_similarity_pct": primary_similarity,
                        "diagnostic_repeat_similarity_pct": diagnostic_similarity,
                    }
                )
                if included:
                    scenario_raw_rows[scenario].append(raw_row)
                    scenario_ids[scenario].append(participant_id)

        for scenario in SCENARIOS:
            retained_rows = scenario_raw_rows.get(scenario, [])
            retained_ids = scenario_ids.get(scenario, [])
            if not retained_rows:
                continue
            if wave == "首轮专家复核":
                ratings, _dictionary = cs.first_expert_long_rows(
                    data, retained_rows, retained_ids, domain
                )
            elif scenario == "second_case05_case07_mean":
                ratings = second_rows_with_case05_case07_mean(
                    data, retained_rows, retained_ids, domain
                )
            else:
                ratings, _dictionary = cs.second_expert_long_rows(
                    data, retained_rows, retained_ids, domain
                )
            scenario_ratings[scenario].extend(ratings)

    output_hashes: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        frame = pd.DataFrame(scenario_ratings.get(scenario, []))
        output_path = OUTPUT_DIR / f"{scenario}.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_hashes[scenario] = sha256_file(output_path)
        if not frame.empty:
            for (wave, domain), group in frame.groupby(["wave", "domain"], sort=False):
                summaries.append(
                    {
                        "scenario": scenario,
                        "wave": wave,
                        "domain": domain,
                        "participants": int(group["participant_id"].nunique()),
                        "items": int(group["item_local_id"].nunique()),
                        "rating_rows": int(len(group)),
                    }
                )

    inclusion = pd.DataFrame(inclusion_rows)
    inclusion_path = OUTPUT_DIR / "scenario_inclusion_manifest.csv"
    inclusion.to_csv(inclusion_path, index=False, encoding="utf-8-sig")

    # The regenerated primary panel must be byte-for-value identical to the expert
    # subset of the already accepted cleaned long file (row order is canonicalized).
    existing = pd.read_csv(cs.ANALYSIS_ROOT / "cleaned_human_ratings_long.csv")
    expert = existing[existing["wave"].isin(["首轮专家复核", "二次专家复核"])].copy()
    primary = pd.read_csv(OUTPUT_DIR / "primary_75_90.csv")
    sort_keys = [
        "wave",
        "domain",
        "participant_id",
        "item_local_id",
        "dimension",
    ]
    expert = expert.sort_values(sort_keys).reset_index(drop=True)
    primary = primary.sort_values(sort_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expert, primary, check_dtype=False)

    manifest = {
        "randomness": "none",
        "selection_rules_frozen_before_substantive_analysis": True,
        "legacy_human_derivatives_read": False,
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "primary_matches_cleaned_long_exactly": True,
        "summaries": summaries,
        "inclusion_manifest_sha256": sha256_file(inclusion_path),
    }
    manifest_path = OUTPUT_DIR / "sensitivity_panel_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
