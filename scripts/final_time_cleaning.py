#!/usr/bin/env python3
"""Publish the final result-blind duration-screened Full Paper panels.

Final rule version 4.0.0:
  * pass the existing attention checks;
  * repeated-item similarity >= 75% in the long first review and >= 90% in the
    short selected review;
  * exclude valid duration < 12 seconds per analyzed unique stimulus
    (first review < 432 s; second review < 72 s);
  * retain missing/nonpositive duration;
  * no cross-domain identity exclusion.

Raw response IDs are read transiently only to reproduce the existing SHA-256
participant pseudonym. No raw identifier, timestamp, or individual duration is
written to an output.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2.0.0"
RULE_VERSION = "4.0.0"
SECONDS_PER_UNIQUE_STIMULUS = 12
FIRST_UNIQUE_STIMULI = 36
SECOND_UNIQUE_STIMULI = 6
FIRST_FLOOR_SECONDS = 432
SECOND_FLOOR_SECONDS = 72

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SURVEY_ROOT = WORKSPACE_ROOT / "数据与问卷源文件"
SECOND_ROOT = SURVEY_ROOT / "专家复核问卷"
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
SOURCE_PANEL_ROOT = ANALYSIS_ROOT / "sensitivity_panels"
OUTPUT_ROOT = ANALYSIS_ROOT / "final_sensitivity_panels"
ROOT_CLEANED = ANALYSIS_ROOT / "cleaned_human_ratings_long.csv"
ROOT_MANIFEST = ANALYSIS_ROOT / "final_cleaning_manifest.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_surveys as cleaning  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def positive_duration(value: Any) -> float | None:
    duration = cleaning.as_float(value)
    if duration is None or duration <= 0:
        return None
    return float(duration)


def base_qc(data: cleaning.WorkbookData, wave: str, row: list[Any]) -> bool:
    attention_columns, attention_expected, _rule = cleaning.attention_spec(data, wave)
    if attention_columns:
        attention_pass, _failures = cleaning.exact_attention_result(
            [row[index] for index in attention_columns], attention_expected
        )
    else:
        attention_pass = True
    primary_specs, _diagnostic_specs = cleaning.repeat_specs(data, wave)
    metrics = [cleaning.row_repeat_metrics(row, spec) for spec in primary_specs]
    threshold = cleaning.similarity_threshold_for_wave(wave)
    repeat_pass = (
        all(
            metric["similarity"] is not None
            and threshold is not None
            and float(metric["similarity"]) >= threshold
            for metric in metrics
        )
        if primary_specs
        else True
    )
    duration = cleaning.positive_duration_seconds(data.headers, row)
    duration_floor = cleaning.duration_floor_for_wave(wave)
    duration_pass = (
        duration_floor is None
        or duration is None
        or duration >= duration_floor
    )
    return bool(attention_pass and repeat_pass and duration_pass)


def eligible_paths() -> list[Path]:
    first = sorted(
        [path for path in SURVEY_ROOT.glob("*.xlsx") if path.name in cleaning.FIRST_EXPERT_FILES],
        key=lambda path: path.name,
    )
    second = sorted(
        [path for path in SECOND_ROOT.glob("*.xlsx") if not path.name.startswith("~$")],
        key=lambda path: path.name,
    )
    if len(first) != 5 or len(second) != 5:
        raise AssertionError(f"Expected 5+5 eligible rating workbooks, got {len(first)}+{len(second)}")
    return first + second


def duration_metadata() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in eligible_paths():
        data = cleaning.read_credamo_workbook(path)
        wave, domain, _sheet = cleaning.source_kind(path, data.headers)
        relative_source = str(path.relative_to(SURVEY_ROOT))
        headers = {str(value or "").strip(): index for index, value in enumerate(data.headers)}
        duration_header = "作答总时长(秒)"
        if duration_header not in headers:
            raise AssertionError(f"Missing platform duration column in {relative_source}")
        is_first = path.parent == SURVEY_ROOT
        floor = FIRST_FLOOR_SECONDS if is_first else SECOND_FLOOR_SECONDS
        for raw_row, excel_row in zip(data.rows, data.excel_row_numbers):
            raw_response_id = raw_row[0] if raw_row else excel_row
            participant_id = cleaning.pseudonym(relative_source, raw_response_id)
            duration = positive_duration(raw_row[headers[duration_header]])
            rows.append(
                {
                    "participant_id": participant_id,
                    "wave": wave,
                    "domain": domain,
                    "base_keep": base_qc(data, wave, raw_row),
                    "duration_missing": duration is None,
                    "below_absolute_floor": duration is not None and duration < floor,
                }
            )
        inputs.append(
            {
                "file": relative_source,
                "wave": wave,
                "domain": domain,
                **file_record(path),
            }
        )
    metadata = pd.DataFrame(rows)
    if len(metadata) != 184 or metadata["participant_id"].duplicated().any():
        raise AssertionError(f"Expected 184 unique eligible records, got {metadata.shape}")
    return metadata, inputs


def remove_abandoned_outputs() -> None:
    for directory in (
        ANALYSIS_ROOT / "revised_sensitivity_panels",
        ANALYSIS_ROOT / "posthoc_identity_time",
    ):
        if directory.exists():
            resolved = directory.resolve()
            if ANALYSIS_ROOT.resolve() not in resolved.parents:
                raise AssertionError(f"Refusing to remove path outside analysis root: {resolved}")
            shutil.rmtree(resolved)
    for stale in (
        ANALYSIS_ROOT / "revised_cleaning_manifest.json",
    ):
        if stale.exists():
            stale.unlink()


def privacy_column_scan(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    exact = {"ip", "ip地址"}
    fragments = (
        "用户id", "作答id", "发布id", "开始时间", "结束时间",
        "作答总时长", "答题时长", "经度", "纬度", "省份", "城市",
        "设备", "操作系统", "浏览器", "屏幕分辨率",
    )
    result: dict[str, list[str]] = {}
    for name, frame in frames.items():
        bad: list[str] = []
        for column in frame.columns:
            normalized = str(column).casefold().replace(" ", "")
            if normalized in exact or any(fragment in normalized for fragment in fragments):
                bad.append(str(column))
        result[name] = bad
    return result


def main() -> None:
    remove_abandoned_outputs()
    if OUTPUT_ROOT.exists():
        resolved = OUTPUT_ROOT.resolve()
        if ANALYSIS_ROOT.resolve() not in resolved.parents:
            raise AssertionError(f"Refusing to refresh path outside analysis root: {resolved}")
        shutil.rmtree(resolved)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    metadata, workbook_inputs = duration_metadata()
    metadata_ids = set(metadata["participant_id"].astype(str))
    time_exclusion_ids = set(
        metadata.loc[metadata["below_absolute_floor"], "participant_id"].astype(str)
    )

    source_mapping = {
        "final_primary": "primary_75_90",
        "threshold_70": "first_threshold_70",
        "threshold_80": "first_threshold_80",
        "threshold_90": "first_threshold_90",
        "threshold_95": "first_threshold_95",
        "attention_only": "no_repeat_attention_only",
        "strict_first_both_pairs": "strict_first_both_pairs_75",
        "second_case05_case07_mean": "second_case05_case07_mean",
    }
    panels: dict[str, pd.DataFrame] = {}
    input_panel_paths: dict[str, Path] = {}
    for output_name, source_name in source_mapping.items():
        source_path = SOURCE_PANEL_ROOT / f"{source_name}.csv"
        source = pd.read_csv(source_path)
        unknown = set(source["participant_id"].astype(str)) - metadata_ids
        if unknown:
            raise AssertionError(f"{source_name} contains participants outside the ten eligible forms")
        panels[output_name] = source[
            ~source["participant_id"].astype(str).isin(time_exclusion_ids)
        ].copy().reset_index(drop=True)
        input_panel_paths[source_name] = source_path

    source_primary = pd.read_csv(SOURCE_PANEL_ROOT / "primary_75_90.csv")
    reconstructed_base = set(metadata.loc[metadata["base_keep"], "participant_id"].astype(str))
    source_base = set(source_primary["participant_id"].astype(str))
    if reconstructed_base != source_base:
        raise AssertionError("Raw attention/repeat/duration reconstruction differs from primary_75_90")

    expected_columns = list(source_primary.columns)
    for name, panel in panels.items():
        if list(panel.columns) != expected_columns:
            raise AssertionError(f"Schema mismatch in {name}")

    output_panel_paths: dict[str, Path] = {}
    for name, panel in panels.items():
        path = OUTPUT_ROOT / f"{name}.csv"
        write_csv(path, panel)
        output_panel_paths[name] = path
    write_csv(ROOT_CLEANED, panels["final_primary"])

    privacy_scan = privacy_column_scan(panels)
    if any(privacy_scan.values()):
        raise AssertionError(f"Forbidden privacy columns in final outputs: {privacy_scan}")

    root_record = file_record(ROOT_CLEANED, len(panels["final_primary"]))
    primary_record = file_record(
        output_panel_paths["final_primary"], len(panels["final_primary"])
    )
    if root_record["sha256"] != primary_record["sha256"] or root_record["bytes"] != primary_record["bytes"]:
        raise AssertionError("Root cleaned data is not byte-identical to final_primary")

    final_participants = {
        str(wave): int(group["participant_id"].nunique())
        for wave, group in panels["final_primary"].groupby("wave", sort=True)
    }
    base_metadata = metadata[metadata["base_keep"]]
    base_time_exclusions = int(base_metadata["below_absolute_floor"].sum())
    if len(panels["final_primary"]) != 7308 or base_time_exclusions != 0:
        raise AssertionError(
            f"Expected 7,308-row primary with duration already applied upstream; got {len(panels['final_primary'])}, {base_time_exclusions}"
        )

    outputs: dict[str, Any] = {
        "cleaned_human_ratings_long": root_record,
        "final_primary": primary_record,
    }
    for name, path in output_panel_paths.items():
        if name == "final_primary":
            continue
        outputs[name] = file_record(path, len(panels[name]))

    manifest = {
        "analysis_label": "final_wave_specific_repeat_and_duration_cleaning",
        "rule_version": RULE_VERSION,
        "script_version": SCRIPT_VERSION,
        "qa_status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "base_attention": "pass_all_keyed_attention_checks",
            "first_repeat_similarity_pct": 75,
            "second_repeat_similarity_pct": 90,
            "absolute_seconds_per_unique_stimulus": SECONDS_PER_UNIQUE_STIMULUS,
            "first_unique_stimuli": FIRST_UNIQUE_STIMULI,
            "first_duration_floor_seconds": FIRST_FLOOR_SECONDS,
            "second_unique_stimuli": SECOND_UNIQUE_STIMULI,
            "second_duration_floor_seconds": SECOND_FLOOR_SECONDS,
            "duration_operator": "exclude_if_valid_duration_strictly_below_floor",
            "missing_duration": "retain",
            "cross_domain_identity_exclusion": "none",
        },
        "counts": {
            "eligible_raw_records": int(len(metadata)),
            "base_primary_participants": int(len(base_metadata)),
            "base_primary_time_exclusions": base_time_exclusions,
            "eligible_raw_records_below_time_floor": int(
                metadata["below_absolute_floor"].sum()
            ),
            "final_participants": final_participants,
            "final_participants_total": int(panels["final_primary"]["participant_id"].nunique()),
            "final_long_rows": int(len(panels["final_primary"])),
            "time_excluded_participants_by_panel": {
                output_name: int(
                    pd.read_csv(input_panel_paths[source_name])["participant_id"]
                    .astype(str)
                    .isin(time_exclusion_ids)
                    .groupby(
                        pd.read_csv(input_panel_paths[source_name])["participant_id"].astype(str)
                    )
                    .max()
                    .sum()
                )
                for output_name, source_name in source_mapping.items()
            },
        },
        "inputs": {
            "source_panels": {
                name: file_record(path, len(pd.read_csv(path)))
                for name, path in input_panel_paths.items()
            },
            "eligible_workbooks": workbook_inputs,
        },
        "outputs": outputs,
        "privacy": {
            "raw_response_id_written": False,
            "platform_user_id_read": False,
            "platform_user_id_written": False,
            "individual_duration_written": False,
            "timestamp_written": False,
            "output_column_scan": privacy_scan,
        },
        "qa": {
            "status": "PASS",
            "raw_base_qc_matches_primary_75_90": True,
            "root_cleaned_equals_final_primary_bytes": True,
            "all_panel_schemas_match": True,
            "all_panel_participants_map_to_eligible_forms": True,
            "no_forbidden_privacy_columns": True,
            "abandoned_identity_outputs_removed": True,
        },
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "script": file_record(Path(__file__).resolve()),
    }
    ROOT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qa_status": manifest["qa_status"],
                "counts": manifest["counts"],
                "root_cleaned": root_record,
                "final_primary": primary_record,
                "manifest": file_record(ROOT_MANIFEST),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
