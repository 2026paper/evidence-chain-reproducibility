#!/usr/bin/env python3
"""Fail-closed audit of nested provenance locks and final-paper data version."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path: Path, expected: str, label: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        ERRORS.append(f"missing {label}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        return
    actual = sha256(path)
    if actual != expected:
        ERRORS.append(f"sha256 mismatch {label}: {actual} != {expected}")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        ERRORS.append(f"byte-size mismatch {label}: {path.stat().st_size} != {expected_bytes}")


def resolve(base: Path, value: str) -> Path:
    relative = Path(str(value).replace("\\", "/"))
    candidates = [ROOT / relative, base / relative]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        return candidates[0]
    return existing[0]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def check_record(base: Path, record: dict, label: str) -> None:
    check(
        resolve(base, record["path"]),
        record["sha256"],
        label,
        record.get("bytes"),
    )


def check_core_manifests() -> None:
    for name in ("alignment_manifest.json", "reliability_manifest.json"):
        path = ROOT / "analysis" / "source_manifests" / name
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for section in ("inputs", "outputs"):
            for key, record in manifest.get(section, {}).items():
                check_record(path.parent, record, f"{name}:{section}:{key}")
        script = resolve(ROOT, manifest["script_path"])
        check(script, manifest["script_sha256"], f"{name}:script")


def check_final_human_version() -> None:
    manifest = load("analysis/final_cleaning_manifest.json")
    rules = manifest.get("rules", {})
    if rules.get("first_repeat_similarity_pct") != 75:
        ERRORS.append("broad repeat gate is not 75%")
    if rules.get("second_repeat_similarity_pct") != 90:
        ERRORS.append("selected repeat gate is not 90%")
    path = ROOT / "analysis" / "cleaned_human_ratings_long.csv"
    participants: dict[str, set[str]] = defaultdict(set)
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            participants[row["wave"]].add(row["participant_id"])
    expected = {"首轮专家复核": 18, "二次专家复核": 95}
    observed = {wave: len(ids) for wave, ids in participants.items()}
    if rows != 7308 or observed != expected:
        ERRORS.append(f"final human table is not the 75% release: rows={rows}, participants={observed}")
    effects = ROOT / "analysis" / "alignment_effects.csv"
    with effects.open(encoding="utf-8-sig", newline="") as handle:
        headline = [
            row for row in csv.DictReader(handle)
            if row["wave"] == "首轮专家复核" and row["family"] == "overall"
        ]
    if len(headline) != 1 or abs(float(headline[0]["slope"]) - 0.231563853146348) > 1e-12:
        ERRORS.append("broad alignment headline is not beta=.231563853146348")


def check_reader_bridge() -> None:
    base = ROOT / "experiments" / "api_reader_bridge_18"
    frozen = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    for record in frozen["input_files"]:
        check(base / record["path"], record["sha256"], f"reader input:{record['path']}", record.get("bytes"))
    for record in frozen["source_artifacts"]:
        path = resolve(base, record["path_from_experiment_dir"])
        check(path, record["sha256"], f"reader source:{path.name}")

    manifest = json.loads((base / "reader_bridge_analysis_manifest.json").read_text(encoding="utf-8"))
    check(base / manifest["script"]["path"], manifest["script"]["sha256"], "reader analysis script")
    sources = manifest["sources"]
    for record in sources["api_sources"]:
        check(resolve(base, record["path"]), record["sha256"], f"reader API:{record['provider_directory']}")
    for key in ("reader_source", "mapping", "runner_inputs"):
        record = sources[key]
        check(resolve(base, record["path"]), record["sha256"], f"reader source:{key}")
    for filename, expected in manifest["outputs"].items():
        check(base / filename, expected, f"reader output:{filename}")


def check_repeat_stability() -> None:
    base = ROOT / "experiments" / "api_repeat_stability"
    manifest = json.loads((base / "repeat_stability_analysis_manifest.json").read_text(encoding="utf-8"))
    check(base / manifest["script"]["path"], manifest["script"]["sha256"], "repeat script")
    for filename, expected in manifest["sources"]["input_sources"].items():
        check(base / filename, expected, f"repeat input:{filename}")
    for record in manifest["sources"]["run_snapshots"]:
        check(resolve(base, record["path"]), record["sha256"], f"repeat scores:{record['judge_provider']}")
        check(resolve(base, record["run_manifest_path"]), record["run_manifest_sha256"], f"repeat run manifest:{record['judge_provider']}")
    for filename, expected in manifest["outputs"].items():
        check(base / filename, expected, f"repeat output:{filename}")

    sampling = json.loads((base / "sampling_manifest.json").read_text(encoding="utf-8"))
    source_locations = {
        "api_test_scores_7290.csv": ROOT / "data/40_GitHub/rebuttal_update_20260714/api_test_scores_7290.csv",
        "judge_configuration_public.json": ROOT / "models_and_prompts/judge_configuration_public.json",
        "judge_output_schema.json": ROOT / "models_and_prompts/judge_output_schema.json",
    }
    for filename, expected in sampling.get("source_hashes_sha256", {}).items():
        check(source_locations[filename], expected, f"sampling source:{filename}")
    for relative, expected in sampling.get("released_source_hashes_sha256", {}).items():
        check(ROOT / relative, expected, f"sampling released source:{relative}")
    for filename, expected in sampling["committed_outputs_sha256"].items():
        check(base / filename, expected, f"sampling output:{filename}")
    runner = json.loads((base / "runner_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in runner["sha256"].items():
        check(base / filename, expected, f"runner input:{filename}")


def check_aggregation() -> None:
    base = ROOT / "experiments" / "api_aggregation_human_baselines"
    manifest = json.loads((base / "aggregation_human_manifest.json").read_text(encoding="utf-8"))
    check(base / manifest["script"]["path"], manifest["script"]["sha256"], "aggregation script")
    inputs = {
        "alignment_human_api_item_dimension.csv": ROOT / "analysis/alignment_human_api_item_dimension.csv",
        "human_api_crosswalk.csv": ROOT / "analysis/human_api_crosswalk.csv",
        "cleaned_human_ratings_long.csv": ROOT / "analysis/cleaned_human_ratings_long.csv",
        "api_test_scores_7290.csv": ROOT / "data/40_GitHub/rebuttal_update_20260714/api_test_scores_7290.csv",
        "api_stimulus_equivalence_810.csv": ROOT / "analysis/api_stimulus_equivalence_810.csv",
    }
    for filename, path in inputs.items():
        check(path, manifest["inputs"][filename], f"aggregation input:{filename}")
    for filename, expected in manifest["outputs"].items():
        check(base / filename, expected, f"aggregation output:{filename}")


def check_figures_and_source_map() -> None:
    base = ROOT / "output" / "figures"
    mappings = {
        "fig1_evidence_manifest.json": {
            "pdf": "fig1_five_layer_evidence.pdf", "png": "fig1_five_layer_evidence.png",
            "svg": "fig1_five_layer_evidence.svg", "source": "source_data/fig1_five_layer_evidence_source.csv",
        },
        "fig2_reliability_manifest.json": {"pdf": "fig2_reliability.pdf", "png": "fig2_reliability.png"},
        "fig3_alignment_manifest.json": {
            "pdf": "fig3_alignment.pdf", "png": "fig3_alignment.png", "source": "source_data/fig3_alignment_source.csv",
        },
        "fig4_public_validation_manifest.json": {"pdf": "fig4_public_validation.pdf", "png": "fig4_public_validation.png"},
    }
    for filename, targets in mappings.items():
        manifest = json.loads((base / filename).read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            ERRORS.append(f"figure manifest is not PASS: {filename}")
        for key, relative in targets.items():
            check(base / relative, manifest["sha256"][key], f"{filename}:{key}")

    source_map = ROOT / "manifests" / "source_to_release_map.csv"
    with source_map.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            check(ROOT / row["release_path"], row["release_sha256"], f"source map:{row['release_path']}", int(row["release_bytes"]))


def main() -> None:
    check_final_human_version()
    check_core_manifests()
    check_reader_bridge()
    check_repeat_stability()
    check_aggregation()
    check_figures_and_source_map()
    if ERRORS:
        print("FAIL")
        for error in ERRORS:
            print(f"- {error}")
        raise SystemExit(1)
    print("PASS: final 75%/18/178 data version and all nested provenance locks verified")


if __name__ == "__main__":
    main()
