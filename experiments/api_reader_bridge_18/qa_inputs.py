from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
MATERIALS = HERE.parents[1] / "materials"
INPUTS_JSONL = HERE / "inputs.jsonl"
INPUTS_CSV = HERE / "inputs.csv"
RUNNER_INPUTS_CSV = HERE / "runner_inputs.csv"
MAPPING_CSV = HERE / "mapping.csv"
MANIFEST = HERE / "manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def normalized_crosswalk_text(text: str) -> str:
    return re.sub(r"\n+", "\n", canonical_text(text).strip())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_crosswalk() -> list[dict]:
    path = MATERIALS / "public_semantic_crosswalk_full.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["layer"] == "public_reader_ABC"]


def extract_question_segment(instrument: str, question_code: str) -> str:
    text = instrument.replace("\r\n", "\n").replace("\r", "\n")
    start_match = re.search(rf"(?m)^{re.escape(question_code)}$", text)
    assert start_match, f"missing {question_code}"
    start = start_match.end()
    if start < len(text) and text[start] == "\n":
        start += 1
    next_code = f"Q{int(question_code[1:]) + 1}"
    end_match = re.search(rf"(?m)^{re.escape(next_code)}(?:\s|$)", text[start:])
    end = start + end_match.start() if end_match else len(text)
    return text[start:end]


def assert_no_secret_like_tokens(paths: list[Path]) -> None:
    patterns = (
        re.compile(r"(?i)(?:sk|ark)-[A-Za-z0-9_-]{16,}"),
        re.compile(r"AIza[A-Za-z0-9_-]{24,}"),
    )
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(pattern.search(content) for pattern in patterns), f"credential-like token in {path.name}"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = read_jsonl(INPUTS_JSONL)
    csv_records = read_csv(INPUTS_CSV)
    runner_records = read_csv(RUNNER_INPUTS_CSV)
    mapping_records = read_csv(MAPPING_CSV)
    crosswalk = load_crosswalk()
    answer_key = json.loads((MATERIALS / "public_abc_answer_key.json").read_text(encoding="utf-8"))

    assert manifest["record_count"] == 18
    assert len(records) == len(csv_records) == len(runner_records) == len(mapping_records) == len(crosswalk) == 18
    assert len({row["item_id"] for row in records}) == 18
    assert Counter(row["version"] for row in records) == Counter({"A": 6, "B": 6, "C": 6})

    expected_concepts = manifest["design"]["concept_keys_in_material_order"]
    expected_cells = {(version, number) for version in "ABC" for number in range(1, 7)}
    assert {(row["version"], row["material_no"]) for row in records} == expected_cells
    assert {(row["concept_key"], row["version"]) for row in records} == {
        (concept, version) for concept in expected_concepts for version in "ABC"
    }

    source_by_name = {Path(src["path_from_experiment_dir"]).name: src for src in manifest["source_artifacts"]}
    for filename, source in source_by_name.items():
        path = MATERIALS / filename
        assert path.exists(), f"missing source: {filename}"
        assert sha256_file(path) == source["sha256"], f"source hash mismatch: {filename}"

    for frozen in manifest["input_files"]:
        path = HERE / frozen["path"]
        assert path.stat().st_size == frozen["bytes"], f"size mismatch: {path.name}"
        assert sha256_file(path) == frozen["sha256"], f"file hash mismatch: {path.name}"

    answer_by_index = {entry["concept_index"]: entry for entry in answer_key["concepts"]}
    crosswalk_index = {
        (row["form_or_case"], int(row["material_no"])): row for row in crosswalk
    }
    csv_index = {row["item_id"]: row for row in csv_records}
    runner_index = {row["item_id"]: row for row in runner_records}
    mapping_by_source = {row["source_item_id"]: row for row in mapping_records}
    manifest_index = {row["item_id"]: row for row in manifest["items"]}
    instrument_cache: dict[str, str] = {}

    for row in records:
        version = row["version"]
        number = row["material_no"]
        item_id = row["item_id"]
        text = row["visible_text"]
        provenance = row["source_provenance"]

        assert row["experiment_id"] == manifest["experiment_id"]
        assert item_id == f"reader_{version}_{number:02d}"
        assert row["question_code"] == f"Q{12 + number}"
        assert row["concept_key"] == expected_concepts[number - 1]
        assert text == canonical_text(text) == text.strip()
        assert "\r" not in text
        assert sha256_bytes(text.encode("utf-8")) == row["visible_text_sha256"]
        assert len(text) == row["visible_text_char_count"]
        assert len(text.encode("utf-8")) == row["visible_text_utf8_bytes"]

        manifest_row = manifest_index[item_id]
        assert manifest_row["version"] == version
        assert manifest_row["material_no"] == number
        assert manifest_row["concept_key"] == row["concept_key"]
        assert manifest_row["visible_text_sha256"] == row["visible_text_sha256"]

        csv_row = csv_index[item_id]
        for key in ("version", "question_code", "concept_key", "concept_label_zh", "concept_id", "domain",
                    "visible_text", "visible_text_sha256"):
            assert csv_row[key] == str(row[key]), f"CSV/JSONL mismatch: {item_id}/{key}"
        assert int(csv_row["material_no"]) == number
        assert csv_row["mapped_item_id"] == provenance["mapped_item_id"]
        assert csv_row["instrument_file"] == provenance["instrument_file"]
        assert csv_row["crosswalk_row_key"] == provenance["crosswalk_row_key"]

        mapping_row = mapping_by_source[item_id]
        opaque_id = "rb18_" + row["visible_text_sha256"][:12]
        assert mapping_row["item_id"] == opaque_id
        assert mapping_row["version"] == version
        assert int(mapping_row["material_no"]) == number
        assert mapping_row["visible_text_sha256"] == row["visible_text_sha256"]
        assert mapping_row["mapped_item_id"] == provenance["mapped_item_id"]
        assert mapping_row["instrument_file"] == provenance["instrument_file"]
        assert mapping_row["crosswalk_row_key"] == provenance["crosswalk_row_key"]

        runner_row = runner_index[opaque_id]
        assert runner_row["domain"] == row["domain"]
        assert runner_row["concept"] == row["concept_key"]
        assert runner_row["task_type"] == "reader_facing_science_explanation"
        assert runner_row["generated_text"] == text

        source_row = crosswalk_index[(version, number)]
        assert normalized_crosswalk_text(source_row["lay_or_controlled_text"]) == text
        assert source_row["comprehension_or_pair_question"] == row["comprehension_question"]
        assert source_row["concept"] == row["concept_label_zh"]
        assert source_row["concept_id"] == row["concept_id"]
        assert source_row["domain"] == row["domain"]
        for key in ("mapped_item_id", "legacy_item_id", "generator", "task", "map_status", "confidence"):
            target_key = {"task": "upstream_task", "confidence": "mapping_confidence"}.get(key, key)
            assert source_row[key] == provenance[target_key]
        expected_crosswalk_key = f"public_reader_ABC|{source_row['instrument']}|{version}|{number}"
        assert provenance["crosswalk_row_key"] == expected_crosswalk_key

        instrument_file = provenance["instrument_file"]
        if instrument_file not in instrument_cache:
            instrument_cache[instrument_file] = (MATERIALS / instrument_file).read_text(encoding="utf-8")
        instrument = instrument_cache[instrument_file]
        assert sha256_file(MATERIALS / instrument_file) == provenance["instrument_sha256"]
        source_container = re.search(r"Source container SHA-256: ([0-9a-f]{64})", instrument)
        assert source_container and source_container.group(1) == provenance["source_container_sha256"]
        segment = extract_question_segment(instrument, row["question_code"])
        assert segment.startswith(text + "\n"), f"instrument text mismatch: {item_id}"
        assert row["comprehension_question"] in segment

        answer_entry = answer_by_index[number]
        assert answer_entry["concept"] == row["concept_key"]
        assert answer_entry["post_correct_answer"] in segment

    assert len({row["visible_text_sha256"] for row in records}) == 18
    assert [row["item_id"] for row in runner_records] == sorted(row["item_id"] for row in runner_records)
    assert_no_secret_like_tokens([INPUTS_CSV, INPUTS_JSONL, RUNNER_INPUTS_CSV, MAPPING_CSV, MANIFEST])
    print("PASS: 18 records; 6 concepts x 3 versions; all source, text, and file hashes verified.")


if __name__ == "__main__":
    main()
