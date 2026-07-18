from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


EXPERIMENT_ID = "api_repeat_stability_90x9x3"
OPAQUE_SEED = "20260717|opaque-v1"
ORDER_SEED = "20260717|runner-order-v1"
RUNNER_FIELDS = ["item_id", "domain", "concept", "task_type", "generated_text"]
MAPPING_FIELDS = [
    "item_id",
    "source_stimulus_id",
    "source_item_uid",
    "api_item_id",
    "replicate_index",
    "runner_order",
    "domain",
    "concept_id",
    "task_type",
    "generator",
    "text_sha256",
    "content_json_sha256",
]


def base_dir() -> Path:
    return Path(__file__).resolve().parent


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base() -> list[dict[str, str]]:
    path = base_dir() / "repeat_sample_90.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise ValueError(f"expected 90 base rows, found {len(rows)}")
    return rows


def build_entries() -> tuple[list[dict[str, object]], int]:
    candidates: list[dict[str, object]] = []
    for row in load_base():
        for replicate_index in (1, 2, 3):
            digest = hashlib.sha256(
                f"{OPAQUE_SEED}|{row['source_item_uid']}|{replicate_index}".encode("utf-8")
            ).hexdigest()[:20]
            candidates.append(
                {
                    "item_id": f"ri_{digest}",
                    "source_stimulus_id": row["repeat_item_id"],
                    "source_item_uid": row["source_item_uid"],
                    "api_item_id": row["api_item_id"],
                    "replicate_index": replicate_index,
                    "domain": row["domain"],
                    "concept": row["concept"],
                    "concept_id": row["concept_id"],
                    "task_type": row["task_type"],
                    "generator": row["generator"],
                    "generated_text": row["content_json"],
                    "text_sha256": row["text_sha256"],
                    "content_json_sha256": row["content_json_sha256"],
                }
            )

    if len({entry["item_id"] for entry in candidates}) != 270:
        raise ValueError("opaque item_id collision")

    selected_attempt = -1
    ordered: list[dict[str, object]] = []
    for attempt in range(10000):
        ordered = sorted(
            candidates,
            key=lambda entry: hashlib.sha256(
                f"{ORDER_SEED}|{attempt}|{entry['source_stimulus_id']}|{entry['replicate_index']}".encode("utf-8")
            ).hexdigest(),
        )
        adjacent = sum(
            ordered[index - 1]["source_stimulus_id"] == ordered[index]["source_stimulus_id"]
            for index in range(1, len(ordered))
        )
        if adjacent == 0:
            selected_attempt = attempt
            break
    if selected_attempt < 0:
        raise RuntimeError("could not find a no-adjacency deterministic order")

    for runner_order, entry in enumerate(ordered, 1):
        entry["runner_order"] = runner_order
    return ordered, selected_attempt


def csv_lines(kind: str) -> tuple[list[str], list[dict[str, object]], int]:
    entries, attempt = build_entries()
    fields = RUNNER_FIELDS if kind == "runner" else MAPPING_FIELDS
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        writer.writerow({field: entry[field] for field in fields})
    return stream.getvalue().splitlines(), entries, attempt


def target_for(kind: str) -> Path:
    return base_dir() / ("runner_inputs_270.csv" if kind == "runner" else "mapping_270.csv")


def emit_chunk(kind: str, start: int, end: int, mode: str) -> None:
    lines, _, _ = csv_lines(kind)
    header, data = lines[0], lines[1:]
    selected = data[start:end]
    target = target_for(kind)
    rel = target.relative_to(Path.cwd()).as_posix()
    print("*** Begin Patch")
    if mode == "add":
        print(f"*** Add File: {rel}")
        print("+" + header)
        for line in selected:
            print("+" + line)
    else:
        previous = target.read_text(encoding="utf-8").splitlines()[-1]
        print(f"*** Update File: {rel}")
        print("@@")
        print("-" + previous)
        print("+" + previous)
        for line in selected:
            print("+" + line)
    print("*** End Patch")


def qa() -> dict[str, object]:
    entries, attempt = build_entries()
    sources = collections.Counter(str(entry["source_stimulus_id"]) for entry in entries)
    reps = collections.Counter(int(entry["replicate_index"]) for entry in entries)
    adjacent = sum(
        entries[index - 1]["source_stimulus_id"] == entries[index]["source_stimulus_id"]
        for index in range(1, len(entries))
    )
    positions = collections.defaultdict(list)
    for entry in entries:
        positions[str(entry["source_stimulus_id"])].append(int(entry["runner_order"]))
    min_separation = min(
        right - left - 1
        for values in positions.values()
        for left, right in zip(sorted(values), sorted(values)[1:])
    )
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "opaque_seed_namespace": OPAQUE_SEED,
        "runner_order_seed_namespace": ORDER_SEED,
        "selected_order_attempt": attempt,
        "rows": len(entries),
        "unique_opaque_item_ids": len({entry["item_id"] for entry in entries}),
        "unique_source_stimuli": len(sources),
        "source_repetition_counts": dict(collections.Counter(sources.values())),
        "replicate_counts": dict(sorted(reps.items())),
        "adjacent_same_source_pairs": adjacent,
        "minimum_intervening_rows_between_same_source": min_separation,
        "historical_matrix_used_as_rep1": False,
    }


def emit_manifest() -> None:
    report = qa()
    runner = base_dir() / "runner_inputs_270.csv"
    mapping = base_dir() / "mapping_270.csv"
    manifest = {
        "schema_version": "1.0",
        **report,
        "design": {
            "base_stimuli": 90,
            "fresh_replicates_per_stimulus": 3,
            "runner_rows": 270,
            "judges": 9,
            "planned_fresh_api_calls": 2430,
            "runner_columns": RUNNER_FIELDS,
            "opaque_id_rule": "ri_ + first 20 hex characters of SHA256(opaque namespace | source_item_uid | replicate_index)",
            "order_rule": "first SHA256-sorted attempt with zero adjacent rows from the same source stimulus",
            "generated_text_rule": "canonical task-aware content_json from repeat_sample_90.csv",
            "mapping_boundary": "source and replicate identifiers occur only in mapping_270.csv, not runner_inputs_270.csv",
        },
        "sha256": {
            "repeat_sample_90.csv": sha256_bytes(base_dir() / "repeat_sample_90.csv"),
            "runner_inputs_270.csv": sha256_bytes(runner),
            "mapping_270.csv": sha256_bytes(mapping),
        },
    }
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    rel = (base_dir() / "runner_manifest.json").relative_to(Path.cwd()).as_posix()
    print("*** Begin Patch")
    print(f"*** Add File: {rel}")
    for line in content.splitlines():
        print("+" + line)
    print("*** End Patch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", choices=["runner", "mapping", "manifest"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--mode", choices=["add", "append"], default="add")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(qa(), ensure_ascii=False, indent=2))
    elif args.emit == "manifest":
        emit_manifest()
    elif args.emit:
        emit_chunk(args.emit, args.start, args.end, args.mode)
    else:
        parser.error("choose --check or --emit")


if __name__ == "__main__":
    main()
