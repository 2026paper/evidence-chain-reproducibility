#!/usr/bin/env python3
"""Build an outcome-blind, text-verified crosswalk from questionnaire items to API items.

This script intentionally does not read any previous human-rating derivative or old
crosswalk.  Each questionnaire item is matched against the current 810-item source
corpus by an exact semantic signature reconstructed from ``content_json``.  The few
source records that are text-identical across generators are disambiguated using the
generator schedule recoverable from the other four domains at the same questionnaire
position.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


DOMAIN_MAP = {
    "物理": "Physics",
    "化学": "Chemistry",
    "生物": "Biology",
    "地理": "Geoscience",
    "大气科学": "Climate/Environment",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: object) -> str:
    """Keep only Unicode letters/numbers and case-fold Latin text."""
    return "".join(ch.casefold() for ch in str(value) if ch.isalnum())


def task_letter(task: str) -> str:
    return str(task).replace("Task", "").strip()


def normalized_options(payload: dict) -> dict[str, str]:
    options = payload.get("options", {})
    if isinstance(options, list):
        return {chr(65 + idx): str(value) for idx, value in enumerate(options)}
    if isinstance(options, dict):
        return {str(key).strip().upper(): str(value) for key, value in options.items()}
    raise TypeError(f"Unsupported options representation: {type(options)!r}")


def render_source_signature(payload: dict, task: str, wave: str) -> str:
    """Render the source fields using the questionnaire's field order and labels."""
    if task == "A":
        if wave == "首轮专家复核":
            rendered = (
                f"概念{payload.get('concept', '')}"
                f"解释{payload.get('text', '')}"
                f"常见误解{payload.get('misconception', '')}"
                f"正确说明{payload.get('correction', '')}"
            )
        else:
            rendered = (
                f"主题{payload.get('concept', '')}"
                f"概念说明{payload.get('text', '')}"
                f"该AI内容列出的常见误解{payload.get('misconception', '')}"
                f"该AI内容给出的澄清说明{payload.get('correction', '')}"
            )
        return normalize(rendered)

    if task == "B":
        tags = payload.get("tags", [])
        tags_text = "".join(str(value) for value in tags) if isinstance(tags, list) else str(tags)
        if wave == "首轮专家复核":
            rendered = (
                f"概念{payload.get('concept', '')}"
                f"标题{payload.get('title', '')}"
                f"场景{payload.get('scene', '')}"
                f"标签{tags_text}"
                f"提醒{payload.get('warning', '')}"
            )
        else:
            rendered = (
                f"主题{payload.get('concept', '')}"
                f"标题{payload.get('title', '')}"
                f"情境{payload.get('scene', '')}"
                f"关键词{tags_text}"
                f"提醒{payload.get('warning', '')}"
            )
        return normalize(rendered)

    if task != "C":
        raise ValueError(f"Unexpected task: {task}")

    options = normalized_options(payload)
    answer = str(payload.get("answer", "")).strip().upper()
    if wave == "首轮专家复核":
        rendered = (
            f"概念{payload.get('concept', '')}"
            f"题目{payload.get('question', '')}"
            f"选项A{options.get('A', '')}B{options.get('B', '')}"
            f"C{options.get('C', '')}D{options.get('D', '')}"
            f"答案{answer}"
            f"解析{payload.get('explain', '')}"
        )
    else:
        rendered = (
            f"主题{payload.get('concept', '')}"
            f"题干{payload.get('question', '')}"
            f"选项A{options.get('A', '')}B{options.get('B', '')}"
            f"C{options.get('C', '')}D{options.get('D', '')}"
            f"该内容给出的答案{answer}{options.get(answer, '')}"
            f"该内容给出的解释{payload.get('explain', '')}"
        )
    return normalize(rendered)


def render_stimulus_signature(payload: dict, task: str) -> str:
    """Wave-neutral signature of the visible scientific content and field order."""
    if task == "A":
        rendered = (
            f"{payload.get('concept', '')}|{payload.get('text', '')}|"
            f"{payload.get('misconception', '')}|{payload.get('correction', '')}"
        )
    elif task == "B":
        tags = payload.get("tags", [])
        tags_text = "|".join(str(value) for value in tags) if isinstance(tags, list) else str(tags)
        rendered = (
            f"{payload.get('concept', '')}|{payload.get('title', '')}|"
            f"{payload.get('scene', '')}|{tags_text}|{payload.get('warning', '')}"
        )
    elif task == "C":
        options = normalized_options(payload)
        answer = str(payload.get("answer", "")).strip().upper()
        rendered = (
            f"{payload.get('concept', '')}|{payload.get('question', '')}|"
            f"A|{options.get('A', '')}|B|{options.get('B', '')}|"
            f"C|{options.get('C', '')}|D|{options.get('D', '')}|"
            f"{answer}|{payload.get('explain', '')}"
        )
    else:
        raise ValueError(f"Unexpected task: {task}")
    return sha256_text(normalize(rendered))


def questionnaire_signature(text: str, wave: str) -> str:
    if wave == "首轮专家复核":
        return normalize(text)
    start = "【AI 生成内容开始】"
    end = "【AI 生成内容结束】"
    if start not in text or end not in text:
        raise ValueError("Second-wave item is missing the AI-content boundary markers")
    core = text.split(start, 1)[1].split(end, 1)[0]
    return normalize(core)


def locate_inputs(root: Path) -> tuple[Path, Path, Path, list[Path]]:
    project_candidates = sorted(root.glob("ADMA2026_*"))
    if len(project_candidates) != 1:
        raise RuntimeError(f"Expected one ADMA2026 project folder, found {project_candidates}")
    project = project_candidates[0]
    dictionary_path = project / "analysis" / "questionnaire_item_dictionary.csv"
    generated_candidates = [
        path
        for path in root.rglob("generated_items_810.csv")
        if "main_study_data" in path.parts
    ]
    aggregate_candidates = list(root.rglob("api_test_item_aggregates_810.csv"))
    provenance_candidates = sorted(
        path
        for path in root.rglob("最终问卷可见案例溯源清单.csv")
        if path.parent.name.startswith("06_专家复核问卷最终版_")
    )
    if len(generated_candidates) != 1:
        raise RuntimeError(f"Expected one current generated-items file, found {generated_candidates}")
    if len(aggregate_candidates) != 1:
        raise RuntimeError(f"Expected one current API aggregate file, found {aggregate_candidates}")
    if len(provenance_candidates) < 2:
        raise RuntimeError(
            "Expected at least two independently retained final-questionnaire provenance manifests"
        )
    return (
        dictionary_path,
        generated_candidates[0],
        aggregate_candidates[0],
        provenance_candidates,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dictionary_path, generated_path, aggregate_path, provenance_paths = locate_inputs(root)
    project = Path(__file__).resolve().parents[1]

    questionnaire = pd.read_csv(dictionary_path, dtype=str).fillna("")
    generated = pd.read_csv(generated_path, dtype=str).fillna("")
    aggregate = pd.read_csv(aggregate_path, dtype=str).fillna("")
    provenance_frames = []
    for provenance_path in provenance_paths:
        frame = pd.read_csv(provenance_path, dtype=str).fillna("")
        frame["provenance_manifest"] = str(provenance_path)
        provenance_frames.append(frame)
    provenance = pd.concat(provenance_frames, ignore_index=True)

    required_provenance_columns = {"domain_cn", "visible_case_code", "source_item_uid"}
    if not required_provenance_columns.issubset(provenance.columns):
        raise AssertionError(
            f"Questionnaire provenance is missing columns: "
            f"{required_provenance_columns - set(provenance.columns)}"
        )
    provenance_consensus: dict[tuple[str, str], tuple[str, str, int]] = {}
    for key, group in provenance.groupby(["domain_cn", "visible_case_code"], sort=False):
        source_uids = sorted(group["source_item_uid"].drop_duplicates().tolist())
        if len(source_uids) != 1:
            raise AssertionError(f"Final-questionnaire provenance disagrees at {key}: {source_uids}")
        roles = sorted(group["selection_role"].drop_duplicates().tolist())
        if len(roles) != 1:
            raise AssertionError(f"Final-questionnaire selection roles disagree at {key}: {roles}")
        provenance_consensus[key] = (
            source_uids[0],
            roles[0],
            int(group["provenance_manifest"].nunique()),
        )

    if len(questionnaire) != 210:
        raise AssertionError(f"Expected 210 questionnaire items, found {len(questionnaire)}")
    if len(generated) != 810 or generated["item_id"].nunique() != 810:
        raise AssertionError("Current generated-item corpus must contain 810 unique item_id values")
    if len(aggregate) != 810 or aggregate["source_item_uid"].nunique() != 810:
        raise AssertionError("Current API aggregate must contain 810 unique source_item_uid values")

    source_rows: list[dict] = []
    for row in generated.to_dict("records"):
        payload = json.loads(row["content_json"])
        stimulus_signature = render_stimulus_signature(payload, row["task"])
        for wave in ("首轮专家复核", "二次专家复核"):
            source_rows.append(
                {
                    "source_item_uid": row["item_id"],
                    "domain": row["domain"],
                    "task": row["task"],
                    "generator": row["generator"],
                    "model": row["model"],
                    "concept_id": row["concept_id"],
                    "concept": row["concept"],
                    "wave": wave,
                    "source_signature": render_source_signature(payload, row["task"], wave),
                    "stimulus_signature_sha256": stimulus_signature,
                }
            )
    source_signatures = pd.DataFrame(source_rows)
    equivalence_sizes = (
        source_signatures.drop_duplicates("source_item_uid")
        .groupby("stimulus_signature_sha256")["source_item_uid"]
        .nunique()
        .to_dict()
    )
    stimulus_map = (
        source_signatures.drop_duplicates("source_item_uid")
        [[
            "source_item_uid",
            "domain",
            "task",
            "generator",
            "model",
            "concept_id",
            "concept",
            "stimulus_signature_sha256",
        ]]
        .copy()
    )
    stimulus_map["source_equivalence_size"] = stimulus_map[
        "stimulus_signature_sha256"
    ].map(equivalence_sizes)
    stimulus_map = stimulus_map.merge(
        aggregate[["item_id", "source_item_uid", "api_source_regime", "n_judges"]],
        on="source_item_uid",
        how="left",
        validate="one_to_one",
    )
    stimulus_map_path = project / "analysis" / "api_stimulus_equivalence_810.csv"
    stimulus_map.to_csv(stimulus_map_path, index=False, encoding="utf-8-sig")

    unresolved: list[dict] = []
    provisional: list[dict] = []
    for row in questionnaire.to_dict("records"):
        wave = row["wave"]
        domain = DOMAIN_MAP[row["domain"]]
        task = task_letter(row["task"])
        q_signature = questionnaire_signature(row["item_text"], wave)
        candidates = source_signatures.loc[
            (source_signatures["wave"] == wave)
            & (source_signatures["domain"] == domain)
            & (source_signatures["task"] == task)
            & (source_signatures["source_signature"] == q_signature)
        ].copy()
        candidate_records = candidates.to_dict("records")
        if not candidate_records:
            unresolved.append(
                {
                    "wave": wave,
                    "domain_cn": row["domain"],
                    "task": task,
                    "item_local_id": row["item_local_id"],
                    "reason": "no_exact_semantic_signature",
                }
            )
        provisional.append(
            {
                **row,
                "domain_en": domain,
                "task_letter": task,
                "questionnaire_signature": q_signature,
                "candidates": candidate_records,
            }
        )

    if unresolved:
        raise AssertionError(f"Questionnaire items without an exact source signature: {unresolved}")

    # Recover the questionnaire's generator schedule without reading an old crosswalk.
    # At positions with duplicated text, the unique matches in the other domains all
    # identify the same generator; this schedule resolves the duplicated candidate set.
    known_schedule: dict[tuple[str, str], str] = {}
    first_wave = [row for row in provisional if row["wave"] == "首轮专家复核"]
    for task in ("A", "B", "C"):
        local_ids = sorted(
            {row["item_local_id"] for row in first_wave if row["task_letter"] == task},
            key=lambda value: int(value.lstrip("Q")),
        )
        for local_id in local_ids:
            generators = [
                row["candidates"][0]["generator"]
                for row in first_wave
                if row["task_letter"] == task
                and row["item_local_id"] == local_id
                and len(row["candidates"]) == 1
            ]
            if generators:
                counts = Counter(generators)
                expected, count = counts.most_common(1)[0]
                if count != len(generators):
                    raise AssertionError(
                        f"Inconsistent generator schedule at {(task, local_id)}: {counts}"
                    )
                known_schedule[(task, local_id)] = expected

    resolved_rows: list[dict] = []
    for row in provisional:
        candidates = row.pop("candidates")
        original_count = len(candidates)
        duplicate_uids = ";".join(candidate["source_item_uid"] for candidate in candidates)
        schedule_generator = ""
        provenance_manifest_count = 0
        provenance_uid = ""
        selection_role = "broad_scheduled_item"
        if row["wave"] == "二次专家复核":
            provenance_key = (row["domain"], row["item_local_id"])
            if provenance_key not in provenance_consensus:
                raise AssertionError(f"No final-questionnaire provenance for {provenance_key}")
            provenance_uid, selection_role, provenance_manifest_count = provenance_consensus[
                provenance_key
            ]
            if provenance_uid not in {candidate["source_item_uid"] for candidate in candidates}:
                raise AssertionError(
                    f"Provenance source {provenance_uid} is not an exact signature candidate "
                    f"for {provenance_key}: {duplicate_uids}"
                )
        if original_count == 1:
            selected = candidates[0]
            if provenance_uid and selected["source_item_uid"] != provenance_uid:
                raise AssertionError(
                    f"Unique text match disagrees with final provenance at "
                    f"{row['domain']} {row['item_local_id']}"
                )
            match_status = "unique_exact"
            match_basis = (
                "direct_all_field_exact_plus_final_questionnaire_provenance"
                if provenance_uid
                else "direct_all_field_exact"
            )
        elif row["wave"] == "首轮专家复核":
            key = (row["task_letter"], row["item_local_id"])
            schedule_generator = known_schedule.get(key, "")
            scheduled = [
                candidate for candidate in candidates if candidate["generator"] == schedule_generator
            ]
            if len(scheduled) != 1:
                raise AssertionError(
                    f"Could not resolve duplicate at {row['domain']} {key}; "
                    f"expected={schedule_generator!r}, candidates={duplicate_uids}"
                )
            selected = scheduled[0]
            match_status = "exact_via_cross_domain_schedule"
            match_basis = "all_field_exact_plus_cross_domain_schedule"
        else:
            provenance_matches = [
                candidate
                for candidate in candidates
                if candidate["source_item_uid"] == provenance_uid
            ]
            if len(provenance_matches) != 1:
                raise AssertionError(
                    f"Final-questionnaire provenance did not uniquely resolve "
                    f"{row['domain']} {row['item_local_id']}: {duplicate_uids}"
                )
            selected = provenance_matches[0]
            match_status = "exact_via_final_questionnaire_provenance"
            match_basis = "all_field_exact_plus_final_provenance_consensus"

        resolved_rows.append(
            {
                "wave": row["wave"],
                "domain_cn": row["domain"],
                "domain": row["domain_en"],
                "task": row["task_letter"],
                "item_local_id": row["item_local_id"],
                "item_text_hash": row["item_text_hash"],
                "questionnaire_signature_sha256": sha256_text(row["questionnaire_signature"]),
                "source_item_uid": selected["source_item_uid"],
                "generator": selected["generator"],
                "model": selected["model"],
                "concept_id": selected["concept_id"],
                "concept": selected["concept"],
                "match_status": match_status,
                "match_basis": match_basis,
                "exact_candidate_count": original_count,
                "duplicate_candidate_uids": duplicate_uids if original_count > 1 else "",
                "candidate_source_uids": duplicate_uids,
                "schedule_expected_generator": schedule_generator,
                "provenance_source_item_uid": provenance_uid,
                "provenance_manifest_count": provenance_manifest_count,
                "selection_role": selection_role,
                "text_identifiable": original_count == 1,
                "stimulus_signature_sha256": selected["stimulus_signature_sha256"],
                "source_equivalence_size": equivalence_sizes[
                    selected["stimulus_signature_sha256"]
                ],
            }
        )

    crosswalk = pd.DataFrame(resolved_rows)
    crosswalk = crosswalk.merge(
        aggregate[
            ["item_id", "source_item_uid", "api_source_regime", "n_judges"]
        ],
        on="source_item_uid",
        how="left",
        validate="many_to_one",
    )
    if crosswalk["item_id"].eq("").any() or crosswalk["item_id"].isna().any():
        raise AssertionError("At least one questionnaire item is missing from the API aggregate")
    if len(crosswalk) != 210:
        raise AssertionError(f"Crosswalk row count changed unexpectedly: {len(crosswalk)}")
    if crosswalk.duplicated(["wave", "domain_cn", "item_local_id"]).any():
        raise AssertionError("Questionnaire keys are not unique in the crosswalk")

    output_path = project / "analysis" / "human_api_crosswalk.csv"
    crosswalk.to_csv(output_path, index=False, encoding="utf-8-sig")

    audit = {
        "questionnaire_dictionary": str(dictionary_path),
        "generated_item_source": str(generated_path),
        "api_aggregate_source": str(aggregate_path),
        "n_questionnaire_items": int(len(crosswalk)),
        "n_first_wave": int((crosswalk["wave"] == "首轮专家复核").sum()),
        "n_second_wave": int((crosswalk["wave"] == "二次专家复核").sum()),
        "n_direct_unique_exact": int((crosswalk["match_status"] == "unique_exact").sum()),
        "n_schedule_resolved_exact": int(
            (crosswalk["match_status"] == "exact_via_cross_domain_schedule").sum()
        ),
        "n_provenance_resolved_exact": int(
            (
                crosswalk["match_status"]
                == "exact_via_final_questionnaire_provenance"
            ).sum()
        ),
        "n_final_questionnaire_provenance_manifests": len(provenance_paths),
        "final_questionnaire_provenance_manifests": [str(path) for path in provenance_paths],
        "all_api_items_present": bool(crosswalk["item_id"].notna().all()),
        "all_api_records_have_nine_judges": bool(
            pd.to_numeric(crosswalk["n_judges"], errors="coerce").eq(9).all()
        ),
        "crosswalk_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "api_stimulus_equivalence_sha256": hashlib.sha256(
            stimulus_map_path.read_bytes()
        ).hexdigest(),
    }
    audit_path = project / "analysis" / "human_api_crosswalk_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
