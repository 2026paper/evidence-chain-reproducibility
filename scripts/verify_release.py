#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

errors = []
manifest = ROOT / "manifests/sha256_manifest.csv"
with manifest.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        path = ROOT / row["path"]
        if not path.exists():
            errors.append(f"missing: {row['path']}")
        elif sha256(path) != row["sha256"]:
            errors.append(f"hash mismatch: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            errors.append(f"size mismatch: {row['path']}")

for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".doc", ".docx"}:
        errors.append(f"forbidden raw/document extension: {path.relative_to(ROOT)}")

individual = [
    ROOT / "analysis/cleaned_human_ratings_long.csv",
    ROOT / "analysis/public_abc_cleaned_long.csv",
    ROOT / "analysis/difference_survey_cleaned_long.csv",
]
banned_columns = {
    "duration_seconds", "source_file", "education", "broad_field", "science_reading_frequency",
    "self_reported_role", "self_reported_field", "science_review_familiarity",
    "ai_evaluation_familiarity", "ip", "longitude", "latitude", "device", "timestamp",
    "platform_user_id", "raw_response_id", "open_text", "email", "phone",
}
for path in individual:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        bad = sorted(fields & banned_columns)
        if bad:
            errors.append(f"banned columns in {path.name}: {bad}")
        rows = list(reader)
    id_fields = [field for field in fields if field in {"participant_id", "response_id", "respondent_id"}]
    patterns = {
        "participant_id": re.compile(r"^(HR|ABCP)_[0-9A-F]{16}$"),
        "response_id": re.compile(r"^ABCR_[0-9A-F]{16}$"),
        "respondent_id": re.compile(r"^DIFF_[0-9A-F]{16}$"),
    }
    for field in id_fields:
        if any(not patterns[field].fullmatch(row[field]) for row in rows):
            errors.append(f"invalid release key in {path.name}:{field}")

crosswalk = ROOT / "analysis/public_semantic_crosswalk_hash_release.csv"
with crosswalk.open(encoding="utf-8", newline="") as f:
    fields = set(csv.DictReader(f).fieldnames or [])
for forbidden in ["core_claim", "target_misconception", "boundary_conditions", "adaptation_or_intervention",
                  "lay_or_controlled_text", "comprehension_or_pair_question"]:
    if forbidden in fields:
        errors.append(f"full-text field retained: {forbidden}")

material_csv_counts = {
    "materials/generated_items_810.csv": 810,
    "materials/human_review_questionnaire_items.csv": 210,
    "materials/public_semantic_crosswalk_full.csv": 34,
}
for relative, expected_rows in material_csv_counts.items():
    path = ROOT / relative
    if not path.exists():
        errors.append(f"missing scientific material: {relative}")
        continue
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_rows:
        errors.append(f"unexpected material row count in {relative}: {len(rows)} != {expected_rows}")

material_json_counts = {
    "materials/public_abc_answer_key.json": ("concepts", 6),
    "materials/difference_survey_answer_key.json": ("cases", 8),
}
for relative, (key, expected_rows) in material_json_counts.items():
    path = ROOT / relative
    if not path.exists():
        errors.append(f"missing scientific material: {relative}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get(key, [])) != expected_rows:
        errors.append(f"unexpected key count in {relative}:{key}")

controlled_pairs = ROOT / "materials/controlled_ab_pairs_full.json"
if not controlled_pairs.exists():
    errors.append("missing scientific material: materials/controlled_ab_pairs_full.json")
elif len(json.loads(controlled_pairs.read_text(encoding="utf-8"))) != 8:
    errors.append("controlled A/B material count is not 8")

for form in ["A", "B", "C"]:
    path = ROOT / f"materials/public_form_{form}_instrument.txt"
    if not path.exists() or "Q20" not in path.read_text(encoding="utf-8"):
        errors.append(f"public form {form} instrument is missing or incomplete")
difference_instrument = ROOT / "materials/difference_survey_instrument.txt"
if not difference_instrument.exists() or "材料八" not in difference_instrument.read_text(encoding="utf-8"):
    errors.append("difference-survey instrument is missing or incomplete")
for domain in ["physics", "chemistry", "biology", "geography", "atmospheric_science"]:
    path = ROOT / f"materials/human_review_wave1_{domain}_instrument.txt"
    if not path.exists() or path.read_text(encoding="utf-8").count("\nQ") < 30:
        errors.append(f"broad human-review instrument is missing or incomplete: {domain}")

secret_patterns = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
allowed_absolute_source = ROOT / "scripts/build_review_package.py"
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".csv"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    if any(pattern.search(text) for pattern in secret_patterns):
        errors.append(f"credential-shaped value: {path.relative_to(ROOT)}")
    if path != allowed_absolute_source and re.search(r"(?i)[A-Z]:\\(?:Users|Desktop|API_test|Desktop)", text):
        errors.append(f"absolute local path: {path.relative_to(ROOT)}")

audit = json.loads((ROOT / "manifests/privacy_audit.json").read_text(encoding="utf-8"))
if audit.get("overall_status") != "PASS":
    errors.append("privacy_audit overall_status is not PASS")

public_check = ROOT / "verification/cleaned_only_public_recalculation.json"
if not public_check.exists():
    errors.append("cleaned-only public recalculation output is missing")
elif json.loads(public_check.read_text(encoding="utf-8")).get("qa_status") != "PASS":
    errors.append("cleaned-only public recalculation did not pass")

if errors:
    print("FAIL")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)
print("PASS: hashes, schemas, release keys, package contents, and privacy assertions verified")
