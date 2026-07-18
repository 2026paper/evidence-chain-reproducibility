#!/usr/bin/env python3
"""Read-only anonymity, secret, path, and forbidden-artifact audit."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".bib", ".csv", ".json", ".jsonl", ".md", ".py", ".tex", ".txt", ".yaml", ".yml"}
BANNED_PARTS = {"__pycache__", "failed_attempts", "pilot"}
BANNED_FRAGMENTS = {"api_call_log", "stderr", "stdout", "raw_payload"}
BANNED_EXTENSIONS = {".doc", ".docx", ".env", ".key", ".p12", ".pem", ".xls", ".xlsx"}
SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAQ\.[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bark-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
ABSOLUTE_PATHS = [
    re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:Users|Desktop|API_test|Documents|Downloads)\\"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s]+/"),
]
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")

errors: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT)
    lower_parts = {part.lower() for part in relative.parts}
    lower_name = path.name.lower()
    if lower_parts & BANNED_PARTS:
        errors.append(f"forbidden directory: {relative.as_posix()}")
    if any(fragment in lower_name for fragment in BANNED_FRAGMENTS):
        errors.append(f"forbidden filename: {relative.as_posix()}")
    if path.suffix.lower() in BANNED_EXTENSIONS:
        errors.append(f"forbidden extension: {relative.as_posix()}")
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        continue
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        errors.append(f"credential-shaped value: {relative.as_posix()}")
    if any(pattern.search(text) for pattern in ABSOLUTE_PATHS):
        errors.append(f"absolute local path: {relative.as_posix()}")
    if EMAIL.search(text):
        errors.append(f"email address: {relative.as_posix()}")

if errors:
    print("FAIL")
    for error in sorted(set(errors)):
        print(f"- {error}")
    sys.exit(1)
print("PASS: no credentials, direct emails, local absolute paths, raw logs, or forbidden artifacts detected")
