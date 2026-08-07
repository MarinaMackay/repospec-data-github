#!/usr/bin/env python3
"""Create a stricter QA subset for formal experiments.

This does not replace human/LLM data generation. It filters an existing SFT JSONL
into a cleaner subset and keeps the rules explicit so later generated QA can go
through the same gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_low_signal_path(path: str) -> bool:
    parts = path.split("/")
    return (
        parts[0] in {"tests", "test", "asv_bench", "examples", "example", "benchmarks", "doc", "docs"}
        or "/tests/" in path
        or "/test/" in path
        or "benchmark" in path.lower()
        or path in {"conftest.py", "setup.py", "versioneer.py"}
        or path.endswith("/conftest.py")
    )


def rejection_reason(row: dict[str, Any], eval_files_by_repo: dict[str, set[str]], min_answer_chars: int) -> str | None:
    question_id = row.get("question_id", "")
    answer = row.get("answer", "")
    evidence_files = row.get("evidence_files") or []
    repo_id = row.get("repo_id", "")

    if ":symbol:" not in question_id:
        return "not_symbol_level"
    if ":top_symbols" in question_id:
        return "list_symbols_question"
    if any(is_low_signal_path(path) for path in evidence_files):
        return "low_signal_path"
    if set(evidence_files).intersection(eval_files_by_repo.get(repo_id, set())):
        return "eval_file_overlap"
    if "Its docstring says:" not in answer:
        return "not_docstring_grounded"
    if len(answer) < min_answer_chars:
        return "answer_too_short"
    if any(token in answer.lower() for token in ["no docstring", "unknown", "not detected"]):
        return "unsupported_or_empty_answer"
    return None


def dev_bucket(row: dict[str, Any]) -> bool:
    key = f"{row['repo_id']}::{row['question_id']}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest(), 16) % 20 == 0


def message_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": row["messages"],
        "repo_id": row["repo_id"],
        "question_id": row["question_id"],
        "loss_on": "assistant_only",
        "curation_tier": row["curation"]["tier"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PACKAGE_DIR / "data/target_sft_train.bootstrap.jsonl")
    parser.add_argument("--manifest", type=Path, default=PACKAGE_DIR / "data/pilot_split_manifest.json")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "data/target_sft_train.curated_v0.jsonl")
    parser.add_argument("--train-output", type=Path, default=PACKAGE_DIR / "training/target_sft_curated_train.messages.jsonl")
    parser.add_argument("--dev-output", type=Path, default=PACKAGE_DIR / "training/target_sft_curated_dev.messages.jsonl")
    parser.add_argument("--report", type=Path, default=PACKAGE_DIR / "reports/curation_report.json")
    parser.add_argument("--min-answer-chars", type=int, default=150)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    eval_files_by_repo = {
        repo_id: set(info.get("eval_evidence_files_extracted", []))
        for repo_id, info in manifest["pilot_repos"].items()
    }

    kept = []
    rejected = Counter()
    for row in rows:
        reason = rejection_reason(row, eval_files_by_repo, args.min_answer_chars)
        if reason:
            rejected[reason] += 1
            continue
        clean = dict(row)
        clean["curation"] = {
            "tier": "curated_v0",
            "rules": [
                "symbol-level QA only",
                "exclude tests, examples, benchmarks, docs, setup/conftest/versioneer",
                "require docstring-grounded answer",
                "require explicit evidence file and no extracted eval-file overlap",
                f"minimum answer length {args.min_answer_chars} chars",
            ],
            "intended_use": "formal-experiment candidate; replace/augment with LLM-generated QA before final paper runs",
        }
        kept.append(clean)

    train_rows = []
    dev_rows = []
    for row in kept:
        if dev_bucket(row):
            dev_rows.append(message_row(row))
        else:
            train_rows.append(message_row(row))

    write_jsonl(args.output, kept)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)

    report = {
        "source": str(args.input.relative_to(PACKAGE_DIR) if args.input.is_relative_to(PACKAGE_DIR) else args.input),
        "output": str(args.output.relative_to(PACKAGE_DIR) if args.output.is_relative_to(PACKAGE_DIR) else args.output),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "kept_by_repo": dict(Counter(row["repo_id"] for row in kept)),
        "train_by_repo": dict(Counter(row["repo_id"] for row in train_rows)),
        "dev_by_repo": dict(Counter(row["repo_id"] for row in dev_rows)),
        "rejected_by_reason": dict(rejected),
        "notes": [
            "curated_v0 is a stricter bootstrap subset, not a claim that the data is final paper-quality synthetic QA.",
            "Use the same script on LLM-generated QA after generation to produce the formal experiment dataset.",
        ],
    }
    write_json(args.report, report)
    print(f"kept {len(kept)} / {len(rows)} rows")
    print(f"train/dev {len(train_rows)} / {len(dev_rows)}")


if __name__ == "__main__":
    main()
