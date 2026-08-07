#!/usr/bin/env python3
"""Export training-friendly message-only files from the annotated SFT JSONL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
TRAINING_DIR = PACKAGE_DIR / "training"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dev_bucket(row: dict) -> bool:
    key = f"{row['repo_id']}::{row['question_id']}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest(), 16) % 20 == 0


def main() -> None:
    rows = load_jsonl(DATA_DIR / "target_sft_train.bootstrap.jsonl")
    train_rows = []
    dev_rows = []
    for row in rows:
        exported = {
            "messages": row["messages"],
            "repo_id": row["repo_id"],
            "question_id": row["question_id"],
            "loss_on": "assistant_only",
        }
        if dev_bucket(row):
            dev_rows.append(exported)
        else:
            train_rows.append(exported)

    write_jsonl(TRAINING_DIR / "target_sft_train.messages.jsonl", train_rows)
    write_jsonl(TRAINING_DIR / "target_sft_dev.messages.jsonl", dev_rows)

    report = {
        "source": "data/target_sft_train.bootstrap.jsonl",
        "format": "chat messages JSONL with assistant-only loss expected by trainer",
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_by_repo": dict(Counter(row["repo_id"] for row in train_rows)),
        "dev_by_repo": dict(Counter(row["repo_id"] for row in dev_rows)),
        "notes": [
            "Use the bootstrap JSONL if the trainer wants evidence metadata.",
            "Use these message files if the trainer expects chat-style SFT rows.",
        ],
    }
    (PACKAGE_DIR / "reports/training_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train_rows)} train rows and {len(dev_rows)} dev rows")


if __name__ == "__main__":
    main()
