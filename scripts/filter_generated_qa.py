#!/usr/bin/env python3
"""Normalize and filter generated RepoSpec QA examples.

This script is for replacing or extending the bootstrap Stage 1 data with
LLM-generated QA while preserving the same schema and leakage controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fingerprint(question: str, answer: str) -> str:
    return hashlib.sha1((norm_text(question) + "\n" + norm_text(answer)).encode()).hexdigest()


def canonicalize_row(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    repo_id = row.get("repo_id")
    if not repo_id or repo_id not in manifest["pilot_repos"]:
        return None
    repo_info = manifest["pilot_repos"][repo_id]
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    evidence_spans = row.get("evidence_spans") or []
    evidence_files = row.get("evidence_files") or [span.get("path") for span in evidence_spans if span.get("path")]
    evidence_files = [str(path) for path in evidence_files if path]

    if len(question) < 20 or len(answer) < 40 or not evidence_files:
        return None

    eval_files = set(repo_info.get("eval_evidence_files_extracted", []))
    if set(evidence_files).intersection(eval_files):
        return None

    qid = row.get("question_id") or f"{repo_id}:generated:{fingerprint(question, answer)[:12]}"
    return {
        "repo": repo_info["repo"],
        "repo_id": repo_id,
        "repo_url": repo_info["repo_url"],
        "commit": repo_info["commit"],
        "split": "target_sft_train",
        "question_id": qid,
        "question": question,
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "loss_on": "assistant_only",
        "evidence_files": evidence_files,
        "evidence_spans": evidence_spans,
        "task_type": "repo_qa",
        "source": row.get("source", "llm_generated_from_repo"),
        "generator": row.get("generator", "pending_generator_name"),
        "quality_flags": {
            "has_file_evidence": True,
            "answer_mentions_unknown": "unknown" in answer.lower(),
            "requires_cross_file_context": bool(row.get("requires_cross_file_context", False)),
            "excluded_due_to_eval_file_overlap": False,
            "near_duplicate_of_eval": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Raw generated QA JSONL.")
    parser.add_argument("--manifest", required=True, type=Path, help="pilot_split_manifest.json.")
    parser.add_argument("--output", required=True, type=Path, help="Filtered target SFT JSONL.")
    parser.add_argument("--report", required=True, type=Path, help="Filtering report JSON.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    rows = load_jsonl(args.input)
    kept = []
    seen = set()
    dropped = 0
    duplicate = 0
    for row in rows:
        clean = canonicalize_row(row, manifest)
        if clean is None:
            dropped += 1
            continue
        fp = fingerprint(clean["question"], clean["answer"])
        if fp in seen:
            duplicate += 1
            continue
        seen.add(fp)
        kept.append(clean)

    write_jsonl(args.output, kept)
    report = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_invalid_or_leaky_rows": dropped,
        "dropped_duplicate_rows": duplicate,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
