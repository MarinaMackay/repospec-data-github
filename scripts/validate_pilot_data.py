#!/usr/bin/env python3
"""Validate the generated RepoSpec pilot data package."""

from __future__ import annotations

import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"


def load_jsonl(path: Path) -> list[dict]:
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    required_files = [
        PACKAGE_DIR / "README.md",
        PACKAGE_DIR / "DATASET_CARD.md",
        PACKAGE_DIR / "config/pilot_config.json",
        PACKAGE_DIR / "schemas/target_sft.schema.json",
        PACKAGE_DIR / "schemas/distill_prompt.schema.json",
        PACKAGE_DIR / "schemas/eval.schema.json",
        PACKAGE_DIR / "schemas/teacher_materialized.schema.json",
        PACKAGE_DIR / "prompts/repo_qa_generation_prompt.md",
        PACKAGE_DIR / "prompts/teacher_distillation_contract.md",
        PACKAGE_DIR / "reports/official_sft_trajectories_report.json",
        PACKAGE_DIR / "training/target_sft_train.messages.jsonl",
        PACKAGE_DIR / "training/target_sft_dev.messages.jsonl",
        PACKAGE_DIR / "training/target_sft_curated_train.messages.jsonl",
        PACKAGE_DIR / "training/target_sft_curated_dev.messages.jsonl",
        PACKAGE_DIR / "reports/training_export_report.json",
        PACKAGE_DIR / "reports/curation_report.json",
    ]
    for path in required_files:
        require(path.exists(), f"required package file missing: {path}")

    manifest = json.loads((DATA_DIR / "pilot_split_manifest.json").read_text(encoding="utf-8"))
    eval_rows = load_jsonl(DATA_DIR / "pilot_eval.official_sweqapro.jsonl")
    train_rows = load_jsonl(DATA_DIR / "target_sft_train.bootstrap.jsonl")
    distill_rows = load_jsonl(DATA_DIR / "distill_prompts.pending_teacher.jsonl")
    training_rows = load_jsonl(PACKAGE_DIR / "training/target_sft_train.messages.jsonl")
    dev_rows = load_jsonl(PACKAGE_DIR / "training/target_sft_dev.messages.jsonl")
    curated_rows = load_jsonl(DATA_DIR / "target_sft_train.curated_v0.jsonl")
    curated_train_rows = load_jsonl(PACKAGE_DIR / "training/target_sft_curated_train.messages.jsonl")
    curated_dev_rows = load_jsonl(PACKAGE_DIR / "training/target_sft_curated_dev.messages.jsonl")
    sft_report = json.loads((PACKAGE_DIR / "reports/official_sft_trajectories_report.json").read_text(encoding="utf-8"))

    require(len(eval_rows) == 30, f"expected 30 official eval rows, got {len(eval_rows)}")
    require(len(train_rows) > 0, "target SFT train data is empty")
    require(len(distill_rows) == len(train_rows), "distillation prompts must match train rows")
    require(len(training_rows) + len(dev_rows) == len(train_rows), "train/dev exports must cover all SFT rows")
    require(len(curated_rows) > 0, "curated QA data is empty")
    require(
        len(curated_train_rows) + len(curated_dev_rows) == len(curated_rows),
        "curated train/dev exports must cover all curated rows",
    )

    required_train_keys = {
        "repo",
        "repo_id",
        "repo_url",
        "commit",
        "split",
        "question_id",
        "question",
        "answer",
        "messages",
        "loss_on",
        "evidence_files",
        "evidence_spans",
        "task_type",
        "source",
        "generator",
        "quality_flags",
    }
    for idx, row in enumerate(train_rows):
        missing = required_train_keys - set(row)
        require(not missing, f"train row {idx} missing keys: {sorted(missing)}")
        require(row["loss_on"] == "assistant_only", f"train row {idx} has wrong loss mask")
        require(row["messages"][-1]["role"] == "assistant", f"train row {idx} message format invalid")
        require(row["evidence_files"], f"train row {idx} missing evidence files")

    train_files_by_repo = {}
    for row in train_rows:
        train_files_by_repo.setdefault(row["repo_id"], set()).update(row["evidence_files"])

    for repo_id, repo_manifest in manifest["pilot_repos"].items():
        eval_files = set(repo_manifest["eval_evidence_files_extracted"])
        overlap = train_files_by_repo.get(repo_id, set()).intersection(eval_files)
        require(not overlap, f"{repo_id} train/eval evidence overlap: {sorted(overlap)[:20]}")
        require(repo_manifest["official_eval_count"] == 10, f"{repo_id} expected 10 eval rows")
        require(repo_manifest["commit"], f"{repo_id} missing commit")

    for idx, row in enumerate(distill_rows):
        require(row["teacher_sequence"] is None, f"distill row {idx} should not fake teacher sequence")
        require(
            row["training_signal"]["status"] == "requires_training_output",
            f"distill row {idx} should mark teacher data dependency",
        )

    for idx, row in enumerate(training_rows + dev_rows):
        require(set(row.keys()) == {"messages", "repo_id", "question_id", "loss_on"}, f"export row {idx} has unexpected keys")
        require(row["loss_on"] == "assistant_only", f"export row {idx} has wrong loss mask")
        require(row["messages"][-1]["role"] == "assistant", f"export row {idx} final message is not assistant")

    for idx, row in enumerate(curated_rows):
        require(row.get("curation", {}).get("tier") == "curated_v0", f"curated row {idx} missing tier")
        require(":symbol:" in row["question_id"], f"curated row {idx} is not symbol-level")
        require("Its docstring says:" in row["answer"], f"curated row {idx} is not docstring-grounded")

    for idx, row in enumerate(curated_train_rows + curated_dev_rows):
        require(row.get("curation_tier") == "curated_v0", f"curated export row {idx} missing tier")

    require(sft_report["row_count"] == 1000, "official SFT trajectories report should show 1000 rows")
    require(
        all(v == 0 for v in sft_report["pilot_repo_exact_basename_counts"].values()),
        "official SFT trajectories should not be treated as exact pilot repo data",
    )

    print("VALIDATION PASSED")
    print(f"official_eval_rows={len(eval_rows)}")
    print(f"target_sft_train_rows={len(train_rows)}")
    print(f"distill_prompt_rows={len(distill_rows)}")
    print(f"message_train_rows={len(training_rows)}")
    print(f"message_dev_rows={len(dev_rows)}")
    print(f"curated_rows={len(curated_rows)}")
    print(f"curated_message_train_rows={len(curated_train_rows)}")
    print(f"curated_message_dev_rows={len(curated_dev_rows)}")


if __name__ == "__main__":
    main()
