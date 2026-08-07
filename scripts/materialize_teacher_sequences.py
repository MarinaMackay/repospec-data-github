#!/usr/bin/env python3
"""Materialize Stage 2 teacher sequences from target+repo-LoRA.

This script intentionally has optional heavy dependencies. It is ready for the
training machine where transformers/peft and the target adapters are available.
Use --dry-run locally to validate inputs without loading models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def dry_run_rows(rows: list[dict[str, Any]], target_model_id: str, adapter_root: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "example_id": row["example_id"],
                "repo": row["repo"],
                "repo_id": row["repo_id"],
                "commit": row["commit"],
                "target_model_id": target_model_id,
                "target_adapter_id": f"{adapter_root}/{row['repo_id']}",
                "prompt": row["prompt"],
                "teacher_sequence": None,
                "teacher_decode_policy": {"mode": "dry_run", "temperature": 0.0},
                "training_signal": {
                    "status": "not_materialized_dry_run",
                    "reason": "Run without --dry-run on a machine with target LoRA adapters.",
                },
            }
        )
    return out


def materialize(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers/peft/torch. Install them on the training machine or run with --dry-run."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    adapters: dict[str, Any] = {}
    outputs = []
    for row in rows:
        repo_id = row["repo_id"]
        adapter_path = str(Path(args.adapter_root) / repo_id)
        if repo_id not in adapters:
            adapters[repo_id] = PeftModel.from_pretrained(base_model, adapter_path)
        model = adapters[repo_id]
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        teacher_sequence = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        outputs.append(
            {
                "example_id": row["example_id"],
                "repo": row["repo"],
                "repo_id": repo_id,
                "commit": row["commit"],
                "target_model_id": args.target_model_id,
                "target_adapter_id": adapter_path,
                "prompt": row["prompt"],
                "teacher_sequence": teacher_sequence,
                "teacher_decode_policy": {
                    "mode": "greedy",
                    "temperature": 0.0,
                    "max_new_tokens": args.max_new_tokens,
                },
                "training_signal": {
                    "status": "teacher_sequence_materialized",
                    "format": "text_sequence_only",
                    "note": "For KL or DFlash cache training, extend this script to persist logits or hidden states at the required positions.",
                },
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--adapter-root", required=True, help="Directory containing one subdir per repo_id.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        out = dry_run_rows(rows, args.target_model_id, args.adapter_root)
    else:
        out = materialize(rows, args)
    write_jsonl(args.output, out)
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
