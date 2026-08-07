#!/usr/bin/env python3
"""Summarize the official SWE-QA-Pro SFT trajectories format."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IN_PATH = ROOT / "work/raw/SWE-QA-Pro-SFT-Trajectories/train.jsonl"
OUT_PATH = ROOT / "outputs/repo_spec_data_package/reports/official_sft_trajectories_report.json"
PATH_RE = re.compile(r"Repository Path:\s*([^\n]+)")


def main() -> None:
    role_counts = Counter()
    repo_counts = Counter()
    tool_counts = Counter()
    row_count = 0
    first_shape = None
    with IN_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            if first_shape is None:
                first_shape = {
                    "keys": sorted(row.keys()),
                    "tool_count": len(row.get("tools", [])),
                    "message_count": len(row.get("messages", [])),
                    "first_message_roles": [m.get("role") for m in row.get("messages", [])[:8]],
                }
            for tool in row.get("tools", []):
                name = tool.get("function", {}).get("name") or tool.get("name")
                if name:
                    tool_counts[name] += 1
            user_text = "\n".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user")
            match = PATH_RE.search(user_text)
            if match:
                basename = match.group(1).strip().rstrip("/").split("/")[-1]
                repo_counts[basename] += 1
            for msg in row.get("messages", []):
                role_counts[msg.get("role", "missing")] += 1

    report = {
        "source": "TIGER-Lab/SWE-QA-Pro-SFT-Trajectories",
        "local_path": str(IN_PATH.relative_to(ROOT)),
        "row_count": row_count,
        "first_shape": first_shape,
        "message_role_counts": dict(role_counts),
        "tool_counts": dict(tool_counts),
        "unique_repository_basenames": len(repo_counts),
        "pilot_repo_exact_basename_counts": {
            name: repo_counts.get(name, 0) for name in ["xarray", "numba", "sphinx"]
        },
        "interpretation": (
            "The official trajectories are useful as an agentic SFT format reference. "
            "They do not contain exact pilot repo trajectories by basename, so they should not be treated as per-repo LoRA data for xarray/numba/sphinx."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
