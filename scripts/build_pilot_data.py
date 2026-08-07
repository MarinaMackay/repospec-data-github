#!/usr/bin/env python3
"""Build the RepoSpec pilot data package from local SWE-QA-Pro assets.

Inputs expected under the workspace:
- work/raw/SWE-QA-Pro-Bench/data/test.jsonl
- work/raw/SWE-QA-Pro/eval/repos.txt
- work/raw/repos/{xarray,numba,sphinx} checked out at benchmark commits

Outputs are written under outputs/repo_spec_data_package/data and reports.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BENCH_PATH = ROOT / "work/raw/SWE-QA-Pro-Bench/data/test.jsonl"
REPOS_TXT = ROOT / "work/raw/SWE-QA-Pro/eval/repos.txt"
REPO_ROOT = ROOT / "work/raw/repos"
OUT_DIR = ROOT / "outputs/repo_spec_data_package/data"
REPORT_DIR = ROOT / "outputs/repo_spec_data_package/reports"

PILOT_REPOS = {
    "pydata/xarray": {
        "repo_id": "xarray",
        "local_dir": "xarray",
        "url": "https://github.com/pydata/xarray",
    },
    "numba/numba": {
        "repo_id": "numba",
        "local_dir": "numba",
        "url": "https://github.com/numba/numba",
    },
    "sphinx-doc/sphinx": {
        "repo_id": "sphinx",
        "local_dir": "sphinx",
        "url": "https://github.com/sphinx-doc/sphinx",
    },
}

SOURCE_EXTS = {
    ".py",
    ".pyx",
    ".pxd",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rst",
    ".md",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
}

SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    "node_modules",
    ".eggs",
}

PATH_RE = re.compile(
    r"(?P<path>/?(?:[A-Za-z0-9_.+-]+/)*[A-Za-z0-9_.+-]+"
    r"\.(?:py|pyx|pxd|c|cc|cpp|h|hpp|rst|md|toml|cfg|ini|yaml|yml))"
)


@dataclass
class Symbol:
    kind: str
    name: str
    lineno: int
    end_lineno: int
    doc: str


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_repo_commits() -> dict[str, str]:
    commits = {}
    with REPOS_TXT.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            url, commit = stripped.split()
            for repo_name, spec in PILOT_REPOS.items():
                if spec["url"] == url:
                    commits[repo_name] = commit
    return commits


def normalize_evidence_path(path: str) -> str:
    if path.startswith("/testbed/"):
        return path[len("/testbed/") :]
    return path.lstrip("/")


def extract_evidence_files(answer: str) -> list[str]:
    files = []
    for match in PATH_RE.finditer(answer):
        candidate = normalize_evidence_path(match.group("path"))
        if candidate not in files:
            files.append(candidate)
    return files


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def list_source_files(repo_path: Path) -> list[str]:
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_path)
        if should_skip(rel):
            continue
        if rel.suffix in SOURCE_EXTS:
            files.append(rel.as_posix())
    return sorted(files)


def safe_read(path: Path, limit_chars: int = 200_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return text[:limit_chars]


def one_line(text: str, max_len: int = 260) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def parse_python_symbols(path: Path) -> tuple[str, list[Symbol]]:
    text = safe_read(path)
    if not text:
        return "", []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "", []
    module_doc = ast.get_docstring(tree) or ""
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(
                Symbol(
                    "class",
                    node.name,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    ast.get_docstring(node) or "",
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                Symbol(
                    "function",
                    node.name,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    ast.get_docstring(node) or "",
                )
            )
    return module_doc, symbols


def make_stage1_examples(repo_name: str, spec: dict[str, str], commit: str, train_files: list[str]) -> list[dict[str, Any]]:
    repo_id = spec["repo_id"]
    repo_path = REPO_ROOT / spec["local_dir"]
    rows: list[dict[str, Any]] = []
    for rel in train_files:
        if not rel.endswith(".py"):
            continue
        module_doc, symbols = parse_python_symbols(repo_path / rel)
        if not module_doc and not symbols:
            continue

        top_symbols = [s for s in symbols if not s.name.startswith("_")]
        symbol_names = [f"{s.kind} `{s.name}`" for s in top_symbols[:12]]
        base_id = f"{repo_id}:{rel.replace('/', '__')}"

        if module_doc:
            q = f"What is the role of `{rel}` in the `{repo_id}` repository?"
            a = (
                f"At commit {commit}, `{rel}` is a repository source file. "
                f"Its module documentation states: {one_line(module_doc)} "
                f"Visible top-level symbols include {', '.join(symbol_names[:8]) if symbol_names else 'no public top-level classes or functions detected'}. "
                f"Evidence: `{rel}`."
            )
            rows.append(example_row(repo_name, spec, commit, f"{base_id}:module_role", q, a, rel, 1, 1))

        documented_symbols = [item for item in top_symbols if item.doc]
        if documented_symbols:
            s = documented_symbols[0]
            q = f"What does the {s.kind} `{s.name}` in `{rel}` do?"
            a = (
                f"In `{repo_id}`, `{s.name}` is a top-level {s.kind} defined in `{rel}`. "
                f"Its docstring says: {one_line(s.doc)} "
                f"Evidence: `{rel}` lines {s.lineno}-{s.end_lineno}."
            )
            rows.append(example_row(repo_name, spec, commit, f"{base_id}:symbol:{s.name}", q, a, rel, s.lineno, s.end_lineno))

        if len(top_symbols) >= 2:
            q = f"Which public top-level Python symbols are defined in `{rel}`?"
            names = ", ".join(symbol_names[:16])
            a = (
                f"`{rel}` defines the following public top-level symbols detected by AST parsing: {names}. "
                f"Evidence: `{rel}`."
            )
            rows.append(example_row(repo_name, spec, commit, f"{base_id}:top_symbols", q, a, rel, 1, 1))
    return rows


def example_row(
    repo_name: str,
    spec: dict[str, str],
    commit: str,
    question_id: str,
    question: str,
    answer: str,
    evidence_file: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    return {
        "repo": repo_name,
        "repo_id": spec["repo_id"],
        "repo_url": spec["url"],
        "commit": commit,
        "split": "target_sft_train",
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "loss_on": "assistant_only",
        "evidence_files": [evidence_file],
        "evidence_spans": [
            {
                "path": evidence_file,
                "start_line": start_line,
                "end_line": end_line,
            }
        ],
        "task_type": "repo_qa",
        "source": "deterministic_ast_bootstrap",
        "generator": "build_pilot_data.py",
        "quality_flags": {
            "has_file_evidence": True,
            "answer_mentions_unknown": False,
            "requires_cross_file_context": False,
            "excluded_due_to_eval_file_overlap": False,
            "near_duplicate_of_eval": False,
        },
    }


def make_distill_prompts(stage1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = []
    for i, row in enumerate(stage1_rows):
        prompts.append(
            {
                "repo": row["repo"],
                "repo_id": row["repo_id"],
                "commit": row["commit"],
                "split": "distill_train",
                "example_id": f"{row['repo_id']}:distill:{i:06d}",
                "source_question_id": row["question_id"],
                "target_model_id": "pending_training_config",
                "target_adapter_id": f"target_lora_{row['repo_id']}_pending",
                "draft_model_id": "pending_training_config",
                "prompt_template_id": "repo_qa_direct_v1",
                "prompt": (
                    "Answer the repository question using the pinned repository state.\n\n"
                    f"Repository: {row['repo']}\n"
                    f"Commit: {row['commit']}\n"
                    f"Question: {row['question']}"
                ),
                "reference_answer": row["answer"],
                "reference_answer_usage": "prompt_scaffolding_only_not_draft_ce_target",
                "teacher_sequence_source": "requires_frozen_target_plus_repo_lora",
                "teacher_sequence": None,
                "training_signal": {
                    "status": "requires_training_output",
                    "accepted_formats": [
                        "teacher_logits",
                        "target_hidden_state_cache",
                        "projected_context_feature_cache",
                        "dflash_native_cache",
                    ],
                },
                "draft_block": {
                    "block_size": "pending_dflash_config",
                    "anchor_positions": [],
                    "masked_positions": [],
                },
            }
        )
    return prompts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    commits = load_repo_commits()
    bench_rows = load_jsonl(BENCH_PATH)

    pilot_eval = []
    by_repo_eval: dict[str, list[dict[str, Any]]] = {repo: [] for repo in PILOT_REPOS}
    for row in bench_rows:
        repo = row["repo"]
        if repo not in PILOT_REPOS:
            continue
        spec = PILOT_REPOS[repo]
        evidence_files = extract_evidence_files(row.get("answer", ""))
        out = {
            "repo": repo,
            "repo_id": spec["repo_id"],
            "repo_url": spec["url"],
            "commit": row["commit_id"],
            "split": "eval_quality_official",
            "cluster": row.get("cluster"),
            "qa_type": row.get("qa_type"),
            "question": row.get("question"),
            "gold_answer": row.get("answer"),
            "evidence_files_extracted_from_answer": evidence_files,
            "allowed_for_training": False,
            "source": "TIGER-Lab/SWE-QA-Pro-Bench test split",
        }
        by_repo_eval[repo].append(out)
        pilot_eval.append(out)

    manifest: dict[str, Any] = {
        "source_benchmark": "TIGER-Lab/SWE-QA-Pro-Bench",
        "source_repo_list": str(REPOS_TXT.relative_to(ROOT)),
        "pilot_repos": {},
    }
    all_stage1: list[dict[str, Any]] = []

    for repo, spec in PILOT_REPOS.items():
        commit = commits.get(repo)
        repo_path = REPO_ROOT / spec["local_dir"]
        source_files = list_source_files(repo_path)
        eval_files = sorted(
            {
                f
                for row in by_repo_eval[repo]
                for f in row["evidence_files_extracted_from_answer"]
                if f in source_files
            }
        )
        train_files = [f for f in source_files if f not in set(eval_files)]
        stage1_rows = make_stage1_examples(repo, spec, commit or "missing_commit", train_files)
        all_stage1.extend(stage1_rows)

        manifest["pilot_repos"][spec["repo_id"]] = {
            "repo": repo,
            "repo_url": spec["url"],
            "local_path": str(repo_path.relative_to(ROOT)),
            "commit": commit,
            "official_eval_count": len(by_repo_eval[repo]),
            "source_file_count": len(source_files),
            "eval_evidence_files_extracted_count": len(eval_files),
            "eval_evidence_files_extracted": eval_files,
            "target_sft_train_file_count": len(train_files),
            "target_sft_train_example_count": len(stage1_rows),
            "split_policy": "file_level_exclusion_for_extracted_eval_evidence_files",
        }

    distill_prompts = make_distill_prompts(all_stage1)

    write_jsonl(OUT_DIR / "pilot_eval.official_sweqapro.jsonl", pilot_eval)
    write_json(OUT_DIR / "pilot_split_manifest.json", manifest)
    write_jsonl(OUT_DIR / "target_sft_train.bootstrap.jsonl", all_stage1)
    write_jsonl(OUT_DIR / "distill_prompts.pending_teacher.jsonl", distill_prompts)

    train_files_by_repo = {}
    for row in all_stage1:
        train_files_by_repo.setdefault(row["repo_id"], set()).update(row["evidence_files"])
    leakage = {}
    for repo_id, repo_manifest in manifest["pilot_repos"].items():
        train_files = train_files_by_repo.get(repo_id, set())
        eval_files = set(repo_manifest["eval_evidence_files_extracted"])
        leakage[repo_id] = sorted(train_files.intersection(eval_files))

    report = {
        "status": "complete_with_explicit_stage2_dependency",
        "created_files": [
            "data/pilot_eval.official_sweqapro.jsonl",
            "data/pilot_split_manifest.json",
            "data/target_sft_train.bootstrap.jsonl",
            "data/distill_prompts.pending_teacher.jsonl",
        ],
        "counts": {
            "pilot_eval_examples": len(pilot_eval),
            "target_sft_train_examples": len(all_stage1),
            "distill_prompt_examples": len(distill_prompts),
        },
        "leakage_check": {
            "policy": "train evidence files must not overlap extracted official eval evidence files",
            "overlaps": leakage,
            "passed": all(len(v) == 0 for v in leakage.values()),
        },
        "limitations": [
            "Stage 1 QA examples are deterministic AST/docstring bootstrap examples for smoke-test training and loader integration; publication-quality synthetic QA still requires the team's chosen generator and filtering pass.",
            "Official SWE-QA-Pro rows do not provide structured evidence files, so evidence files are extracted from answer text and may under-approximate true evidence.",
            "Stage 2 teacher logits/hidden/cache cannot be materialized until target LoRA adapters are trained and the training owner confirms the required DFlash cache format.",
        ],
    }
    write_json(REPORT_DIR / "validation_report.json", report)

    md_lines = [
        "# RepoSpec Pilot Data Validation Report",
        "",
        f"- Official eval examples: {len(pilot_eval)}",
        f"- Target SFT bootstrap examples: {len(all_stage1)}",
        f"- Distillation prompt examples: {len(distill_prompts)}",
        f"- File-level leakage check passed: {report['leakage_check']['passed']}",
        "",
        "## Per Repo",
        "",
    ]
    for repo_id, repo_manifest in manifest["pilot_repos"].items():
        md_lines.extend(
            [
                f"### {repo_id}",
                "",
                f"- Repo: `{repo_manifest['repo']}`",
                f"- Commit: `{repo_manifest['commit']}`",
                f"- Official eval examples: {repo_manifest['official_eval_count']}",
                f"- Source files: {repo_manifest['source_file_count']}",
                f"- Extracted eval evidence files: {repo_manifest['eval_evidence_files_extracted_count']}",
                f"- Target SFT train examples: {repo_manifest['target_sft_train_example_count']}",
                f"- Leakage overlaps: {len(leakage[repo_id])}",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Limitations",
            "",
            "- The generated Stage 1 data is a deterministic bootstrap dataset, suitable for pipeline smoke tests and initial adapter-format integration.",
            "- Publication-quality QA should be regenerated with the team's chosen LLM generator, then passed through evidence, duplicate, and leakage filters.",
            "- Stage 2 teacher data is represented as prompts plus required cache schema; actual teacher logits/hidden states require the frozen target+LoRA adapters.",
            "",
        ]
    )
    (REPORT_DIR / "validation_report.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
