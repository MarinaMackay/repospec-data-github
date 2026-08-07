# Dataset Card: RepoSpec Pilot Data

## Summary

This package provides a pilot data bundle for the RepoSpec experiment on three SWE-QA-Pro repositories: `xarray`, `numba`, and `sphinx`.

It is designed for the revised two-stage training plan:

1. Target adaptation: train target model per-repo LoRA on repo QA SFT data.
2. Draft alignment: train the draft/drafter against the frozen target+repo-LoRA policy.

## Data Files

- `data/pilot_eval.official_sweqapro.jsonl`: official held-out SWE-QA-Pro eval rows for the three pilot repos.
- `data/pilot_split_manifest.json`: source repo commits, local paths, eval evidence files, training file counts, and leakage policy.
- `data/target_sft_train.bootstrap.jsonl`: deterministic Stage 1 bootstrap SFT data from pinned repo source.
- `data/target_sft_train.curated_v0.jsonl`: stricter symbol-level subset filtered for formal-experiment candidates.
- `data/distill_prompts.pending_teacher.jsonl`: Stage 2 prompts awaiting target+LoRA teacher materialization.
- `training/target_sft_train.messages.jsonl`: message-only Stage 1 train export.
- `training/target_sft_dev.messages.jsonl`: message-only Stage 1 dev export.
- `training/target_sft_curated_train.messages.jsonl`: message-only curated train export.
- `training/target_sft_curated_dev.messages.jsonl`: message-only curated dev export.

## Source Data

- Official benchmark: `TIGER-Lab/SWE-QA-Pro-Bench`
- Official codebase: `TIGER-AI-Lab/SWE-QA-Pro`
- Official SFT trajectories: `TIGER-Lab/SWE-QA-Pro-SFT-Trajectories`
- Pilot source repos:
  - `pydata/xarray`
  - `numba/numba`
  - `sphinx-doc/sphinx`

## Intended Use

Use `training/target_sft_train.messages.jsonl` and `training/target_sft_dev.messages.jsonl` for a first target-LoRA SFT run.

Use `training/target_sft_curated_train.messages.jsonl` and `training/target_sft_curated_dev.messages.jsonl` for a stricter first formal-experiment candidate run.

Use `data/target_sft_train.bootstrap.jsonl` if the training code wants evidence metadata.

Use `distill_prompts.pending_teacher.jsonl` to materialize Stage 2 teacher data after target LoRA adapters are trained and frozen.

Use `pilot_eval.official_sweqapro.jsonl` only for held-out evaluation.

## Not Intended Use

Do not use official eval rows for training.

Do not train the draft directly against Claude/gold answers as the final Stage 2 objective.

Do not treat `distill_prompts.pending_teacher.jsonl` as completed teacher data. It is the input to teacher materialization.

## Leakage Controls

The split manifest excludes files parsed from official eval answers. The validation script checks zero overlap between training evidence files and extracted eval evidence files.

Important limitation: SWE-QA-Pro Bench does not publish structured evidence fields, so evidence files are extracted from answer text. This is conservative where paths are explicit but may under-approximate hidden evidence.

## Current Validation

- Official eval examples: 30
- Stage 1 target SFT bootstrap examples: 1877
- Message-only train/dev export: 1790 / 87
- Curated v0 examples: 350
- Curated v0 train/dev export: 332 / 18
- Stage 2 distillation prompts: 1877
- File-level leakage overlap over extracted evidence files: 0

## Known Limitations

The Stage 1 bootstrap data is deterministic and grounded. The curated v0 subset removes the lowest-signal bootstrap questions, but it is still not a substitute for a final LLM-generated and reviewed paper dataset. For paper results, generate richer QA with the agreed LLM generator using `prompts/repo_qa_generation_prompt.md`, then filter it through `scripts/filter_generated_qa.py` and `scripts/curate_publication_qa.py`.

Actual Stage 2 teacher logits/hidden states/DFlash cache are not included because they require trained/frozen target LoRA adapters and the exact DFlash format.
