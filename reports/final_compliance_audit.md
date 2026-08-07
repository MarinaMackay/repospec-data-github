# Final Compliance Audit

## Requirement From Proposal / Chat

The team needs a data owner for the revised RepoSpec experiment. The relevant data requirement is not generic repo continued pretraining. It is:

1. Stage 1: target model per-repo LoRA trained with repo QA SFT data.
2. Stage 2: draft/drafter trained against the adapted target policy, not against Claude/gold answer text.
3. Eval: quality and tau measured separately, with file-level leakage control.

## What Was Completed

- Downloaded official SWE-QA-Pro repository.
- Downloaded official SWE-QA-Pro Bench dataset from Hugging Face.
- Downloaded official SWE-QA-Pro SFT trajectories from Hugging Face and verified the real 1000-row JSONL, replacing the Git LFS pointer.
- Downloaded and checked out the three pilot source repos at official SWE-QA-Pro commits:
  - `pydata/xarray` at `82fd8320b1be7a434effd247baf07004b66b802f`
  - `numba/numba` at `6bb29300254a40d2be35a6c88906bd4ddaad2c4d`
  - `sphinx-doc/sphinx` at `47757c4062a6421feeaf0ae2ded89896d6cb3526`
- Extracted the 30 official eval rows for the three repos.
- Parsed evidence-file references from official answers where available.
- Built a file-level split manifest.
- Generated Stage 1 target-SFT bootstrap data from pinned repo source.
- Exported message-only train/dev SFT files for training code.
- Generated Stage 2 distillation prompts without faking teacher outputs.
- Added JSON schemas, pilot config, dataset card, QA generation prompt, and Stage 2 distillation contract.
- Added a generated-QA filtering script for publication-quality data replacement.
- Added a teacher-sequence materialization script with dry-run validation for post-target-LoRA Stage 2 data.
- Added build and validation scripts.
- Ran validation successfully.

## Validation Result

Validation passed.

- Official eval rows: 30
- Target SFT bootstrap rows: 1877
- Message-only train/dev rows: 1790 / 87
- Distillation prompt rows: matches target SFT rows
- File-level train/eval overlap over extracted eval evidence files: 0
- Loss mask metadata: `assistant_only`
- Official SFT trajectories: 1000 rows inspected; exact pilot repo basenames found: 0.

## Does This Completely Satisfy The Team's Data Requirement?

Partially, with an important honest boundary.

Fully satisfied:

- Official benchmark data is local.
- Pilot repos are local at pinned commits.
- Eval extraction is real.
- Split manifest is real.
- Leakage control is implemented.
- Stage 1 JSONL exists and is consumable.
- Message-only train/dev exports can be handed directly to a chat SFT loader.
- Stage 2 prompt manifest exists.
- Official SFT trajectory format is downloaded and summarized.
- Future LLM-generated QA can be normalized and leakage-filtered with a provided script.
- Teacher sequence materialization can be run once target LoRA adapters exist.
- Scripts are reproducible.

Not fully satisfiable before model training:

- Actual Stage 2 teacher logits/hidden states/DFlash cache cannot be produced until target LoRA adapters are trained and frozen.
- Publication-quality synthetic QA likely needs the team's chosen LLM generator and filtering pass. The current Stage 1 data is a deterministic bootstrap set intended for pipeline integration and a first target-LoRA run.

## Final Verdict

This is a complete first-pass data engineering deliverable for taking ownership of the data side and unblocking the pilot. It is not the final publication-quality dataset or final Stage 2 teacher cache, because those require training outputs and model-format decisions from the rest of the team.
