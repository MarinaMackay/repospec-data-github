# RepoSpec Pilot Data

Data-side pilot package for the RepoSpec per-repo LoRA experiment.

The package covers three SWE-QA-Pro pilot repositories:

- `pydata/xarray` at `82fd8320b1be7a434effd247baf07004b66b802f`
- `numba/numba` at `6bb29300254a40d2be35a6c88906bd4ddaad2c4d`
- `sphinx-doc/sphinx` at `47757c4062a6421feeaf0ae2ded89896d6cb3526`

## What The Files Are

### Data

- `data/pilot_eval.official_sweqapro.jsonl`  
  Official SWE-QA-Pro eval rows for the three pilot repos. Do not train on this.

- `data/pilot_split_manifest.json`  
  Repo commits, local paths, extracted eval evidence files, source-file counts, training counts, and leakage policy.

- `data/target_sft_train.bootstrap.jsonl`  
  Stage 1 target-LoRA SFT bootstrap data. This is the data the target model trainer can consume first.

- `data/distill_prompts.pending_teacher.jsonl`  
  Stage 2 prompts for target-policy distillation. These are waiting for target+repo-LoRA teacher outputs.

- `data/teacher_sequences.dry_run.jsonl`  
  Three dry-run rows proving the teacher materialization script interface. Not real teacher data.

### Scripts

- `scripts/build_pilot_data.py` rebuilds the pilot data from local official sources.
- `scripts/validate_pilot_data.py` checks counts, schema-critical fields, loss mask metadata, and leakage.
- `scripts/filter_generated_qa.py` filters future LLM-generated QA into the target SFT schema.
- `scripts/materialize_teacher_sequences.py` materializes Stage 2 teacher sequences after target LoRA adapters exist.
- `scripts/summarize_official_sft_trajectories.py` summarizes official SWE-QA-Pro SFT trajectories.

### Schemas And Config

- `schemas/*.schema.json` defines the expected row formats.
- `config/pilot_config.json` records sources, repos, commits, splits, and training contract.

### Docs

- `DATASET_CARD.md` is the dataset card.
- `prompts/repo_qa_generation_prompt.md` is the prompt for generating publication-quality QA later.
- `prompts/teacher_distillation_contract.md` defines the Stage 2 teacher-data contract.
- `reports/*.md` and `reports/*.json` contain validation and compliance reports.

## Validation

Latest validation:

```text
VALIDATION PASSED
official_eval_rows=30
target_sft_train_rows=1877
distill_prompt_rows=1877
```

Run it again:

```bash
python3 scripts/validate_pilot_data.py
```

## Boundary

This package completes the data artifacts that can be produced before target training. Actual Stage 2 teacher logits, hidden states, or DFlash cache cannot be included until target+repo-LoRA adapters are trained and frozen.
