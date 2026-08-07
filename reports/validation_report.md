# RepoSpec Pilot Data Validation Report

- Official eval examples: 30
- Target SFT bootstrap examples: 1877
- Distillation prompt examples: 1877
- Message-only train/dev export: 1790 / 87
- File-level leakage check passed: True

## Per Repo

### xarray

- Repo: `pydata/xarray`
- Commit: `82fd8320b1be7a434effd247baf07004b66b802f`
- Official eval examples: 10
- Source files: 186
- Extracted eval evidence files: 18
- Target SFT train examples: 171
- Leakage overlaps: 0

### numba

- Repo: `numba/numba`
- Commit: `6bb29300254a40d2be35a6c88906bd4ddaad2c4d`
- Official eval examples: 10
- Source files: 845
- Extracted eval evidence files: 21
- Target SFT train examples: 902
- Leakage overlaps: 0

### sphinx

- Repo: `sphinx-doc/sphinx`
- Commit: `47757c4062a6421feeaf0ae2ded89896d6cb3526`
- Official eval examples: 10
- Source files: 1104
- Extracted eval evidence files: 30
- Target SFT train examples: 804
- Leakage overlaps: 0

## Limitations

- The generated Stage 1 data is a deterministic bootstrap dataset, suitable for pipeline smoke tests and initial adapter-format integration.
- Publication-quality QA should be regenerated with the team's chosen LLM generator, then passed through evidence, duplicate, and leakage filters.
- Stage 2 teacher data is represented as prompts plus required cache schema; actual teacher logits/hidden states require the frozen target+LoRA adapters.
