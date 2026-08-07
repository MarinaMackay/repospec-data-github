# Stage 2 Teacher Distillation Contract

The Stage 2 draft-training signal must come from the frozen target model with the repo-specific target LoRA loaded.

Do not train the draft against Claude-written answers or benchmark gold answers as the final objective. Gold/reference answers may be used only to construct realistic prompts.

## Input

`data/distill_prompts.pending_teacher.jsonl`

Each row contains:

- repo id
- commit
- prompt
- source question id
- target adapter id placeholder
- reference answer for prompt scaffolding only

## Required Teacher Run

For each row:

1. Load target base model.
2. Load the matching target repo LoRA adapter.
3. Run the exact serving/eval prompt template.
4. Decode with the agreed policy, usually greedy for deterministic pilot measurements.
5. Store the generated `teacher_sequence`.
6. If using KL distillation, store logits or top-k logits at draft prediction positions.
7. If using DFlash, store the target hidden-state/cache representation required by the DFlash training loader.

## Output

Materialized teacher data should conform to `schemas/teacher_materialized.schema.json`.

## Reason

Speculative decoding acceptance depends on draft-target distribution agreement. A draft model can improve gold-answer likelihood while reducing tau if it moves away from the target policy. Stage 2 therefore has to be policy-aligned to target+repo-LoRA.

