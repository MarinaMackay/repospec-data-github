# Repo QA Generation Prompt

Use this prompt to generate publication-quality Stage 1 target-SFT data from an eligible training file.

## System

You generate repository-grounded QA data for target-model LoRA adaptation. You must only ask questions that can be answered from the supplied repository file context. Every answer must cite exact file paths and line ranges. Do not invent behavior that is not supported by the context.

## User Template

Repository: `{repo}`

Commit: `{commit}`

File: `{file_path}`

Allowed context:

```text
{numbered_file_excerpt}
```

Generate up to 3 question-answer pairs.

Requirements:

- The question must require repository-specific knowledge.
- The answer must be grounded only in the provided excerpt.
- Include `evidence_spans` with exact `path`, `start_line`, and `end_line`.
- Prefer questions about behavior, API contracts, configuration, edge cases, control flow, data flow, or interactions between symbols.
- Do not ask about style, obvious file names, or facts visible without reading the code.
- If the excerpt lacks enough substance, return an empty list.

Return strict JSON:

```json
{
  "examples": [
    {
      "question": "...",
      "answer": "...",
      "evidence_spans": [
        {"path": "...", "start_line": 1, "end_line": 10}
      ],
      "requires_cross_file_context": false
    }
  ]
}
```

