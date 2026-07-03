# Evals

A minimal, honest starting point for checking the agent's behavior — not a full
eval suite. Each case in `dataset.json` runs through the *real* agent (the same
entry point the HTTP transport uses) and its reply is checked against loose
substring expectations.

## Run

From the sample root, with a Gemini key set:

```bash
export GEMINI_API_KEY="your_api_key_here"
uv run python evals/run_evals.py
```

Exit code is non-zero if any case fails, so this drops into CI as-is.

## How it works

- `dataset.json` — the cases. Each has a `message` and expectations:
  - `expect_substrings` — every string must appear in the reply
  - `expect_any` — at least one must appear
- `run_evals.py` — runs each case against the demo user's context (so the
  user-scoped tools return the demo catalog) and grades the reply.

## Extend it

Substring checks are deliberately simple and a bit brittle — good enough to
catch "the agent stopped calling its tools," not "the tone regressed." As your
agent gets more nuanced, grow this in place:

1. **More cases.** Add entries to `dataset.json` for each behavior you care about.
2. **LLM-as-judge.** Replace `_grade()` with a call to a model that scores the
   reply against a rubric — better for open-ended answers.
3. **Tool-call assertions.** Inspect the streamed chunks (not just the final
   text) to assert *which* tools ran.
4. **Managed evaluation.** When you outgrow a script, Google Cloud's managed
   evaluation tooling can run larger datasets and track results over time.
