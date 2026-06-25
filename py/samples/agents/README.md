# Genkit Agents Samples

Backend examples demonstrating the Genkit agents runtime: `stream_bidi`, `send_text`, `send_resume`, `detach`, `close`, `receive`, `output`, and session stores.

Requires `GEMINI_API_KEY` for the basic examples.

## Getting Started

From the sample directory:

```bash
cd py/samples/agents
uv sync
```

To run any example in Dev UI:

```bash
genkit start -- uv run basic_examples/01_define_agent_with_store.py
```

## Available Examples

### 1. Basic Examples — `basic_examples/`

These demonstrate the core agent APIs and require a `GEMINI_API_KEY`.

| File | Shows |
|------|-------|
| `01_define_agent_with_store.py` | Two `stream_bidi` calls on the same `session_id` to demonstrate history. |
| `02_define_agent_no_store.py` | Client-managed state using `AgentInit(state=out.state)`. |
| `03_interrupt_resume_with_store.py` | `ToolApproval` interrupt → client approval → resume with store. |
| `04_interrupt_resume_no_store.py` | `ToolApproval` middleware with client-managed state. |
| `05_define_prompt_agent.py` | Defining an agent using a Prompt template (`define_prompt_agent`). |
| `06_define_custom_agent.py` | Defining a custom agent (`define_custom_agent`). |
| `07_artifacts_custom_patch.py` | `customPatch` and `artifact` chunks. |
| `08_graceful_failure.py` | Handling execution errors (`finish_reason=failed`). |
| `09_detach.py` | Moving execution to the background (`conn.detach()`). |
| `10_abort.py` | Aborting a backgrounded task (`store.abort_snapshot()`). |
| `11_write_artifact_tool.py` | Using `Artifacts()` middleware to automatically write artifacts. |

Each example defines its agent inline.

### 2. Branching Examples — `branching_examples/`

These demonstrate how to fork `SessionSnapshots` to compare directions, regenerate a turn, explore options in parallel, and time-travel a build — building up to two end-to-end demos. They require a `GEMINI_API_KEY`.

See [`branching_examples/README.md`](branching_examples/README.md) for more details.
