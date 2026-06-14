# Agent examples

Three areas:

## 1. Basic samples — `basic_samples/`

Core agent APIs: store, resume, middleware, artifacts, detach/abort. Most need
`GEMINI_API_KEY`.

```bash
cd py/samples/agents
uv sync
uv run python src/examples/basic_samples/01_define_agent_with_store.py
```

| File | Shows |
|------|-------|
| `01_define_agent_with_store.py` | Two `stream_bidi`, same `session_id` |
| `02_define_agent_no_store.py` | `AgentInit(state=out.state)` |
| `03_interrupt_resume_with_store.py` | ToolApproval interrupt → client approve → second POST with `resume` |
| `04_interrupt_resume_no_store.py` | ToolApproval middleware + client state |
| `05_define_prompt_agent.py` | `define_prompt_agent` |
| `06_define_custom_agent.py` | `define_custom_agent` |
| `07_artifacts_custom_patch.py` | `customPatch` / `artifact` chunks (runtime smoke test) |
| `08_graceful_failure.py` | `finish_reason=failed` |
| `09_detach.py` | `conn.detach()` |
| `10_abort.py` | `store.abort_snapshot()` after detach |
| `11_write_artifact_tool.py` | `define_agent` + `Artifacts()` middleware → `write_artifact` → `artifact` chunks |

Each example defines its agent inline.

## 2. Branching samples — `branching_samples/`

Snapshot forks for regenerate / compare / continue-on-branch. **No API key** (echo agent).

See [`branching_samples/README.md`](branching_samples/README.md) for the product map.

## 3. FastAPI / HTTP — `fastapi/`

See [`fastapi/README.md`](fastapi/README.md) and [`../main.py`](../main.py): wrap any `Agent` with `@genkit_fastapi_handler` — one POST per turn, SSE wire format for `GenkitChatTransport`.

## In-process pattern

```python
conn = await agent.stream_bidi(AgentInit(session_id=...))
await conn.send_text('...')
await conn.close()
async for chunk in conn.receive():
    ...
out = await conn.output()
```

Also: `conn.send_resume(...)`, `conn.detach()`, and `store.abort_snapshot(id)`.
