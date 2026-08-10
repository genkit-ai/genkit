# Flow Fundamentals (`basic-flows`)

Python port of [`js/testapps/flow-sample1`](../../../js/testapps/flow-sample1).
No model is used; these flows exercise the framework itself — traced steps,
streaming, context propagation, error handling (caught and uncaught), and a
long-running flow you can stare at in Dev UI to confirm spans appear live.

## Flows

| Flow | What it shows |
|------|---------------|
| `basic` | Two `ai.run()` traced steps |
| `parent` | One flow calling another |
| `withInputSchema` | Typed object input via Pydantic |
| `withContext` | Reading the request context inside a flow |
| `streamy` | Streaming `count` chunks at 1s intervals |
| `streamyThrowy` | Stream a few chunks, then raise mid-stream |
| `throwy` | Run a step, then raise from the flow body |
| `throwy2` | Raise from inside a traced step |
| `flowMultiStepCaughtError` | Catch an error from a middle step and keep going |
| `multiSteps` | Several traced steps with reused span names |
| `largeSteps` | ~1MB string outputs per step (stress the trace pipe) |
| `test-long-broadcast` | Multi-minute flow with nested spans (broadcast test) |

## Run once

```bash
uv sync
uv run src/main.py
```

This runs `basic`, `parent`, `withInputSchema`, `multiSteps`, and
`flowMultiStepCaughtError` and prints their results. The streaming, throwing,
and long-broadcast flows are skipped here — pick them from Dev UI.

## Run in Dev UI

```bash
genkit start -- uv run src/main.py
```

Open http://localhost:4000 and pick a flow from the sidebar.

## Suggested manual checks

1. **basic** — `"hello"` → returns `foo: subject: hello`. Two spans show.
2. **streamy** — `5` with streaming on → five `{count: N}` chunks at 1s
   intervals, then `done: 5, streamed: 5 times`.
3. **streamyThrowy** — `5` with streaming on → three chunks, then a
   `RuntimeError: whoops` that surfaces in the trace.
4. **throwy** / **throwy2** — `"hello"` → flow errors out; the failing span
   is highlighted in Dev UI.
5. **multiSteps** — `"world"` → returns `42`; check that the reused `step1`
   name appears twice in the trace.
6. **test-long-broadcast** — `{"steps": 5, "step_delay_ms": 5000}` → ~25s
   flow with nested fetch/process/save spans you can watch arrive live.
