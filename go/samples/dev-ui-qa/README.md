# dev-ui-qa

Consolidated QA testapp for driving the Dev UI against the Go SDK. Registers
every action type in one process so the full surface is visible at once.

Companion to the "Dev UI x Go SDK Audit" tab of the plugin gap-analysis doc;
files here mirror its sections:

| File           | Audit sections                                              |
| -------------- | ----------------------------------------------------------- |
| `models.go`    | Model runner - config editor; Model runner - request/response |
| `flows.go`     | Flows and tools; Traces and logs                            |
| `embedders.go` | Embedder rows of Discovery and Model runner                 |
| `lifecycle.go` | Crash and lifecycle; discovery failure modes                |

## Running

```bash
genkit start -- go run .
```

Plugins initialize only when their credentials are present, so the app starts
with whatever subset you have:

| Env var                                     | Enables            |
| ------------------------------------------- | ------------------ |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)      | googleai backend   |
| `GOOGLE_CLOUD_PROJECT` (+ `GOOGLE_CLOUD_LOCATION`) | vertexai backend |
| `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`)    | anthropic        |

The missing-key Init panic (ANT-52, GGA-48) is itself an audit case; to
reproduce it, set `DEV_UI_QA_FORCE_PLUGINS=1`, which registers all three
plugins unconditionally (see Tier A below).

## Tier A repros (no API keys)

Tier A covers everything that works keyless: local registrations only. With
no credentials set, `genkit start -- go run .` logs one "skipping" line per
plugin and the Dev UI should list Flows(4), Prompts(1), Tools(1),
Retrievers(1), Models(0).

| Check | Steps | Expect |
| ----- | ----- | ------ |
| Discovery | Open the Dev UI, check the sidebar counts | Flows `smoke`, `streamingCounter`, `loggingFlow`, `panicFlow`; prompt `qa-joke`; tool `shoutTool`; retriever `staticRetriever`; no models/embedders |
| Streaming | Run `streamingCounter` with input `5` and "Stream response" checked | One chunk every 400ms, then final `streamed 5 chunks` |
| Log streaming | Run `loggingFlow` with any string input; look for its log lines in the UI, then `curl http://127.0.0.1:4033/api/traces/<traceId>/logs` | Three records at INFO/WARN/ERROR, span-correlated, on the telemetry server; whether the UI renders them is the finding |
| Trace spans | Open the trace for any run (`View trace`, or `/traces/<id>`) | Root flow span with Input, Output, and Attributes sections |
| Flow panic | Run `panicFlow` with any input | Panic message `DEV_UI_QA_PANIC_MARKER: ...` in the terminal; record what the UI shows and whether the process survived |
| Init panic | `DEV_UI_QA_FORCE_PLUGINS=1 genkit start -- go run .` with no keys | Go panic `Google AI requires setting GEMINI_API_KEY...` at Init; record what happens to the CLI and the Dev UI |

Tool and retriever runs are direct: `shoutTool` uppercases its string input;
`staticRetriever` substring-matches the query against a three-document corpus
(try `{"content":[{"text":"dev ui"}]}`). `qa-joke` renders its template
keyless but fails generation with "model is required" - also a useful row.

## Recording results

Each check produces a row in the audit tab's Findings section: surface,
action, expected, observed, verdict, linked gap ID, trace link. Tag findings
Go-side or UI-side - Dev UI defects belong to genkit-ui, a separate repo.
