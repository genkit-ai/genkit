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
reproduce it, bypass the conditional init by constructing the plugin with no
key in the environment.

## Recording results

Each check produces a row in the audit tab's Findings section: surface,
action, expected, observed, verdict, linked gap ID, trace link. Tag findings
Go-side or UI-side - Dev UI defects belong to genkit-ui, a separate repo.
