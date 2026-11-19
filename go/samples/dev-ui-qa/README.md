# dev-ui-qa

Consolidated QA testapp for driving the Dev UI against the Go SDK. It is a
work in progress: `flows.go` is populated, while `models.go`, `embedders.go`
and `lifecycle.go` are stubs (a TODO each) that register nothing yet.

Companion to the "Dev UI x Go SDK Audit" tab of the plugin gap-analysis doc;
files here mirror its sections:

| File           | Audit sections                                              |
| -------------- | ----------------------------------------------------------- |
| `models.go` (stub) | Model runner - config editor; Model runner - request/response |
| `flows.go`     | Flows and tools; Traces and logs                            |
| `embedders.go` (stub) | Embedder rows of Discovery and Model runner                 |
| `lifecycle.go` (stub) | Crash and lifecycle; discovery failure modes                |

## Running

```bash
genkit start -- go run .
```

Plugins initialize only when their credentials are present, so the app starts
with whatever subset you have:

| Env var                                     | Enables            |
| ------------------------------------------- | ------------------ |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)      | googleai backend   |
| `GOOGLE_CLOUD_PROJECT` + (`GOOGLE_CLOUD_LOCATION` or `GOOGLE_CLOUD_REGION`) | vertexai backend |
| `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`)    | anthropic        |

Two ports are in play: the Genkit reflection runtime that the Dev UI talks to
(starts at :3100 and probes upward; pin it with `GENKIT_REFLECTION_PORT`), and a
plain HTTP mux on `127.0.0.1:8080` that serves every flow as `POST /<flowName>`.
Use the Dev UI for the audit; the mux is only for curl.

The audit run notes use `GENKIT_ENV=dev go run .` in a separate shell with the
Dev UI started on its own (see `audit/flows-and-tools/results.md`);
`genkit start -- go run .` does both in one command.

Flows are registered per backend, so a missing key removes only the flows that
need it: anthropic gates `toolLoop`, `resumeSpend` and `interruptedSpend`;
googleai gates `providerError` and `codeExecTwoTurn`; `slashToolLoop` needs
either. `smoke`, `triageTicket`, `streamProgress` and `failingFlow` always load.

The missing-key Init panic (ANT-52, GGA-48) is itself an audit case; to
reproduce it, bypass the conditional init by constructing the plugin with no
key in the environment.

## Expected-to-fail flows

These fail on purpose (or document a known gap); a failure is the observation,
not a broken sample. Also marked in `flows.go`.

| Flow / tool | Needs | Expected outcome |
| --- | --- | --- |
| `failingFlow` | none | typed errors by mode (`invalid`, `unauthenticated`, `plain`) to see how the UI classifies them |
| `providerError` | googleai | provider rejects the nonexistent model `gemini-does-not-exist` (404) |
| `slashToolLoop` | googleai or anthropic | client-side rejection of tool name `math/add` (GGA-25) |
| `math/add` (tool) | none | runs fine standalone; only fails once attached to a model call |
| `resumeSpend` / `interruptedSpend` | anthropic | interrupt is the intended outcome with an empty decision / amount over $100 |
| `codeExecTwoTurn` | googleai | should now pass; it is a regression guard for GGA-4 |

## Audit IDs

IDs such as ANT-9, ANT-25, GGA-4 and GGA-25 come from the external plugin
gap-analysis doc (ANT = anthropic, GGA = googlegenai); this repo does not define
them. See the `audit/` results for per-finding notes.

## Recording results

Each check produces a row in the audit tab's Findings section: surface,
action, expected, observed, verdict, linked gap ID, trace link. Tag findings
Go-side or UI-side - Dev UI defects belong to genkit-ui, a separate repo.
