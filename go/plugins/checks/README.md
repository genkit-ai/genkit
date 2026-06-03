# Google Checks plugin

The `checks` plugin integrates [Google Checks AI Safety](https://checks.google.com/ai-safety)
with Genkit. It provides two shapes over the Checks `aisafety:classifyContent`
v1alpha REST API:

- a **`checks/guardrails` evaluator** for batch evaluation of a dataset, and
- an exported **`Guardrails`** client (synchronous classifier) plus a ready
  **guardrail middleware** that blocks violative model input/output in-flight.

It is a Go port of the JS `@genkit-ai/checks` plugin, built directly on
`net/http` + Application Default Credentials (there is no first-party Go SDK).

## Prerequisites

- A Google Cloud project with the **Checks API** enabled and quota.
- Application Default Credentials (e.g. `gcloud auth application-default login`,
  or a service account). The plugin requests the `cloud-platform` and `checks`
  OAuth scopes.

Project ID is resolved in this order: explicit `Checks.ProjectID` >
`GOOGLE_CLOUD_PROJECT` > the ADC project.

## Policies

`DANGEROUS_CONTENT`, `PII_SOLICITING_RECITING`, `HARASSMENT`,
`SEXUALLY_EXPLICIT`, `HATE_SPEECH`, `MEDICAL_INFO`, `VIOLENCE_AND_GORE`,
`OBSCENITY_AND_PROFANITY` (constants `checks.DangerousContent`, … on `PolicyType`).

## Evaluator

```go
import (
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/plugins/checks"
)

threshold := 0.55
g := genkit.Init(ctx, genkit.WithPlugins(&checks.Checks{
    ProjectID: "your-project-id", // or GOOGLE_CLOUD_PROJECT
    Metrics: []checks.ChecksMetricConfig{
        {Type: checks.DangerousContent},
        {Type: checks.Harassment},
        {Type: checks.ViolenceAndGore, Threshold: &threshold},
    },
}))

ev := genkit.LookupEvaluator(g, "checks/guardrails")
resp, err := ev.Evaluate(ctx, &ai.EvaluatorRequest{
    Dataset: []*ai.Example{
        {TestCaseId: "1", Output: "some model output to classify"},
    },
})
```

The single `checks/guardrails` evaluator scores each datapoint's `Output`
against **all** configured policies in one API call. Each policy yields a
`Score{Id: <policyType>, Score: <0–1, if returned>, Details: {"reasoning":
"Status <VIOLATIVE|NON_VIOLATIVE>"}}`. As in the JS plugin, the score is the raw
likelihood and the violation result is surfaced as reasoning only — no Pass/Fail
status is synthesized. A score absent from the API response is omitted.

## Guardrails client & middleware

Use the exported `Guardrails` client for synchronous classification, or
`GuardrailMiddleware` to block generations whose input or output is VIOLATIVE:

```go
import "golang.org/x/oauth2/google"

creds, _ := google.FindDefaultCredentials(ctx,
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/checks")
gr := checks.NewGuardrails(creds.TokenSource, "your-project-id")

policies := []checks.ChecksMetricConfig{{Type: checks.DangerousContent}}

// Synchronous classification:
res, _ := gr.ClassifyContent(ctx, "text to classify", policies)

// As guardrail middleware (blocks VIOLATIVE input/output):
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithUse(checks.GuardrailMiddleware(gr, policies)),
    ai.WithPrompt("Write a poem about kittens."),
)
// On a violation, resp.FinishReason == ai.FinishReasonBlocked and
// resp.FinishMessage names the violated policies.
```

`ClassifyContent` retries on `429`/`503` with exponential backoff, honoring
`Retry-After`.

> **Streaming note:** the input guardrail runs before the model is called, so
> violative prompts are blocked outright. The output guardrail runs after
> generation; for **streamed** responses the model's chunks have already been
> delivered to your callback by the time the output is classified, so output
> blocking is best-effort for streaming. Use non-streaming generation when the
> output guardrail must prevent any content from reaching the client.

## Tests

Unit tests run offline against an `httptest.Server` with a fake token source:

```bash
go test ./plugins/checks/
```

A runnable sample (needs a real project + ADC) lives at
[`go/samples/checks`](../../samples/checks).
