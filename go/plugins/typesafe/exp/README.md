# TypeSafe plugin for Genkit Go

Adds [TypeSafe AI](https://typesafe.ai)'s System One models to Genkit Go. Jev,
the first of them, does not generate text. It evaluates a state against typed
questions and returns one typed answer per question, with calibrated
probabilities, in a few hundred milliseconds. That makes it a fit for the
decisions inside an application: routing, classification, scoring, guardrails,
and verification.

> Status: in preview. The package lives under `go/plugins/typesafe/exp` and its
> APIs may change in any minor version release. Import it as `typesafex`.

## Design principle: the output type is the question set

Jev is served as a model that speaks only constrained JSON, which is the subset
of the generate API it fits exactly. The questions are the fields of the output
type, declared with `Choice`, `Score`, and `Noul`. Each field's description is
the question's instructions, and each field's JSON is the wire answer, so a
decision is one typed generate call and the answer lands in typed fields.

```go
import typesafex "github.com/firebase/genkit/go/plugins/typesafe/exp"

// The options of a choice belong to their type, with their criteria.
type Dept string

func (Dept) Criteria() map[Dept]string {
	return map[Dept]string{
		"billing":   "Payments, invoicing, refunds",
		"technical": "Bugs, outages, integrations",
		"other":     "None of the above",
	}
}

// The levels of a score belong to their rubric, lowest first.
type Anger int

func (Anger) Levels() []string { return []string{"Calm", "Concerned but civil", "Very angry"} }

// The decision: one question per field.
type Triage struct {
	Department  typesafex.Choice[Dept] `json:"department"  jsonschema_description:"Which team should handle this?"`
	IsUrgent    typesafex.Noul         `json:"is_urgent"   jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
	Frustration typesafex.Score[Anger] `json:"frustration" jsonschema_description:"How frustrated is the customer?"`
}

g := genkit.Init(ctx, genkit.WithPlugins(&typesafex.TypeSafe{})) // TYPESAFE_API_KEY

out, resp, err := genkit.GenerateData[Triage](ctx, g,
	ai.WithModel(typesafex.Model(g, "jev-1.13.0")),
	ai.WithOutputFormat(typesafex.OutputFormatDecision),
	ai.WithPrompt(ticket))
if err != nil {
	return err
}
if out.Department.Confidence < 0.6 { // the caller owns thresholds
	return escalate(ticket)
}
switch out.Department.Choice { // typed
case "billing":
	// ...
}
```

Everything else is the generate API as it already is: prompt files and the Dev
UI prompts page, model middleware such as fallback across gateways, the trace
with the question set on the request and the distributions on the response, and
the token counters.

## Question types

| Field type       | Question                  | Answer fields                                   |
| ---------------- | ------------------------- | ----------------------------------------------- |
| `Choice[T]`      | pick one of `T.Criteria()` | `Choice`, `Probabilities` per option, `Confidence` |
| `Score[L]`       | rate on `L.Levels()`       | `Score` (expected level, fractional), `Probabilities`, `Confidence`, `Legend` |
| `Noul`           | is this true?              | `Probability`; near 0.5 means "could not tell"    |
| `NoulOf[C]`      | is this true, where `C.Criteria()` says what yes and no mean | as `Noul`, which is `NoulOf` with no criteria |

The criteria of every question belong to a type, so a pair of yes and no
criteria is a type too, and it is reused wherever the question is asked:

```go
type Urgent struct{}

func (Urgent) Criteria() (yes, no string) {
	return "Names a deadline, or says now or today", "No time pressure is expressed"
}

IsUrgent typesafex.NoulOf[Urgent] `json:"is_urgent" jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
```

The built-in `enum` output format also works, with no decision type: the enum
values are the options of one choice question, the system message is the
question, and `resp.Text()` is the option.

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModel(model),
	ai.WithSystem("Which team should handle this ticket?"),
	ai.WithOutputEnums(Billing, Technical, Sales),
	ai.WithPrompt(ticket))
team := Dept(resp.Text())
```

The same call runs unchanged on a chat model. What the enum format does not
give is criteria per option or a probability per option; a `Choice` field in a
decision type gives both.

## State

The state is built from the user and model messages, with no instruction text
mixed in. A system message is never state: it goes in front of every
question's instructions, and for the enum format it is the question. That is
the place for shared context, such as what the state is and what its fields
mean. To judge a transcript that has its own system prompt, leave that prompt
out of the messages.

- One message is sent as its value: the string of a text part, or the JSON of a
  data part. `ai.WithPromptParts(ai.NewDataPart(v))` sends any value as an
  object state, which TypeSafe recommends so that a question can name a field.
- Several messages are sent as an array of `{role, content}` records, roles
  included, which is the shape TypeSafe documents for a conversation.
- With documents attached, the state is `{messages, context}`, each document as
  its text or as `{content, metadata}` when it has metadata.

A prompt template renders text, so a template that renders JSON needs
`stateJSON: true` in its config to produce an object state.

Media parts are rejected, as are tools: the model takes text and JSON only, and
never calls anything.

## Prompt files

The decision type is a registered schema, so a prompt file can name it:

```go
genkit.DefineSchemasFor(g, TicketInput{}, Triage{})
```

```yaml
---
model: typesafe/jev-1.13.0
input:
  schema: TicketInput
output:
  format: decision
  schema: Triage
---
{{ticket}}
```

## Endpoints and gateways

The same questions reach jev through TypeSafe's own API or through a gateway.
Pick the endpoint on the plugin; the model names stay `typesafe/<id>`.

| Endpoint                          | Key                     | Notes                                                     |
| --------------------------------- | ----------------------- | --------------------------------------------------------- |
| `Direct()` (default)              | `TYPESAFE_API_KEY`      | `TYPESAFE_BASE_URL` is read too; lists models              |
| `OpenRouter()`                    | `OPENROUTER_API_KEY`    | alpha Decisions API; `jev-1.13.0` is served as `typesafe/jev-1.13` |
| `Cloudflare(accountID)`           | `CLOUDFLARE_API_TOKEN`  | alias `typesafe/jev` only; a pinned version is refused     |

```go
genkit.WithPlugins(&typesafex.TypeSafe{Endpoint: typesafex.OpenRouter()})
```

A proxy that forwards the native protocol, such as LiteLLM, is the default
endpoint with `BaseURL` set to the proxy root plus its prefix. `HTTPClient` and
`Headers` are the escape hatches to the transport, and `Config.Extra` merges
fields into the request body that the plugin does not model, such as
OpenRouter's `session_id` or `trace`.

Requests that fail to connect, time out, are rate limited, or hit a server
error are retried twice, with `Retry-After` honored.

Pin a version in production. Confidence thresholds tuned against one release do
not carry over to the next, and the resolved version is on
`resp.Custom["model"]` for every call.

## The decision format on other models

`decision` is an output format, so the same decision type runs on any model,
which is useful before a TypeSafe key is at hand. The format keeps the contract
honest: on jev the questions ride on the output schema and the answers come
back calibrated; on another model the questions are rendered as instructions,
and the answer is hardened before it is parsed. Probabilities and confidence
are removed, a score is rounded to a whole level, and a noul becomes 0 or 1. A
zero confidence therefore means unknown, and code that gates on it routes to a
human rather than trusting a number the model invented.

## Limits

- No per-item questions. Score a list of passages with one call per passage.
- No nested decision types: a question is a top-level field.
- Text only, English mostly, 32k tokens of state per request.
- No streaming; the answer arrives whole.

## Tests

`go test ./plugins/typesafe/...` runs against a fake endpoint. With
`OPENROUTER_API_KEY` set, `TestOpenRouterLive` runs the decision, enum, and
history paths against jev through OpenRouter.
