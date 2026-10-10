# System One decision models for Genkit Go

Adds decision models to Genkit Go: models that speak System One, the protocol
TypeSafe AI introduced with jev and that Liquid AI's d1, Cloudflare's Clef, and
the gateways serving them also speak. A decision model does not generate text.
It evaluates a state against typed questions and returns one typed answer per
question, with calibrated probabilities, in a few hundred milliseconds. That
makes it a fit for the decisions inside an application: routing,
classification, scoring, guardrails, and verification.

> Status: in preview. The package lives under `go/plugins/systemone/exp` and its
> APIs may change in any minor version release. Import it as `systemonex`.

One plugin type, `SystemOne`, serves any server that speaks the protocol.
Constructors set it up for the known ones, and any other server takes a few
fields; see [Servers](#servers).

## Design principle: the output type is the question set

A decision model is served as a model that speaks only constrained JSON, which
is the subset of the generate API it fits exactly. The questions are the fields of the output
type, declared with `Choice`, `Score`, and `Noul`. Each field's description is
the question's instructions, and each field's JSON is the wire answer, so a
decision is one typed generate call and the answer lands in typed fields.

The question types belong to the protocol, not to a server, so one decision
type serves every model that speaks it, and a fallback from one to another
needs no second type.

```go
import systemonex "github.com/firebase/genkit/go/plugins/systemone/exp"

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
	Department  systemonex.Choice[Dept] `json:"department"  jsonschema_description:"Which team should handle this?"`
	IsUrgent    systemonex.Noul         `json:"is_urgent"   jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
	Frustration systemonex.Score[Anger] `json:"frustration" jsonschema_description:"How frustrated is the customer?"`
}

g := genkit.Init(ctx, genkit.WithPlugins(systemonex.TypeSafe())) // TYPESAFE_API_KEY

out, resp, err := genkit.GenerateData[Triage](ctx, g,
	ai.WithModelName("typesafe/jev-1.13.0"),
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

That is the whole call: the model, the state, and the type. The questions ride
on the output schema, which the model reads back, so no output format needs
naming.

Everything else is the generate API as it already is: prompt files and the Dev
UI prompts page, model middleware such as fallback across servers, the trace
with the question set on the request and the distributions on the response, and
the token counters.

## Question types

| Field type       | Question                  | Answer fields                                   |
| ---------------- | ------------------------- | ----------------------------------------------- |
| `Choice[T]`      | pick one of `T.Criteria()` | `Choice`, `Probabilities` per option, `Confidence` |
| `Score[L]`       | rate on `L.Levels()`       | `Score` (expected level, fractional), `Probabilities`, `Confidence`, `Legend` |
| `Noul`           | is this true?              | `Probability`; near 0.5 means "could not tell"    |
| `NoulOf[C]`      | is this true, where `C.Criteria()` says what yes and no mean | as `Noul`, which is `NoulOf` with no criteria |

A `Choice` also answers `Ranked()`, the options from most to least likely,
and `Margin()`, the probability gap between the top two, which is how
decisive the choice is. A `Score` answers `Level()`, the nearest whole level,
and `Label()`, that level's text from the rubric type, with or without a
legend on the answer.

The criteria of every question belong to a type, so a pair of yes and no
criteria is a type too, and it is reused wherever the question is asked:

```go
type Urgent struct{}

func (Urgent) Criteria() (yes, no string) {
	return "Names a deadline, or says now or today", "No time pressure is expressed"
}

IsUrgent systemonex.NoulOf[Urgent] `json:"is_urgent" jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
```

Criteria and levels are strings, and the schema and the answers show those
strings. The wire format also takes a structured description, most often an
object with labeled parts. A type adds one with `Guidance()`, the optional
companion of its interface: `GuidedOption` keyed by option, `GuidedRubric`
by level index, and `GuidedYesNo` by side.

```go
func (Dept) Guidance() map[Dept]any {
	return map[Dept]any{
		"billing": map[string]any{
			"not_for":  "Where an order is, or when it arrives",
			"examples": []string{"I was charged twice for one order."},
		},
	}
}
```

An object with no `what` gets the string as its `what`, so guidance adds to
the description. Any other value goes out as it is. A score's legend keeps
the rubric's strings, and the guidance the API echoes back is on
`systemonex.ResponseInfo(resp).Answers`.

The built-in `enum` output format also works, with no decision type: the enum
values are the options of one choice question, the system message is the
question, and `resp.Text()` is the option.

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModelName("typesafe/jev-1.13.0"),
	ai.WithSystem("Which team should handle this ticket?"),
	ai.WithOutputEnums(Billing, Technical, Sales),
	ai.WithPrompt(ticket))
team := Dept(resp.Text())
```

What the enum format does not give is criteria per option or a probability per
option; a `Choice` field in a decision type gives both.

## Questions built at run time

A question whose options come from data, such as the tools on hand, a
tenant's categories, or the nodes of a taxonomy, has no type to declare it
with. `systemonex.Schema` builds the output schema from values instead, and the
answers come back as a map of `systemonex.Answer`:

```go
options := make([]systemonex.ChoiceOption, 0, len(tools))
for _, tool := range tools {
	options = append(options, systemonex.ChoiceOption{Name: tool.Name, Criteria: tool.Description})
}
answers, _, err := genkit.GenerateData[map[string]systemonex.Answer](ctx, g,
	ai.WithModelName("typesafe/jev-1.13.0"),
	ai.WithOutputSchema(systemonex.Schema(map[string]systemonex.Question{
		"tool": systemonex.ChoiceQuestion{Instructions: "Which tool serves the request?", Options: options},
		"personal": systemonex.NoulQuestion{
			Instructions: "Does the request involve the user's own data?",
			Yes:          "Names the user's files, mail, or calendar",
			No:           "Asks about the world at large",
		},
	})),
	ai.WithPrompt(request))
if err != nil {
	return err
}
if a := (*answers)["tool"]; a.Confidence >= 0.8 {
	return run(a.Choice)
}
```

`ChoiceQuestion`, `ScoreQuestion`, and `NoulQuestion` take what the question
types take from their type parameters: options with criteria and guidance,
levels with guidance by index, and yes and no criteria. A choice's options go
out in the order given. Instructions can be a string or any JSON value, such
as an object that describes the field of the state a question is about; a
system message goes beside structured instructions as the first element of an
array.

## State

The state is built from the user and model messages, with no instruction text
mixed in. A system message is never state: it goes in front of every
question's instructions, and for the enum format it is the question. That is
the place for shared context, such as what the state is and what its fields
mean. A description on the output schema itself, such as one a prompt file's
schema carries, follows the system message in every question's instructions.
To judge a transcript that has its own system prompt, leave that prompt out of
the messages.

- One message is sent as its value: the string of a text part, or the JSON of a
  data part. `ai.WithPromptParts(ai.NewDataPart(v))` sends any value as an
  object state, which the vendors recommend so that a question can name a field.
- Several messages are sent as an array of `{role, content}` records, roles
  included, so a question can refer to what the user said and what the model
  said.
- With documents attached, the state is `{messages, context}`, each document as
  its content, read as a message's is, or as `{content, metadata}` when it has
  metadata.

A prompt template renders text, so a template that renders JSON needs
`stateJSON: true` in its config to produce an object state.

Tools are rejected: the model never calls anything.

### Media

Media parts become the request's media, by kind: images go in `images`, audio
in `audio`, and video in `videos`, each in the order its parts appear, messages
first and documents after. The servers place them before the state, and the
text of the same messages stays the state; a message of media alone is a turn
with empty text. A request may carry media and no text, except to Ollama. Media
goes out as a base64 data URL, or as raw base64 to Ollama; a part given as a URL
is refused rather than fetched, so download it first, for example with the
`ai.DownloadRequestMedia` middleware. Any other media type is refused.

```go
damage, _, err := genkit.GenerateData[Damage](ctx, g,
	ai.WithModelName("liquid/d1"),
	ai.WithMessages(ai.NewUserMessage(
		ai.NewTextPart("Customer's note: arrived like this."),
		ai.NewMediaPart("image/jpeg", photoDataURL))))
```

Media is an extension to the protocol, and a server that does not know a field
can drop it and answer as if nothing had been sent. A plugin therefore refuses
media before the request is sent, kind by kind, unless its `Images`, `Audio`, or
`Video` is `systemonex.MediaSupported`. `Ollama()` sets `Images`, and so does a
server of your own that takes images, such as Liquid's in
[Any other server](#any-other-server). Then a model is sent that kind, and the
server refuses it for a model that cannot read it, so a new model needs no
setup. A model's `ModelSpec` overrides the plugin's setting, kind by kind,
either way. Its zero value keeps the plugin's, so an entry that only sets a
`Label` changes nothing else. How much media a request takes, and how large, is
the server's to say.

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
  schema: Triage
---
{{ticket}}
```

## Servers

Models are named by the server they are reached through: the plugin's
`Provider`, then the ID the server uses, so a model a server adds later works
on the day it ships.

| Constructor    | Models                                  | Key                  | Notes                                         |
| -------------- | --------------------------------------- | -------------------- | --------------------------------------------- |
| `TypeSafe()`   | `typesafe/jev-1.13.0`                   | `TYPESAFE_API_KEY`   | `TYPESAFE_BASE_URL` is read too; lists models; text only |
| `OpenRouter()` | `openrouter-decisions/liquid/d1`, `openrouter-decisions/typesafe/jev-1.13`, `openrouter-decisions/~typesafe/jev-latest` | `OPENROUTER_API_KEY` | alpha Decisions API; lists OpenRouter's decision models; cost in `resp.Usage.Custom["cost"]`; text only |
| `Ollama()`     | `ollama-decisions/clef`                 | none                 | `http://localhost:11434` unless `BaseURL` is set; images as raw base64; a request needs text |

A gateway serves several vendors' models under one key, so one plugin reaches
all of them. OpenRouter's is named `openrouter-decisions` because the plugin for
its chat models already has `openrouter`, and an app often uses both; Ollama's
is `ollama-decisions` for the same reason.

A constructor returns a `*SystemOne` set up for its server, and its fields can
still be changed before `genkit.Init`, such as to pass a key from a secret
manager rather than the environment.

### TypeSafe

TypeSafe's own API serves jev, by version and as `jev-latest`, which follows
each release. Pin a version in production.

```go
ts := systemonex.TypeSafe()
ts.APIKey = key // when not in TYPESAFE_API_KEY
g := genkit.Init(ctx, genkit.WithPlugins(ts))

out, resp, err := genkit.GenerateData[Triage](ctx, g,
	ai.WithModelName("typesafe/jev-1.13.0"),
	ai.WithPrompt(ticket))
```

### OpenRouter

Models go by OpenRouter's IDs, and each response reports what the request cost.
An alias that follows the latest release starts with a tilde, as
`~typesafe/jev-latest` does; `typesafe/jev-latest` is not an ID.

```go
g := genkit.Init(ctx, genkit.WithPlugins(systemonex.OpenRouter())) // OPENROUTER_API_KEY

out, resp, err := genkit.GenerateData[Triage](ctx, g,
	ai.WithModelName("openrouter-decisions/liquid/d1"),
	ai.WithPrompt(ticket))
if err != nil {
	return err
}
cost := resp.Usage.Custom["cost"] // in OpenRouter credits
```

The API is in alpha, and its models are text only here. It takes images inside
the state, and can route them to a provider that drops them and answers anyway,
so the plugin refuses media before the request is sent.

### Ollama

A local Ollama serves the decision models it has pulled, such as Clef, with no
key, and they read images. The Dev UI lists Clef, and `Models` adds the other
models pulled.

```go
ollama := systemonex.Ollama()
ollama.BaseURL = "http://gpu-box:11434" // when not on localhost
g := genkit.Init(ctx, genkit.WithPlugins(ollama))

damage, _, err := genkit.GenerateData[Damage](ctx, g,
	ai.WithModelName("ollama-decisions/clef"),
	ai.WithMessages(ai.NewUserMessage(
		ai.NewTextPart("Customer's note: arrived like this."), // Ollama needs text beside images
		ai.NewMediaPart("image/jpeg", photoDataURL))))
```

### Any other server

Any other server that speaks the protocol takes a few fields. `Provider` names
the plugin and prefixes its models, and must differ from every other plugin's
name; `BaseURL` is the root that `Path` (default `/v1/systemone`, or `/` for
`BaseURL` itself) and `ModelsPath` hang under; `APIKey` is optional for a server
that takes none.

```go
// Liquid AI's d1, from Liquid's own API.
liquid := &systemonex.SystemOne{
	Provider:   "liquid",
	BaseURL:    "https://api.liquid.ai/decisions",
	ModelsPath: "/v1/models",
	APIKey:     os.Getenv("LIQUID_API_KEY"),
	Images:     systemonex.MediaSupported, // d1 reads images
}

g := genkit.Init(ctx, genkit.WithPlugins(liquid)) // liquid/d1
```

A proxy that forwards the native protocol, such as LiteLLM, is reached the same
way. A server whose wire is not the native one takes two hooks over the native
body, so it needs no Genkit release:

```go
wrapped := &systemonex.SystemOne{
	Provider: "wrapped",
	BaseURL:  "https://decisions.example.com",
	Path:     "/run",
	// The request: the native body under input, the model in the path.
	Route: func(model string, body map[string]any) (string, any, error) {
		delete(body, "model")
		return "/" + model, map[string]any{"input": body}, nil
	},
	// The response: the native body out of the server's envelope.
	Unwrap: func(body []byte) ([]byte, error) {
		var envelope struct{ Result json.RawMessage }
		err := json.Unmarshal(body, &envelope)
		return envelope.Result, err
	},
}
```

### Every server

`Models` describes the models known ahead, keyed by the server's ID or by the
full model name; they are listed in the Dev UI with whatever the server's
listing adds, which is kept for five minutes. The fields are read once, when
Genkit initializes the plugin. `HTTPClient` and `Headers` are the escape hatches
to the transport, and `Config.Extra` merges fields into the request body that
the plugin does not model, such as a gateway's `session_id` or `trace`; it
cannot replace a field the request builds itself.

Requests that fail to connect, time out, are rate limited, or hit a server
error are retried twice, with `Retry-After` honored.

Pin a version in production where the server offers one. Confidence thresholds
tuned against one release do not carry over to the next, nor from one model to
another, and `systemonex.ResponseInfo(resp).Model` is the version that answered,
on every call. A gateway's cost is `resp.Usage.Custom["cost"]`. Where a
reference with a config is needed, such as a fallback list, the plugin's
`ModelRef("jev-1.13.0", &cfg)` builds one. A fallback takes each model's own
config, so give every reference the `StateJSON` it needs, or send the state as a
data part.

## Limits

- No per-item questions. Score a list of passages with one call per passage.
- No nested decision types: a question is a top-level field.
- The instructions of a field in a decision type are a string, since a
  field's description is a tag. Structured instructions need a runtime
  question.
- Each server sets its own limits on questions and tokens; jev takes up to 64k
  tokens of state and questions together, and up to 32k of state and its
  longest question.
- A server can cut a long state to fit its limit and answer from what is left,
  with no error. Check a long document against the model's limit, or split it.
- No streaming; the answer arrives whole.

## Tests

`go test ./plugins/systemone/... ./plugins/internal/systemone/...` runs against
a fake endpoint. With `OPENROUTER_API_KEY` set, `TestOpenRouterLive` runs the
decision, guidance, enum, history, runtime-question, document, listing, and
model-version paths against jev and d1 through OpenRouter.
