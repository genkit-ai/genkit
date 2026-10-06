<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../docs/resources/genkit-logo-dark.png">
    <img alt="Genkit logo" src="../docs/resources/genkit-logo.png" width="400">
  </picture>
  <br>
  <strong>Genkit Go</strong>
  <br>
  <em>AI SDK for Go &bull; LLM Framework &bull; AI Agent Toolkit</em>
</p>

<p align="center">
  <a href="https://pkg.go.dev/github.com/firebase/genkit/go"><img src="https://pkg.go.dev/badge/github.com/firebase/genkit/go.svg" alt="Go Reference"></a>
  <a href="https://goreportcard.com/report/github.com/firebase/genkit/go"><img src="https://goreportcard.com/badge/github.com/firebase/genkit/go" alt="Go Report Card"></a>
</p>

<p align="center">
  Build production-ready AI-powered applications in Go with a unified interface for text generation, structured output, tool calling, and agentic workflows.
  <br><br>
  One interface over Google AI and Vertex AI (Gemini), Anthropic (Claude), OpenAI (GPT), xAI (Grok), DeepSeek, DashScope (Qwen), Moonshot (Kimi), Z.ai (GLM), Vertex AI Model Garden (Llama, Mistral), Ollama for local models, and any OpenAI-compatible endpoint.
</p>

<p align="center">
  <a href="https://genkit.dev/docs/overview/?lang=go">Documentation</a> &bull;
  <a href="https://pkg.go.dev/github.com/firebase/genkit/go">API Reference</a> &bull;
  <a href="https://discord.gg/qXt5zzQKpc">Discord</a>
</p>

---

> **Building with a coding agent? Install the Genkit Go skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-go
> ```
>
> It teaches your agent the current Genkit Go APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Installation

```bash
go get github.com/firebase/genkit/go
```

## Quick Start

Get up and running in under a minute:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
    ctx := context.Background()
    g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

    answer, err := genkit.GenerateText(ctx, g,
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithPrompt("Why is Go a great language for AI applications?"),
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(answer)
}
```

```bash
export GEMINI_API_KEY="your-api-key"
go run main.go
```

---

## Samples

Each sample runs with `go run .`. Start with the `basic-*` set: together they cover the whole framework.

| Sample | Description |
|--------|-------------|
| [basic](samples/basic/main.go) | Text generation, streaming, and flows |
| [basic‑structured](samples/basic-structured/main.go) | Typed JSON output with `GenerateData` and `GenerateDataStream` |
| [basic‑formats](samples/basic-formats/main.go) | Output formats and how each one streams |
| [basic‑decisions](samples/basic-decisions/main.go) | Typed decisions with calibrated probabilities from TypeSafe jev |
| [basic‑media](samples/basic-media) | Image input, image generation and editing, and video generation |
| [basic‑prompts](samples/basic-prompts) | Handlebars templates, `.prompt` files, partials, and helpers |
| [basic‑prompt‑content](samples/basic-prompt-content/main.go) | Prompt content built in Go from your data |
| [basic‑tools](samples/basic-tools/main.go) | A tool that streams progress and attaches a chart |
| [basic‑agents](samples/basic-agents) | Inline, prompt-file, and custom agents with snapshots, background runs, and delegation |
| [basic‑agents‑server](samples/basic-agents-server/main.go) | Store-backed and stateless agents over HTTP |
| [basic‑tool‑interrupts](samples/basic-tool-interrupts/main.go) | Human in the loop: a tool that pauses for approval |
| [basic‑middleware](samples/basic-middleware) | [Retry and fallback](samples/basic-middleware/retry-fallback/main.go), [filesystem](samples/basic-middleware/filesystem), and [skills](samples/basic-middleware/skills) middleware |
| [basic‑errors](samples/basic-errors/main.go) | Error classification with sentinels and `errors.Is` |
| [basic‑durable‑streaming‑exp](samples/basic-durable-streaming-exp/main.go) | Reconnectable streams with replay *(preview)* |

---

<details>
<summary><strong>Contents</strong> &middot; everything Genkit Go can do, at a glance</summary>

**[Agents](#agents)** *(preview)*

Multi-turn conversations with snapshots you can resume, branch, and run in the background.

[Define an Agent](#define-an-agent) &middot;
[Multi-Turn Conversations](#multi-turn-conversations) &middot;
[Snapshots](#snapshots) &middot;
[Background Agents](#background-agents) &middot;
[Delegate to Sub-Agents](#delegate-to-sub-agents) &middot;
[Load the Prompt from a File](#load-the-prompt-from-a-file) &middot;
[Custom Turn Loops](#custom-turn-loops) &middot;
[Redact on the Way Out](#redact-on-the-way-out) &middot;
[Serve Agents over HTTP](#serve-agents-over-http)

**[Features](#features)**

**Generating**
[Generate and Stream](#generate-and-stream) &middot;
[Structured Output](#structured-output) &middot;
[Output Formats](#output-formats)

**Tools**
[Define Tools](#define-tools) &middot;
[Tool Interrupts](#tool-interrupts) &middot;
[Middleware](#middleware)

**Prompts**
[Define Prompts](#define-prompts) &middot;
[Build Prompts from Your Data](#build-prompts-from-your-data) &middot;
[Prompt Files](#prompt-files)

**Flows**
[Flows](#flows) &middot;
[Traces and Logs](#traces-and-logs) &middot;
[Error Handling](#error-handling)

**[Model Providers](#model-providers)**

Gemini, Claude, GPT, Grok, DeepSeek, Qwen, Kimi, GLM, Llama, Mistral, local models, and anything OpenAI-compatible.

**[Development Tools](#development-tools)**

[Genkit CLI](#genkit-cli) &middot;
[Developer UI](#developer-ui)

</details>

---

## Agents

Agents run multi-turn conversations. Each one owns its turn loop and session state, so your code sends messages and reads results. Every turn writes a snapshot, so a conversation can resume in any process, branch from an earlier turn, recover from a failure, or keep running in the background.

> [!WARNING]
> This API is in preview and may experience breaking changes in minor releases.

Constructors live in `genkit/exp` (`genkitx`) and types and options in `ai/exp` (`aix`). Enable them with `genkit.WithExperimental()`.

[Docs](https://genkit.dev/docs/go/agents/overview/)

### Define an Agent

A prompt-backed agent with an inline prompt and a session store:

```go
// *aix.Agent[ChatState]: the State type comes from the session store.
chatAgent := genkitx.DefineAgent(g, "chat",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("You are a sarcastic pirate. Keep responses concise."),
    },
    aix.WithSessionStore(localstore.NewInMemorySessionStore[ChatState]()),
)

// *aix.AgentOutput[ChatState]
out, err := chatAgent.RunText(ctx, "What's the best way to learn Go?")
fmt.Println(out.Message.Text())
```

[Docs](https://genkit.dev/docs/go/agents/define/) &middot; [Example](samples/basic-agents/pirate.go)

### Multi-Turn Conversations

`Connect` opens a streaming session. Send a message, read chunks until `TurnEnd`, then send the next one:

```go
conn, err := chatAgent.Connect(ctx)

conn.SendText("What is Go's concurrency model?")
for chunk, err := range conn.Receive() {
    if chunk.ModelChunk != nil {
        fmt.Print(chunk.ModelChunk.Text())
    }
    if chunk.TurnEnd != nil {
        break
    }
}

conn.SendText("Show me an example with goroutines.")
// ...

out, err := conn.Output()
fmt.Println(out.Message.Text())
```

[Docs](https://genkit.dev/docs/go/agents/run/) &middot; [Example](samples/basic-agents)

### Snapshots

Every turn writes a snapshot: the conversation and its state at that point. Continue a session from its latest snapshot in any process, or branch from any earlier one:

```go
plan, err := chatAgent.RunText(ctx, "Plan a day in Kyoto.")

// Continue the session, in this process or another. Options are typed to the
// agent's State, so an option for a different State type does not compile.
next, err := chatAgent.RunText(ctx, "Add a tea ceremony.",
    aix.WithSessionID[ChatState](plan.SessionID))

// Branch from the first plan. The tea ceremony is not in this timeline.
rainy, err := chatAgent.RunText(ctx, "Assume it rains.",
    aix.WithSnapshotID[ChatState](plan.SnapshotID))
fmt.Println(rainy.Message.Text())
```

Failed and stopped runs land as snapshots too, keeping the work they finished, so you resume them instead of starting over. Background runs, sub-agent tasks, and HTTP clients all address work by snapshot ID. Stores ship for memory, files, and Firestore.

[Docs](https://genkit.dev/docs/go/agents/state/) &middot; [Stores](https://genkit.dev/docs/go/agents/session-stores/) &middot; [Failures](https://genkit.dev/docs/go/agents/errors/#resume-after-a-failure-or-a-stop) &middot; [Example](samples/basic-agents)

### Background Agents

`Detach` hands the work to the server and returns a snapshot ID right away. Check on it with `GetSnapshot`, wait with `WaitForSnapshot`, or stop it with `Abort`:

```go
conn, err := chatAgent.Connect(ctx)
conn.SendText("Draft a detailed two-week Japan itinerary.")
conn.Detach()
out, err := conn.Output()

// Later, from anywhere. snap is *aix.SessionSnapshot[ChatState].
snap, err := chatAgent.WaitForSnapshot(ctx, out.SnapshotID)
```

[Docs](https://genkit.dev/docs/go/agents/background/) &middot; [Example](samples/basic-agents)

### Delegate to Sub-Agents

The `Agents` middleware (`plugins/middleware/exp`) gives an agent one `delegate_to_<name>` tool per sub-agent:

```go
researcher := genkitx.DefineAgent(g, "researcher",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("Research the topic and summarize well-sourced findings."),
    },
    aix.WithDescription[any]("Researches a topic."),
)

orchestrator := genkitx.DefineAgent(g, "orchestrator",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("Delegate research, then write the final answer."),
        ai.WithUse(&middlewarex.Agents{Agents: []aix.AgentRef{researcher.Ref()}}),
    },
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)
```

Set `Async: true` to run sub-agents in the background, with tools to check, wait for, abort, and continue them.

[Docs](https://genkit.dev/docs/go/agents/multi-agent/) &middot; [Example](samples/basic-agents/orchestrator.go) &middot; [Async example](samples/basic-agents/commander.go)

### Load the Prompt from a File

`DefinePromptAgent` renders the `.prompt` file named after the agent, so prompt authors can tune it without touching Go:

```yaml
# prompts/chat.prompt
---
model: googleai/gemini-flash-latest
---
You are a Michelin-starred chef. Keep responses concise.
```

```go
chatAgent := genkitx.DefinePromptAgent(g, "chat",
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)
```

`aix.WithNamedPrompt` points several agents at one shared prompt, each with its own input.

[Docs](https://genkit.dev/docs/go/agents/define/#wrap-an-existing-prompt) &middot; [Example](samples/basic-agents/chef.go)

### Custom Turn Loops

`DefineCustomAgent` hands you the turn body. Session state, snapshots, and background runs still come for free:

```go
// *aix.Agent[any]: the State type comes from the SessionRunner in the signature.
chatAgent := genkitx.DefineCustomAgent(g, "chat",
    func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
        err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
            for chunk, err := range genkit.GenerateStream(ctx, g,
                ai.WithModelName("googleai/gemini-flash-latest"),
                ai.WithMessages(sess.Messages()...),
            ) {
                if err != nil {
                    return nil, err
                }
                if chunk.Done {
                    sess.AddMessages(chunk.Response.Message)
                } else {
                    resp.SendModelChunk(chunk.Chunk)
                }
            }
            return nil, nil
        })
        if err != nil {
            return nil, err
        }
        return sess.Result(), nil
    },
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)
```

With a typed `State`, `sess.UpdateCustom` changes your own state and streams the change to the client.

[Docs](https://genkit.dev/docs/go/agents/custom-orchestration/) &middot; [Example](samples/basic-agents/coder.go)

### Redact on the Way Out

`WithStateTransform` rewrites session state before it leaves the server. Stored snapshots stay raw:

```go
// The store and the transform both set State to ChatState.
// If they disagree, the code does not compile.
chatAgent := genkitx.DefineAgent(g, "chat",
    aix.InlinePrompt{ai.WithModelName("googleai/gemini-flash-latest")},
    aix.WithSessionStore(store),
    aix.WithStateTransform(func(ctx context.Context, s *aix.SessionState[ChatState]) (*aix.SessionState[ChatState], error) {
        return redactPII(ctx, s) // ctx carries the caller's identity
    }),
)
```

`WithStreamTransform` does the same for streamed chunks.

[Docs](https://genkit.dev/docs/go/agents/state/#state-and-stream-transforms)

### Serve Agents over HTTP

`AllAgentRoutes` serves every agent, one turn per request:

```go
mux := http.NewServeMux()
for _, r := range genkitx.AllAgentRoutes(g) {
    mux.HandleFunc(r.Pattern(), r.Handler())
}
// POST /agents/chat                   one turn (?stream=true for SSE)
// POST /agents/chat/getSnapshot       read a snapshot
// POST /agents/chat/waitForSnapshot   wait for a snapshot to settle
// POST /agents/chat/abort             abort background work
log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
```

[Docs](https://genkit.dev/docs/go/agents/http/) &middot; [Example](samples/basic-agents-server/main.go)

---

## Features

Genkit Go gives you everything you need to build AI applications with confidence.

### Generate and Stream

`Generate` returns the whole response. `GenerateStream` takes the same options and yields the response chunk by chunk:

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Write a short story about a robot learning to paint."),
)
fmt.Println(resp.Text())

for result, err := range genkit.GenerateStream(ctx, g, /* same options */) {
    if result.Done {
        break // result.Response holds the same *ai.ModelResponse as Generate
    }
    fmt.Print(result.Chunk.Text())
}
```

`genkit.GenerateText` returns only the string, as in the [Quick Start](#quick-start). To stream to a callback instead of a loop, pass `ai.WithStreaming(cb)` to any of them.

[Docs](https://genkit.dev/docs/go/models/#streaming) &middot; [Example](samples/basic-tools/main.go)

### Structured Output

Get type-safe JSON output that maps directly to your Go structs:

```go
type Recipe struct {
    Title       string   `json:"title"`
    Ingredients []string `json:"ingredients"`
}

// The output schema is inferred from Recipe, and the reply is decoded into a *Recipe.
recipe, resp, err := genkit.GenerateData[Recipe](ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Create a recipe for chocolate chip cookies."),
)
fmt.Println(recipe.Title)

for result, err := range genkit.GenerateDataStream[Recipe](ctx, g, /* same options */) {
    if result.Done {
        break // result.Output is the final Recipe
    }
    fmt.Println(len(result.Chunk.Ingredients)) // result.Chunk is the Recipe so far
}
```

[Docs](https://genkit.dev/docs/go/models/#structured-output) &middot; [Example](samples/basic-structured/main.go)

### Output Formats

A format decides how the model's text is parsed and what each streamed chunk holds. With `jsonl`, every item arrives once, as soon as it is complete:

```go
for result, err := range genkit.GenerateDataStream[[]Character](ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Invent four characters for a heist movie."),
    ai.WithOutputFormat(ai.OutputFormatJSONL),
) {
    for _, c := range result.Chunk { // only the characters finished since the last chunk
        fmt.Println(c.Name)
    }
}

// WithOutputEnums selects the enum format and limits the answer to these labels.
rating, resp, err := genkit.GenerateData[Rating](ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Which audience is this story for? %s", story),
    ai.WithOutputEnums(RatingAllAges, RatingYoungAdult, RatingAdult),
)
```

Built in: `json` (the default for typed output, where each chunk is the whole value so far), `jsonl` and `array` for lists, `enum`, and `text`. To add your own, implement `ai.Formatter`, which gives each request an `ai.FormatHandler` that writes the model instructions and parses the output:

```go
genkit.DefineFormats(g, csvFormatter{})

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("List three colors."),
    ai.WithOutputFormat("csv"), // the name that csvFormatter.Name returns
)
```

[Docs](https://genkit.dev/docs/go/models/#output-formats) &middot; [Custom formats](https://genkit.dev/docs/go/models/#custom-formats) &middot; [Example](samples/basic-formats/main.go)

### Define Tools

Give models the ability to take actions and access external data:

```go
type WeatherInput struct {
    Location string `json:"location"`
}

// *ai.ToolAction[WeatherInput, string].
// The model sees a JSON schema inferred from WeatherInput.
weatherTool := genkit.DefineTool(g, "getWeather",
    "Gets the current weather for a location",
    func(ctx *ai.ToolContext, input WeatherInput) (string, error) {
        return fmt.Sprintf("Weather in %s: 72°F and sunny", input.Location), nil
    },
)

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("What's the weather like in San Francisco?"),
    ai.WithTools(weatherTool),
)
fmt.Println(resp.Text())
```

Return `tool.Fail` (package `ai/tool`) to send an error back to the model so it can try again:

```go
pop, err := db.Population(ctx, input.City)
if errors.Is(err, ErrNoSuchCity) {
    return 0, tool.Fail(ctx, err) // the model tries another spelling
}
return pop, err // any other error stops the generation
```

`tool.SendChunk` streams progress while a tool runs, and `tool.AttachParts` adds media to its response.

[Docs](https://genkit.dev/docs/go/tool-calling/) &middot; [Example](samples/basic-tools/main.go)

### Tool Interrupts

Pause a tool for a person's approval, then resume with their answer:

```go
// *ai.ResumableToolAction[TransferInput, string, Approval].
// The third type is the answer that the tool resumes with.
transfer := genkit.DefineResumableTool(g, "transfer", "Transfers money.",
    func(ctx context.Context, input TransferInput, approval *Approval) (string, error) {
        if approval == nil {
            return "", tool.Interrupt(ctx, nil) // pause and ask
        }
        if !approval.Approved {
            return "Transfer cancelled.", nil
        }
        return "Transfer completed.", nil
    },
)

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Transfer $5000 to account ABC123"),
    ai.WithTools(transfer),
)

var answers []*ai.Part
for _, part := range resp.Interrupts() {
    if call, ok := transfer.Interrupted(part); ok {
        // call.Input is a TransferInput, and Restart accepts only an Approval.
        approved := askHuman(call.Input)
        answers = append(answers, call.Restart(Approval{Approved: approved}))
    }
}

resp, err = genkit.Generate(ctx, g,
    ai.WithMessages(resp.History()...),
    ai.WithTools(transfer),
    ai.WithResume(answers...),
)
```

`call.Respond` answers without running the tool again. The `ToolApproval` middleware pauses calls the same way.

[Docs](https://genkit.dev/docs/go/interrupts/) &middot; [Example](samples/basic-tool-interrupts/main.go)

### Middleware

Middleware wraps generation, model calls, and tool execution. Register the `middleware` plugin to see the built-ins in the Dev UI, then attach them per call with `ai.WithUse`:

```go
g := genkit.Init(ctx, genkit.WithPlugins(
    &googlegenai.GoogleAI{},
    &middleware.Middleware{},
))

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Explain quantum computing."),
    ai.WithUse(
        &middleware.Retry{MaxRetries: 3},
        &middleware.Fallback{Models: []ai.ModelRef{
            googlegenai.ModelRef("googleai/gemini-3.5-flash", nil),
        }},
    ),
)
```

Also built in:

- [`ToolApproval`](plugins/middleware/tool_approval.go): pauses tool calls not on an allow list until a person or a judge model decides.
- [`SoftToolErrors`](plugins/middleware/soft_tool_errors.go): sends tool errors back to the model so it can correct itself.
- [`Filesystem`](samples/basic-middleware/filesystem): gives the model file tools confined to one directory.
- [`Skills`](samples/basic-middleware/skills): loads [Agent Skills](https://agentskills.io) `SKILL.md` files on demand.

To build your own, implement [`ai.Middleware`](https://genkit.dev/docs/go/middleware/#building-your-own-custom-middleware) or wrap a function with `ai.MiddlewareFunc`.

[Docs](https://genkit.dev/docs/go/middleware/) &middot; [Example](samples/basic-middleware/retry-fallback/main.go)

### Define Prompts

Create reusable prompts with Handlebars templates and typed input and output:

```go
jokePrompt := genkit.DefineDataPrompt[JokeRequest, Joke](g, "joke",
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Tell a joke about {{topic}}."),
)

joke, resp, err := jokePrompt.Execute(ctx, JokeRequest{Topic: "cats"})
fmt.Println(joke.Punchline)
```

`ExecuteStream` streams typed chunks of the output. `genkit.DefinePrompt` is the untyped form.

[Docs](https://genkit.dev/docs/go/dotprompt/#defining-prompts-in-code) &middot; [Example](samples/basic-prompts)

### Build Prompts from Your Data

Fill any part of a prompt with a Go function instead of a template. What it returns is sent as written, so user text never needs escaping:

```go
supportPrompt := genkit.DefinePrompt(g, "support",
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithInputType(Ticket{}),
    ai.WithPromptPartsFn(func(ctx context.Context, t Ticket) ([]*ai.Part, error) {
        return []*ai.Part{
            ai.NewTextPart(t.Question),
            ai.NewMediaPart("image/png", t.Screenshot),
        }, nil
    }),
)
```

Every slot has a function form: `WithSystemFn`, `WithPromptFn`, `WithMessagesFn`, and `WithDocsFn`.

[Docs](https://genkit.dev/docs/go/dotprompt/#defining-prompts-in-code) &middot; [Example](samples/basic-prompt-content/main.go)

### Prompt Files

Keep prompts in `.prompt` files with YAML frontmatter, and embed them in your binary:

```yaml
# prompts/recipe.prompt
---
model: googleai/gemini-flash-latest
input:
  schema: RecipeRequest
output:
  format: json
  schema: Recipe
---
{{role "system"}}
You are an experienced chef.

{{role "user"}}
Create a {{cuisine}} {{dish}} recipe for {{servingSize}} people.
```

```go
//go:embed prompts
var prompts embed.FS

g := genkit.Init(ctx,
    genkit.WithPlugins(&googlegenai.GoogleAI{}),
    genkit.WithPromptFS(prompts),
)
genkit.DefineSchemasFor(g, RecipeRequest{}, Recipe{})

recipePrompt := genkit.LookupDataPrompt[RecipeRequest, *Recipe](g, "recipe")
recipe, resp, err := recipePrompt.Execute(ctx, RecipeRequest{
    Dish:        "tacos",
    Cuisine:     "Mexican",
    ServingSize: 4,
})
fmt.Println(recipe.Title)
```

Without `genkit.WithPromptFS`, prompts load from the `prompts` directory on disk. `genkit.DefinePartial` and `genkit.DefineHelper` add partials and helpers that every prompt shares.

[Docs](https://genkit.dev/docs/go/dotprompt/) &middot; [Example](samples/basic-prompts)

### Flows

A flow is a typed, traced function that you can test in the Dev UI and serve as an HTTP endpoint with one line:

```go
// *core.Flow[string, string, string].
// The input, output, and stream types come from the function.
storyFlow := genkit.DefineStreamingFlow(g, "story",
    func(ctx context.Context, topic string, send core.StreamCallback[string]) (string, error) {
        return genkit.GenerateText(ctx, g,
            ai.WithModelName("googleai/gemini-flash-latest"),
            ai.WithPrompt("Write a story about %s", topic),
            ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
                return send(ctx, chunk.Text())
            }),
        )
    },
)

mux := http.NewServeMux()
mux.HandleFunc("POST /story", genkit.Handler(storyFlow))
log.Fatal(http.ListenAndServe(":8080", mux))
```

```bash
curl -N "http://localhost:8080/story?stream=true" -d '{"data": "a robot who paints"}'
```

`genkit.Handler` returns a standard `http.HandlerFunc`, so it works with Gin, Echo, Chi, or any router. [`genkit.WithStreamManager`](https://genkit.dev/docs/go/durable-streaming/) lets clients reconnect to a stream *(preview)*.

[Docs](https://genkit.dev/docs/go/flows/) &middot; [Example](samples/basic/main.go) &middot; [Durable streaming example](samples/basic-durable-streaming-exp/main.go)

### Traces and Logs

Every flow, model call, and tool call is traced. `genkit.Run` adds your own steps, and `core/logger` attaches logs to the span that wrote them:

```go
genkit.DefineFlow(g, "summarize",
    func(ctx context.Context, doc string) (string, error) {
        logger.Info(ctx, "summarizing", "bytes", len(doc))

        // A traced step. Its output type comes from the function, so clean is a string.
        clean, err := genkit.Run(ctx, "clean", func() (string, error) {
            return strings.TrimSpace(doc), nil
        })
        return genkit.GenerateText(ctx, g,
            ai.WithModelName("googleai/gemini-flash-latest"),
            ai.WithPrompt("Summarize: %s", clean),
        )
    },
)
```

Set `GENKIT_LOG_LEVEL=debug` to see Genkit's own model and tool detail in the terminal.

[Docs](https://genkit.dev/docs/go/local-observability/) &middot; [Steps](https://genkit.dev/docs/go/flows/#flow-steps)

### Error Handling

Genkit classifies its failures with sentinels, so you branch with `errors.Is` instead of matching message text:

```go
text, err := genkit.GenerateText(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Summarize this."),
)
switch {
case errors.Is(err, ai.ErrModelNotFound):
    // The plugin providing this model isn't registered in genkit.Init.
case errors.Is(err, ai.ErrToolFailed):
    // A tool returned an error. It's wrapped, so errors.As reaches yours.
case errors.Is(err, status.ErrResourceExhausted):
    // Rate limited or out of quota: back off and retry.
}
```

A failed generation still returns its partial response, so `resp.History()` keeps the tool rounds that finished.

Classify your own errors the same way. Over HTTP, the status sets the response code, and only `PublicErrorf` messages reach the client:

```go
var ErrRecipeNotFound = status.ErrNotFound.Subtype("recipe not found") // HTTP 404

return "", status.PublicErrorf(ErrRecipeNotFound, "no recipe for %q", dish)
```

[Docs](https://genkit.dev/docs/go/error-types/) &middot; [Example](samples/basic-errors/main.go)

---

## Model Providers

Genkit provides a unified interface across all major AI providers. Use whichever model fits your needs:

| Provider | Plugin | Models |
|----------|--------|--------|
| **Google AI** | `googlegenai.GoogleAI` | Gemini 3.5 Flash, Gemini 3.1 Pro, and more |
| **Vertex AI** | `googlegenai.VertexAI` | Gemini 3.5 Flash, Gemini 3.1 Pro via Google Cloud |
| **Anthropic** | `anthropic.Anthropic` | Claude Opus 5, Claude Sonnet 5, Claude Fable 5, Claude Haiku 4.5 |
| **Vertex AI Model Garden** | `modelgarden.Anthropic`, `.Llama`, `.Mistral` | Claude, Llama, and Mistral via Google Cloud |
| **Ollama** | `ollama.Ollama` | Llama 4, Qwen 3, DeepSeek, and other local models |
| **OpenAI Compatible** | `compat_oai` | GPT-5.6, Grok, DeepSeek, Qwen, Kimi, GLM, the OpenRouter gateway, and any OpenAI-compatible API |
| **System One** *(preview)* | `systemonex.SystemOne`, `systemonex.TypeSafe()`, `systemonex.OpenRouter()` | Decision models such as TypeSafe's jev and Liquid AI's d1, with calibrated answers ([README](plugins/systemone/exp/README.md)) |

```go
// Google AI
g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

// Anthropic
g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{}))

// Ollama (local models)
g := genkit.Init(ctx, genkit.WithPlugins(&ollama.Ollama{
    ServerAddress: "http://localhost:11434",
}))

// Multiple providers at once
g := genkit.Init(ctx, genkit.WithPlugins(
    &googlegenai.GoogleAI{},
    &anthropic.Anthropic{},
))
```

Use `ai.WithModelName` for simple cases, or pair a model with provider-specific config using `ModelRef`:

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
        Temperature:     genai.Ptr(float32(0.7)),
        MaxOutputTokens: 1000,
    })),
    ai.WithPrompt("Hello!"),
)
```

[Docs](https://genkit.dev/docs/go/integrations/model-providers/)

---

## Development Tools

### Genkit CLI

Use the Genkit CLI to run your app with tracing and a local development UI:

```bash
curl -sL cli.genkit.dev | bash
genkit start -- go run main.go
```

### Developer UI

The local developer UI lets you:

- **Test flows** with different inputs interactively
- **Inspect traces** to debug complex multi-step operations
- **Compare models** by switching providers in real-time
- **Evaluate prompts** against datasets

[Docs](https://genkit.dev/docs/go/devtools/)

---

<p align="center">
  Built by Google with contributions from the <a href="https://github.com/genkit-ai/genkit/graphs/contributors">Open Source Community</a>
</p>
