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
</p>

<p align="center">
  <a href="https://genkit.dev/docs/overview/?lang=go">Documentation</a> &bull;
  <a href="https://pkg.go.dev/github.com/firebase/genkit/go">API Reference</a> &bull;
  <a href="https://discord.gg/qXt5zzQKpc">Discord</a>
</p>

---

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
        fmt.Println("could not generate: %s", err)
    }
    fmt.Println(answer)
}
```

```bash
export GEMINI_API_KEY="your-api-key"
go run main.go
```

---

## Agents

Agents are Genkit's primitive for multi-turn, stateful conversations. An agent owns the per-turn loop (render the prompt, append history, call the model, stream the reply) and the conversation's session state, so your code sends messages and reads results instead of re-threading history on every call.

Beyond a plain chat loop, agents give you:

- **Managed session state** that persists across turns, with typed custom state of your own.
- **Snapshots** written at the end of every successful turn, so a conversation can be resumed later by session or snapshot ID.
- **Background execution** via `Detach`: hand a long-running turn to the server, walk away, and poll, resume, or abort it later.
- **One definition, many transports**: the same agent runs in-process (`RunText`, `Connect`) or over HTTP, one turn per request.

> [!WARNING]
> This API is in preview and may experience breaking changes in minor releases.

The constructors (`DefineAgent`, `DefinePromptAgent`, `DefineCustomAgent`) live in `github.com/firebase/genkit/go/genkit/exp` (aliased `genkitx` below); the agent types and options live in `github.com/firebase/genkit/go/ai/exp` (aliased `aix`). Initialize Genkit with `genkit.WithExperimental()` to enable the `genkit/exp` surface.

### Define an Agent

The shortest path is a prompt-backed agent with an inline prompt and a session store. `aix.InlinePrompt` declares the prompt right next to the agent; the store persists each turn so the conversation can resume later:

```go
import (
    "github.com/firebase/genkit/go/ai"
    aix "github.com/firebase/genkit/go/ai/exp"
    "github.com/firebase/genkit/go/ai/exp/localstore"
    "github.com/firebase/genkit/go/genkit"
    genkitx "github.com/firebase/genkit/go/genkit/exp"
)

chatAgent := genkitx.DefineAgent(g, "chat",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("You are a sarcastic pirate. Keep responses concise."),
    },
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)

// Single turn: RunText drives the whole connection lifecycle for you.
out, _ := chatAgent.RunText(ctx, "What's the best way to learn Go?")
fmt.Println(out.Message.Text())
```

The `State` type parameter is inferred from the typed options (`aix.WithSessionStore`, `aix.WithStateTransform`), so the explicit `genkitx.DefineAgent[State]` is only needed when no typed option is supplied.

[See full example](samples/basic-agents)

### Multi-Turn Conversations

`Connect` opens a streaming session you drive turn by turn: send a message, iterate chunks until `TurnEnd`, then send the next one. The agent carries the history between turns. `Output` ends the conversation and returns the final result:

```go
conn, _ := chatAgent.Connect(ctx)

conn.SendText("What is Go's concurrency model?")
for chunk, err := range conn.Receive() {
    if err != nil {
        log.Fatal(err)
    }
    if chunk.ModelChunk != nil {
        fmt.Print(chunk.ModelChunk.Text()) // stream tokens as they arrive
    }
    if chunk.TurnEnd != nil {
        break // turn complete, ready for the next input
    }
}

conn.SendText("Show me an example with goroutines.")
// ... iterate conn.Receive() again ...

out, _ := conn.Output() // closes input, drains, returns the final AgentOutput
fmt.Println(out.Message.Text())
```

[See full example](samples/basic-agents)

### Load the Prompt from a File

`genkitx.DefinePromptAgent` backs the agent with a prompt from the registry instead of an inline one. By default it uses the prompt registered under the agent's own name, including one loaded from a `.prompt` file, so prompt authors can tune the model, config, template, and default input without touching the Go wiring:

```yaml
# prompts/chat.prompt
---
model: googleai/gemini-flash-latest
input:
  schema: ChatInput
  default:
    personality: a Michelin-starred chef
---
{{role "system"}}
You are {{personality}}. Keep responses concise.
```

```go
type ChatInput struct {
    Personality string `json:"personality"`
}

// Register the schema so the .prompt file can reference it by name.
genkit.DefineSchemaFor[ChatInput](g)

// Agent "chat" renders ./prompts/chat.prompt every turn (no source option needed).
chatAgent := genkitx.DefinePromptAgent(g, "chat",
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)
```

To back several agents with one shared prompt, point each at it with `aix.WithNamedPrompt` and give each its own input. The prompt name need not match the agent name:

```go
for _, p := range []struct{ name, persona string }{
    {"pirate", "a sarcastic pirate"},
    {"chef", "a Michelin-starred chef"},
} {
    genkitx.DefinePromptAgent(g, p.name,
        aix.WithNamedPrompt[any]("chat", ChatInput{Personality: p.persona}),
        aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
    )
}
```

[See full example](samples/basic-agents)

### Custom Turn Loops

When the prompt-backed loop isn't enough (custom models per turn, pre/post processing, bespoke tool plumbing), `DefineCustomAgent` hands you the turn body. You still get managed session state, snapshots, and the detach lifecycle for free. A typed `State` parameter carries structured state across turns, and mutating it with `UpdateCustom` streams the delta to the client automatically:

```go
type ChatState struct {
    TopicsDiscussed []string `json:"topicsDiscussed"`
}

chatAgent := genkitx.DefineCustomAgent(g, "chat",
    func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[ChatState]) (*aix.AgentResult, error) {
        err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
            for chunk, err := range genkit.GenerateStream(ctx, g,
                ai.WithModelName("googleai/gemini-flash-latest"),
                ai.WithMessages(sess.Messages()...), // the history is yours to manage
            ) {
                if err != nil {
                    return nil, err
                }
                if chunk.Done {
                    sess.AddMessages(chunk.Response.Message)
                    if input.Message != nil {
                        sess.UpdateCustom(func(s ChatState) ChatState {
                            s.TopicsDiscussed = append(s.TopicsDiscussed, input.Message.Text())
                            return s
                        })
                    }
                    // Report how the turn ended so the framework can forward it
                    // on the TurnEnd chunk and persist it on the snapshot.
                    return &aix.TurnResult{
                        FinishReason: aix.AgentFinishReason(chunk.Response.FinishReason),
                    }, nil
                }
                resp.SendModelChunk(chunk.Chunk) // stream tokens to the client
            }
            return nil, nil
        })
        if err != nil {
            return nil, err
        }
        return sess.Result(), nil
    },
    aix.WithSessionStore(localstore.NewInMemorySessionStore[ChatState]()),
)
```

[See full example](samples/basic-agents)

### Persist and Resume

With a session store configured, every successful turn writes a snapshot. The caller only needs the `SessionID` from a previous result to pick the conversation back up:

```go
first, _ := chatAgent.RunText(ctx, "My name is Alex and I'm planning a trip to Japan.")

// Later, in another request or process: resume from the latest snapshot.
second, _ := chatAgent.RunText(ctx, "What is my name?",
    aix.WithSessionID[any](first.SessionID))
fmt.Println(second.Message.Text()) // "Your name is Alex."
```

Resume from one specific point in history with `aix.WithSnapshotID`, or skip the server store entirely and round-trip the state yourself with `aix.WithState` (the conversation's identity travels inside the state object).

[See full example](samples/basic-agents)

### Redact on the Way Out

`WithStateTransform` rewrites session state as it leaves the server, on `GetSnapshot` reads, on a client-managed `out.State`, and on the streamed `CustomPatch` diffs, while the persisted snapshot and the state your agent function sees stay raw:

```go
chatAgent := genkitx.DefineAgent(g, "chat",
    aix.InlinePrompt{ai.WithModelName("googleai/gemini-flash-latest")},
    aix.WithSessionStore(store),
    aix.WithStateTransform(func(ctx context.Context, s *aix.SessionState[ChatState]) (*aix.SessionState[ChatState], error) {
        return redactPII(ctx, s) // ctx carries caller identity for RBAC-aware redaction
    }),
)
```

`WithStreamTransform[State]` is the stream-side counterpart, rewriting each `AgentStreamChunk` (model tokens, artifacts, custom patches, turn-end) on its way to the client. It takes `State` as an explicit type argument because a chunk carries no state type to infer it from, unlike `WithStateTransform`, whose `State` is derived from the transform's signature. Both transforms own a fresh deep copy: mutate it in place, return a new value, or return `nil` to omit that state (or drop that chunk) from the client's view. A non-nil error fails closed, so the read or invocation fails with the transform's status (e.g. `PERMISSION_DENIED`) instead of leaking unredacted data.

### Background Agents

`Detach` hands the rest of the work to the server and closes the connection promptly with a pending snapshot ID. The agent keeps processing in the background on a context decoupled from the client's, so a long task survives the caller walking away:

```go
conn, _ := chatAgent.Connect(ctx)
conn.SendText("Draft a detailed two-week Japan itinerary.")
conn.Detach() // server takes ownership of the remaining work

out, _ := conn.Output() // returns immediately; FinishReason is "detached"
snapshotID := out.SnapshotID

// Later: poll the snapshot, then resume once it has finalized.
snap, _ := chatAgent.GetSnapshot(ctx, snapshotID)
switch snap.Status {
case aix.SnapshotStatusPending:   // still working
case aix.SnapshotStatusCompleted: // snap.State holds the final state; resume it
case aix.SnapshotStatusFailed:    // snap.Error holds the structured failure
}

// Or stop it early; the runtime observes the abort and cancels the work.
chatAgent.Abort(ctx, snapshotID)
```

Detach requires a store that implements `SnapshotSubscriber` (both bundled local stores do). A detached turn refreshes a heartbeat while it runs, so a crashed worker surfaces as `expired` instead of orphaning the conversation forever.

[See full example](samples/basic-agents)

### Delegate to Sub-Agents

> [!WARNING]
> This API is in preview and may experience breaking changes in minor releases.

The experimental `Agents` middleware (in `plugins/middleware/exp`) lets one agent delegate to others. It injects one `delegate_to_<name>` tool per sub-agent and a `<sub-agents>` listing into the orchestrator's system prompt, then runs the chosen sub-agent and returns its result when the model calls the tool. Each sub-agent's `aix.WithDescription` (captured by `agent.Ref()`) tells the orchestrator when to reach for it:

```go
import (
    "github.com/firebase/genkit/go/ai"
    aix "github.com/firebase/genkit/go/ai/exp"
    "github.com/firebase/genkit/go/ai/exp/localstore"
    genkitx "github.com/firebase/genkit/go/genkit/exp"
    middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
)

researcher := genkitx.DefineAgent(g, "researcher",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("You are a thorough research assistant. Summarize well-sourced findings."),
    },
    aix.WithDescription[any]("Researches a topic and summarizes well-sourced findings."),
)

// The orchestrator delegates instead of answering directly: the model calls
// delegate_to_researcher and the middleware runs the sub-agent.
orchestrator := genkitx.DefineAgent(g, "orchestrator",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("You are a project coordinator. Delegate research to the " +
            "researcher sub-agent, then synthesize a final answer."),
        ai.WithUse(&middlewarex.Agents{
            Agents:         []aix.AgentRef{researcher.Ref()},
            MaxDelegations: 5, // cap delegation tool calls per turn (0 = unlimited)
            HistoryLength:  4, // recent messages forwarded to client-managed sub-agents
        }),
    },
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)

out, _ := orchestrator.RunText(ctx, "Research goroutine scheduling and summarize the key ideas.")
fmt.Println(out.Message.Text())
```

Sub-agents are named by `aix.AgentRef`, either captured from an agent value with `agent.Ref()` or written by hand (`aix.AgentRef{Name: "researcher"}`). The middleware composes with the `Artifacts` middleware: give a sub-agent `&middlewarex.Artifacts{}` so it can save output, set `ArtifactStrategy: middlewarex.ArtifactStrategySession` to merge those artifacts into the orchestrator's session instead of inlining them in the tool result, and add `&middlewarex.Artifacts{Readonly: true}` on the orchestrator so it can review them before answering.

[See full example](samples/basic-agents)

### Serve Agents over HTTP

An `Agent` is an `api.BidiAction`, so it serves over HTTP one turn per request. The `genkit/exp` package lays out a default route surface for every registered agent, including the snapshot companion endpoints for store-backed agents:

```go
import (
    genkitx "github.com/firebase/genkit/go/genkit/exp"
    "github.com/firebase/genkit/go/plugins/server"
)

mux := http.NewServeMux()
for _, r := range genkitx.AllAgentRoutes(g) {
    mux.HandleFunc(r.Pattern(), r.Handler())
}
// POST /agents/chat                one turn per request (?stream=true for SSE)
// POST /agents/chat/getSnapshot    read a snapshot by ID
// POST /agents/chat/abort          abort background work
log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
```

A client starts a conversation by POSTing a turn, then continues it by sending the returned `sessionId` in the request's `init` field. Agents with no store return the full state instead and the client round-trips it, so stateless and store-backed agents deploy side by side.

[See full example](samples/basic-agents-server)

---

## Features

Genkit Go gives you everything you need to build AI applications with confidence.

### Generate Text

Call any model with a simple, unified API:

```go
text, _ := genkit.GenerateText(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Explain quantum computing in simple terms."),
)
fmt.Println(text)
```

### Generate Structured Data

Get type-safe JSON output that maps directly to your Go structs:

```go
type Recipe struct {
    Title       string   `json:"title"`
    Ingredients []string `json:"ingredients"`
    Steps       []string `json:"steps"`
}

recipe, _ := genkit.GenerateData[Recipe](ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Create a recipe for chocolate chip cookies."),
)
fmt.Printf("Recipe: %s\n", recipe.Title)
```

[See full example](samples/basic-structured)

### Stream Responses

Stream text as it's generated for responsive user experiences:

```go
stream := genkit.GenerateStream(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Write a short story about a robot learning to paint."),
)

for result, err := range stream {
    if err != nil {
        log.Fatal(err)
    }
    if result.Done {
        break
    }
    fmt.Print(result.Chunk.Text())
}
```

[See full example](samples/basic-structured)

### Stream Structured Data

Stream typed JSON objects as they're being generated:

```go
type Ingredient struct {
    Name   string `json:"name"`
    Amount string `json:"amount"`
}

type Recipe struct {
    Title       string        `json:"title"`
    Ingredients []*Ingredient `json:"ingredients"`
}

stream := genkit.GenerateDataStream[*Recipe](ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Create a recipe for spaghetti carbonara."),
)

for result, err := range stream {
    if err != nil {
        log.Fatal(err)
    }
    if result.Done {
        fmt.Printf("\nComplete recipe: %s\n", result.Output.Title)
        break
    }
    // Access partial data as it streams in
    if result.Chunk != nil && len(result.Chunk.Ingredients) > 0 {
        fmt.Printf("Found ingredient: %s\n", result.Chunk.Ingredients[0].Name)
    }
}
```

[See full example](samples/basic-structured)

### Define Tools

Give models the ability to take actions and access external data:

```go
type WeatherInput struct {
    Location string `json:"location"`
}

weatherTool := genkit.DefineTool(g, "getWeather",
    "Gets the current weather for a location",
    func(ctx *ai.ToolContext, input WeatherInput) (string, error) {
        // Call your weather API here
        return fmt.Sprintf("Weather in %s: 72°F and sunny", input.Location), nil
    },
)

response, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("What's the weather like in San Francisco?"),
    ai.WithTools(weatherTool),
)
fmt.Println(response.Text())
```

[See full example](samples/basic)

### Tool Interrupts

Pause execution for human approval, then resume with modified inputs or direct responses:

```go
type TransferInput struct {
    ToAccount string  `json:"toAccount"`
    Amount    float64 `json:"amount"`
}

type TransferInterrupt struct {
    Reason  string  `json:"reason"`
    Amount  float64 `json:"amount"`
    Balance float64 `json:"balance"`
}

transferTool := genkit.DefineTool(g, "transfer",
    "Transfer money to an account",
    func(ctx *ai.ToolContext, input TransferInput) (string, error) {
        // Confirm large transfers
        if !ctx.IsResumed() && input.Amount > 1000 {
            return "", ai.InterruptWith(ctx, TransferInterrupt{
                Reason:  "confirm_large",
                Amount:  input.Amount,
                Balance: currentBalance,
            })
        }
        return "Transfer completed", nil
    },
)

// Handle interrupts in your flow
resp, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Transfer $5000 to account ABC123"),
    ai.WithTools(transferTool),
)

if resp.FinishReason == ai.FinishReasonInterrupted {
    for _, interrupt := range resp.Interrupts() {
        meta, _ := ai.InterruptAs[TransferInterrupt](interrupt)

        // Get user confirmation, then resume
        part, _ := transferTool.RestartWith(interrupt)
        resp, _ = genkit.Generate(ctx, g,
            ai.WithMessages(resp.History()...),
            ai.WithTools(transferTool),
            ai.WithToolRestarts(part),
        )
    }
}
```

[See full example](samples/intermediate-interrupts)

### Streaming, Multipart, and Interruptible Tools

> [!WARNING]
> This API is in preview and may experience breaking changes in minor releases.

The experimental tool constructors in `genkit/exp` (aliased `genkitx`) hand your function a plain `context.Context` instead of `ai.ToolContext`, with helpers in `ai/exp/tool` for streaming progress, attaching media, and typed interrupts. This is a preview of Genkit Go's next-generation tools API: it is slated to replace the current `genkit.DefineTool` (shown above) as the default in the next major version. Initialize Genkit with `genkit.WithExperimental()` to enable them.

`genkitx.DefineTool` infers its input and output types from the function. Inside the tool, `tool.SendPartial` streams partial results mid-execution and `tool.AttachParts` adds extra content parts to the response, neither of which changes the function signature:

```go
import (
    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/ai/exp/tool"
    genkitx "github.com/firebase/genkit/go/genkit/exp"
)

type AnalyzeInput struct {
    Symbol string `json:"symbol"`
}

analyzeTool := genkitx.DefineTool(g, "analyzeStock",
    "Analyzes a stock and returns a summary with a chart.",
    func(ctx context.Context, input AnalyzeInput) (string, error) {
        // Stream progress to the client while the tool runs. It is a no-op when
        // the caller isn't streaming; the return value is always authoritative.
        tool.SendPartial(ctx, map[string]any{"status": "fetching prices", "progress": 50})

        // Attach media to the tool's response without a multipart signature.
        tool.AttachParts(ctx, ai.NewMediaPart("image/png", chartDataURI))

        return fmt.Sprintf("%s closed up 4%% this week.", input.Symbol), nil
    },
)
```

`genkitx.DefineInterruptibleTool` adds a typed resume parameter: it is `nil` on the first call and carries the caller's decision when the tool resumes. Reusing the `TransferInput`/`TransferInterrupt` types from above, the tool pauses with `tool.Interrupt` and the caller resumes it with typed data via the tool's `Resume`:

```go
type Confirmation struct {
    Approved bool `json:"approved"`
}

// The third parameter (*Confirmation) is the resume payload: nil on the first
// call, populated when the caller resumes after an interrupt.
transferTool := genkitx.DefineInterruptibleTool(g, "transfer",
    "Transfers money to another account.",
    func(ctx context.Context, input TransferInput, confirm *Confirmation) (string, error) {
        if confirm == nil && input.Amount > 1000 {
            // Pause and hand typed data to the caller.
            return "", tool.Interrupt(TransferInterrupt{Reason: "confirm_large", Amount: input.Amount})
        }
        if confirm != nil && !confirm.Approved {
            return "Transfer cancelled.", nil
        }
        return "Transfer completed.", nil
    },
)

resp, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Transfer $5000 to account ABC123"),
    ai.WithTools(transferTool),
)

// Interrupts() yields nothing unless the tool paused for input.
var restarts []*ai.Part
for _, interrupt := range resp.Interrupts() {
    meta, _ := tool.InterruptAs[TransferInterrupt](interrupt)

    // Use meta to ask the user for a decision, then resume with their answer.
    // The typed data arrives as the tool's *Confirmation parameter.
    restart, _ := transferTool.Resume(interrupt, Confirmation{Approved: true})
    restarts = append(restarts, restart)
}
if len(restarts) > 0 {
    resp, _ = genkit.Generate(ctx, g,
        ai.WithMessages(resp.History()...),
        ai.WithTools(transferTool),
        ai.WithToolRestarts(restarts...),
    )
}
```

[See the banker example](samples/basic-agents) for an interruptible tool wired into an agent.

### Middleware

Middleware wraps generation, model calls, and tool execution to add cross-cutting behavior without touching your flows. Register the `middleware` plugin during `Init` to expose the built-ins in the Dev UI, then attach them per call with `ai.WithUse`:

```go
import "github.com/firebase/genkit/go/plugins/middleware"

g := genkit.Init(ctx, genkit.WithPlugins(
    &googlegenai.GoogleAI{},
    &middleware.Middleware{},
))

// Retry transient failures, then fall back to a secondary model if the primary
// stays down. Middleware composes outer-to-inner: Retry { Fallback { model } }.
response, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Explain quantum computing."),
    ai.WithUse(
        &middleware.Retry{MaxRetries: 3},
        &middleware.Fallback{Models: []ai.ModelRef{
            googlegenai.ModelRef("googleai/gemini-2.5-flash", nil),
        }},
    ),
)
```

The `middleware` plugin also ships with:

- [`ToolApproval`](plugins/middleware/tool_approval.go) — interrupts any tool not on an allow list and resumes once the call is explicitly approved on restart.
- [`Filesystem`](samples/basic-middleware/filesystem) — gives the model `list_files` and `read_file` tools (plus `write_file` and `edit_file` when `AllowWriteAccess` is set), all confined to a single `RootDir` via `os.Root` (Go 1.25+) so paths cannot escape via `..`, absolute paths, or symlinks.
- [`Skills`](samples/basic-middleware/skills) — exposes a library of `SKILL.md` files through a `use_skill` tool so the model can pull in specialised instructions on demand.

[See the retry + fallback sample](samples/basic-middleware/retry-fallback) for a full composition.

### Custom Middleware

Implement the `ai.Middleware` interface — `Name()` plus `New(ctx)` — to build your own. `New` returns a `Hooks` bundle whose four fields (`Tools`, `WrapGenerate`, `WrapModel`, `WrapTool`) are all optional; nil hooks pass through:

```go
type Logger struct {
    Prefix string `json:"prefix,omitempty"`
}

func (l *Logger) Name() string { return "mine/logger" }

func (l *Logger) New(ctx context.Context) (*ai.Hooks, error) {
    return &ai.Hooks{
        WrapModel: func(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
            start := time.Now()
            resp, err := next(ctx, params)
            log.Printf("%s model call took %s", l.Prefix, time.Since(start))
            return resp, err
        },
    }, nil
}

// Use it like any built-in middleware.
ai.WithUse(&Logger{Prefix: "[trace]"})
```

`Name()` must be unique and stable since it's the key used to register the middleware and reference it from the Dev UI and across runtimes. `New()` is called once per `Generate` invocation, so per-call state (counters, caches, message queues) can be allocated inside it and closed over by the hooks — just guard anything mutable, since `WrapTool` may run concurrently when tools execute in parallel. For ad-hoc, inline middleware that doesn't need to surface in the Dev UI, wrap a factory closure with `ai.MiddlewareFunc`.

### Define Flows

Wrap your AI logic in flows for better observability, testing, and deployment:

```go
jokeFlow := genkit.DefineFlow(g, "tellJoke",
    func(ctx context.Context, topic string) (string, error) {
        return genkit.GenerateText(ctx, g,
            ai.WithModelName("googleai/gemini-flash-latest"),
            ai.WithPrompt("Tell me a joke about %s", topic),
        )
    },
)

joke, _ := jokeFlow.Run(ctx, "programming")
fmt.Println(joke)
```

[See full example](samples/basic)

### Streaming Flows

Stream data from your flows using Server-Sent Events (SSE):

```go
genkit.DefineStreamingFlow(g, "streamStory",
    func(ctx context.Context, topic string, send core.StreamCallback[string]) (string, error) {
        stream := genkit.GenerateStream(ctx, g,
            ai.WithModelName("googleai/gemini-flash-latest"),
            ai.WithPrompt("Write a story about %s", topic),
        )

        for result, err := range stream {
            if err != nil {
                return "", err
            }
            if result.Done {
                return result.Response.Text(), nil
            }
            send(ctx, result.Chunk.Text())
        }
        return "", nil
    },
)
```

[See full example](samples/basic)

### Traced Sub-steps

Add observability to complex flows by breaking them into traced operations:

```go
genkit.DefineFlow(g, "processDocument",
    func(ctx context.Context, doc string) (string, error) {
        // Each Run call creates a traced step visible in the Dev UI
        summary, _ := genkit.Run(ctx, "summarize", func() (string, error) {
            return genkit.GenerateText(ctx, g,
                ai.WithModelName("googleai/gemini-flash-latest"),
                ai.WithPrompt("Summarize: %s", doc),
            )
        })

        keywords, _ := genkit.Run(ctx, "extractKeywords", func() ([]string, error) {
            return genkit.GenerateData[[]string](ctx, g,
                ai.WithModelName("googleai/gemini-flash-latest"),
                ai.WithPrompt("Extract keywords from: %s", summary),
            )
        })

        return fmt.Sprintf("Summary: %s\nKeywords: %v", summary, keywords), nil
    },
)
```

[See full example](samples/basic)

### Define Prompts

Create reusable prompts with Handlebars templating:

```go
greetingPrompt := genkit.DefinePrompt(g, "greeting",
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Write a {{style}} greeting for {{name}}."),
)

response, _ := greetingPrompt.Execute(ctx, ai.WithInput(map[string]any{
    "name":  "Alice",
    "style": "formal",
}))
fmt.Println(response.Text())
```

[See full example](samples/basic-prompts)

### Type-Safe Data Prompts

Get compile-time type safety for your prompt inputs and outputs:

```go
type JokeRequest struct {
    Topic string `json:"topic"`
}

type Joke struct {
    Setup     string `json:"setup"`
    Punchline string `json:"punchline"`
}

jokePrompt := genkit.DefineDataPrompt[JokeRequest, *Joke](g, "joke",
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Tell a joke about {{topic}}."),
)

for result, err := range jokePrompt.ExecuteStream(ctx, JokeRequest{Topic: "cats"}) {
    if err != nil {
        log.Fatal(err)
    }
    if result.Done {
        fmt.Printf("Punchline: %s\n", result.Output.Punchline)
        break
    }
    // Access typed partial data as it streams
    if result.Chunk != nil && result.Chunk.Setup != "" {
        fmt.Printf("Got setup: %s\n", result.Chunk.Setup)
    }
}
```

[See full example](samples/basic-prompts)

### Load Prompts from Files

Keep prompts separate from code using `.prompt` files with YAML frontmatter:

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
{{#if dietaryRestrictions}}
Dietary restrictions: {{#each dietaryRestrictions}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}.
{{/if}}
```

```go
// Register schemas so .prompt files can reference them by name
genkit.DefineSchemaFor[RecipeRequest](g)
genkit.DefineSchemaFor[Recipe](g)

// Look up and execute the prompt
recipePrompt := genkit.LookupDataPrompt[RecipeRequest, *Recipe](g, "recipe")
recipe, _ := recipePrompt.Execute(ctx, RecipeRequest{
    Dish:        "tacos",
    Cuisine:     "Mexican",
    ServingSize: 4,
})
fmt.Printf("%s (%s)\n", recipe.Title, recipe.PrepTime)
```

[See full example](samples/basic-prompts)

### Embed Prompts in Your Binary

Ship a single binary with prompts compiled in using Go's embed package:

```go
//go:embed prompts/*
var promptsFS embed.FS

func main() {
    ctx := context.Background()
    g := genkit.Init(ctx,
        genkit.WithPlugins(&googlegenai.GoogleAI{}),
        genkit.WithPromptFS(promptsFS),
    )

    prompt := genkit.LookupPrompt(g, "greeting")
    response, _ := prompt.Execute(ctx)
    fmt.Println(response.Text())
}
```

[See full example](samples/prompts-embed)

### Expose Flows as HTTP Endpoints

Serve your flows over HTTP with automatic JSON serialization:

```go
mux := http.NewServeMux()
for _, flow := range genkit.ListFlows(g) {
    mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
}
log.Fatal(http.ListenAndServe(":8080", mux))
```

```bash
curl -X POST http://localhost:8080/tellJoke \
  -H "Content-Type: application/json" \
  -d '{"data": "programming"}'
```

### Works with Any HTTP Framework

`genkit.Handler` returns a standard `http.HandlerFunc`, so it works with any Go HTTP framework:

```go
// net/http (standard library)
mux := http.NewServeMux()
mux.HandleFunc("POST /joke", genkit.Handler(jokeFlow))
log.Fatal(http.ListenAndServe(":8080", mux))

// Gin
r := gin.Default()
r.POST("/joke", gin.WrapF(genkit.Handler(jokeFlow)))
r.Run(":8080")

// Echo
e := echo.New()
e.POST("/joke", echo.WrapHandler(genkit.Handler(jokeFlow)))
e.Start(":8080")

// Chi
r := chi.NewRouter()
r.Post("/joke", genkit.Handler(jokeFlow))
http.ListenAndServe(":8080", r)
```

### Frameworks with Centralized Error Handling

`genkit.HandlerFunc` returns:

```go
func(http.ResponseWriter, *http.Request) error
```

This is useful for frameworks that support centralized error handling and middleware chains, and expect handlers to return an error.

For example, Echo natively supports handlers that return error, so `genkit.HandlerFunc` can be adapted directly:

```go
e := echo.New()
h := genkit.HandlerFunc(jokeFlow)

e.POST("/joke", func(c *echo.Context) error {
	  return h(c.Response(), c.Request())
})

e.Start(":8080")
```

Any error returned by `genkit.HandlerFunc` will be handled by Echo's middleware stack.

### Durable Streaming

> [!WARNING]
> This API is in preview and may experience breaking changes in minor releases.

Allow clients to reconnect to in-progress or completed streams using a stream ID. The stream manager lives in `core/x/streaming`:

```go
import "github.com/firebase/genkit/go/core/x/streaming"

mux.HandleFunc("POST /myFlow", genkit.Handler(myStreamingFlow,
    genkit.WithStreamManager(streaming.NewInMemoryStreamManager(
        streaming.WithTTL(10*time.Minute),
    )),
))
```

Clients receive a stream ID in the `X-Genkit-Stream-Id` header and can reconnect to replay buffered chunks.

[See full example](samples/durable-streaming)

---

## Model Providers

Genkit provides a unified interface across all major AI providers. Use whichever model fits your needs:

| Provider | Plugin | Models |
|----------|--------|--------|
| **Google AI** | `googlegenai.GoogleAI` | Gemini 3.5 Flash, Gemini 3.1 Pro, and more |
| **Vertex AI** | `vertexai.VertexAI` | Gemini 3.5 Flash, Gemini 3.1 Pro via Google Cloud |
| **Anthropic** | `anthropic.Anthropic` | Claude Opus 4.8, Claude Sonnet 4.6, Claude Haiku 4.5 |
| **Ollama** | `ollama.Ollama` | Llama 4, Qwen 3, DeepSeek, and other local models |
| **OpenAI Compatible** | `compat_oai` | GPT-5.5, and any OpenAI-compatible API |

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
import "google.golang.org/genai"

// Simple: just the model name
response, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Hello!"),
)

// Advanced: model name + provider-specific configuration
response, _ := genkit.Generate(ctx, g,
    ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
        Temperature:     genai.Ptr(float32(0.7)),
        MaxOutputTokens: genai.Ptr(int32(1000)),
        TopP:            genai.Ptr(float32(0.9)),
    })),
    ai.WithPrompt("Hello!"),
)
```

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

---

## Samples

Explore working examples to see Genkit in action:

| Sample | Description |
|--------|-------------|
| [basic](samples/basic) | Simple text generation with streaming |
| [basic-structured](samples/basic-structured) | Typed JSON output with `GenerateData` and `GenerateDataStream` |
| [basic-prompts](samples/basic-prompts) | Prompt templates with Handlebars and `.prompt` files |
| [basic-agents](samples/basic-agents) | Multi-turn agents (inline, prompt-file, and custom-loop) with snapshots and background detach |
| [basic-agents-server](samples/basic-agents-server) | Serving store-backed and stateless agents over HTTP |
| [intermediate-interrupts](samples/intermediate-interrupts) | Human-in-the-loop with tool interrupts |
| [basic-middleware/retry-fallback](samples/basic-middleware/retry-fallback) | Composing `Retry` and `Fallback` middleware |
| [basic-middleware/filesystem](samples/basic-middleware/filesystem) | Scoped filesystem tools for the model |
| [basic-middleware/skills](samples/basic-middleware/skills) | On-demand loadable `SKILL.md` personas |
| [prompts-embed](samples/prompts-embed) | Embed prompts in your binary |
| [durable-streaming](samples/durable-streaming) | Reconnectable streams with replay |

---

<p align="center">
  Built by Google with contributions from the <a href="https://github.com/genkit-ai/genkit/graphs/contributors">Open Source Community</a>
</p>
