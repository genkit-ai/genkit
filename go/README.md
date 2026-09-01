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

## Samples

Every sample below runs on its own with `go run .`, and its package comment explains what it demonstrates and how to call it. The `basic-*` set is the canonical one: together it covers the whole framework, and each program is small enough to read in one sitting.

| Sample | Description |
|--------|-------------|
| [basic](samples/basic/main.go) | Simple text generation with streaming |
| [basic‑structured](samples/basic-structured/main.go) | Typed JSON output with `GenerateData` and `GenerateDataStream` |
| [basic‑formats](samples/basic-formats/main.go) | Output formats and what each one makes a streamed chunk mean |
| [basic‑media](samples/basic-media) | Reading, drawing, and redrawing pictures, plus generating video with a polled background model |
| [basic‑prompts](samples/basic-prompts) | Prompt templates with Handlebars and `.prompt` files, shared partials and helpers, and prompts embedded in the binary |
| [basic‑prompt‑content](samples/basic-prompt-content/main.go) | Prompt content computed from your data, with media and retrieved docs |
| [basic‑tools](samples/basic-tools/main.go) | A slow tool whose answer is more than one value, returned as a multipart response |
| [basic‑tools‑exp](samples/basic-tools-exp/main.go) | The same program on the in-preview tools API, so the diff between the two is the API |
| [basic‑agents](samples/basic-agents) | Multi-turn agents (inline, prompt-file, and custom-loop) with snapshots and background detach |
| [basic‑agents‑server](samples/basic-agents-server/main.go) | Serving store-backed and stateless agents over HTTP |
| [basic‑tool‑interrupts](samples/basic-tool-interrupts/main.go) | Human in the loop (HITL): a tool that pauses for approval and resumes with the answer |
| [basic‑tool‑interrupts‑exp](samples/basic-tool-interrupts-exp/main.go) | The same HITL program on the in-preview tools API, so the diff between the two is the API |
| [basic‑middleware](samples/basic-middleware) | Model middleware, one program each: [retry-fallback](samples/basic-middleware/retry-fallback/main.go) composes `Retry` and `Fallback` into a cascade that survives a dead model, [filesystem](samples/basic-middleware/filesystem) gives the model file access scoped to a single directory, and [skills](samples/basic-middleware/skills) loads `SKILL.md` personas on demand |
| [basic‑errors](samples/basic-errors/main.go) | Classifying failures with sentinels and recovering with `errors.Is` |
| [basic‑durable‑streaming‑exp](samples/basic-durable-streaming-exp/main.go) | Reconnectable streams with replay, on the in-preview `core/x/streaming` API |

---

<details>
<summary><strong>Contents</strong> &middot; everything Genkit Go can do, at a glance</summary>

**[Agents](#agents)** *(preview)*

Multi-turn conversations that own their own loop and state.

[Define an Agent](#define-an-agent) &middot;
[Multi-Turn Conversations](#multi-turn-conversations) &middot;
[Load the Prompt from a File](#load-the-prompt-from-a-file) &middot;
[Custom Turn Loops](#custom-turn-loops) &middot;
[Persist and Resume](#persist-and-resume) &middot;
[Re-attempt Failed Turns](#re-attempt-failed-turns) &middot;
[Stop a Run and Continue It](#stop-a-run-and-continue-it) &middot;
[Redact on the Way Out](#redact-on-the-way-out) &middot;
[Background Agents](#background-agents) &middot;
[Delegate to Sub-Agents](#delegate-to-sub-agents) &middot;
[Serve Agents over HTTP](#serve-agents-over-http)

**[Features](#features)**

**Generating**
[Generate Text](#generate-text) &middot;
[Generate Structured Data](#generate-structured-data) &middot;
[Stream Responses](#stream-responses) &middot;
[Stream Structured Data](#stream-structured-data)

**Tools**
[Define Tools](#define-tools) &middot;
[Tool Interrupts](#tool-interrupts) &middot;
[Streaming, Multipart, and Interruptible Tools](#streaming-multipart-and-interruptible-tools) *(preview)*

**Middleware**
[Middleware](#middleware) &middot;
[Custom Middleware](#custom-middleware)

**Flows**
[Define Flows](#define-flows) &middot;
[Streaming Flows](#streaming-flows) &middot;
[Traced Sub-steps](#traced-sub-steps) &middot;
[Logging](#logging)

**Prompts**
[Define Prompts](#define-prompts) &middot;
[Type-Safe Data Prompts](#type-safe-data-prompts) &middot;
[Build Prompts from Your Data](#build-prompts-from-your-data) &middot;
[Load Prompts from Files](#load-prompts-from-files) &middot;
[Embed Prompts in Your Binary](#embed-prompts-in-your-binary)

**Serving**
[Expose Flows as HTTP Endpoints](#expose-flows-as-http-endpoints) &middot;
[Works with Any HTTP Framework](#works-with-any-http-framework) &middot;
[Frameworks with Centralized Error Handling](#frameworks-with-centralized-error-handling) &middot;
[Error Handling](#error-handling) &middot;
[Durable Streaming](#durable-streaming) *(preview)*

**[Model Providers](#model-providers)**

Gemini, Claude, GPT, Grok, DeepSeek, Qwen, Kimi, GLM, Llama, Mistral, local models, and anything OpenAI-compatible.

**[Development Tools](#development-tools)**

[Genkit CLI](#genkit-cli) &middot;
[Developer UI](#developer-ui)

</details>

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

{{history}}

{{role "user"}}
If the question is ambiguous, ask one clarifying question instead of guessing.
```

```go
type ChatInput struct {
    Personality string `json:"personality"`
}

// Register the schema so the .prompt file can reference it by name.
genkit.DefineSchemasFor(g, ChatInput{})

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

With a session store configured, each turn writes a snapshot as it ends. The caller only needs the `SessionID` from a previous result to pick the conversation back up:

```go
first, _ := chatAgent.RunText(ctx, "My name is Alex and I'm planning a trip to Japan.")

// Later, in another request or process: resume from the latest snapshot.
second, _ := chatAgent.RunText(ctx, "What is my name?",
    aix.WithSessionID[any](first.SessionID))
fmt.Println(second.Message.Text()) // "Your name is Alex."
```

Resume from one specific point in history with `aix.WithSnapshotID`, or skip the server store entirely and round-trip the state yourself with `aix.WithState` (the conversation's identity travels inside the state object).

[See full example](samples/basic-agents)

### Re-attempt Failed Turns

A turn that fails keeps the tool rounds it completed. The generate call hands back the conversation as it stood at the last turn seam, the agent commits that, and it persists as a `failed` snapshot carrying the error. Resume that snapshot like any other and send an input with no payload: the turn runs again on the committed messages, so the tool calls that already succeeded are not repeated.

```go
out, _ := agent.RunText(ctx, "Book the full itinerary.")
if out.FinishReason == aix.AgentFinishReasonFailed {
    // out.Error carries the status. Retry the transient ones.
    if out.Error.Status == status.Unavailable {
        retried, _ := agent.Run(ctx, &aix.AgentInput{},
            aix.WithSessionID[any](out.SessionID))
    }
}
```

Whether a failure is worth another attempt is yours to decide; the framework records the error and leaves the snapshot resumable either way. A new message works there too, and rewinding past the failure is a resume from the previous snapshot ID.

Without a store, the failed output's `State` carries the same resume point inline; initialize the next invocation with it. A turn rejected before it reached the model (an invalid input, for example) commits nothing, and the resume point stays the turn before it. Custom agents opt in by returning a `TurnResult` alongside the turn's error; a bare error keeps the discard-the-turn behavior.

### Stop a Run and Continue It

A run the caller stopped is `aborted`; a run that broke is `failed`. Stopping covers every way the caller ends it: cancelling the context on a live connection, closing the transport, letting a deadline expire, hitting a limit it set such as `ai.WithMaxTurns`, or calling `Abort` on a detached one. All of them land an `aborted` snapshot holding the turns that finished, and all of them resume like a `failed` one. A detached run gets there in two writes: `Abort` flips its snapshot to `aborting`, which stops the work while the worker keeps heartbeating as it winds down, and the worker's finalize then lands the `aborted` snapshot with the state. An `aborting` snapshot is not yet a resume point; wait for it to settle, and if its heartbeat goes stale the worker died and the snapshot reads as `expired`:

```go
ctx, cancel := context.WithCancel(ctx)
out, err := chatAgent.Run(ctx, &aix.AgentInput{Message: msg})  // cancel() elsewhere

// Both come back: err is what stopped the run, out names where it stopped.
if out.FinishReason == aix.AgentFinishReasonAborted {
    resumed, _ := chatAgent.Run(context.Background(), &aix.AgentInput{},
        aix.WithSnapshotID[any](out.SnapshotID))
}
```

The turn that was in flight is discarded whole, so the conversation ends at a turn seam. A tool that had already run inside it loses its response along with the rest of the round, and re-sending the conversation calls it again.

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

// Later: read where it stands, or block until it settles.
snap, _ := chatAgent.GetSnapshot(ctx, snapshotID)     // one read, no waiting
snap, _ = chatAgent.WaitForSnapshot(ctx, snapshotID)  // blocks until terminal
switch snap.Status {
case aix.SnapshotStatusPending:   // still working (GetSnapshot only)
case aix.SnapshotStatusCompleted: // snap.State holds the final state; resume it
case aix.SnapshotStatusFailed:    // snap.Error holds the failure; still resumable
case aix.SnapshotStatusAborted:   // stopped early; resumable from the last finished turn
case aix.SnapshotStatusExpired:   // the worker stopped heartbeating
}

// Or stop it early; the runtime observes the abort and cancels the work.
// The snapshot keeps the turns that finished, so it can be resumed later.
chatAgent.Abort(ctx, snapshotID)
```

Detach requires a store that implements `SnapshotSubscriber` (both bundled local stores do). A detached turn refreshes a heartbeat while it runs and while it winds down after an abort, so a crashed worker surfaces as `expired` instead of orphaning the conversation forever. Reads that only need to know where a snapshot stands can skip its state: `GetSnapshot(ctx, id, WithMetadataOnly())` returns the status, finish reason, parent, and timestamps. A session store that implements the optional `SnapshotMetadataReader` (the bundled stores and the Firestore store do) answers it without loading the conversation; any other store is read in full and the state dropped.

[See full example](samples/basic-agents)

### Delegate to Sub-Agents

> [!WARNING]
> This API is in preview and may experience breaking changes in minor releases.

The experimental `Agents` middleware (in `plugins/middleware/exp`) lets one agent delegate to others. It injects one `delegate_to_<name>` tool per sub-agent and a `<sub-agents>` listing into the orchestrator's system prompt, then runs the chosen sub-agent and returns its result when the model calls the tool. Each sub-agent's `aix.WithDescription` (captured by `agent.Ref()`) tells the orchestrator when to reach for it. Delegation composes: a sub-agent that carries its own `Agents` middleware delegates further, so orchestrations nest without extra wiring. Delegations also accept an optional `name`, a human-readable label echoed on results and background-task reports next to the `taskId`.

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

Set `Async: true` to let the orchestrator keep working while a sub-agent runs. Each delegation tool gains a `background` flag that returns a task ID instead of waiting, and three shared tools give the orchestrator one control per thing it can do with a launched task: `check_background_tasks` reads statuses without waiting, `wait_for_background_tasks` blocks until they settle, and `abort_background_tasks` stops the ones whose results are no longer needed. The *sub-agent* is what needs the session store here: background work is tracked by a snapshot, so a sub-agent without one can only be delegated to synchronously (see `AgentMetadata.Abortable`).

```go
// The sub-agent needs its own store to be delegated to in the background.
researcher := genkitx.DefineAgent(g, "researcher",
    aix.InlinePrompt{
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithSystem("You are a thorough research assistant."),
    },
    aix.WithDescription[any]("Researches a topic and summarizes well-sourced findings."),
    aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
)

ai.WithUse(&middlewarex.Agents{
    Agents: []aix.AgentRef{researcher.Ref()},
    Async:  true,
})
```

The orchestrator then launches with `{"task": "...", "background": true}`, posts an update while the sub-agent runs, and collects the result later. `wait_for_background_tasks` takes an optional `timeoutSeconds`, so a slow task becomes an interim answer instead of a blocked turn, and `waitFor: "first"` turns the join into a race: the tool returns as soon as any listed task settles while the rest keep running, so the orchestrator can act on the first answer. An abort is safe to call on any task, and it never blocks: one that had already finished is left alone and reports its result, so the orchestrator never loses an answer by giving up on it, and a live one reports `aborting` while it winds down and saves its progress, settling as a resumable `aborted` that the wait tool can collect.

Delegations that settle leave a continuable handle behind. Every delegation result and background-task report carries a `taskId`, and the shared `continue_task` tool spends it: a failed or aborted task continues from its last saved progress (with no `instructions` the committed turn is re-attempted as it stood; with `instructions` the retry is steered), and a completed task accepts follow-up instructions inside the sub-agent's own session, so pressing on never repeats finished work. With `Async` set, `continue_task` also takes `background: true` and returns a fresh task ID in the same session. Only server-managed sub-agents are continuable: their handles survive across turns and process restarts, while a client-managed delegation settles inline, carries no `taskId`, and is redone by delegating again. A task that stopped on an interrupt cannot be continued, since continuing it would mean answering the interrupt; the tool refuses it and the orchestrator delegates a more self-contained task instead. An expired task (its worker died) is fenced with an abort and continued from its parent snapshot, the last one committed before the detach; a background launch that died before committing anything has nothing saved, and the tool says to delegate again.

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
// POST /agents/chat                     one turn per request (?stream=true for SSE)
// POST /agents/chat/getSnapshot         read a snapshot by ID
// POST /agents/chat/waitForSnapshot     block until a snapshot settles
// POST /agents/chat/abort               abort background work
log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
```

A client starts a conversation by POSTing a turn, then continues it by sending the returned `sessionId` in the request's `init` field. Agents with no store return the full state instead and the client round-trips it, so stateless and store-backed agents deploy side by side.

[See full example](samples/basic-agents-server/main.go)

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

Options compose, so you can build a request up from several helpers. Options
carrying multiple items (`ai.WithMessages`, `ai.WithTools`, `ai.WithDocs`,
`ai.WithUse`) accumulate across repeats, while single-value options
(`ai.WithConfig`, `ai.WithModelName`, `ai.WithSystem`) take the last one set.
Repeating an option is how a request gets assembled in pieces, not a mistake to
be reported:

```go
opts := []ai.GenerateOption{
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithSystem("You are a helpful assistant."),
    ai.WithTools(searchTool, weatherTool),
}

if isAdmin {
    // Appends to the tools above; the model sees all of them.
    opts = append(opts, ai.WithTools(adminTools...))
}
if terse {
    // Fills the same slot as the system prompt above, so this one wins.
    opts = append(opts, ai.WithSystem("You are a terse assistant. One sentence."))
}

response, _ := genkit.Generate(ctx, g, append(opts, ai.WithPrompt("What should I pack for Tokyo?"))...)
```

What still fails is a combination no merge could make sense of, rather than a
repeat. Tool names must be unique across the merged list, so the same tool from
two helpers is rejected when the request runs, and `genkit.DefinePrompt` panics
if given a conversation template alongside separately supplied messages, since
the template already is the conversation. These rules cover a single options
list; APIs that layer two lists, like a prompt's define-time options against
`Execute`-time options, document their own precedence.

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

[See full example](samples/basic-structured/main.go)

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

Both forms are supported, and the choice is about how much of the stream you need
to see. `ai.WithStreaming` takes a callback and still returns the finished
response, so it stays shorter whenever forwarding the text is all you do.
`GenerateStream` hands you the loop instead, which is what you want to inspect
individual parts, recover from a failure part-way through, or act on the chunks
before passing them on.

[See full example](samples/basic-tools/main.go)

### Stream Structured Data

Stream typed JSON objects as they're being generated:

```go
type Ingredient struct {
    Name   string `json:"name"`
    Amount string `json:"amount"`
}

type Recipe struct {
    Title       string       `json:"title"`
    Ingredients []Ingredient `json:"ingredients"`
}

// Asking for a value rather than a pointer means a chunk is always usable:
// the zero Recipe reads the same as one whose fields have not arrived yet.
stream := genkit.GenerateDataStream[Recipe](ctx, g,
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
    if len(result.Chunk.Ingredients) > 0 {
        fmt.Printf("Found ingredient: %s\n", result.Chunk.Ingredients[0].Name)
    }
}
```

[See full example](samples/basic-structured/main.go)

The output format decides how the text is parsed and what a chunk holds: the default `json` gives a growing value like the one above, while `ai.WithOutputFormat(ai.OutputFormatJSONL)` on a slice type hands over one item at a time instead of restating the whole list, and `ai.WithOutputEnums` constrains the answer to one label.

Streaming a flow over HTTP needs `?stream=true` on the URL or an `Accept: text/event-stream` header; without one the flow returns only its final result.

[See full example](samples/basic-formats/main.go)

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

A tool error fails the whole generation rather than being reported to the model, so a miss the model could work around (no such city, no rows matched) belongs in the result rather than in an `error`.

`genkit.DefineMultipartTool` is for a result that is more than one value. It returns an `ai.MultipartToolResponse` instead: `Output` is what a plain tool would have returned, and `Content` carries parts that are not values, such as an image or a document. Those parts reach the model and the client both, and must be media or data parts.

```go
chartTool := genkit.DefineMultipartTool(g, "chartWeather",
    "Charts a location's temperatures for the last week",
    func(ctx *ai.ToolContext, input WeatherInput) (*ai.MultipartToolResponse, error) {
        return &ai.MultipartToolResponse{
            Output:  Trend{Low: 61, High: 78},
            Content: []*ai.Part{ai.NewMediaPart("image/png", chartDataURI)},
        }, nil
    },
)
```

[See full example](samples/basic-tools/main.go)

### Tool Interrupts

Interrupts are how Genkit does human in the loop (HITL). A tool pauses execution for a person's approval, and the flow resumes it with modified inputs or a direct response:

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
        // The answer travels as metadata, so it is read back a key at a time.
        if approved, ok := ai.ResumedValue[bool](ctx, "approved"); ok && !approved {
            return "Transfer declined", nil
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

// Interrupts() yields nothing unless the tool paused for input.
var restarts []*ai.Part
for _, interrupt := range resp.Interrupts() {
    meta, _ := ai.InterruptAs[TransferInterrupt](interrupt)

    // Use meta to get user confirmation, then resume with their answer.
    approved := askAHuman(meta)
    part, _ := transferTool.RestartWith(interrupt,
        ai.WithResumedMetadata[TransferInput](map[string]any{"approved": approved}))
    restarts = append(restarts, part)
}

// Collect every restart first, then resume once: generating inside the loop
// would resume later interrupts against a history that already moved on.
if len(restarts) > 0 {
    resp, _ = genkit.Generate(ctx, g,
        ai.WithMessages(resp.History()...),
        ai.WithTools(transferTool),
        ai.WithToolRestarts(restarts...),
    )
}
```

[See full example](samples/basic-tool-interrupts/main.go)

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

`tool.SendChunk` is the third helper: where `tool.SendPartial` wraps a value as a partial tool response, `SendChunk` hands the caller an `ai.ModelResponseChunk` the tool built itself, which is what to reach for when the update is a line of prose. Both are best-effort, so a tool that reports progress through them still works when the caller is not streaming, and neither is written to history.

[See full example](samples/basic-tool-interrupts-exp/main.go), which is the HITL sample above rewritten against this API, or [the banker example](samples/basic-agents) for an interruptible tool wired into an agent.

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
            googlegenai.ModelRef("googleai/gemini-3.5-flash", nil),
        }},
    ),
)
```

The `middleware` plugin also ships with:

- [`ToolApproval`](plugins/middleware/tool_approval.go) — interrupts any tool not on an allow list and resumes once the call is explicitly approved on restart.
- [`Filesystem`](samples/basic-middleware/filesystem) — gives the model `list_files` and `read_file` tools (plus `write_file` and `edit_file` when `AllowWriteAccess` is set), all confined to a single `RootDir` via `os.Root` (Go 1.25+) so paths cannot escape via `..`, absolute paths, or symlinks.
- [`Skills`](samples/basic-middleware/skills) — exposes a library of `SKILL.md` files through a `use_skill` tool so the model can pull in specialised instructions on demand.

[See the retry + fallback sample](samples/basic-middleware/retry-fallback/main.go) for a full composition.

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

[See full example](samples/basic/main.go)

### Streaming Flows

Stream data from your flows using Server-Sent Events (SSE):

```go
genkit.DefineStreamingFlow(g, "streamStory",
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
```

[See full example](samples/basic/main.go)

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

[See full example](samples/basic/main.go)

### Logging

Log through the context-aware logger (package `github.com/firebase/genkit/go/core/logger`) and records reach the terminal and, during development, the Dev UI attached to the trace span that emitted them:

```go
genkit.DefineFlow(g, "importDocuments",
    func(ctx context.Context, source string) (int, error) {
        logger.Info(ctx, "starting import", "source", source)

        count, err := importAll(ctx, source)
        if err != nil {
            logger.Error(ctx, "import failed", "source", source, "error", err)
            return 0, err
        }

        logger.Debug(ctx, "import finished", "documents", count)
        return count, nil
    },
)
```

The terminal shows info and above by default; run with `GENKIT_LOG_LEVEL=debug` to also see Genkit's per-request detail (model calls, tool runs, span timings) there. The Dev UI always receives debug and above, so an interactive app can keep a quiet terminal while the full narrative lands in the trace viewer.

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

jokePrompt := genkit.DefineDataPrompt[JokeRequest, Joke](g, "joke",
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
    if result.Chunk.Setup != "" {
        fmt.Printf("Got setup: %s\n", result.Chunk.Setup)
    }
}
```

[See full example](samples/basic-prompts)

### Build Prompts from Your Data

Fill any part of a prompt with a Go function instead of a template. The function receives your input type, and what it returns is sent as written, so text from a user or a database never has to be escaped:

```go
type Ticket struct {
    Question   string `json:"question"`
    Screenshot string `json:"screenshot,omitempty"` // data URI, optional
}

supportPrompt := genkit.DefinePrompt(g, "support",
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithInputType(Ticket{}),
    ai.WithSystem("You are a support agent. Answer in two sentences."),

    // Text plus an image, assembled per request.
    ai.WithPromptPartsFn(func(ctx context.Context, t Ticket) ([]*ai.Part, error) {
        parts := []*ai.Part{ai.NewTextPart(t.Question)}
        if t.Screenshot != "" {
            parts = append(parts, ai.NewMediaPart("image/png", t.Screenshot))
        }
        return parts, nil
    }),
)

// The braces below reach the model as typed, not as a template.
resp, _ := supportPrompt.Execute(ctx, ai.WithInput(Ticket{
    Question:   "Why does my {{template}} not render?",
    Screenshot: screenshotDataURI,
}))
```

Every slot has a function form: `WithSystemFn` and `WithPromptFn` for the single-message slots (with `WithSystemPartsFn` and `WithPromptPartsFn` for multi-part content), `WithMessagesFn` for the conversation, and `WithDocsFn` to attach retrieved documents.

Each of the two single-message slots holds one message, so its text, parts, and function forms are four ways to write the same thing and the last one set wins. Documents and conversation messages accumulate instead.

[See full example](samples/basic-prompt-content/main.go)

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
genkit.DefineSchemasFor(g, RecipeRequest{}, Recipe{})

// Look up and execute the prompt
recipePrompt := genkit.LookupDataPrompt[RecipeRequest, *Recipe](g, "recipe")
recipe, _ := recipePrompt.Execute(ctx, RecipeRequest{
    Dish:        "tacos",
    Cuisine:     "Mexican",
    ServingSize: 4,
})
fmt.Printf("%s (%s)\n", recipe.Title, recipe.PrepTime)
```

`genkit.DefinePartial` registers a named block of template text to pull into any prompt with `{{> name}}`, and `genkit.DefineHelper` registers a Go function a template can call as `{{name arg}}`; both live on the Genkit instance, so code-defined prompts and `.prompt` files share the same ones.

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

    prompt := genkit.LookupPrompt(g, "joke")
    response, _ := prompt.Execute(ctx)
    fmt.Println(response.Text())
}
```

[See full example](samples/basic-prompts)

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

### Error Handling

The framework classifies its own failures with sentinels, so you can tell what went wrong with `errors.Is` instead of matching message text. Each sentinel also matches the base it derives from, so you can branch at whichever granularity you need:

```go
import (
    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/core/status"
)

_, err := genkit.GenerateText(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithPrompt("Summarize this."),
)
switch {
case errors.Is(err, ai.ErrModelNotFound):
    // The plugin providing this model isn't registered in genkit.Init.
case errors.Is(err, ai.ErrMaxTurnsExceeded):
    // The tool loop hit its limit; raise it with ai.WithMaxTurns.
case errors.Is(err, ai.ErrToolFailed):
    // A tool returned an error. It's wrapped, so errors.As reaches yours.
case errors.Is(err, status.ErrResourceExhausted):
    // Rate limited or out of quota: back off and retry.
}
```

Models, tools, prompts, and provider APIs all report failures this way, so recovery logic reads as a switch rather than a string match.

A generation failure keeps the progress the loop made: once the request has resolved, the classified error comes back alongside a partial `ai.ModelResponse`. When the loop stopped early, the response reports `ai.FinishReasonFailed` with the cause in `FinishMessage` if something broke (a failed model call, a failed tool), or `ai.FinishReasonAborted` if the caller stopped it: a cancelled context, an expired deadline, or a limit it set such as max turns. `History()` carries the conversation up to that point:

```go
resp, err := genkit.Generate(ctx, g /* ... */)
if err != nil && resp != nil {
    transcript := resp.History() // A conversation you can send again.
    cause := resp.Error          // The same failure, classified.
}
```

`resp.Error` is the structured form of `FinishMessage`, so a response that travelled as data (a trace, a persisted turn) still says why it stopped without anyone matching a string. That history stops at a turn seam: the completed tool rounds, and nothing from the turn that failed. A model message whose tools did not all answer is dropped along with the round it opened, because no provider accepts a conversation that ends in an unanswered tool request. Send the history back to retry the failed step without repeating the tool calls that already succeeded. Text streamed before the failure reached your callback, so watch the stream if you want to show it.

Your own failures work the same way. Derive a subtype to keep a parent's status, and use `PublicErrorf` when the message is safe to return to a client:

```go
// Keeps NOT_FOUND (so HTTP 404), and matches both ErrRecipeNotFound
// and status.ErrNotFound.
var ErrRecipeNotFound = status.ErrNotFound.Subtype("recipe not found")

genkit.DefineFlow(g, "recipeFlow", func(ctx context.Context, dish string) (string, error) {
    recipe, ok := cookbook[dish]
    if !ok {
        return "", status.PublicErrorf(ErrRecipeNotFound, "no recipe for %q", dish)
    }
    return recipe, nil
})
```

Wrapping with `fmt.Errorf` and `%w` preserves the classification, so context added up the stack costs you nothing. Served over HTTP, the status picks the response code and only `PublicErrorf` messages reach the client: everything else is redacted and logged server-side, so provider text and internal identifiers stay out of responses. Set `GENKIT_ENV=dev` to see them unredacted while developing.

[See full example](samples/basic-errors/main.go)

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

[See full example](samples/basic-durable-streaming-exp/main.go)

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

<p align="center">
  Built by Google with contributions from the <a href="https://github.com/genkit-ai/genkit/graphs/contributors">Open Source Community</a>
</p>
