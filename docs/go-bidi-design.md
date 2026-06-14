# Genkit Go Bidirectional Streaming Features - Design Document

## Overview

This document describes the design for bidirectional streaming features in Genkit Go. The implementation introduces three new primitives:

1. **BidiAction** - Core primitive for bidirectional operations (`go/core/x`)
2. **BidiFlow** - BidiAction with observability, intended for user definition (`go/core/x`)
3. **BidiModel** - Specialized bidi action for real-time LLM APIs (`go/ai/x`)

For stateful multi-turn agents with session persistence, see [go-agent-design.md](go-agent-design.md).

## Package Location

```
go/core/x/
├── bidi.go           # BidiAction, BidiFunc, BidiConnection
├── bidi_flow.go      # BidiFlow
├── bidi_options.go   # Options
├── bidi_test.go      # Tests

go/ai/x/
├── bidi_model.go     # BidiModel, BidiModelFunc
├── bidi_model_test.go
```

Import as:
- `corex "github.com/firebase/genkit/go/core/x"`
- `aix "github.com/firebase/genkit/go/ai/x"`

---

## 1. Core Type Definitions

### 1.1 BidiAction

```go
// BidiAction represents a bidirectional streaming action.
// Type parameters:
//   - Init: Type of initialization data (use struct{} if not needed)
//   - In: Type of each message sent to the action
//   - Out: Type of the final output
//   - Stream: Type of each streamed output chunk
type BidiAction[Init, In, Out, Stream any] struct {
    name     string
    fn       BidiFunc[Init, In, Out, Stream]
    registry api.Registry
    desc     *api.ActionDesc
}

// BidiFunc is the function signature for bidi actions.
type BidiFunc[Init, In, Out, Stream any] func(
    ctx context.Context,
    init Init,
    inCh <-chan In,
    outCh chan<- Stream,
) (Out, error)
```

### 1.2 BidiConnection

```go
// BidiConnection represents an active bidirectional streaming session.
type BidiConnection[In, Out, Stream any] struct {
    inputCh  chan In                     // Internal, accessed via Send()
    streamCh chan Stream                 // Internal output stream channel
    doneCh   chan struct{}               // Closed when action completes
    output   Out                         // Final output (valid after done)
    err      error                       // Error if any (valid after done)
    ctx      context.Context
    cancel   context.CancelFunc
    span     tracing.Span               // Trace span, ended on completion
    mu       sync.Mutex
    closed   bool
}

// Send sends an input message to the bidi action.
func (c *BidiConnection[In, Out, Stream]) Send(input In) error

// Close signals that no more inputs will be sent.
func (c *BidiConnection[In, Out, Stream]) Close() error

// Receive returns an iterator for receiving streamed response chunks.
// The iterator completes when the action finishes or signals end of turn.
func (c *BidiConnection[In, Out, Stream]) Receive() iter.Seq2[Stream, error]

// Output returns the final output after the action completes.
// Blocks until done or context cancelled.
func (c *BidiConnection[In, Out, Stream]) Output() (Out, error)

// Done returns a channel closed when the connection completes.
func (c *BidiConnection[In, Out, Stream]) Done() <-chan struct{}
```

### 1.3 BidiFlow

```go
type BidiFlow[Init, In, Out, Stream any] struct {
    *BidiAction[Init, In, Out, Stream]
}
```

---

## 2. BidiModel

### 2.1 Overview

`BidiModel` is a specialized bidi action for real-time LLM APIs like Gemini Live and OpenAI Realtime. These APIs establish a persistent connection where configuration (temperature, system prompt, tools) must be provided upfront, and then the conversation streams bidirectionally.

### 2.2 The Role of `init`

For real-time sessions, the connection to the model API often requires configuration to be established *before* the first user message is received. The `init` payload fulfills this requirement:

- **`init`**: `ModelRequest` (contains config, tools, system prompt)
- **`inputStream`**: Stream of `ModelRequest` (contains user messages/turns)
- **`stream`**: Stream of `ModelResponseChunk`

### 2.3 Type Definitions

```go
// In go/ai/x/bidi_model.go

// BidiModel represents a bidirectional streaming model for real-time LLM APIs.
type BidiModel struct {
    *corex.BidiAction[*ai.ModelRequest, *ai.ModelRequest, *ai.ModelResponse, *ai.ModelResponseChunk]
}

// BidiModelFunc is the function signature for bidi model implementations.
type BidiModelFunc func(
    ctx context.Context,
    init *ai.ModelRequest,
    inCh <-chan *ai.ModelRequest,
    outCh chan<- *ai.ModelResponseChunk,
) (*ai.ModelResponse, error)
```

### 2.4 Defining a BidiModel

```go
// DefineBidiModel creates and registers a BidiModel for real-time LLM interactions.
// The opts parameter follows the same pattern as DefineModel for consistency.
func DefineBidiModel(r api.Registry, name string, opts *ai.ModelOptions, fn BidiModelFunc) *BidiModel
```

**Example Plugin Implementation:**

```go
func (g *GoogleAI) defineBidiModel(r api.Registry) *aix.BidiModel {
    return aix.DefineBidiModel(r, "googleai/gemini-2.0-flash-live",
        &ai.ModelOptions{
            Label: "Gemini 2.0 Flash Live",
            Supports: &ai.ModelSupports{
                Multiturn:  true,
                Tools:      true,
                SystemRole: true,
                Media:      true,
            },
        },
        func(ctx context.Context, init *ai.ModelRequest, inCh <-chan *ai.ModelRequest, outCh chan<- *ai.ModelResponseChunk) (*ai.ModelResponse, error) {
            session, err := g.client.Live.Connect(ctx, "gemini-2.0-flash-live", &genai.LiveConnectConfig{
                SystemInstruction:  toContent(init.Messages),
                Tools:              toTools(init.Tools),
                Temperature:        toFloat32Ptr(init.Config),
                ResponseModalities: []genai.Modality{genai.ModalityText},
            })
            if err != nil {
                return nil, err
            }
            defer session.Close()

            var totalUsage ai.GenerationUsage

            for request := range inCh {
                err := session.SendClientContent(genai.LiveClientContentInput{
                    Turns:        toContents(request.Messages),
                    TurnComplete: true,
                })
                if err != nil {
                    return nil, err
                }

                for {
                    msg, err := session.Receive()
                    if err != nil {
                        return nil, err
                    }

                    if msg.ToolCall != nil {
                        outCh <- &ai.ModelResponseChunk{
                            Content: toToolCallParts(msg.ToolCall),
                        }
                        continue
                    }

                    if msg.ServerContent != nil {
                        if msg.ServerContent.ModelTurn != nil {
                            outCh <- &ai.ModelResponseChunk{
                                Content: fromParts(msg.ServerContent.ModelTurn.Parts),
                            }
                        }
                        if msg.ServerContent.TurnComplete {
                            break
                        }
                    }

                    if msg.UsageMetadata != nil {
                        totalUsage.InputTokens += int(msg.UsageMetadata.PromptTokenCount)
                        totalUsage.OutputTokens += int(msg.UsageMetadata.CandidatesTokenCount)
                    }
                }
            }

            return &ai.ModelResponse{
                Usage: &totalUsage,
            }, nil
        },
    )
}
```

### 2.5 Using BidiModel (`GenerateBidi`)

`GenerateBidi` is the high-level API for interacting with bidi models. It provides a session-like interface for real-time conversations.

```go
// In go/genkit/generate.go or go/ai/x/generate_bidi.go

// ModelBidiConnection wraps BidiConnection with model-specific convenience methods.
type ModelBidiConnection struct {
    conn *corex.BidiConnection[*ai.ModelRequest, *ai.ModelResponse, *ai.ModelResponseChunk]
}

// Send sends a user message to the model.
func (s *ModelBidiConnection) Send(messages ...*ai.Message) error {
    return s.conn.Send(&ai.ModelRequest{Messages: messages})
}

// SendText is a convenience method for sending a text message.
func (s *ModelBidiConnection) SendText(text string) error {
    return s.Send(ai.NewUserTextMessage(text))
}

// Stream returns an iterator for receiving response chunks.
func (s *ModelBidiConnection) Receive() iter.Seq2[*ai.ModelResponseChunk, error] {
    return s.conn.Receive()
}

// Close signals that the conversation is complete.
func (s *ModelBidiConnection) Close() error {
    return s.conn.Close()
}

// Output returns the final response after the session completes.
func (s *ModelBidiConnection) Output() (*ai.ModelResponse, error) {
    return s.conn.Output()
}
```

**Usage:**

`GenerateBidi` uses the same shared option types as regular `Generate` calls. Options like `WithModel`, `WithConfig`, `WithSystem`, and `WithTools` work the same way - they configure the initial session setup.

```go
// GenerateBidi starts a bidirectional streaming session with a model.
// Uses the existing shared option types from ai/option.go.
func GenerateBidi(ctx context.Context, g *Genkit, opts ...ai.GenerateBidiOption) (*ModelBidiConnection, error)
```

**Example:**

```go
conn, err := genkit.GenerateBidi(ctx, g,
    ai.WithModel(geminiLive),
    ai.WithConfig(&genai.LiveConnectConfig{Temperature: genai.Ptr[float32](0.7)}),
    ai.WithSystem("You are a helpful voice assistant"),
    ai.WithTools(weatherTool),
)
if err != nil {
    return err
}
defer conn.Close()

conn.SendText("Hello!")

for chunk, err := range conn.Receive() {
    if err != nil {
        return err
    }
    fmt.Print(chunk.Text())
}

conn.SendText("Tell me more about that.")
for chunk, err := range conn.Receive() {
    // ...
}

response, _ := conn.Output()
fmt.Printf("Total tokens: %d\n", response.Usage.TotalTokens)
```

### 2.6 Tool Calling in BidiModel

Real-time models may support tool calling. The pattern follows the standard generate flow but within the streaming context:

```go
conn, _ := genkit.GenerateBidi(ctx, g,
    ai.WithModel(geminiLive),
    ai.WithTools(weatherTool, calculatorTool),
)

conn.SendText("What's the weather in NYC?")

for chunk, err := range conn.Receive() {
    if err != nil {
        return err
    }

    if toolCall := chunk.ToolCall(); toolCall != nil {
        result, _ := toolCall.Tool.Execute(ctx, toolCall.Input)
        conn.Send(ai.NewToolResponseMessage(toolCall.ID, result))
    } else {
        fmt.Print(chunk.Text())
    }
}
```

---

## 3. API Surface

### 3.1 Defining Bidi Actions

```go
// In go/core/x/bidi.go

// NewBidiAction creates a BidiAction without registering it.
func NewBidiAction[Init, In, Out, Stream any](
    name string,
    fn BidiFunc[Init, In, Out, Stream],
) *BidiAction[Init, In, Out, Stream]

// DefineBidiAction creates and registers a BidiAction.
func DefineBidiAction[Init, In, Out, Stream any](
    r api.Registry,
    name string,
    fn BidiFunc[Init, In, Out, Stream],
) *BidiAction[Init, In, Out, Stream]
```

Schemas for `In`, `Out`, `Init`, and `Stream` types are automatically inferred from the type parameters using the existing JSON schema inference in `go/internal/base/json.go`.

### 3.2 Defining Bidi Flows

```go
// In go/core/x/bidi_flow.go

// DefineBidiFlow creates a BidiFlow with tracing and registers it.
// Use this for user-defined bidirectional streaming operations.
func DefineBidiFlow[Init, In, Out, Stream any](
    r api.Registry,
    name string,
    fn BidiFunc[Init, In, Out, Stream],
) *BidiFlow[Init, In, Out, Stream]
```

### 3.3 Starting Connections

All bidi types (BidiAction, BidiFlow, BidiModel) use the same `StreamBidi` method to start connections:

```go
func (ba *BidiAction[Init, In, Out, Stream]) StreamBidi(ctx context.Context, init Init) (*BidiConnection[In, Out, Stream], error)
```

### 3.4 High-Level Genkit API

```go
// In go/genkit/bidi.go

func DefineBidiFlow[Init, In, Out, Stream any](
    g *Genkit,
    name string,
    fn corex.BidiFunc[Init, In, Out, Stream],
) *corex.BidiFlow[Init, In, Out, Stream]

// GenerateBidi uses shared options from ai/option.go
// Options like WithModel, WithConfig, WithSystem, WithTools configure the session init.
func GenerateBidi(
    ctx context.Context,
    g *Genkit,
    opts ...ai.GenerateBidiOption,
) (*ModelBidiConnection, error)
```

---

## 4. Integration with Existing Infrastructure

### 4.1 Tracing Integration

BidiFlows create spans that remain open for the lifetime of the connection, enabling streaming trace visualization in the Dev UI.

**Key behaviors:**
- Span starts when `StreamBidi()` is called
- Span ends when the bidi function returns (via `defer` in the connection goroutine)
- Flow context is injected so `core.Run()` works inside the bidi function
- Nested spans for sub-operations (e.g., each LLM call) work normally

**Important**: The span stays open while the connection is active, allowing:
- Streaming traces to the Dev UI in real-time
- Nested spans for sub-operations (e.g., each LLM call)
- Events recorded as they happen

### 4.2 Action Registration

Add new action types and schema fields:

```go
// In go/core/api/action.go
const (
    ActionTypeBidiFlow  ActionType = "bidi-flow"
    ActionTypeBidiModel ActionType = "bidi-model"
)

// ActionDesc gets two new optional fields
type ActionDesc struct {
    // ... existing fields ...
    StreamSchema map[string]any `json:"streamSchema,omitempty"` // NEW: schema for streamed chunks
    InitSchema   map[string]any `json:"initSchema,omitempty"`   // NEW: schema for initialization data
}
```

---

## 5. Example Usage

### 5.1 Basic Echo Bidi Flow

```go
package main

import (
    "context"
    "fmt"

    "github.com/firebase/genkit/go/genkit"
)

func main() {
    ctx := context.Background()
    g := genkit.Init(ctx)

    echoFlow := genkit.DefineBidiFlow(g, "echo",
        func(ctx context.Context, init struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
            var count int
            for input := range inCh {
                count++
                outCh <- fmt.Sprintf("echo: %s", input)
            }
            return fmt.Sprintf("processed %d messages", count), nil
        },
    )

    conn, err := echoFlow.StreamBidi(ctx)
    if err != nil {
        panic(err)
    }

    conn.Send("hello")
    conn.Send("world")
    conn.Close()

    for chunk, err := range conn.Receive() {
        if err != nil {
            panic(err)
        }
        fmt.Println(chunk)
    }

    output, _ := conn.Output()
    fmt.Println(output)
}
```

### 5.2 Bidi Flow with Initialization Data

```go
type ChatInit struct {
    SystemPrompt string  `json:"systemPrompt"`
    Temperature  float64 `json:"temperature"`
}

configuredChat := genkit.DefineBidiFlow(g, "configuredChat",
    func(ctx context.Context, init ChatInit, inCh <-chan string, outCh chan<- string) (string, error) {
        for input := range inCh {
            resp, _ := genkit.GenerateText(ctx, g,
                ai.WithSystem(init.SystemPrompt),
                ai.WithConfig(&genai.GenerateContentConfig{Temperature: &init.Temperature}),
                ai.WithPrompt(input),
            )
            outCh <- resp
        }
        return "done", nil
    },
)

conn, _ := configuredChat.StreamBidi(ctx, ChatInit{
    SystemPrompt: "You are a helpful assistant.",
    Temperature:  0.7,
})
```

---

## 6. Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `go/core/x/bidi.go` | BidiAction, BidiFunc, BidiConnection |
| `go/core/x/bidi_flow.go` | BidiFlow with tracing |
| `go/core/x/bidi_options.go` | BidiOption types |
| `go/core/x/bidi_test.go` | Tests |
| `go/ai/x/bidi_model.go` | BidiModel, BidiModelFunc, ModelBidiConnection |
| `go/ai/x/bidi_model_test.go` | Tests |
| `go/genkit/bidi.go` | High-level API wrappers |

### Modified Files

| File | Change |
|------|--------|
| `go/core/api/action.go` | Add `ActionTypeBidiFlow`, `ActionTypeBidiModel` constants |

---

## 7. Implementation Notes

### Error Handling
- Errors from the bidi function propagate to both `Responses()` iterator and `Output()`
- Context cancellation closes all channels and terminates the action
- Send after Close returns an error
- Errors are yielded as the second value in the `iter.Seq2[Stream, error]` iterator

### Goroutine Management
- BidiConnection spawns a goroutine to run the action
- Proper cleanup on context cancellation using `defer` and `sync.Once`
- Channel closure follows Go idioms (sender closes)
- Trace span is ended in the goroutine's defer

### Thread Safety
- BidiConnection uses mutex for state (closed flag)
- Send is safe to call from multiple goroutines

### Channels and Backpressure
- Both input and output channels are **unbuffered** by default (size 0)
- This provides natural backpressure: `Send()` blocks until the action reads, output blocks until consumer reads
- If needed, `WithInputBufferSize` / `WithOutputBufferSize` options could be added later for specific use cases

### Tracing
- Span is started when connection is created, ended when action completes
- Nested spans work normally within the bidi function
- Events can be recorded throughout the connection lifecycle
- Dev UI can show traces in real-time as they stream
- Implementation uses the existing tracer infrastructure (details left to implementation)

### Shutdown Sequence
When `Close()` is called on a BidiConnection:
1. The input channel is closed, signaling no more inputs
2. The bidi function's `for range inputStream` loop exits
3. The function returns its final output
4. The stream channel is closed
5. The `Done()` channel is closed
6. `Output()` unblocks and returns the result

On context cancellation:
1. Context error propagates to the bidi function
2. All channels are closed
3. `Output()` returns the context error

---

## 8. Integration with Reflection API

These features align with **Reflection API V2**, which uses WebSockets to support bidirectional streaming between the Runtime and the CLI/Manager.

- `runAction` now supports an `input` stream
- `streamChunk` notifications are bidirectional (Manager <-> Runtime)
