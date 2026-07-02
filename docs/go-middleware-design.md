# Middleware Design for Genkit Go

## Problem

The current `ModelMiddleware` only wraps the raw model call. We need middleware that can:

1. Wrap the entire generation (including tool loop)
2. Wrap individual tool executions
3. Share state across these hooks within a single `ai.Generate()` invocation

## Design

### Core Interface

```go
// Middleware provides hooks for different stages of generation.
type Middleware interface {
    // Name returns the middleware's unique identifier.
    Name() string
    // New returns a fresh instance for each ai.Generate() call, enabling per-invocation state.
    New() Middleware
    // Generate wraps each iteration of the tool loop.
    Generate(ctx context.Context, state *GenerateState, next GenerateNext) (*ModelResponse, error)
    // Model wraps each model API call.
    Model(ctx context.Context, state *ModelState, next ModelNext) (*ModelResponse, error)
    // Tool wraps each tool execution.
    Tool(ctx context.Context, state *ToolState, next ToolNext) (*ToolResponse, error)
}

// State structs - can be extended without breaking the interface
type GenerateState struct {
    Options   *GenerateActionOptions  // original options passed to ai.Generate()
    Request   *ModelRequest           // current request for this iteration (has accumulated messages)
    Iteration int                     // current loop iteration (0-indexed)
}

type ModelState struct {
    Request  *ModelRequest
    Callback ModelStreamCallback
}

type ToolState struct {
    Request *ToolRequest
    Tool    Tool  // provides Name(), Definition(), etc.
}

// Next function types for each hook
type GenerateNext func(ctx context.Context, state *GenerateState) (*ModelResponse, error)
type ModelNext func(ctx context.Context, state *ModelState) (*ModelResponse, error)
type ToolNext func(ctx context.Context, state *ToolState) (*ToolResponse, error)
```

Note: `GenerateActionOptions`, `ModelRequest`, `ModelResponse`, `ToolRequest`, `ToolResponse`, and `ToolDefinition` are existing types in `ai/gen.go`.

### Base Implementation

Embed this to get default pass-through behavior for hooks you don't need. You must still implement `Name()` and `New()` yourself.

```go
type BaseMiddleware struct{}

func (b *BaseMiddleware) Generate(ctx context.Context, state *GenerateState, next GenerateNext) (*ModelResponse, error) {
    return next(ctx, state)
}

func (b *BaseMiddleware) Model(ctx context.Context, state *ModelState, next ModelNext) (*ModelResponse, error) {
    return next(ctx, state)
}

func (b *BaseMiddleware) Tool(ctx context.Context, state *ToolState, next ToolNext) (*ToolResponse, error) {
    return next(ctx, state)
}
```

### Usage

```go
resp, err := ai.Generate(ctx, r,
    ai.WithModel(myModel),
    ai.WithPrompt("Hello"),
    ai.WithUse(RetryMiddleware{MaxRetries: 3}),
)
```

### Registration

```go
ai.DefineMiddleware(r, "Retries failed tool calls", &RetryMiddleware{})
```

```go
type MiddlewareDesc struct {
    Name           string         `json:"name"`
    Description    string         `json:"description,omitempty"`
    ConfigSchema   map[string]any `json:"configSchema,omitempty"`
    configFromJSON func([]byte) (Middleware, error) // not serialized
}

func (d *MiddlewareDesc) Register(r api.Registry) {
    r.RegisterValue("/middleware/"+d.Name, d)
}

func NewMiddleware[T Middleware](description string, prototype T) *MiddlewareDesc {
    return &MiddlewareDesc{
        Name:         prototype.Name(),
        Description:  description,
        ConfigSchema: core.InferSchemaMap(*new(T)),
        configFromJSON: func(configJSON []byte) (Middleware, error) {
            inst := prototype.New()
            if len(configJSON) > 0 {
                if err := json.Unmarshal(configJSON, inst); err != nil {
                    return nil, err
                }
            }
            return inst, nil
        },
    }
}

func DefineMiddleware[T Middleware](r api.Registry, description string, prototype T) *MiddlewareDesc {
    d := NewMiddleware(description, prototype)
    d.Register(r)
    return d
}
```

### MiddlewarePlugin Interface

Plugins that provide middleware implement `MiddlewarePlugin`:

```go
type MiddlewarePlugin interface {
    ListMiddleware(ctx context.Context) ([]*MiddlewareDesc, error)
}
```

During `genkit.Init()`, the framework calls `ListMiddleware` on plugins that implement this interface and registers each returned descriptor.

**Plugin example:**
```go
func (p *MyPlugin) ListMiddleware(ctx context.Context) ([]*ai.MiddlewareDesc, error) {
    return []*ai.MiddlewareDesc{
        ai.NewMiddleware("Distributed tracing", &TracingMiddleware{exporter: p.exporter}),
    }, nil
}
```

### API/Schema Integration

To support middleware via the API and Dev UI, we add schemas for middleware descriptors and references.

**Zod schema (genkit-tools/common/src/types/middleware.ts):**
```ts
import { z } from 'zod';
import { JSONSchema7Schema } from './action';

/** Descriptor for a registered middleware, returned by reflection API. */
export const MiddlewareDescSchema = z.object({
  /** Unique name of the middleware. */
  name: z.string(),
  /** Human-readable description of what the middleware does. */
  description: z.string().optional(),
  /** JSON Schema for the middleware's configuration. */
  configSchema: JSONSchema7Schema.optional(),
});
export type MiddlewareDesc = z.infer<typeof MiddlewareDescSchema>;

/** Reference to a registered middleware with optional configuration. */
export const MiddlewareRefSchema = z.object({
  /** Name of the registered middleware. */
  name: z.string(),
  /** Configuration for the middleware (schema defined by the middleware). */
  config: z.any().optional(),
});
export type MiddlewareRef = z.infer<typeof MiddlewareRefSchema>;
```

**GenerateActionOptions addition (genkit-tools/common/src/types/model.ts):**
```ts
export const GenerateActionOptionsSchema = z.object({
  // ... existing fields ...

  /** Middleware to apply to this generation. */
  use: z.array(MiddlewareRefSchema).optional(),
});
```

**Go types:**
```go
type GenerateActionOptions struct {
    // ... existing fields ...
    Use []*MiddlewareRef `json:"use,omitempty"`
}

type MiddlewareRef struct {
    Name   string `json:"name"`
    Config any    `json:"config,omitempty"`
}
```

### Reflection API

General values endpoint (matches JS), filtered by type:

```
GET /api/values?type=middleware → map[string]MiddlewareDesc
```

Dev UI uses this to list middleware and render config forms from `configSchema`.

## Example: Tool Retry Middleware

```go
type RetryMiddleware struct {
    ai.BaseMiddleware
    MaxRetries   int           `json:"maxRetries"`
    Backoff      time.Duration `json:"backoff"`
    totalRetries int           // per-invocation state
}

func (m *RetryMiddleware) Name() string { return "retry" }

func (m *RetryMiddleware) New() ai.Middleware {
    return &RetryMiddleware{
        MaxRetries: m.MaxRetries,
        Backoff:    m.Backoff,
    }
}

// Override Generate to log iteration info
func (m *RetryMiddleware) Generate(ctx context.Context, state *ai.GenerateState, next ai.GenerateNext) (*ai.ModelResponse, error) {
    if state.Iteration == 0 {
        log.Printf("Starting generation with model %s", state.Options.Model)
    }
    resp, err := next(ctx, state)
    if m.totalRetries > 0 {
        log.Printf("Iteration %d complete, total retries so far: %d", state.Iteration, m.totalRetries)
    }
    return resp, err
}

// Override Tool to add retry logic
func (m *RetryMiddleware) Tool(ctx context.Context, state *ai.ToolState, next ai.ToolNext) (*ai.ToolResponse, error) {
    var lastErr error
    for attempt := 0; attempt <= m.MaxRetries; attempt++ {
        if attempt > 0 {
            m.totalRetries++
            time.Sleep(m.Backoff * time.Duration(attempt))
        }
        resp, err := next(ctx, state)
        if err == nil {
            return resp, nil
        }
        lastErr = err
    }
    return nil, lastErr
}
```

**Usage:**

```go
resp, err := ai.Generate(ctx, r,
    ai.WithModel(myModel),
    ai.WithTools(weatherTool),
    ai.WithPrompt("What's the weather?"),
    ai.WithUse(RetryMiddleware{MaxRetries: 3, Backoff: time.Second}),
)
```

## Multiple Middleware

```go
ai.WithUse(
    LoggingMiddleware{},
    RetryMiddleware{MaxRetries: 3},
)
```

First middleware is outermost: `Logging.Tool(Retry.Tool(actual))`.

## Plugin Integration

Plugins implement `MiddlewarePlugin` to register middleware automatically, and provide factory methods to inject dependencies:

```go
type TracingMiddleware struct {
    ai.BaseMiddleware
    SampleRate float64        `json:"sampleRate"`
    exporter   trace.Exporter // injected by plugin, not serializable
}

func (m *TracingMiddleware) Name() string { return "tracing" }

func (m *TracingMiddleware) New() ai.Middleware {
    return &TracingMiddleware{exporter: m.exporter}
}

func (m *TracingMiddleware) Model(ctx context.Context, state *ai.ModelState, next ai.ModelNext) (*ai.ModelResponse, error) {
    ctx, span := m.exporter.Start(ctx, "model-call")
    defer span.End()
    return next(ctx, state)
}

// Plugin registers middleware with injected dependencies via prototype
func (p *MyPlugin) ListMiddleware(ctx context.Context) ([]*ai.MiddlewareDesc, error) {
    return []*ai.MiddlewareDesc{
        ai.NewMiddleware("Distributed tracing", &TracingMiddleware{exporter: p.exporter}),
    }, nil
}
```

**Usage:**

```go
// Inline: exporter comes from prototype via New()
ai.WithUse(TracingMiddleware{SampleRate: 0.1})

// Dev UI: sends {"name": "tracing", "config": {"sampleRate": 0.1}}
// → configFromJSON calls prototype.New() (preserves exporter), unmarshals config on top
```

## Deprecation of Existing Middleware

The existing `core.Middleware` and `ai.ModelMiddleware` types will be deprecated:

**This release:**
- Deprecate `core.Middleware`, `ai.ModelMiddleware`, and `WithMiddleware()` option
- Refactor internal middleware to use new `Middleware` type:
  - `addAutomaticTelemetry()` → uses Model hook
  - `simulateSystemPrompt()` → uses Model hook
  - `validateSupport()` → uses Model hook
  - `augmentWithContext()` → uses Model hook
  - `DownloadRequestMedia()` → uses Model hook
- New user-facing middleware uses `Middleware` exclusively

**Next major version:**
- Remove deprecated `core.Middleware` and `ai.ModelMiddleware` types

## Implementation Notes

- `New()` called once at start of each `ai.Generate()`
- Tool hooks run in parallel—implementations must be thread-safe if mutating shared state
- Hooks that just pass through can embed `BaseMiddleware` or return `next()` directly
