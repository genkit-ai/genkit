# Genkit Go v2 — Summary of Changes

## New repository

Genkit Go is moving to a dedicated repository: **github.com/genkit-ai/genkit-go**. The current monorepo contains code for all Genkit languages; the new repo will contain only Go code. This change is meant to improve discoverability of Genkit Go through search engines and LLMs, keep issues and PRs focused on Go, and foster a stronger community around the Go SDK specifically.

---

The sections below summarize every breaking change in Genkit Go v2, with a brief rationale and before/after code for each.

---

## Concrete types replace interfaces

v1 defines `Model`, `Tool`, `Embedder`, `Retriever`, `Evaluator`, and `Prompt` as interfaces with unexported implementations. This has two compounding problems. First, the interfaces are bloated — every one carries a `Register()` method alongside domain methods like `Generate()` or `Embed()`, even though users never call `Register()` themselves. Second, because users can never hold the concrete type, every `With*` option function needs a parallel `*Arg` interface (e.g. `ModelArg`, `EmbedderArg`) plus a `*Ref` struct for lazy references. That's three types per concept just to pass one around.

v2 exports the concrete structs under the clean names. The interfaces that remain (in the `api` package, for registry internals) can be thin and focused — just `Name()` plus the domain method — because infrastructure methods like `Register()` live on the concrete struct. The concrete types can still be passed directly to functions like `genkit.RegisterAction()` or `genkit.Handler()` without casting, since Go resolves methods on the concrete type. One shared `Named` interface and one `ActionRef` struct replace the entire family of `*Arg`/`*Ref` types.

**Before:**

```go
type Model interface {
    Name() string
    Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error)
    Register(r api.Registry) // users never call this, but it's in the interface
}

var m Model = ai.DefineModel(r, "my-model", opts, fn)  // returns interface
ai.Generate(ctx, r, ai.WithModel(m))                   // WithModel accepts ModelArg

ref := ai.NewModelRef("google/gemini-2.5-flash", nil)
ai.Generate(ctx, r, ai.WithModel(ref))                 // ModelRef also satisfies ModelArg
```

**After:**

```go
// api.Model interface is just Name() + Generate()
var m *Model = ai.DefineModel(r, "my-model", fn, opts) // returns concrete *Model
ai.Generate(ctx, r, ai.WithModel(m))                   // WithModel accepts Named

ref := ai.NewActionRef("google/gemini-2.5-flash", nil)
ai.Generate(ctx, r, ai.WithModel(ref))                 // ActionRef also satisfies Named

// Register() is a method on the concrete *Model, not on any interface.
// Works naturally with genkit.RegisterAction(), genkit.Handler(), etc.
```

The same pattern applies to `Embedder`, `Retriever`, `Evaluator`, and `Prompt` — each drops its bloated interface and `*Arg`/`*Ref` types in favor of the concrete struct + `Named`/`ActionRef`. The `api` package interfaces stay thin (just the domain contract), while the concrete types carry infrastructure methods without polluting the interface.

---

## Unified Prompt type

v1 has two separate prompt concepts: `Prompt` (an interface, untyped input, returns `*ModelResponse`) and `DataPrompt[In, Out]` (a generic struct, typed input and output). They have different `Execute` signatures and can't be used interchangeably.

v2 merges them into a single `Prompt[In, Out, Stream]` type. Users never specify `Stream` — the constructor determines it. `DefinePrompt[In]` produces text output; `DefineDataPrompt[In, Out]` produces structured output. Both share the same `Execute` and `ExecuteStream` signatures.

**Before:**

```go
p := ai.DefinePrompt(r, "greeting", ...)                  // returns Prompt (interface)
resp, err := p.Execute(ctx, ai.WithInput(myInput))        // untyped input via option, returns *ModelResponse

dp := ai.DefineDataPrompt[MyIn, MyOut](r, "extract", ...) // returns *DataPrompt[MyIn, MyOut]
out, resp, err := dp.Execute(ctx, myInput)                // typed input as param, returns (MyOut, *ModelResponse)
```

**After:**

```go
p := ai.DefinePrompt[MyIn](r, "greeting", ...)            // returns *Prompt[MyIn, string, *ModelResponseChunk]
text, resp, err := p.Execute(ctx, myInput)                // typed input, returns (string, *ModelResponse)

dp := ai.DefineDataPrompt[MyIn, MyOut](r, "extract", ...) // returns *Prompt[MyIn, MyOut, MyOut]
out, resp, err := dp.Execute(ctx, myInput)                // same Execute signature
```

This cuts the API surface in half while keeping strong types for all prompts. The naming mirrors `Generate`/`GenerateData` at the generation level. As part of this change, `WithInput` is removed -- input is a typed parameter on `Execute` and `ExecuteStream` directly rather than an untyped option.

---

## Streaming: callbacks → channels

v1 streams via a callback function (`func(context.Context, S) error`) passed to action implementations. Channels are the idiomatic Go primitive for streaming data between goroutines and compose naturally with `select`, `range`, and cancellation. This also brings consistency with newer v2 types like `BidiAction`, `BidiFlow`, and `AgentFlow`, which are built on channels from the start.

**Before:**

```go
type StreamingFunc[In, Out, Stream any] = func(ctx context.Context, input In, cb StreamCallback[Stream]) (Out, error)
// where StreamCallback[S] = func(context.Context, S) error
```

**After:**

```go
type StreamingFunc[In, Out, Stream any] = func(ctx context.Context, input In, streamCh chan<- Stream) (Out, error)
```

---

## Define\* argument order: name, fn, opts

v1 places options before the function in some APIs. v2 standardizes all `Define*` and `New*` functions to put required arguments first (name, function) and optional config last.

**Before:**

```go
ai.DefineModel(r, "my-model", opts, fn)
ai.DefineEmbedder(r, "my-embedder", opts, fn)
```

**After:**

```go
ai.DefineModel(r, "my-model", fn, opts)
ai.DefineEmbedder(r, "my-embedder", fn, opts)
```

---

## Action creation functions: required params only, ActionOptions for the rest

v1's action creation functions (`NewAction`, `NewStreamingAction`, etc.) accept a mix of required and optional parameters as top-level function arguments. This makes signatures long and inconsistent -- some functions take metadata, custom schemas, and other optional fields directly, while others omit them.

v2 narrows top-level parameters to only the required ones (name and function). Everything else -- `Metadata`, `InputSchema`, `OutputSchema`, `StreamSchema` -- moves into an `ActionOptions` struct passed as the final argument. Schema fields in `ActionOptions` override the schemas the framework derives from the type parameters, giving plugin authors an escape hatch for types whose default JSON schema reflection is insufficient without cluttering the common path.

**Before:**

```go
action := core.NewAction(name, atype, metadata, inputSchema, fn)
streaming := core.NewStreamingAction(name, atype, metadata, inputSchema, fn)
```

**After:**

```go
action := core.NewAction(name, atype, fn, &core.ActionOptions{
    Metadata:     map[string]any{"label": "my action"},
    InputSchema:  customInputSchema,   // overrides schema derived from In type param
    OutputSchema: customOutputSchema,  // overrides schema derived from Out type param
})

// common case -- no options needed
action := core.NewAction(name, atype, fn, nil)
```

This applies uniformly to all action creation functions (`NewAction`, `NewStreamingAction`, and any higher-level wrappers that create actions internally).

`NewBackgroundAction` follows the same pattern. The three lifecycle functions (`startFn`, `checkFn`, `cancelFn`) remain top-level parameters since they define the action's behavioral contract, even though `cancelFn` is optional (nil when cancellation is not supported). Metadata moves into the options struct.

**Before:**

```go
core.NewBackgroundAction(name, atype, metadata, startFn, checkFn, cancelFn)
```

**After:**

```go
core.NewBackgroundAction(name, atype, startFn, checkFn, cancelFn, &core.ActionOptions{
    Metadata: map[string]any{"supports": "polling"},
})
```

Separately, `LookupAction` is removed. v1 has both `LookupAction` (registry only) and `ResolveAction` (registry first, then dynamic plugin resolution). v2 keeps only `ResolveAction` as the single lookup path, so callers never accidentally miss dynamically-resolved actions.

---

## Typed config for models, embedders, and evaluators

`DefineModel`, `DefineEmbedder`, `DefineEvaluator`, and `DefineBatchEvaluator` gain a `Config` type parameter. The JSON schema for the config is inferred from the type automatically, and the framework deserializes the config before calling the plugin function. This eliminates manual `ConfigSchema` declaration and the `configFromRequest()` boilerplate every plugin currently has. It also prevents accidentally passing one provider's config type to another -- only the exact `Config` type or `map[string]any` (from the dev UI / JSON callers) are accepted; mismatched struct types are rejected.

`ConfigSchema` remains in each options struct as an optional override for types whose default JSON schema reflection is insufficient (e.g. third-party types with custom wrapper generics like `Opt[float64]`).

**Before:**

```go
// DefineModel is not generic. Plugin manually specifies ConfigSchema and
// deserializes config from req.Config (which is `any`) in every implementation.
func DefineModel(g *Genkit, name string, fn ModelFunc, opts *ModelOptions) *Model

type ModelFunc = func(context.Context, *ModelRequest, ModelStreamCallback) (*ModelResponse, error)

genkit.DefineModel(g, "my-model", func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
    cfg, err := configFromRequest(req) // type-switch on any, MapToStruct, etc.
    // ...
}, &ModelOptions{
    ConfigSchema: configToMap(&MyConfig{}),
})
```

**After:**

```go
// DefineModel gains a Config type parameter. Config is inferred from the fn
// signature -- no need to specify it explicitly. The framework deserializes
// req.Config into the typed Config before calling fn.
func DefineModel[Config any](g *Genkit, name string, fn ModelFunc[Config], opts *ModelOptions) *Model

type ModelFunc[Config any] = func(context.Context, *ModelRequest, Config, ModelStreamCallback) (*ModelResponse, error)

genkit.DefineModel(g, "my-model", func(ctx context.Context, req *ModelRequest, cfg MyConfig, cb ModelStreamCallback) (*ModelResponse, error) {
    // cfg is already MyConfig -- no manual deserialization
    // ...
}, &ModelOptions{})
```

The same pattern applies to `DefineEmbedder` (`EmbedderFunc[Config]`), `DefineEvaluator` (`EvaluatorFunc[Config]`), `DefineBatchEvaluator` (`BatchEvaluatorFunc[Config]`), and `DefineBackgroundModel` (which embeds `ModelOptions`). The serialization structs (`ModelRequest.Config`, `EmbedRequest.Config`, etc.) remain `any` -- the typed deserialization happens in the wrapper that `Define*` generates.

---

## Field naming cleanup

v1 has inconsistent field names across generated and hand-written types. v2 fixes these to follow Go conventions.

- **`.Options` to `.Config`**: Fields like `EmbedRequest.Options`, `EvaluatorCallbackRequest.Options`, and `EvaluatorRequest.Options` are renamed to `.Config` for consistency with `ModelRequest.Config`.
- **`Id` to `ID`**: Generated types use `Id` (e.g. `EvaluationId`, `SpanId`, `TraceId`) which violates Go naming conventions. v2 renames these to `ID` (e.g. `EvaluationID`, `SpanID`, `TraceID`).
- **`Url` to `URL`**, **`Api` to `API`**, and similar acronym fields follow the same pattern.

This is a mechanical pass across generated types in `gen.go` and any hand-written types that drifted from convention.

---

## Sealed Registry interface

v1's `Registry` is a public interface that anyone can implement. Adding a method to it is a breaking change. v2 adds an unexported `sealed()` method, preventing external implementations while allowing the interface to grow safely.

**Before:**

```go
type Registry interface {
    RegisterAction(key string, action Action)
    LookupAction(key string) Action
    // ...
}
```

**After:**

```go
type Registry interface {
    RegisterAction(typ ActionType, a Action)
    ResolveAction(ctx context.Context, typ ActionType, name string) (Action, error)
    // ...
    sealed() // prevents external implementations
}
```

---

## Remove WithOutputType / WithOutputSchema

v1 allows specifying structured output types via `WithOutputType(val)` or `WithOutputSchema(schema)` as generate options. This creates a class of runtime type errors — the schema and the variable you unmarshal into can easily drift apart. v2 removes both in favor of `GenerateData[Out]` and `DefineDataPrompt[In, Out]`, where the output type is a compile-time generic parameter.

**Before:**

```go
resp, err := ai.Generate(ctx, r, ai.WithOutputType(MyStruct{}), ...)
var out MyStruct
resp.Output(&out)
```

**After:**

```go
out, resp, err := ai.GenerateData[MyStruct](ctx, r, ...)
// out is already typed -- no separate unmarshal step
```

---

## GenerateData returns value, not pointer

v1's `GenerateData[Out]` returns `(*Out, *ModelResponse, error)`. On error or empty output, the caller gets a nil pointer that's easy to dereference without checking. v2 returns `(Out, *ModelResponse, error)` — zero value on error.

**Before:**

```go
out, resp, err := ai.GenerateData[MyStruct](ctx, r, ...)
// out is *MyStruct -- nil on error, easy to panic
```

**After:**

```go
out, resp, err := ai.GenerateData[MyStruct](ctx, r, ...)
// out is MyStruct -- zero value on error, no nil pointer
```

---

## New tool APIs

v1 introduced experimental tool APIs in `ai/x` ([PR #4797](https://github.com/firebase/genkit/pull/4797)) that simplify tool definitions, add typed interrupt/resume, and provide runtime helpers for multipart responses and streaming progress. These APIs are built on top of the existing v1 tool infrastructure. v2 promotes them to the standard tool APIs and replaces the v1 internals underneath.

The main changes from v1's `DefineTool` / `DefineMultipartTool` / `ToolContext` / untyped interrupt API:

- **`ToolContext` removed.** v1 wraps `context.Context` in a `ToolContext` struct solely to carry interrupt/resume metadata, forcing every tool function to accept a non-standard context type. Tool functions now receive a plain `context.Context`. Interrupt/resume data and other tool-specific state are accessed through typed helpers in the `tool` package (see below).
- **Unified `DefineTool`** replaces both `DefineTool` and `DefineMultipartTool`. Tool functions return `(Out, error)`. Multipart responses (images, media) are attached via `tool.AttachParts(ctx, parts...)` instead of requiring a separate function signature.
- **`DefineInterruptibleTool`** replaces the untyped `Restart`/`Respond`/`Interrupt` pattern with a typed `*Res` parameter. When the tool is called normally, `*Res` is nil; when resumed after an interrupt, it carries the caller's typed resume data.
- **`tool` runtime helper package** provides functions for use inside tool bodies: `tool.Interrupt(data)` to pause execution, `tool.AttachParts(ctx, ...)` for multipart content, `tool.SendPartial(ctx, data)` for streaming progress updates during execution, and `tool.OriginalInput[In](ctx)` for accessing the pre-restart input.
- **Typed interrupt handling on the caller side** via `tool.InterruptAs[T](part)` for extracting interrupt metadata, and `tool.Resume[Res](part, data)` / `tool.Respond(part, output)` for building restart/response parts. The tool definition itself also has `.Resume()` and `.Respond()` methods that validate the part belongs to that tool.
- **`GenerateActionResume` fields use `[]*Part` directly.** v1's `GenerateActionResume.Restart` and `.Respond` fields are typed as `[]*toolRequestPart` and `[]*toolResponsePart` -- unexported wrapper structs that exist because Go lacks union types and the original implementation modeled tool requests and responses as separate types rather than using `Part` directly. v2 replaces both with `[]*Part`, matching the wire format and aligning with how `tool.Resume` and `tool.Respond` already return `*Part`. The fields are also renamed to `Restarts` and `Responses` to reflect that they are plural collections.
- **Partial tool responses** let tools stream progress updates to the client during execution via `tool.SendPartial(ctx, data)`. Callers distinguish progress from final results using `Part.IsPartial()`.

**Before:**

```go
weatherTool := ai.DefineTool(r, "getWeather", "Fetches the weather",
    func(ctx *ai.ToolContext, input WeatherInput) (string, error) { ... })

multipartTool := ai.DefineMultipartTool(r, "screenshot", "Takes a screenshot",
    func(ctx *ai.ToolContext, input ScreenshotInput) (*ai.MultipartToolResponse, error) { ... })

tool.Restart(toolReq, &RestartOptions{ResumedMetadata: rawMeta}) // untyped
```

**After:**

```go
weatherTool := genkit.DefineTool(g, "getWeather", "Fetches the weather",
    func(ctx context.Context, input WeatherInput) (string, error) {
        return "Sunny, 25°C", nil
    },
)

// Multipart via AttachParts instead of a separate function signature
screenshotTool := genkit.DefineTool(g, "screenshot", "Takes a screenshot",
    func(ctx context.Context, input ScreenshotInput) (string, error) {
        img := takeScreenshot(input.URL)
        tool.AttachParts(ctx, ai.NewMediaPart("image/png", img))
        return "Screenshot captured", nil
    },
)

// Typed interrupt/resume via DefineInterruptibleTool
transferTool := genkit.DefineInterruptibleTool(g, "transfer", "Transfers money",
    func(ctx context.Context, input TransferInput, confirm *Confirmation) (string, error) {
        if confirm != nil && !confirm.Approved {
            return "cancelled", nil
        }
        if confirm == nil && input.Amount > 100 {
            return "", tool.Interrupt(TransferInterrupt{Reason: "large_amount", Amount: input.Amount})
        }
        return "completed", nil
    },
)

// Caller-side: typed resume
restart, _ := transferTool.Resume(interrupt, Confirmation{Approved: true})
resp, _ = genkit.Generate(ctx, g,
    ai.WithMessages(resp.History()...),
    ai.WithTools(transferTool),
    ai.WithToolRestarts(restart),
)
```

See [PR #4797](https://github.com/firebase/genkit/pull/4797) for the full API reference, agent flow integration, and additional examples.

---

## Merge FormatHandler and StreamingFormatHandler

v1 has two separate interfaces for format handlers: `FormatHandler` (with `ParseMessage`) and `StreamingFormatHandler` (with `ParseOutput`/`ParseChunk`). This split existed to avoid a breaking change in v1. v2 merges them into a single `FormatHandler` with four methods and drops the legacy `ParseMessage`.

**Before:**

```go
type FormatHandler interface {
    ParseMessage(message *Message) (*Message, error)
    Instructions() string
    Config() ModelOutputConfig
}
type StreamingFormatHandler interface {
    ParseOutput(message *Message) (any, error)
    ParseChunk(chunk *ModelResponseChunk) (any, error)
}
```

**After:**

```go
type FormatHandler interface {
    Instructions() string
    Config() ModelOutputConfig
    ParseOutput(message *Message) (any, error)
    ParseChunk(chunk *ModelResponseChunk) (any, error)
}
```

---

## DefineFormat drops redundant name parameter

v1's `DefineFormat` takes a `name` string and a `Formatter`, but `Formatter` already has a `Name()` method. The name parameter is redundant and a source of potential mismatch. v2 drops it.

**Before:**

```go
func DefineFormat(r api.Registry, name string, formatter Formatter)
```

**After:**

```go
func DefineFormat(r api.Registry, formatter Formatter)
// name comes from formatter.Name()
```

---

## Action.RunJSON always returns telemetry

v1 has `RunJSON` (discards telemetry) and `RunJSONWithTelemetry` (returns it). There's no reason to discard telemetry — callers who don't need it can ignore the extra fields. v2 removes `RunJSONWithTelemetry` and makes `RunJSON` always return `*api.ActionRunResult` containing both the output and trace metadata.

**Before:**

```go
out, err := action.RunJSON(ctx, input, cb)                 // json.RawMessage
result, err := action.RunJSONWithTelemetry(ctx, input, cb) // *ActionRunResult[json.RawMessage]
```

**After:**

```go
result, err := action.RunJSON(ctx, input, streamCh)        // *ActionRunResult[json.RawMessage] (always)
```

---

## ToolArg replaces ToolRef

v1 uses `ToolRef` for passing tool references. v2 introduces a sealed `ToolArg` interface that both `*Tool[In, Out]` and `ToolNamed` (a string/glob type) satisfy. This lets `WithTools` accept concrete tools, exact names, and glob patterns (e.g. `"mcp/*"`) through one type.

**Before:**

```go
ai.Generate(ctx, r, ai.WithTools(myTool, ai.ToolRef("other-tool")))
```

**After:**

```go
ai.Generate(ctx, r, ai.WithTools(myTool, ai.ToolNamed("other-tool"), ai.ToolNamed("mcp/*")))
```

---

## genkit.Init and plugin functions return errors

v1's `genkit.Init()` returns only a `*Genkit` instance. Plugin functions like `Init()` and `ListActions()` either panic or silently swallow errors. This is wrong for two reasons: any function that accepts a `context.Context` can perform fallible work (network requests, credential validation), and the Go convention is that such functions return an `error`. Panicking when a user forgets an environment variable is a poor experience — it should be a normal error the caller can handle.

v2 adds `error` to the return value of `genkit.Init()` and all plugin functions that take a `context.Context`. This makes the SDK more idiomatic and gives callers control over error handling instead of crashing the process.

**Before:**

```go
g := genkit.Init(ctx, opts)              // panics on failure
googleai.Init(ctx, &googleai.Config{})   // panics or silently fails
```

**After:**

```go
g, err := genkit.Init(ctx, opts)         // returns error
err := googleai.Init(ctx, &googleai.Config{}) // returns error
```

---

## Unified error type with sentinel causes

v1 puts status codes, HTTP mappings, and error types (`GenkitError`, `UserFacingError`) directly in the `core` package. v2 moves them into a dedicated `status` package so that error handling has a clear home and `core` stays focused on bigger picture concepts.

v1 also has two unrelated error types: `GenkitError` (internal, captures stack traces) and `UserFacingError` (safe to return over HTTP). They share no interface, so error-handling code must check for both via separate `errors.As` calls. Neither type supports wrapping a cause, so callers cannot use `errors.Is` to distinguish specific failure modes -- the only option is string-matching on error messages.

v2 unifies `GenkitError` and `UserFacingError` into a single `GenkitError` type with a `Public` field that controls whether the message is safe to return to clients. `NewError` and `NewPublicError` remain as convenience constructors that set this field accordingly.

The package defines sentinel errors that each carry a fixed status code. The sentinel is the sole parameter to `Errorf` -- no separate status code argument. A base sentinel exists for every status code (`ErrInvalidArgument`, `ErrAborted`, `ErrInternal`, etc.) for cases where no more specific sentinel applies. Domain-specific sentinels like `ErrMaxTurnsExceeded` carry their own status code (e.g. `ABORTED`) and can optionally wrap the base sentinel so that `errors.Is` matches at both levels of granularity.

To add context without reclassifying, use plain `fmt.Errorf` with `%w` -- the sentinel and status survive unwrapping. To intentionally reclassify an error at a boundary, wrap it with `status.Errorf` using a different sentinel. When multiple `GenkitError` wrappers are in the chain, `errors.As` finds the outermost one, so the outermost status code is what the HTTP boundary uses.

**Sentinel errors (after):** each carries a fixed status code.

```go
var (
    ErrNotFound             = status.NewSentinel(NOT_FOUND, "not found")
    ErrInvalidArgument      = status.NewSentinel(INVALID_ARGUMENT, "invalid argument")
    ErrAborted              = status.NewSentinel(ABORTED, "aborted")
    ErrMaxTurnsExceeded     = status.NewSentinel(ABORTED, "max turns exceeded") // wraps ErrAborted
    ErrToolInterrupted      = status.NewSentinel(ABORTED, "tool interrupted")  // wraps ErrAborted
    ErrOutputSchemaMismatch = status.NewSentinel(INTERNAL, "output does not match schema")
    // ...
)
```

**Before:**

```go
err := core.NewError(core.ABORTED, "exceeded maximum tool call iterations (%d)", maxTurns)
if strings.Contains(err.Error(), "exceeded maximum tool call iterations") { ... }
```

**After:**

```go
err := status.Errorf(status.ErrMaxTurnsExceeded, "exceeded max tool call iterations (%d)", maxTurns)
if errors.Is(err, status.ErrMaxTurnsExceeded) { ... } // specific match
if errors.Is(err, status.ErrAborted) { ... }          // broad match (via wrapping)

// Public flag replaces UserFacingError
err := status.PublicErrorf(status.ErrInvalidArgument, "invalid query parameter: %s", param)
// err.Public == true; HTTP handler knows it's safe to return the message

// Adding context without reclassifying -- use fmt.Errorf
return fmt.Errorf("agent %q: %w", name, err) // sentinel and status survive

// Intentional reclassification at a boundary -- use status.Errorf
return status.Errorf(status.ErrInternal, "agent %q failed: %w", name, err)
```

---

## Remove ModelSupports

v1 requires plugins to declare a `ModelSupports` struct listing the capabilities their model supports (multiturn, media, tool calls, system role, constrained output, etc.). The framework uses this declaration to run middleware that compensates for missing capabilities -- for example, merging system messages into user messages when `SystemRole` is false, or simulating constrained output when `Constrained` is `ConstrainedNone`.

v2 removes `ModelSupports`, `Stage`, and `Versions` from `ModelOptions` and drops the associated middleware. Plugins are expected to support all features of the Genkit model contract. If a plugin receives a request it genuinely cannot fulfill (e.g. a model that has no concept of tool calling receives tool definitions), it should return a clear error so the developer knows the limitation up front rather than getting silently degraded behavior.

The one exception is `Constrained`, which is promoted to a top-level field on `ModelOptions`. Unlike the other `ModelSupports` fields, constrained output is not a feature the middleware fakes. It is a strategy selector: the framework must choose between sending a schema for native enforcement or injecting format instructions into the prompt *before* building the request. This decision depends on the model (and on whether tools are present in the request), so the framework needs the hint up front.

**Before:**

```go
ai.DefineModel(g, "my-model", fn, &ai.ModelOptions{
    Stage:    ai.ModelStageStable,
    Versions: []string{"v1", "v2"},
    Supports: &ai.ModelSupports{
        Multiturn:   true,
        Tools:       true,
        SystemRole:  false, // framework will merge system messages into user messages
        Media:       false,
        Constrained: ai.ConstrainedNoTools,
    },
})
```

**After:**

```go
ai.DefineModel(g, "my-model", fn, &ai.ModelOptions{
    Constrained: ai.ConstrainedNoTools, // strategy hint for structured output
    // Other capabilities, stage, and versions are no longer declared.
    // Plugins handle unsupported features by returning errors and manage
    // their own deprecation warnings and version validation.
})
```

---

## Full removed/replaced type reference

| Removed | Replacement |
|---------|-------------|
| `Model` interface | `*Model` struct |
| `ModelArg`, `ModelRef` | `Named`, `ActionRef` |
| `BackgroundModel` interface | `*BackgroundModel` struct |
| `Tool` interface | `*Tool[In, Out]` struct |
| `ToolDef[In, Out]` | `*Tool[In, Out]` (renamed) |
| `ToolRef` | `ToolArg` sealed interface, `ToolNamed` |
| `ToolContext` | `context.Context` + helpers |
| `Embedder` interface | `*Embedder` struct |
| `EmbedderArg`, `EmbedderRef` | `Named`, `ActionRef` |
| `Retriever` interface | `*Retriever` struct |
| `RetrieverArg`, `RetrieverRef` | `Named`, `ActionRef` |
| `Evaluator` interface | `*Evaluator` struct |
| `EvaluatorArg`, `EvaluatorRef` | `Named`, `ActionRef` |
| `Prompt` interface | `*Prompt[In, Out, Stream]` struct |
| `DataPrompt[In, Out]` | `*Prompt[In, Out, Out]` via `DefineDataPrompt` |
| `StreamCallback[S]` | `chan<- S` |
| `RunJSONWithTelemetry` | `RunJSON` (always returns telemetry) |
| `WithOutputType`, `WithOutputSchema` | `GenerateData[Out]`, `DefineDataPrompt[In, Out]` |
| `WithInput` (prompt option) | Typed `input In` parameter |
| `FormatHandler.ParseMessage` | Removed |
| `StreamingFormatHandler` | Merged into `FormatHandler` |
| `DefineTool`, `DefineMultipartTool` | `DefineTool` (unified), `DefineInterruptibleTool` |
| `Interrupt()`, `InterruptOptions` | `tool.Interrupt(data)` |
| `OriginalInput()` | `tool.OriginalInput[In](ctx)` |
| `Respond`, `Restart` (untyped) | `tool.Resume[Res]`, `tool.Respond`, or typed methods on `*InterruptibleTool` |
| `GenerateActionResume.Restart` / `.Respond` (`[]*toolRequestPart` / `[]*toolResponsePart`) | `.Restarts` / `.Responses` (`[]*Part`) |
| `ModelFunc` (untyped) | `ModelFunc[Config]` (typed config param) |
| `EmbedderFunc` (untyped) | `EmbedderFunc[Config]` (typed config param) |
| `EvaluatorFunc` (untyped) | `EvaluatorFunc[Config]` (typed config param) |
| `BatchEvaluatorFunc` (untyped) | `BatchEvaluatorFunc[Config]` (typed config param) |
| `configFromRequest()` per plugin | Removed; framework handles deserialization |
| `NewAction(name, atype, metadata, inputSchema, fn)` | `NewAction(name, atype, fn, *ActionOptions)` |
| `NewStreamingAction(name, atype, metadata, inputSchema, fn)` | `NewStreamingAction(name, atype, fn, *ActionOptions)` |
| `NewBackgroundAction(name, atype, metadata, startFn, checkFn, cancelFn)` | `NewBackgroundAction(name, atype, startFn, checkFn, cancelFn, *ActionOptions)` |
| `LookupAction` | Removed; use `ResolveAction` |
| `UserFacingError` | `GenkitError` with `Public: true` |
| `NewPublicError(status, msg, details)` | `PublicErrorf(sentinel, msg, args...)` sets `Public: true`; status derived from sentinel |
| `NewError(status, msg, args...)` | `Errorf(sentinel, msg, args...)` status derived from sentinel |
| `ModelSupports` | Removed; `Constrained` promoted to `ModelOptions.Constrained` |
| `ModelOptions.Supports` | Removed; only `Constrained` field survives (moved up) |
| `ModelOptions.Stage`, `ModelStage` | Removed; plugins manage their own deprecation warnings |
| `ModelOptions.Versions` | Removed; plugins validate versions in their model function |

## New types

| New | Purpose |
|-----|---------|
| `Named` | Shared interface for all `With*` options that accept a model, embedder, retriever, or evaluator |
| `ActionRef` | Unified lazy reference replacing `ModelRef`, `EmbedderRef`, `RetrieverRef`, `EvaluatorRef` |
| `ToolArg` | Sealed interface for `WithTools` — distinguishes tool references from other named things |
| `ToolNamed` | String/glob tool reference satisfying `ToolArg` |
| `InterruptibleTool[In, Out, Res]` | Tool with typed interrupt/resume support |
| `tool` package | Runtime helpers for tool bodies: `Interrupt`, `AttachParts`, `SendPartial`, `Resume`, `Respond`, etc. |
| `ActionOptions` | Optional config for action creation: `Metadata`, `InputSchema`, `OutputSchema`, `StreamSchema` |
| `ErrNotFound`, `ErrMaxTurnsExceeded`, `ErrOutputSchemaMismatch`, ... | Sentinel errors for `errors.Is` matching on specific failure modes |
