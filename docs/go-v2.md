# Genkit Go v2 — API Changes

## Part 1: New API Surface

### Core Types (core package)

```go
// Function types — CHANGED: StreamingFunc uses chan instead of callback
type Func[In, Out any] = func(ctx context.Context, input In) (Out, error)
type StreamingFunc[In, Out, Stream any] = func(ctx context.Context, input In, streamCh chan<- Stream) (Out, error)

// Action definition — CHANGED: options struct, channel-based streaming, RunJSON returns telemetry.
type Action[In, Out, Stream any] struct{ /* unexported */ }

type ActionOptions struct {
    InputSchema  map[string]any
    OutputSchema map[string]any
    StreamSchema map[string]any
    Metadata     map[string]any
}

func NewAction[In, Out any](atype api.ActionType, name string, fn Func[In, Out], opts *ActionOptions) *Action[In, Out, struct{}]
func NewStreamingAction[In, Out, Stream any](atype api.ActionType, name string, fn StreamingFunc[In, Out, Stream], opts *ActionOptions) *Action[In, Out, Stream]
func DefineAction[In, Out any](r api.Registry, atype api.ActionType, name string, fn Func[In, Out], opts *ActionOptions) *Action[In, Out, struct{}]
func DefineStreamingAction[In, Out, Stream any](r api.Registry, atype api.ActionType, name string, fn StreamingFunc[In, Out, Stream], opts *ActionOptions) *Action[In, Out, Stream]
func LookupActionFor[In, Out, Stream any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, Stream]

// Action methods
func (a *Action[In, Out, Stream]) Name() string
func (a *Action[In, Out, Stream]) Desc() string
func (a *Action[In, Out, Stream]) Run(ctx context.Context, input In, streamCh chan<- Stream) (Out, error)
func (a *Action[In, Out, Stream]) RunJSON(ctx context.Context, input json.RawMessage, streamCh chan<- json.RawMessage) (*api.ActionRunResult[json.RawMessage], error)

// Background action — unchanged
type Operation[Out any] struct {
    Action   string
    ID       string
    Done     bool
    Output   Out
    Error    error
    Metadata map[string]any
}

// Schema — unchanged
func DefineSchema(r api.Registry, name string, schema map[string]any)
func DefineSchemaFor[T any](r api.Registry)
func SchemaRef(name string) map[string]any
func ResolveSchema(r api.Registry, schema map[string]any) (map[string]any, error)
func InferSchemaMap(value any) map[string]any

// Flow step — unchanged
func Run[Out any](ctx context.Context, name string, fn func() (Out, error)) (Out, error)
```

### Registry (api package)

```go
// CHANGED: sealed to allow adding methods without breaking changes.
type Registry interface {
    Name() string
    RegisterAction(typ ActionType, a Action)
    LookupAction(typ ActionType, name string) Action
    // ... other methods
    sealed()
}
```

### Named and ActionRef

```go
// NEW: shared interface for With* option functions.
type Named interface {
    Name() string
}

// NEW: unified lazy reference replacing ModelRef, EmbedderRef, RetrieverRef.
type ActionRef struct {
    name   string
    config any
}

func NewActionRef(name string, config any) ActionRef
func (r ActionRef) Name() string
func (r ActionRef) Config() any
```

### Flow

```go
type Flow[In, Out, Stream any] struct{ /* unexported */ }

func NewFlow[In, Out any](name string, fn core.Func[In, Out]) *Flow[In, Out, struct{}]
func DefineFlow[In, Out any](r api.Registry, name string, fn core.Func[In, Out]) *Flow[In, Out, struct{}]
func NewStreamingFlow[In, Out, Stream any](name string, fn core.StreamingFunc[In, Out, Stream]) *Flow[In, Out, Stream]
func DefineStreamingFlow[In, Out, Stream any](r api.Registry, name string, fn core.StreamingFunc[In, Out, Stream]) *Flow[In, Out, Stream]

func (f *Flow[In, Out, Stream]) Name() string
func (f *Flow[In, Out, Stream]) Run(ctx context.Context, input In) (Out, error)
func (f *Flow[In, Out, Stream]) Stream(ctx context.Context, input In) iter.Seq2[*FlowStreamValue[Out, Stream], error]

// Unchanged
type FlowStreamValue[Out, Stream any] struct {
    Done   bool
    Output Out    // valid if Done is true
    Stream Stream // valid if Done is false
}
```

### Model

```go
// CHANGED: concrete struct, not interface. Arg order: name, fn, opts.
type Model struct{ /* unexported */ }

func NewModel(name string, fn ModelFunc, opts *ModelOptions) *Model
func DefineModel(r api.Registry, name string, fn ModelFunc, opts *ModelOptions) *Model
func LookupModel(r api.Registry, name string) *Model

func (m *Model) Name() string
func (m *Model) Generate(ctx context.Context, req *ModelRequest, streamCh chan<- *ModelResponseChunk) (*ModelResponse, error)

type ModelFunc = core.StreamingFunc[*ModelRequest, *ModelResponse, *ModelResponseChunk]
```

#### Model Types (unchanged)

```go
type ModelOptions struct {
    Label    string
    Supports *ModelSupports
    Info     *ModelInfo
    // ...
}

type ModelRequest struct {
    Messages []*Message
    Config   any
    Tools    []*ToolDefinition
    Output   *ModelOutputConfig
    // ...
}

type ModelResponse struct{ /* fields */ }
func (mr *ModelResponse) Text() string
func (mr *ModelResponse) History() []*Message
func (mr *ModelResponse) Reasoning() string
func (mr *ModelResponse) Output(v any) error
func (mr *ModelResponse) ToolRequests() []*ToolRequest
func (mr *ModelResponse) Interrupts() []*Part
func (mr *ModelResponse) Media() string

type ModelResponseChunk struct{ /* fields */ }
func (c *ModelResponseChunk) Text() string
func (c *ModelResponseChunk) Reasoning() string
func (c *ModelResponseChunk) Output(v any) error

type ModelInfo struct{ /* fields */ }
type ModelSupports struct{ /* fields */ }
type ModelOutputConfig struct{ /* fields */ }
type GenerationCommonConfig struct{ /* fields */ }
type GenerationUsage struct{ /* fields */ }
type FinishReason string
type ConstrainedSupport int
type ModelMiddleware = core.Middleware[*ModelRequest, *ModelResponse, *ModelResponseChunk]
```

### Background Model

```go
// CHANGED: concrete struct, not interface. Arg order: name, fns, opts.
type BackgroundModel struct{ /* unexported */ }

func NewBackgroundModel(name string, startFn StartModelOpFunc, checkFn CheckModelOpFunc, opts *BackgroundModelOptions) *BackgroundModel
func DefineBackgroundModel(r api.Registry, name string, startFn StartModelOpFunc, checkFn CheckModelOpFunc, opts *BackgroundModelOptions) *BackgroundModel
func LookupBackgroundModel(r api.Registry, name string) *BackgroundModel

func (m *BackgroundModel) Name() string
func (m *BackgroundModel) Start(ctx context.Context, req *ModelRequest) (*ModelOperation, error)
func (m *BackgroundModel) Check(ctx context.Context, op *ModelOperation) (*ModelOperation, error)
func (m *BackgroundModel) Cancel(ctx context.Context, op *ModelOperation) (*ModelOperation, error)
func (m *BackgroundModel) SupportsCancel() bool

// Unchanged
type ModelOperation = core.Operation[*ModelResponse]
type StartModelOpFunc = func(ctx context.Context, req *ModelRequest) (*ModelOperation, error)
type CheckModelOpFunc = func(ctx context.Context, op *ModelOperation) (*ModelOperation, error)
type CancelModelOpFunc = func(ctx context.Context, op *ModelOperation) (*ModelOperation, error)

type BackgroundModelOptions struct {
    ModelOptions
    Cancel   CancelModelOpFunc
    Metadata map[string]any
}

// Top-level functions — unchanged
func GenerateOperation(ctx context.Context, r api.Registry, opts ...GenerateOption) (*ModelOperation, error)
func CheckModelOperation(ctx context.Context, r api.Registry, op *ModelOperation) (*ModelOperation, error)
```

### Message and Part (unchanged)

```go
type Message struct {
    Role     Role
    Content  []*Part
    Metadata map[string]any
}
func (m *Message) Text() string

type Role string // "system", "user", "model", "tool"

type Part struct {
    Kind         PartKind
    Text         string
    Media        *Media
    ToolRequest  *ToolRequest
    ToolResponse *ToolResponse
    Data         string
    Metadata     map[string]any
    // ...
}

type PartKind int
type Media struct {
    ContentType string
    URL         string
}

type ToolRequest struct{ /* fields */ }
type ToolResponse struct{ /* fields */ }

type Document struct {
    Content  []*Part
    Metadata map[string]any
}
```

### Tool

```go
// CHANGED: concrete struct, context.Context instead of ToolContext.
type ToolFunc[In, Out any] = func(ctx context.Context, input In) (Out, error)
type MultipartToolFunc[In any] = func(ctx context.Context, input In) (*MultipartToolResponse, error)

type Tool[In, Out any] struct{ /* unexported */ }

func NewTool[In, Out any](name, desc string, fn ToolFunc[In, Out], opts ...ToolOption) *Tool[In, Out]
func DefineTool[In, Out any](r api.Registry, name, desc string, fn ToolFunc[In, Out], opts ...ToolOption) *Tool[In, Out]
func NewMultipartTool[In any](name, desc string, fn MultipartToolFunc[In], opts ...ToolOption) *Tool[In, *MultipartToolResponse]
func DefineMultipartTool[In any](r api.Registry, name, desc string, fn MultipartToolFunc[In], opts ...ToolOption) *Tool[In, *MultipartToolResponse]
func LookupTool[In, Out any](r api.Registry, name string) *Tool[In, Out]

func (t *Tool[In, Out]) Name() string
func (t *Tool[In, Out]) Definition() *ToolDefinition
func (t *Tool[In, Out]) Run(ctx context.Context, input In) (Out, error)
func (t *Tool[In, Out]) RunRaw(ctx context.Context, input any) (any, error)
func (t *Tool[In, Out]) RespondWith(toolReq *Part, output Out, opts ...RespondWithOption[Out]) (*Part, error)
func (t *Tool[In, Out]) RestartWith(toolReq *Part, opts ...RestartWithOption[In]) (*Part, error)
func (t *Tool[In, Out]) IsMultipart() bool
```

#### ToolArg (NEW, sealed, for WithTools)

```go
type ToolArg interface {
    Name() string
    isToolArg()
}

// ToolNamed returns a ToolArg that matches registered tools by exact name
// or glob pattern (e.g. "mcp/*"). Exact names resolve to one tool; patterns
// may match multiple. Logs a warning if a glob pattern matches zero tools.
type ToolNamed string

func (t ToolNamed) Name() string { return string(t) }
func (t ToolNamed) isToolArg()   {}
func (t *Tool[In, Out]) isToolArg() {}
```

#### Tool Context Helpers (NEW, replace ToolContext)

```go
func IsResumed(ctx context.Context) bool
func ResumedMetadata[T any](ctx context.Context) (T, bool)
func OriginalInputAs[T any](ctx context.Context) (T, error)
func InterruptWith[T any](ctx context.Context, meta T) error
func InterruptAs[T any](p *Part) (T, bool)                        // unchanged
func IsToolInterruptError(err error) (bool, map[string]any)        // unchanged
```

#### Tool Types (unchanged unless noted)

```go
type ToolDefinition struct{ /* fields */ }
type ToolConfig struct{ /* fields */ }
type ToolChoice string // "auto", "required", "none"
type MultipartToolResponse struct{ /* fields */ }

// CHANGED: generic
type RestartOptions[In, Meta any] struct {
    ReplaceInput    In
    ResumedMetadata Meta
}

type RespondWithOption[Out any] func(*respondWithConfig[Out])
type RestartWithOption[In any] func(*restartWithConfig[In])

func WithResponseMetadata[Out any](meta map[string]any) RespondWithOption[Out]  // unchanged
func WithNewInput[In any](input In) RestartWithOption[In]                       // unchanged
func WithResumedMetadata[In any](meta any) RestartWithOption[In]                // unchanged
```

### Embedder

```go
// CHANGED: concrete struct. Arg order: name, fn, opts.
type Embedder struct{ /* unexported */ }

func NewEmbedder(name string, fn EmbedderFunc, opts *EmbedderOptions) *Embedder
func DefineEmbedder(r api.Registry, name string, fn EmbedderFunc, opts *EmbedderOptions) *Embedder
func LookupEmbedder(r api.Registry, name string) *Embedder

func (e *Embedder) Name() string
func (e *Embedder) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error)

// Unchanged
type EmbedderFunc = func(context.Context, *EmbedRequest) (*EmbedResponse, error)
type EmbedderOptions struct{ /* fields */ }
type EmbedRequest struct{ /* fields */ }
type EmbedResponse struct{ /* fields */ }
type Embedding struct{ /* fields */ }
```

### Retriever

```go
// CHANGED: concrete struct. Arg order: name, fn, opts.
type Retriever struct{ /* unexported */ }

func NewRetriever(name string, fn RetrieverFunc, opts *RetrieverOptions) *Retriever
func DefineRetriever(r api.Registry, name string, fn RetrieverFunc, opts *RetrieverOptions) *Retriever
func LookupRetriever(r api.Registry, name string) *Retriever

func (r *Retriever) Name() string
func (r *Retriever) Retrieve(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error)

// Unchanged
type RetrieverFunc = func(context.Context, *RetrieverRequest) (*RetrieverResponse, error)
type RetrieverOptions struct{ /* fields */ }
type RetrieverRequest struct{ /* fields */ }
type RetrieverResponse struct{ /* fields */ }
```

### Resource

```go
// CHANGED: concrete struct. Arg order: name, fn, opts.
type Resource struct{ /* unexported */ }

func NewResource(name string, fn ResourceFunc, opts *ResourceOptions) *Resource
func DefineResource(r api.Registry, name string, fn ResourceFunc, opts *ResourceOptions) *Resource
func LookupResource(r api.Registry, name string) *Resource
func FindMatchingResource(r api.Registry, uri string) (*Resource, *ResourceInput, error)

func (r *Resource) Name() string
func (r *Resource) Execute(ctx context.Context, input *ResourceInput) (*ResourceOutput, error)
func (r *Resource) Matches(uri string) bool
func (r *Resource) ExtractVariables(uri string) (map[string]string, error)

// Unchanged
type ResourceFunc = func(context.Context, *ResourceInput) (*ResourceOutput, error)
type ResourceOptions struct{ /* fields */ }
type ResourceInput struct{ /* fields */ }
type ResourceOutput struct{ /* fields */ }
```

### Evaluator

```go
// CHANGED: concrete struct. Arg order: name, fn, opts.
type Evaluator struct{ /* unexported */ }

func NewEvaluator(name string, fn EvaluatorFunc, opts *EvaluatorOptions) *Evaluator
func DefineEvaluator(r api.Registry, name string, fn EvaluatorFunc, opts *EvaluatorOptions) *Evaluator
func NewBatchEvaluator(name string, fn BatchEvaluatorFunc, opts *EvaluatorOptions) *Evaluator
func DefineBatchEvaluator(r api.Registry, name string, fn BatchEvaluatorFunc, opts *EvaluatorOptions) *Evaluator
func LookupEvaluator(r api.Registry, name string) *Evaluator

func (e *Evaluator) Name() string
func (e *Evaluator) Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error)

// Unchanged
type EvaluatorFunc = func(context.Context, *EvaluatorCallbackRequest) (*EvaluatorCallbackResponse, error)
type BatchEvaluatorFunc = func(context.Context, *EvaluatorRequest) (*EvaluatorResponse, error)
type EvaluatorOptions struct{ /* fields */ }
type EvaluatorRequest struct{ /* fields */ }
type EvaluatorResponse struct{ /* fields */ }
type EvaluatorCallbackRequest struct{ /* fields */ }
type EvaluatorCallbackResponse struct{ /* fields */ }
type Example struct{ /* fields */ }
type EvaluationResult struct{ /* fields */ }
type Score struct{ /* fields */ }
type ScoreStatus string
type ScoreDetails struct{ /* fields */ }
```

### Prompt

```go
// CHANGED: unified generic type replacing both Prompt and DataPrompt.
// In = input type, Out = typed output, Stream = streaming chunk type.
// Users never specify Stream directly — constructors determine it.
type Prompt[In, Out, Stream any] struct{ /* unexported */ }

// DefinePrompt creates a text prompt with typed input.
// Out = string, Stream = *ModelResponseChunk.
// For fully untyped input, use DefinePrompt[any].
func DefinePrompt[In any](r api.Registry, name string, opts ...PromptOption) *Prompt[In, string, *ModelResponseChunk]

// DefineDataPrompt creates a structured output prompt with typed input and output.
// Stream = Out (chunks are the same type as the final output).
func DefineDataPrompt[In, Out any](r api.Registry, name string, opts ...PromptOption) *Prompt[In, Out, Out]

// LookupPrompt looks up a registered prompt by name and wraps it as a text prompt.
func LookupPrompt[In any](r api.Registry, name string) *Prompt[In, string, *ModelResponseChunk]

// LookupDataPrompt looks up a registered prompt by name and wraps it with structured output types.
func LookupDataPrompt[In, Out any](r api.Registry, name string) *Prompt[In, Out, Out]

func (p *Prompt[In, Out, Stream]) Name() string
func (p *Prompt[In, Out, Stream]) Execute(ctx context.Context, input In, opts ...PromptExecuteOption) (Out, *ModelResponse, error)
func (p *Prompt[In, Out, Stream]) ExecuteStream(ctx context.Context, input In, opts ...PromptExecuteOption) iter.Seq2[*StreamValue[Out, Stream], error]
func (p *Prompt[In, Out, Stream]) Render(ctx context.Context, input In) (*GenerateActionOptions, error)

// Prompt loading
func LoadPromptDir(r api.Registry, dir string, namespace string)                          // unchanged
func LoadPromptDirFromFS(r api.Registry, fsys fs.FS, dir, namespace string)               // unchanged
func LoadPrompt[In any](r api.Registry, dir, filename, namespace string) *Prompt[In, string, *ModelResponseChunk]
func LoadPromptFromFS[In any](r api.Registry, fsys fs.FS, dir, filename, namespace string) *Prompt[In, string, *ModelResponseChunk]
func LoadPromptFromSource[In any](r api.Registry, source, name, namespace string) (*Prompt[In, string, *ModelResponseChunk], error)

// Unchanged
type PromptFn = func(context.Context) (string, error)
type MessagesFn = func(context.Context) ([]*Message, error)
type GenerateActionOptions struct{ /* fields */ }
```

### Generate

```go
// CHANGED: GenerateData returns value not pointer.
func Generate(ctx context.Context, r api.Registry, opts ...GenerateOption) (*ModelResponse, error)
func GenerateStream(ctx context.Context, r api.Registry, opts ...GenerateOption) iter.Seq2[*ModelStreamValue, error]
func GenerateData[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) (Out, *ModelResponse, error)
func GenerateDataStream[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) iter.Seq2[*StreamValue[Out, Out], error]
func GenerateText(ctx context.Context, r api.Registry, opts ...GenerateOption) (string, error)    // unchanged

// Unchanged
type StreamValue[Out, Stream any] struct {
    Done     bool
    Chunk    Stream         // valid if Done is false
    Output   Out            // valid if Done is true
    Response *ModelResponse // valid if Done is true
}
type ModelStreamValue = StreamValue[*ModelResponse, *ModelResponseChunk]
```

### Generate Options

```go
// CHANGED: WithModel/WithTools use new Named/ToolArg types.
// REMOVED: WithOutputType, WithOutputSchema.
func WithModel(m Named) GenerateOption
func WithModelName(name string) GenerateOption
func WithTools(tools ...ToolArg) GenerateOption
func WithResources(resources ...*Resource) GenerateOption

// Prompting — unchanged
func WithPrompt(text string, args ...any) GenerateOption
func WithPromptFn(fn PromptFn) GenerateOption
func WithSystem(text string, args ...any) GenerateOption
func WithSystemFn(fn PromptFn) GenerateOption
func WithMessages(messages ...*Message) GenerateOption
func WithMessagesFn(fn MessagesFn) GenerateOption
func WithTextDocs(text ...string) GenerateOption
func WithDocs(docs ...*Document) GenerateOption

// Config — unchanged
func WithConfig(config any) GenerateOption
func WithMiddleware(middleware ...ModelMiddleware) GenerateOption
func WithMaxTurns(maxTurns int) GenerateOption
func WithReturnToolRequests() GenerateOption
func WithToolChoice(choice ToolChoice) GenerateOption
func WithStreaming(streamCh chan<- *ModelResponseChunk) GenerateOption

// Schema — CHANGED: WithOutputType/WithOutputSchema removed.
func WithInputSchema(schema map[string]any) GenerateOption
func WithOutputFormat(format OutputFormat) GenerateOption
func WithOutputInstructions(instructions string) GenerateOption
func WithCustomConstrainedOutput() GenerateOption

// Resume — unchanged
func WithToolResponses(parts ...*Part) GenerateOption
func WithToolRestarts(parts ...*Part) GenerateOption
```

### Prompt Options

```go
// Prompt definition options — unchanged
func WithDescription(description string) PromptOption
func WithMetadata(metadata map[string]any) PromptOption
// Also accepts: WithSystem, WithSystemFn, WithPrompt, WithPromptFn,
//   WithMessages, WithMessagesFn, WithConfig, WithTools, WithOutputFormat,
//   WithInputSchema, WithModel, WithModelName, etc.

// Prompt execution options
// CHANGED: WithInput removed — input is now a typed parameter on Execute/ExecuteStream.
// Also accepts most GenerateOption values.
```

### Embed Options (unchanged except Named)

```go
func WithEmbedder(e Named) EmbedderOption           // CHANGED: Named instead of EmbedderArg
func WithEmbedderName(name string) EmbedderOption
func WithConfig(config any) EmbedderOption

func Embed(ctx context.Context, r api.Registry, opts ...EmbedderOption) (*EmbedResponse, error)
```

### Retrieve Options (unchanged except Named)

```go
func WithRetriever(r Named) RetrieverOption          // CHANGED: Named instead of RetrieverArg
func WithRetrieverName(name string) RetrieverOption
func WithConfig(config any) RetrieverOption

func Retrieve(ctx context.Context, r api.Registry, opts ...RetrieverOption) (*RetrieverResponse, error)
```

### Evaluate Options (unchanged except Named)

```go
func WithEvaluator(e Named) EvaluatorOption          // CHANGED: Named instead of EvaluatorArg
func WithEvaluatorName(name string) EvaluatorOption
func WithDataset(examples ...*Example) EvaluatorOption
func WithID(ID string) EvaluatorOption

func Evaluate(ctx context.Context, r api.Registry, opts ...EvaluatorOption) (*EvaluatorResponse, error)
```

### Formatters

```go
// Unchanged
type Formatter interface {
    Name() string
    Handler(schema map[string]any) (FormatHandler, error)
}

// CHANGED: merged FormatHandler + StreamingFormatHandler, removed ParseMessage.
type FormatHandler interface {
    Instructions() string
    Config() ModelOutputConfig
    ParseOutput(message *Message) (any, error)
    ParseChunk(chunk *ModelResponseChunk) (any, error)
}
```

---

## Part 2: Changes and Motivation

### Streaming: callbacks → channels

Replace `StreamCallback[S] func(context.Context, S) error` with `chan<- S`. Channels are idiomatic Go for streaming data. The caller creates and closes the channel; the action implementation just sends.

### ToolContext → context.Context + helpers

`ToolContext` wraps `context.Context` solely to carry interrupt/resume data. Replacing it with a standard `context.Context` and standalone helper functions (`IsResumed`, `ResumedMetadata[T]`, `InterruptWith[T]`, etc.) eliminates the custom type without losing functionality. The untyped `Interrupt(ctx, *InterruptOptions)` and `OriginalInput(ctx) any` are removed — only the typed variants `InterruptWith[T]` and `OriginalInputAs[T]` remain. Resume metadata is always strongly typed — no `map[string]any`.

### Seal api.Registry

`api.Registry` gains an unexported `sealed()` method. This prevents external implementations, making it safe to add methods without breaking changes. The concrete implementation stays in `internal/registry`. Users interact with the registry through `genkit.*` functions and never implement the interface themselves.

### Concrete types replace interfaces

v1 uses interfaces (`Model`, `Tool`, `Embedder`, `Retriever`, `Prompt`, `Evaluator`, `BackgroundModel`) with unexported implementations, forcing companion types (`ModelArg`, `ModelRef`, `ToolRef`, etc.) for polymorphic APIs. v2 exports the concrete structs directly with the clean names. The interfaces move to the `api` package for registry internals. No public type exposes `Register()` — `Define*` handles registration.

### Named + ActionRef replace \*Ref/\*Arg types

`Named` (`Name() string`) is the single interface accepted by all `With*` option functions. `ActionRef` replaces `ModelRef`, `EmbedderRef`, `RetrieverRef`, and `EvaluatorRef` — one type carrying a name and optional default config. Tools use the sealed `ToolArg` interface (`*Tool[In, Out]` or `ToolName`) because `WithTools` needs to distinguish tool references from other named things.

### Define\* argument order: name, fn, opts

All `Define*` and `New*` functions place the function argument before options. Required args (`name`, `fn`) come first; optional config (`opts`/`*Options`) comes last. This matches user-facing APIs that use variadic options and reads naturally. Plugin-facing APIs use `*Options` structs but follow the same positional convention.

### Typed interrupt API

`RestartOptions` becomes generic (`RestartOptions[In, Meta]`) so both `ReplaceInput` and `ResumedMetadata` are strongly typed. The untyped `Restart`/`Respond` methods are removed — only `RestartWith`/`RespondWith` on `*Tool[In, Out]` remain.

### Unified Prompt type

v1 has two prompt concepts: `Prompt` (untyped, streams `*ModelResponseChunk`) and `DataPrompt[In, Out]` (typed input/output, streams `Out`). v2 unifies them into a single `Prompt[In, Out, Stream]` with three type parameters. The third parameter (`Stream`) is never specified by users — constructors determine it:

- `DefinePrompt[In]` → `Prompt[In, string, *ModelResponseChunk]`. Text output, `*ModelResponseChunk` stream chunks (preserves chunk metadata like `Role`, `Custom`, `Index`). Input is validated against the `In` schema. For fully untyped input, use `DefinePrompt[any]`.
- `DefineDataPrompt[In, Out]` → `Prompt[In, Out, Out]`. Structured output, typed stream chunks. Same behavior as v1 `DataPrompt`.

Both share the same `Execute` signature: `(ctx, input In) → (Out, *ModelResponse, error)`. Both share the same `ExecuteStream` signature: `iter.Seq2[*StreamValue[Out, Stream], error]`. Lookup functions (`LookupPrompt[In]`, `LookupDataPrompt[In, Out]`) are generic and return the concrete `*Prompt` directly — no intermediate non-generic interface is needed since the registry already stores actions as untyped internally.

This matches the `Generate`/`GenerateData` naming convention at the generate level.

### Remove WithOutputType / WithOutputSchema

Structured output is handled exclusively by `GenerateData[Out]`, `GenerateDataStream[Out]`, and `DefineDataPrompt[In, Out]`. Removing `WithOutputType` and `WithOutputSchema` eliminates a class of runtime type errors. `WithOutputFormat` and `WithInputSchema` are retained for format control and MCP definitions.

### Merge FormatHandler and StreamingFormatHandler

v1 split `FormatHandler` and `StreamingFormatHandler` into two interfaces to avoid a breaking change. v2 merges them into a single `FormatHandler` with four methods: `Instructions`, `Config`, `ParseOutput`, `ParseChunk`. `ParseMessage` is removed — it was a legacy method superseded by `ParseOutput`/`ParseChunk`. All format handlers must implement parsing.

### Action.RunJSON always returns telemetry

v1 had `RunJSON` (discards telemetry) and `RunJSONWithTelemetry` (returns `*api.ActionRunResult`). v2 removes the telemetry-discarding variant. `RunJSON` always returns `*api.ActionRunResult[json.RawMessage]` containing both the output and telemetry metadata.

### GenerateData returns value, not pointer

`GenerateData[Out]` returns `(Out, *ModelResponse, error)` instead of `(*Out, *ModelResponse, error)`. Zero value on error or no output. Avoids nil pointer footguns.

### Removed types

| Removed | Replacement |
|---------|-------------|
| `Model` interface | `*Model` struct |
| `ModelArg`, `ModelRef` | `Named`, `ActionRef` |
| `BackgroundModel` interface | `*BackgroundModel` struct |
| `Tool` interface (untyped) | `*Tool[In, Out]` struct |
| `ToolRef` | `ToolArg` sealed interface |
| `ToolContext` | `context.Context` + helpers |
| `Embedder` interface | `*Embedder` struct |
| `EmbedderArg`, `EmbedderRef` | `Named`, `ActionRef` |
| `Retriever` interface | `*Retriever` struct |
| `RetrieverArg`, `RetrieverRef` | `Named`, `ActionRef` |
| `Evaluator` interface | `*Evaluator` struct |
| `EvaluatorArg`, `EvaluatorRef` | `Named`, `ActionRef` |
| `Prompt` interface | `*Prompt[In, Out, Stream]` struct |
| `DataPrompt[In, Out]` | `*Prompt[In, Out, Out]` via `DefineDataPrompt` |
| `AsDataPrompt` | `LookupDataPrompt[In, Out]` |
| `Register()` on public types | Unexported; `Define*` handles it |
| `RunRawMultipart` | Consolidated into `RunRaw` (always returns `*MultipartToolResponse`) |
| `Respond`, `Restart` (untyped) | `RespondWith`, `RestartWith` only |
| `Interrupt()`, `InterruptOptions` | `InterruptWith[T]` (typed) |
| `OriginalInput()` (untyped) | `OriginalInputAs[T]` (typed) |
| `RunJSON` (discards telemetry) | `RunJSON` (always returns telemetry) |
| `RunJSONWithTelemetry` | Renamed to `RunJSON` |
| `WithOutputType`, `WithOutputSchema` | `GenerateData[Out]`, `DefineDataPrompt[In, Out]` |
| `WithInput` (prompt execution) | Typed `input In` parameter on `Execute`/`ExecuteStream` |
| `FormatHandler.ParseMessage` | Removed |
| `StreamingFormatHandler` | Merged into `FormatHandler` |

### New types

| New | Purpose |
|-----|---------|
| `Named` | Shared interface for `With*` options accepting concrete or ref |
| `ActionRef` | Unified lazy reference with name + config |
| `ToolArg` | Sealed interface for `WithTools` |
| `ToolNamed` | String/glob tool reference satisfying `ToolArg` |
