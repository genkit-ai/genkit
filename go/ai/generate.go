// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/invopop/jsonschema"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
)

// Model represents a model that can generate content based on a request. It
// is the type to accept as an argument and to look up by name; implementations
// are created with [NewModelAction], or [genkit.DefineModelAction] in an
// application.
type Model interface {
	// Name returns the registry name of the model.
	Name() string
	// Generate applies the [Model] to provided request, handling tool requests and handles streaming.
	Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error)
	// Register registers the model with the given registry.
	Register(r api.Registry)
}

// ModelArg is the interface for model arguments. It can either be the retriever action itself or a reference to be looked up.
type ModelArg interface {
	Name() string
}

// ModelRef is a struct to hold model name and configuration.
//
// ModelRef supports JSON marshaling: it serializes as a plain string when
// only a name is present, or as {"name": "...", "config": ...} when
// configuration is also set.
type ModelRef struct {
	name   string
	config any
}

// ToolConfig handles configuration around tool calls during generation.
type ToolConfig struct {
	MaxTurns           int  // Maximum number of tool call iterations before erroring.
	ReturnToolRequests bool // Whether to return tool requests instead of making the tool calls and continuing the generation.
}

// ModelFunc is a streaming function that takes in a ModelRequest and generates a ModelResponse, optionally streaming ModelResponseChunks.
type ModelFunc = core.StreamingFunc[*ModelRequest, *ModelResponse, *ModelResponseChunk]

// ModelActionFunc is a [ModelFunc] that additionally receives the
// request's typed Config: the framework deserializes the request's raw config
// into it before calling the function (see [NewModelAction]).
type ModelActionFunc[Config any] = func(context.Context, *ModelRequest, Config, ModelStreamCallback) (*ModelResponse, error)

// ModelStreamCallback is a stream callback of a ModelAction.
type ModelStreamCallback = func(context.Context, *ModelResponseChunk) error

// ModelMiddleware is middleware for model generate requests that takes in a ModelFunc, does something, then returns another ModelFunc.
//
// Deprecated: Use [Middleware] interface with [WithUse] instead, which supports Generate, Model, and Tool hooks.
type ModelMiddleware = core.Middleware[*ModelRequest, *ModelResponse, *ModelResponseChunk]

// action is an unexported alias of [core.Action] used as the embedded field
// in the ai primitives (ModelAction, EmbedderAction, EvaluatorAction).
// Embedding via the alias promotes Action's methods without exporting the
// field itself, so the containment stays an internal detail of each primitive.
//
// Each primitive redeclares the promoted methods that satisfy its interfaces,
// forwarding to the embedded action. Promotion alone would compile, but godoc
// cannot see through an alias into another package: a promoted method appears
// in no documentation and no doc link to it resolves, so a reader would find
// a model with one method that nonetheless claims to be an [api.Action].
type action[In, Out, Stream any] = core.Action[In, Out, Stream]

// ModelAction is a generative model backed by a registry action. It is the
// concrete type returned by [NewModelAction]; pass it to [WithModel] to use it
// for generation, or return it from a plugin's Init for the framework to
// register.
//
// It implements [Model] and [api.Action], so it can be passed anywhere either
// is accepted. It also promotes [core.Action.Run], the typed equivalent of
// [ModelAction.Generate].
type ModelAction struct {
	action[*ModelRequest, *ModelResponse, *ModelResponseChunk]
}

// Pinned here so that breaking either interface fails the build at the type
// rather than at a call site.
var (
	_ api.Action = (*ModelAction)(nil)
	_ Model      = (*ModelAction)(nil)
)

// Name returns the registry name of the model.
func (m *ModelAction) Name() string { return m.action.Name() }

// Register registers the model with r, making it available to lookups and to
// the Dev UI. A plugin that returns the model from its Init does not need to
// call this.
func (m *ModelAction) Register(r api.Registry) { m.action.Register(r) }

// Desc returns the model's action descriptor: its name, schemas, and metadata.
func (m *ModelAction) Desc() api.ActionDesc { return m.action.Desc() }

// RunJSON runs the model on a JSON-encoded [ModelRequest] and returns a
// JSON-encoded [ModelResponse]. The framework uses it to serve reflection and
// registry-driven calls; prefer [ModelAction.Generate].
func (m *ModelAction) RunJSON(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	if m == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Model.RunJSON: model called on a nil model; check that all models are defined")
	}
	return m.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [ModelAction.RunJSON] with the run's telemetry
// returned alongside the output.
func (m *ModelAction) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	if m == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Model.RunJSONWithTelemetry: model called on a nil model; check that all models are defined")
	}
	return m.action.RunJSONWithTelemetry(ctx, input, cb)
}

// generateAction is the type for a utility model generation action that takes in a GenerateActionOptions instead of a ModelRequest.
type generateAction = core.Action[*GenerateActionOptions, *ModelResponse, *ModelResponseChunk]

// result is a generic struct for parallel operation results with index, value, and error.
type result[T any] struct {
	index int
	value T
	err   error
}

// resumeOptionOutput is the return type for resolveResumeOption.
type resumeOptionOutput struct {
	revisedRequest      *GenerateActionOptions
	interruptedResponse *ModelResponse
	toolMessage         *Message
}

// resumedToolRequestOutput is the return type for resolveResumedToolRequest.
type resumedToolRequestOutput struct {
	toolRequest  *Part
	toolResponse *Part
	interrupt    *Part
}

// ModelOptions represents the configuration options for a model.
type ModelOptions struct {
	// ConfigSchema is the JSON schema for the model's config. Inferred from the
	// constructor's Config type parameter when nil.
	ConfigSchema map[string]any
	// Label is a user-friendly name for the model. Defaults to its name.
	Label string
	// Stage indicates the maturity stage of the model.
	Stage ModelStage
	// Supports describes what the model can do. A nil value claims nothing.
	Supports *ModelSupports
	// Versions lists the model versions a request may pin through its config.
	Versions []string
	// Metadata is arbitrary key-value data attached to the action descriptor.
	Metadata map[string]any
}

// DefineGenerateAction defines a utility generate action.
func DefineGenerateAction(ctx context.Context, r api.Registry) *generateAction {
	a := core.NewStreamingActionOf(api.ActionTypeUtil, "generate", nil,
		func(ctx context.Context, actionOpts *GenerateActionOptions, cb ModelStreamCallback) (resp *ModelResponse, err error) {
			// The action's own span records the request and response, and
			// stands in for the first turn's: opening one would nest a
			// duplicate "generate" span directly inside it.
			return generateWithRequest(ctx, r, actionOpts, nil, cb, false /* spanTurnZero */)
		})
	a.Register(r)
	return (*generateAction)(a)
}

// NewModelAction creates an unregistered [ModelAction]: return it from a
// plugin's Init for the framework to register, or call
// [ModelAction.Register] directly. Applications should define models with
// [genkit.DefineModelAction].
//
// Config is the model's typed configuration; it is usually inferred from fn's
// signature. The framework deserializes the request's raw config into Config
// before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The request's [ModelRequest.Config] is
// normalized to the converted value, so it always matches the typed
// parameter. The config's JSON schema is inferred from Config unless
// [ModelOptions.ConfigSchema] overrides it.
//
// The config schema is enforced by input validation on every call, so if
// Config's JSON marshaling diverges from its reflected schema (e.g. SDK
// wrapper types like Opt[float64] that marshal to primitives but reflect as
// objects), set [ModelOptions.ConfigSchema] explicitly or requests will be
// rejected at the action boundary.
func NewModelAction[Config any](
	name string,
	opts *ModelOptions,
	fn ModelActionFunc[Config],
) *ModelAction {
	if name == "" {
		panic("ai.NewModelAction: name is required")
	}

	o := ModelOptions{}
	if opts != nil {
		o = *opts
	}
	if o.Label == "" {
		o.Label = name
	}
	o.Supports = cloneModelSupports(o.Supports)

	configSchema, inputSchema := modelConfigSchemas[Config](o.ConfigSchema, o.Versions)
	metadata := modelActionMetadata(api.ActionTypeModel, &o, configSchema, o.Metadata)

	typedFn := func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		// req.Config was normalized to the exact Config type by
		// normalizeConfig below, so this hits the fast path.
		cfg, err := resolveConfig[Config](req.Config)
		if err != nil {
			return nil, err
		}
		return fn(ctx, req, cfg, cb)
	}

	// normalizeConfig runs outermost so that the built-in wrappers and the
	// model function all see the typed, converted config on the request.
	rawFn := core.ChainMiddleware(
		normalizeConfig[Config](name, o.Versions),
		simulateSystemPrompt(&o, nil),
		augmentWithContext(&o, nil),
		validateSupport(name, &o),
		addAutomaticTelemetry(),
	)(typedFn)

	return &ModelAction{*core.NewStreamingActionOf(api.ActionTypeModel, name, &core.ActionOptions{
		Metadata:    metadata,
		InputSchema: inputSchema,
	}, rawFn)}
}

// modelActionMetadata builds the descriptor metadata shared by model and
// background-model actions.
func modelActionMetadata(actionType api.ActionType, opts *ModelOptions, configSchema map[string]any, callerMetadata ...map[string]any) map[string]any {
	model := map[string]any{
		"label": opts.Label,
		"supports": map[string]any{
			"media":       opts.Supports.Media,
			"context":     opts.Supports.Context,
			"multiturn":   opts.Supports.Multiturn,
			"systemRole":  opts.Supports.SystemRole,
			"tools":       opts.Supports.Tools,
			"toolChoice":  opts.Supports.ToolChoice,
			"constrained": opts.Supports.Constrained,
			"output":      opts.Supports.Output,
			"contentType": opts.Supports.ContentType,
			"longRunning": opts.Supports.LongRunning,
		},
		"versions":      opts.Versions,
		"stage":         opts.Stage,
		"customOptions": configSchema,
	}
	return actionMetadata(actionType, map[string]any{"model": model}, callerMetadata...)
}

// NewModel creates a new [Model].
//
// Deprecated: Use [NewModelAction], which passes the request's config to
// fn as a typed value instead of leaving it type-erased on the request.
func NewModel(name string, opts *ModelOptions, fn ModelFunc) Model {
	if name == "" {
		panic("ai.NewModel: name is required")
	}
	return NewModelAction(name, opts, func(ctx context.Context, req *ModelRequest, _ any, cb ModelStreamCallback) (*ModelResponse, error) {
		return fn(ctx, req, cb)
	})
}

// LookupModel looks up a registered [Model] by name.
// It will try to resolve the model dynamically if the model is not found.
// It returns nil if the model was not resolved.
func LookupModel(r api.Registry, name string) Model {
	action := core.ResolveActionFor[*ModelRequest, *ModelResponse, *ModelResponseChunk](r, api.ActionTypeModel, name)
	if action == nil {
		return nil
	}
	return &ModelAction{*action}
}

// isAbnormal reports whether generation ended in a way known to carry no
// conforming output. Every code path that would otherwise parse a response
// against its output schema consults this predicate first: a schema error
// raised on such a response would mask the FinishReason and FinishMessage the
// caller needs to handle the outcome.
//
// FinishReasonUnknown is not in the set on purpose: plugins map unrecognized
// provider reasons to it, so treating it as abnormal would silently drop
// output validation for responses the model may well have completed.
//
// FinishReasonBlocked is in the set, because a refusal must skip parsing like
// any other abnormal finish, but the typed helpers test for it first and
// report [ErrGenerationBlocked] instead of reaching here.
func (fr FinishReason) isAbnormal() bool {
	switch fr {
	case FinishReasonBlocked, FinishReasonAborted, FinishReasonFailed, FinishReasonInterrupted, FinishReasonOther:
		return true
	default:
		return false
	}
}

// blockedError reports a refusal as [ErrGenerationBlocked], carrying the
// provider's explanation when there is one. Only the typed helpers call it:
// they promise a value that a refusal cannot produce, whereas [Generate]
// hands the response back and lets the caller read FinishReason.
func blockedError(resp *ModelResponse) error {
	if resp.FinishMessage == "" {
		return status.Errorf(ErrGenerationBlocked, "generation blocked")
	}
	return status.Errorf(ErrGenerationBlocked, "generation blocked: %s", resp.FinishMessage)
}

// responseError renders cause as the structured error a response carries
// alongside it. [status.Convert] supplies the status, classified or inferred,
// and the message is replaced with the error's own text so it matches the
// FinishMessage beside it: a wrapped error's sentinel carries only the
// innermost wording. A public error keeps the wording it was given, which was
// chosen for a caller to read.
func responseError(cause error) *status.Error {
	e := status.Convert(cause)
	if e == nil {
		return nil
	}
	if !e.Public && e.Message != cause.Error() {
		ne := *e
		ne.Message = cause.Error()
		return &ne
	}
	return e
}

// callerStopped reports whether the loop ended because the caller stopped it
// rather than because something inside it broke: it cancelled the context, its
// deadline expired, or the loop reached a limit it set ([ErrMaxTurnsExceeded]).
// Those report [FinishReasonAborted]; everything else reports
// [FinishReasonFailed].
//
// It tests the context and the sentinels, never the classified status. A
// service that answers 409 or 504 lands on ABORTED or DEADLINE_EXCEEDED
// through the HTTP mapping in [status], and a provider stopping the request is
// not the caller stopping the run: reporting it aborted tells a retry client
// the one thing that is not true of it.
func callerStopped(ctx context.Context, cause error) bool {
	return ctx.Err() != nil ||
		errors.Is(cause, context.Canceled) ||
		errors.Is(cause, context.DeadlineExceeded) ||
		errors.Is(cause, ErrMaxTurnsExceeded)
}

// failurePartial builds the partial [ModelResponse] that accompanies the
// error when the generate loop stops before it produced a final response.
//
// The partial ends at a turn seam: req carries the conversation as it stood
// when the failing turn began, which is either the caller's own messages or
// a run of completed [model with tool requests, tool with every response]
// rounds, and Message is cleared so nothing half-finished rides along. The
// failing turn's own output is dropped whatever it was, a partially streamed
// model message, a model message whose tools did not all answer, or the tool
// requests [WithMaxTurns] refused to run, because a conversation ending in
// an unanswered tool request is one no provider will accept back. What the
// caller gets is therefore a conversation it can re-send.
//
// base, when non-nil, supplies the accounting the turn already earned (usage
// and custom data); it is copied, not mutated, since the model
// implementation and hooks may retain the original.
//
// The finish reason is [FinishReasonFailed] with the cause as the finish
// message, or [FinishReasonAborted] when the caller stopped the loop rather
// than anything breaking: a cancelled context, an expired deadline, or a
// limit the caller set such as [WithMaxTurns]. Error carries the same cause
// classified, so a consumer reading the response as data branches on a status
// rather than a string. Downstream consumers treat both finishes as abnormal:
// output parsing is skipped and the typed helpers extract nothing from it.
func failurePartial(ctx context.Context, base *ModelResponse, req *ModelRequest, cause error) *ModelResponse {
	p := ModelResponse{}
	if base != nil {
		p = *base
	}
	p.Message = nil
	p.FinishReason = FinishReasonFailed
	if callerStopped(ctx, cause) {
		p.FinishReason = FinishReasonAborted
	}
	p.FinishMessage = cause.Error()
	p.Error = responseError(cause)
	if req != nil {
		p.Request = req
	}
	return &p
}

// GenerateWithRequest is the central generation implementation for ai.Generate(), prompt.Execute(), and the GenerateAction direct call.
//
// Failures follow the partial-response contract documented on [Generate].
func GenerateWithRequest(ctx context.Context, r api.Registry, opts *GenerateActionOptions, mmws []ModelMiddleware, cb ModelStreamCallback) (*ModelResponse, error) {
	return generateWithRequest(ctx, r, opts, mmws, cb, true /* spanTurnZero */)
}

// generateWithRequest runs the tool loop. spanTurnZero reports whether the
// first turn opens its own "generate" span; the generate action passes false
// because its own span already serves as that one.
func generateWithRequest(ctx context.Context, r api.Registry, opts *GenerateActionOptions, mmws []ModelMiddleware, cb ModelStreamCallback, spanTurnZero bool) (*ModelResponse, error) {
	if opts.Model == "" {
		if defaultModel, ok := r.LookupValue(api.DefaultModelKey).(string); ok && defaultModel != "" {
			opts.Model = defaultModel
			logger.Debug(ctx, "no model specified, using default model", "model", opts.Model)
		}
		if opts.Model == "" {
			return nil, status.Errorf(status.ErrInvalidArgument, "ai.GenerateWithRequest: model is required")
		}
	}

	m := LookupModel(r, opts.Model)
	bm := LookupBackgroundModel(r, opts.Model)
	if m == nil && bm == nil {
		return nil, status.Errorf(ErrModelNotFound, "ai.GenerateWithRequest: model %q not found", opts.Model)
	}

	mws, err := resolveRefs(ctx, r, opts.Use)
	if err != nil {
		return nil, err
	}

	// Tools contributed by middleware bundles are registered on a child
	// registry so this Generate() call sees them while outer callers do not.
	// Duplicate names across multiple middleware are rejected explicitly.
	toolDefMap := make(map[string]*ToolDefinition)
	for _, t := range opts.Tools {
		if _, ok := toolDefMap[t]; ok {
			return nil, status.Errorf(status.ErrInvalidArgument, "ai.GenerateWithRequest: duplicate tool %q", t)
		}

		tool := LookupTool(r, t)
		if tool == nil {
			return nil, status.Errorf(ErrToolNotFound, "ai.GenerateWithRequest: tool %q not found", t)
		}

		toolDefMap[t] = tool.Definition()
	}
	var middlewareTools []Tool
	for _, mw := range mws {
		if mw.hooks == nil {
			continue
		}
		for _, t := range mw.hooks.Tools {
			if _, ok := toolDefMap[t.Name()]; ok {
				return nil, status.Errorf(status.ErrInvalidArgument, "ai.GenerateWithRequest: tool %q is contributed by middleware but already declared elsewhere", t.Name())
			}
			toolDefMap[t.Name()] = nil // Reserves the name; the definition is captured after registration.
			middlewareTools = append(middlewareTools, t)
		}
	}
	if len(middlewareTools) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		// Definitions are captured only after registration: Definition resolves
		// schema $refs (e.g. from WithOutputSchemaName) only once the tool has
		// a registry.
		for _, t := range middlewareTools {
			t.Register(r)
			toolDefMap[t.Name()] = t.Definition()
		}
	}
	toolDefs := make([]*ToolDefinition, 0, len(toolDefMap))
	for _, t := range toolDefMap {
		toolDefs = append(toolDefs, t)
	}

	maxTurns := opts.MaxTurns
	if maxTurns < 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "ai.GenerateWithRequest: max turns must be greater than 0, got %d", maxTurns)
	}
	if maxTurns == 0 {
		maxTurns = 50 // Default max turns.
	}

	var outputCfg ModelOutputConfig
	var formatHandler FormatHandler

	if opts.Output != nil {
		formatter, err := resolveFormat(r, opts.Output.JsonSchema, opts.Output.Format)
		if err != nil {
			return nil, err
		}

		formatHandler, err = formatter.Handler(opts.Output.JsonSchema)
		if err != nil {
			return nil, err
		}
		outputCfg = formatHandler.Config()

		// Native constrained output is enabled only when the user has
		// requested it, the model supports it, and there's a JSON schema.
		outputCfg.Constrained = opts.Output.JsonSchema != nil &&
			opts.Output.Constrained && outputCfg.Constrained && m != nil && m.(*ModelAction).supportsConstrained(len(toolDefs) > 0)

		// Add schema instructions to prompt when not using native constraints.
		// This is a no-op for unstructured output requests.
		if !outputCfg.Constrained {
			instructions := ""
			if opts.Output.Instructions != nil {
				instructions = *opts.Output.Instructions
			} else {
				instructions = formatHandler.Instructions()
			}
			if instructions != "" {
				opts.Messages = injectInstructions(opts.Messages, instructions)
			}

			// This is optional to make the output config internally consistent.
			outputCfg.Schema = nil
		}
	}

	req := &ModelRequest{
		Messages:   opts.Messages,
		Config:     opts.Config,
		Docs:       opts.Docs,
		ToolChoice: opts.ToolChoice,
		Tools:      toolDefs,
		Output:     &outputCfg,
	}

	var fn ModelFunc
	if bm != nil {
		if cb != nil {
			logger.Warn(ctx, "background model does not support streaming, ignoring stream callback", "model", bm.Name())
		}
		fn = backgroundModelToModelFn(bm.Start)
	} else {
		fn = m.Generate
	}

	// Build the full hook chains once: wrapping the model function with
	// WrapModel hooks from middleware, and wrapping the generate iteration
	// with WrapGenerate hooks. These chains are reused across every tool-loop
	// iteration rather than rebuilt each turn.
	fn = buildModelChain(mws, fn)
	fn = core.ChainMiddleware(mmws...)(fn)

	// Share one streaming format handler across middleware- and model-emitted
	// chunks so its per-index accumulation (e.g. accumulatedText) spans both.
	var streamingHandler StreamingFormatHandler
	if sfh, ok := formatHandler.(StreamingFormatHandler); ok {
		streamingHandler = sfh
	}

	// middlewareCb is the callback given to WrapGenerate hooks. It attaches the
	// shared streamingHandler so middleware-emitted chunks can be parsed and
	// contribute to accumulation, while preserving any Index/Role the middleware
	// set explicitly (the model path in wrappedCb assigns those from role-based
	// state).
	var middlewareCb ModelStreamCallback
	if cb != nil {
		middlewareCb = func(ctx context.Context, chunk *ModelResponseChunk) error {
			if chunk.Role == "" {
				chunk.Role = RoleModel
			}
			chunk.formatHandler = streamingHandler
			return cb(ctx, chunk)
		}
	}

	runTool := buildToolRunner(mws)

	var generate func(context.Context, *ModelRequest, int, int) (*ModelResponse, error)

	// The loop records the conversation behind each turn so a failure hands
	// back the rounds that completed: runTurn records each failing turn's
	// partial response, and lastReq tracks the conversation entering the
	// current turn as a fallback for errors raised outside a turn (e.g. by a
	// WrapGenerate hook). The loop is sequential, so neither needs
	// synchronization.
	var lastPartial *ModelResponse
	lastReq := req

	turnBody := func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
		req := params.Request
		currentTurn := params.Iteration
		messageIndex := params.MessageIndex
		var wrappedCb ModelStreamCallback
		currentRole := RoleModel
		currentIndex := messageIndex

		if cb != nil {
			wrappedCb = func(ctx context.Context, chunk *ModelResponseChunk) error {
				if chunk.Role != currentRole && chunk.Role != "" {
					currentIndex++
					currentRole = chunk.Role
				}
				chunk.Index = currentIndex
				if chunk.Role == "" {
					chunk.Role = RoleModel
				}
				chunk.formatHandler = streamingHandler
				return cb(ctx, chunk)
			}
		}

		// Resume on the first turn is handled here so that restarted tool
		// execution is both wrapped by WrapGenerate and recorded under this
		// turn's span (generate > tool > generate > model > tool).
		if currentTurn == 0 && opts.Resume != nil && (len(opts.Resume.Respond) > 0 || len(opts.Resume.Restart) > 0) {
			resumeOutput, err := handleResumeOption(ctx, r, opts, runTool, wrappedCb)
			if err != nil {
				return nil, err
			}

			if ir := resumeOutput.interruptedResponse; ir != nil {
				err := status.Errorf(status.ErrFailedPrecondition,
					"One or more tools triggered an interrupt during a restarted execution.")
				ir.Error = responseError(err)
				// ir.Message is the conversation's revised last message, so
				// the request carries the messages before it and History()
				// reproduces the full conversation. Copied from the turn's
				// request so a field added to ModelRequest carries through.
				irReq := *req
				irReq.Messages = opts.Messages[:len(opts.Messages)-1]
				ir.Request = &irReq
				return ir, err
			}

			opts = resumeOutput.revisedRequest

			resumeReq := *req
			resumeReq.Messages = opts.Messages

			if resumeOutput.toolMessage != nil && wrappedCb != nil {
				if err := wrappedCb(ctx, &ModelResponseChunk{
					Content: resumeOutput.toolMessage.Content,
					Role:    RoleTool,
				}); err != nil {
					err = fmt.Errorf("streaming callback failed for resumed tool message: %w", err)
					return failurePartial(ctx, nil, &resumeReq, err), err
				}
			}

			return generate(ctx, &resumeReq, currentTurn+1, currentIndex)
		}

		logger.Debug(ctx, "calling model", "model", opts.Model, "turn", currentTurn, "messages", len(req.Messages))
		resp, err := fn(ctx, req, wrappedCb)
		if err != nil {
			// The model's own output is dropped, complete or not: only the
			// conversation entering this turn survives. Chunks already
			// streamed reached the callback, so nothing observable is lost.
			return failurePartial(ctx, resp, req, err), err
		}

		// ToolRequests allocates a scan of the message, so only build the
		// log arguments when a handler accepts debug records.
		if logger.FromContext(ctx).Enabled(ctx, slog.LevelDebug) {
			modelArgs := []any{"model", opts.Model, "turn", currentTurn, "finishReason", resp.FinishReason, "toolRequests", len(resp.ToolRequests())}
			if resp.Usage != nil {
				modelArgs = append(modelArgs, "inputTokens", resp.Usage.InputTokens, "outputTokens", resp.Usage.OutputTokens)
			}
			logger.Debug(ctx, "model responded", modelArgs...)
		}

		// Ensure all tool requests have unique refs for matching during resume.
		ensureToolRequestRefs(resp.Message)

		// If this is a long-running operation response, return it immediately without further processing
		if bm != nil && resp.Operation != nil {
			return resp, nil
		}

		if formatHandler != nil {
			resp.formatHandler = streamingHandler
			if resp.FinishReason.isAbnormal() {
				// The response passes through as-is so the caller reads the
				// finish reason rather than a schema error. See
				// [FinishReason.isAbnormal].
				logger.Warn(ctx, "model finished abnormally, skipping output parsing",
					"model", opts.Model,
					"finishReason", resp.FinishReason,
					"finishMessage", resp.FinishMessage)
			} else {
				// This is legacy behavior. New format handlers should implement ParseMessage as a passthrough.
				parsed, perr := formatHandler.ParseMessage(resp.Message)
				if perr != nil {
					logger.Debug(ctx, "model output does not match the expected schema", "model", opts.Model, "error", perr)
					// The response rides back with its original message and
					// finish reason, not marked aborted: the model finished,
					// post-processing did not, and the raw output is often
					// exactly what the caller needs to see.
					if resp.Request == nil {
						resp.Request = req
					}
					err := status.Errorf(status.ErrInvalidOutput, "model failed to generate output matching expected schema: %w", perr)
					resp.Error = responseError(err)
					return resp, err
				}
				resp.Message = parsed
			}
		}

		if len(resp.ToolRequests()) == 0 || opts.ReturnToolRequests {
			return resp, nil
		}

		if currentTurn+1 > maxTurns {
			err := status.Errorf(ErrMaxTurnsExceeded, "exceeded maximum tool call iterations (%d)", maxTurns)
			return failurePartial(ctx, resp, req, err), err
		}

		newReq, revisedMsg, err := handleToolRequests(ctx, r, req, resp, wrappedCb, currentIndex, runTool)
		if err != nil {
			// The whole round goes, the model message that opened it
			// included: a failed tool leaves its request unanswered, and
			// [failurePartial] hands back a conversation that can be
			// re-sent.
			return failurePartial(ctx, resp, req, err), err
		}
		if revisedMsg != nil {
			logger.Debug(ctx, "generation paused by tool interrupts", "model", opts.Model, "turn", currentTurn)
			resp.FinishReason = "interrupted"
			resp.FinishMessage = "One or more tool calls resulted in interrupts."
			resp.Message = revisedMsg
			return resp, nil
		}
		if newReq == nil {
			return resp, nil
		}

		return generate(ctx, newReq, currentTurn+1, currentIndex+1)
	}

	// runTurn records the turn's partial result before it enters the
	// WrapGenerate chain, so a hook that drops the response on the way up
	// does not lose it.
	runTurn := func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
		resp, err := turnBody(ctx, params)
		if err != nil && resp != nil {
			lastPartial = resp
		}
		return resp, err
	}

	// runGenerate opens the turn's span around runTurn. The span records the
	// messages this turn sends rather than the ones the call started with, and
	// is built after the WrapGenerate hooks so it includes theirs.
	runGenerate := func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
		if params.Iteration == 0 && !spanTurnZero {
			return runTurn(ctx, params)
		}
		name := opts.StepName
		if name == "" {
			name = "generate"
		}
		// No subtype, as in TypeScript: the type says it all, and a subtype
		// would annotate the trace path as well.
		spanMetadata := &tracing.SpanMetadata{
			Name: name,
			Type: "util",
		}
		spanInput := turnOptions(opts)
		spanInput.Messages = params.Request.Messages

		return tracing.RunInNewSpan(ctx, spanMetadata, spanInput,
			func(ctx context.Context, _ *GenerateActionOptions) (*ModelResponse, error) {
				return runTurn(ctx, params)
			})
	}

	// Compose WrapGenerate hooks once; this chain is invoked for every
	// tool-loop iteration.
	hookedGenerate := buildGenerateChain(mws, runGenerate)

	generate = func(ctx context.Context, req *ModelRequest, currentTurn int, messageIndex int) (*ModelResponse, error) {
		// A fresh turn invalidates the previous turn's recorded partial: a
		// hook may have recovered that failure, and pairing its stale partial
		// with a later error would regress the conversation.
		lastReq = req
		lastPartial = nil
		return hookedGenerate(ctx, &GenerateParams{
			// The hooks get their own copy of the options: a hook writing to
			// it must reach neither the loop nor the turns after it.
			Options:      turnOptions(opts),
			Request:      req,
			Iteration:    currentTurn,
			MessageIndex: messageIndex,
			Callback:     middlewareCb,
		})
	}

	resolvedArgs := []any{
		"model", opts.Model,
		"messages", len(req.Messages),
		"tools", len(toolDefs),
		"maxTurns", maxTurns,
		"format", outputCfg.Format,
		"constrained", outputCfg.Constrained,
		"streaming", cb != nil,
	}
	if len(mws) > 0 {
		resolvedArgs = append(resolvedArgs, "middleware", middlewareNames(mws))
	}
	logger.Debug(ctx, "generate request resolved", resolvedArgs...)

	resp, err := generate(ctx, req, 0, 0)
	if err != nil && resp == nil {
		// Every error after this point comes with a partial response. A
		// failing turn built its own; when it was lost on the way up (a
		// WrapGenerate hook that returns (nil, err)) the recorded one is
		// restored, and an error raised outside a turn gets one synthesized
		// from the conversation entering the current turn.
		if lastPartial != nil {
			resp = lastPartial
		} else {
			resp = failurePartial(ctx, nil, lastReq, err)
		}
	}
	return resp, err
}

// turnOptions returns a per-turn copy of opts for the WrapGenerate hooks and
// the turn's span. Messages and Resume are the fields the loop reads back, so
// the copy detaches both; everything else is shared.
func turnOptions(opts *GenerateActionOptions) *GenerateActionOptions {
	turn := *opts
	if turn.Resume != nil {
		resume := *turn.Resume
		turn.Resume = &resume
	}
	return &turn
}

// middlewareNames returns the names of the resolved middleware in chain
// order, for the request-resolved log line.
func middlewareNames(mws []namedHooks) []string {
	names := make([]string, len(mws))
	for i, mw := range mws {
		names[i] = mw.name
	}
	return names
}

// hookLogArgs builds the attributes for the "middleware hook finished" log
// record: the middleware and hook names, hook-specific extras, the duration,
// whether the hook short-circuited the chain (returned without invoking
// next), and any error it returned. Hooks have no spans of their own, so
// these records are the only per-hook visibility.
func hookLogArgs(name, hook string, start time.Time, nextCalled bool, err error, extra ...any) []any {
	args := make([]any, 0, len(extra)+10)
	args = append(args, "middleware", name, "hook", hook)
	args = append(args, extra...)
	args = append(args, "duration", time.Since(start).Round(time.Millisecond))
	if !nextCalled {
		args = append(args, "shortCircuited", true)
	}
	if err != nil {
		args = append(args, "error", err)
	}
	return args
}

// buildGenerateChain composes the WrapGenerate hooks from mws (outer-to-inner)
// around run. Middleware with a nil WrapGenerate hook is skipped. When debug
// logging is enabled, each hook invocation is bracketed by log records that
// attribute the layer, its duration, and whether it short-circuited the chain.
func buildGenerateChain(mws []namedHooks, run func(ctx context.Context, params *GenerateParams) (*ModelResponse, error)) func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
	chain := run
	for i := len(mws) - 1; i >= 0; i-- {
		mw := mws[i]
		if mw.hooks == nil || mw.hooks.WrapGenerate == nil {
			continue
		}
		hook := mw.hooks.WrapGenerate
		name := mw.name
		next := chain
		chain = func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
			if !logger.FromContext(ctx).Enabled(ctx, slog.LevelDebug) {
				return hook(ctx, params, next)
			}
			logger.Debug(ctx, "middleware hook started", "middleware", name, "hook", "generate", "iteration", params.Iteration)
			start := time.Now()
			var nextCalled atomic.Bool
			resp, err := hook(ctx, params, func(ctx context.Context, p *GenerateParams) (*ModelResponse, error) {
				nextCalled.Store(true)
				return next(ctx, p)
			})
			logger.Debug(ctx, "middleware hook finished",
				hookLogArgs(name, "generate", start, nextCalled.Load(), err, "iteration", params.Iteration)...)
			return resp, err
		}
	}
	return chain
}

// buildModelChain composes the WrapModel hooks from mws (outer-to-inner)
// around fn. Middleware with a nil WrapModel hook is skipped. Hook
// invocations are logged as in [buildGenerateChain].
func buildModelChain(mws []namedHooks, fn ModelFunc) ModelFunc {
	chain := fn
	for i := len(mws) - 1; i >= 0; i-- {
		mw := mws[i]
		if mw.hooks == nil || mw.hooks.WrapModel == nil {
			continue
		}
		hook := mw.hooks.WrapModel
		name := mw.name
		next := chain
		chain = func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			nextFn := func(ctx context.Context, params *ModelParams) (*ModelResponse, error) {
				return next(ctx, params.Request, params.Callback)
			}
			if !logger.FromContext(ctx).Enabled(ctx, slog.LevelDebug) {
				return hook(ctx, &ModelParams{Request: req, Callback: cb}, nextFn)
			}
			logger.Debug(ctx, "middleware hook started", "middleware", name, "hook", "model")
			start := time.Now()
			var nextCalled atomic.Bool
			resp, err := hook(ctx, &ModelParams{Request: req, Callback: cb},
				func(ctx context.Context, params *ModelParams) (*ModelResponse, error) {
					nextCalled.Store(true)
					return nextFn(ctx, params)
				})
			logger.Debug(ctx, "middleware hook finished",
				hookLogArgs(name, "model", start, nextCalled.Load(), err)...)
			return resp, err
		}
	}
	return chain
}

// toolRanKey carries a per-call flag that the innermost runner sets when it
// actually executes the tool, letting the engine attribute a hook that
// short-circuits the call to the tool in traces. It rides the context rather
// than ToolParams so that a WrapTool hook which rebuilds the params struct (to
// rewrite the request, say) cannot silently lose it: every hook that derives
// its context from the one it was handed keeps the flag.
var toolRanKey = base.NewContextKey[*bool]()

// toolStage is one WrapTool hook of a tool call's chain, in chain order; the
// tool itself is the stage after the last one. id identifies the stage on an
// interrupt it raises ([ToolInterrupt.RaisedBy]): the middleware's name, made
// unique within the chain with a "#n" suffix when the name repeats, as inline
// middleware's does.
type toolStage struct {
	id, name string
	hook     func(ctx context.Context, params *ToolParams, next ToolNext) (*MultipartToolResponse, error)
}

// toolStages lists the WrapTool hooks of mws in chain order, with their ids.
func toolStages(mws []namedHooks) []toolStage {
	var stages []toolStage
	seen := map[string]int{}
	for _, mw := range mws {
		if mw.hooks == nil || mw.hooks.WrapTool == nil {
			continue
		}
		seen[mw.name]++
		id := mw.name
		if n := seen[mw.name]; n > 1 {
			id = fmt.Sprintf("%s#%d", mw.name, n)
		}
		stages = append(stages, toolStage{id: id, name: mw.name, hook: mw.hooks.WrapTool})
	}
	return stages
}

// toolStageContext derives the context stage i of a call to the tool named
// toolName runs with, i == len(stages) being the tool itself. It names the
// stage, so that tool.Interrupt attributes an interrupt to it, marks the tool
// stage with a fresh [base.ToolCall] for the tool function to claim, and,
// when the call is a restart, delivers the restart to the stage it answers:
// that stage receives the resume payload and the original input; a stage
// before it, which ran and let the call through, is marked released; a stage
// after it, which never ran, sees a fresh call, as every stage does when the
// middleware that raised the interrupt is no longer in the chain. The keys
// are set on every stage, restart or not, so nothing a hook's context carries
// leaks into the next stage and nothing an enclosing tool call's context
// carries leaks into this one.
func toolStageContext(ctx context.Context, toolName string, stages []toolStage, i int) context.Context {
	id := ""
	var call *base.ToolCall
	if i < len(stages) {
		id = stages[i].id
	} else {
		call = &base.ToolCall{Name: toolName}
	}
	ctx = base.ToolHookKey.NewContext(ctx, id)
	ctx = base.ToolCallKey.NewContext(ctx, call)
	var resume, original any
	released := false
	if restart := base.ToolRestartKey.FromContext(ctx); restart != nil {
		answered := len(stages)
		if restart.RaisedBy != "" {
			answered = slices.IndexFunc(stages, func(s toolStage) bool { return s.id == restart.RaisedBy })
		}
		switch {
		case restart.RaisedBy == "" && restart.Released == nil:
			// The interrupt does not say which stage raised it, so every
			// stage reads the answer, as all did before stages were told
			// apart, and none counts as having let the call through.
			resume, original = restart.Resume, restart.OriginalInput
		case i == answered:
			resume, original = restart.Resume, restart.OriginalInput
		case i < len(stages):
			// A hook is released only if the interrupted call records that
			// it let the call through, and only for the input it saw then.
			// A hook the chain gained since, or one facing a replaced
			// input, sees a fresh call and decides again.
			released = restart.OriginalInput == nil && slices.Contains(restart.Released, stages[i].id)
		}
	}
	ctx = base.ToolResumeKey.NewContext(ctx, resume)
	ctx = base.ToolOriginalInputKey.NewContext(ctx, original)
	return base.ToolReleasedKey.NewContext(ctx, released)
}

// buildToolRunner composes the WrapTool hooks from mws (outer-to-inner) into
// a single function that executes a tool. The returned function is safe to
// invoke from concurrent goroutines; each invocation threads its own params
// through the shared hook chain. Each stage runs with the context
// toolStageContext derives for it. When no WrapTool hooks are configured, the
// tool is invoked directly without allocating a ToolParams wrapper.
func buildToolRunner(mws []namedHooks) func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error) {
	stages := toolStages(mws)
	if len(stages) == 0 {
		return func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error) {
			return tool.RunRawMultipart(toolStageContext(ctx, tool.Name(), nil, 0), req.Input)
		}
	}
	chain := func(ctx context.Context, params *ToolParams) (*MultipartToolResponse, error) {
		if ran := toolRanKey.FromContext(ctx); ran != nil {
			*ran = true
		}
		return params.Tool.RunRawMultipart(toolStageContext(ctx, params.Tool.Name(), stages, len(stages)), params.Request.Input)
	}
	for i := len(stages) - 1; i >= 0; i-- {
		stage := stages[i]
		next := chain
		chain = func(ctx context.Context, params *ToolParams) (*MultipartToolResponse, error) {
			ctx = toolStageContext(ctx, params.Tool.Name(), stages, i)
			debug := logger.FromContext(ctx).Enabled(ctx, slog.LevelDebug)
			var start time.Time
			if debug {
				logger.Debug(ctx, "middleware hook started", "middleware", stage.name, "hook", "tool", "tool", params.Tool.Name())
				start = time.Now()
			}
			var nextCalled atomic.Bool
			resp, err := stage.hook(ctx, params, func(ctx context.Context, p *ToolParams) (*MultipartToolResponse, error) {
				nextCalled.Store(true)
				return next(ctx, p)
			})
			// An interrupt raised further in passed through this hook, which
			// let the call through: recorded on the interrupt, so that the
			// restart answering it releases this hook and no other.
			var tie *base.ToolInterruptError
			if nextCalled.Load() && errors.As(err, &tie) && tie.RaisedBy != stage.id {
				tie.Released = append([]string{stage.id}, tie.Released...)
			}
			if debug {
				logger.Debug(ctx, "middleware hook finished",
					hookLogArgs(stage.name, "tool", start, nextCalled.Load(), err, "tool", params.Tool.Name())...)
			}
			return resp, err
		}
	}
	return func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error) {
		ran := false
		resp, err := chain(toolRanKey.NewContext(ctx, &ran), &ToolParams{Request: req, Tool: tool})
		if !ran {
			return recordToolShortCircuit(ctx, tool.Name(), req.Input, resp, err)
		}
		return resp, err
	}
}

// recordToolShortCircuit emits the tool-shaped span that core/action.go would
// have created, attributing a WrapTool outcome (interrupt, cached response,
// injected error) to the tool in traces even though the tool never ran. The
// span wraps the already-known outcome and returns it unchanged, so it records
// what the tool call resolved to rather than how long the hooks took to
// resolve it.
func recordToolShortCircuit(ctx context.Context, name string, input any, resp *MultipartToolResponse, err error) (*MultipartToolResponse, error) {
	// Mirrors the span core builds for a tool action, subtype included, so the
	// two are indistinguishable in a trace.
	spanMeta := &tracing.SpanMetadata{
		Name:            name,
		Type:            "action",
		Subtype:         string(api.ActionTypeToolV2),
		Metadata:        map[string]string{},
		TelemetryLabels: tracing.TelemetryLabelsFromContext(ctx),
	}
	if flowName := core.FlowNameFromContext(ctx); flowName != "" {
		spanMeta.Metadata["flow:name"] = flowName
	}
	return tracing.RunInNewSpan(ctx, spanMeta, input,
		func(context.Context, any) (*MultipartToolResponse, error) { return resp, err })
}

// Generate generates a model response based on the provided options.
//
// When generation fails after the request has resolved, the classified error
// is returned together with a non-nil partial [ModelResponse].
//
// A loop that stopped early leaves Message nil and reports FinishReason
// [FinishReasonFailed] with the cause as the FinishMessage when something
// broke (a failed model call, a failed tool), or [FinishReasonAborted] when
// the caller stopped it instead: a cancelled context, an expired deadline, or
// a limit it set such as [WithMaxTurns]. Error carries the same cause
// classified, the structured form of the FinishMessage beside it.
// [ModelResponse.History] is then a
// conversation that can be sent again: it ends at a turn seam, meaning the
// messages the failing turn started from, which are the caller's own or a run
// of completed [model with tool requests, tool with every response] rounds.
// Nothing from the failing turn rides along, since a conversation ending in a
// tool request nothing answered is one no provider accepts. Text streamed
// before the failure still reached the callback.
//
// Two errors are not loop failures and keep their response's message: a
// response the model completed but post-processing rejected (structured
// output that does not match the schema), which keeps the model's own finish
// reason, and a resume whose restarted tool interrupted again, which keeps
// FinishReason interrupted under its FAILED_PRECONDITION error and is
// answered with [WithResume] rather than re-sent.
//
// Errors reported before a request is made (unknown model or tool, invalid
// options) carry a nil response.
func Generate(ctx context.Context, r api.Registry, opts ...GenerateOption) (*ModelResponse, error) {
	genOpts := &generateOptions{}
	for _, opt := range opts {
		opt.applyGenerate(genOpts)
	}

	if genOpts.OutputSchema != nil {
		resolved, err := core.ResolveSchema(r, genOpts.OutputSchema)
		if err != nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "ai.Generate: invalid output schema: %w", err)
		}
		genOpts.OutputSchema = resolved
		if genOpts.OutputFormat == "" {
			genOpts.OutputFormat = OutputFormatJSON
		}
	}

	var modelName string
	if genOpts.Model != nil {
		modelName = genOpts.Model.Name()
	}

	toolNames, dynamicTools, err := resolveUniqueTools(r, genOpts.Tools)
	if err != nil {
		return nil, err
	}

	if len(dynamicTools) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, t := range dynamicTools {
			t.Register(r)
		}
	}

	if len(genOpts.Resources) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, res := range genOpts.Resources {
			res.Register(r)
		}
	}

	// Generate has no prompt input, so content functions get a nil raw input,
	// which each turns into the zero value of its own type, and no text is
	// compiled: there would be nothing to render it against.
	messages := []*Message{}
	if genOpts.SystemText != nil {
		messages = append(messages, NewSystemTextMessage(*genOpts.SystemText))
	} else if genOpts.SystemFn != nil {
		parts, err := genOpts.SystemFn(ctx, nil)
		if err != nil {
			return nil, err
		}
		if len(parts) > 0 {
			messages = append(messages, &Message{Role: RoleSystem, Content: parts})
		}
	}
	if genOpts.MessagesFn != nil {
		msgs, err := genOpts.MessagesFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, msgs...)
	}
	if genOpts.PromptText != nil {
		messages = append(messages, NewUserTextMessage(*genOpts.PromptText))
	} else if genOpts.PromptFn != nil {
		parts, err := genOpts.PromptFn(ctx, nil)
		if err != nil {
			return nil, err
		}
		if len(parts) > 0 {
			messages = append(messages, &Message{Role: RoleUser, Content: parts})
		}
	}

	if modelRef, ok := genOpts.Model.(ModelRef); ok && genOpts.Config == nil {
		if cfg := modelRef.Config(); !base.IsNil(cfg) {
			genOpts.Config = cfg
		}
	}

	actionOpts := &GenerateActionOptions{
		Model:              modelName,
		Messages:           messages,
		Tools:              toolNames,
		MaxTurns:           genOpts.MaxTurns,
		Config:             genOpts.Config,
		ToolChoice:         genOpts.ToolChoice,
		Docs:               genOpts.Documents,
		ReturnToolRequests: genOpts.ReturnToolRequests != nil && *genOpts.ReturnToolRequests,
		StepName:           genOpts.StepName,
		Output: &GenerateActionOutputConfig{
			JsonSchema:   genOpts.OutputSchema,
			Format:       genOpts.OutputFormat,
			Instructions: genOpts.OutputInstructions,
			Constrained:  !genOpts.CustomConstrained,
		},
	}

	if len(genOpts.RespondParts) > 0 || len(genOpts.RestartParts) > 0 {
		actionOpts.Resume = &GenerateActionResume{
			Respond: genOpts.RespondParts,
			Restart: genOpts.RestartParts,
		}
	}

	refs, err := configsToRefs(genOpts.Use)
	if err != nil {
		return nil, err
	}
	if len(refs) > 0 {
		actionOpts.Use = refs
	}

	processedMessages, err := processResources(ctx, r, messages)
	if err != nil {
		return nil, status.Errorf(status.ErrInternal, "ai.Generate: error processing resources: %w", err)
	}
	actionOpts.Messages = processedMessages

	return GenerateWithRequest(ctx, r, actionOpts, genOpts.Middleware, genOpts.Stream)
}

// GenerateText run generate request for this model. Returns generated text only.
// On error, the text of the partial response (see [Generate]), usually empty,
// is returned with the error.
func GenerateText(ctx context.Context, r api.Registry, opts ...GenerateOption) (string, error) {
	res, err := Generate(ctx, r, opts...)
	return res.Text(), err
}

// A refusal is an error: when the response finished blocked, the output is nil
// and the error is [ErrGenerationBlocked], carrying the provider's explanation.
// The response is still returned alongside it. A generation failure likewise
// returns its error alongside the partial response [Generate] documents, with
// a nil output.
//
// Every other finish that yields nothing to extract is not an error. If the
// response carries no text (tool requests or interrupts instead), or ended
// aborted, interrupted, or other, the output is nil and the error is nil.
// Check resp.FinishReason, resp.Interrupts(), and resp.ToolRequests().
//
// The output format is JSON with a schema inferred from Out; an explicit
// [WithOutputSchema] or [WithOutputSchemaName] overrides the schema while
// extraction into Out keeps working. Overriding the format itself with a
// non-JSON [WithOutputFormat] or [WithOutputEnums] breaks that extraction:
// the response text will not parse into Out.
func GenerateData[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) (*Out, *ModelResponse, error) {
	var value Out
	// Prepend the inferred output type so an explicit WithOutputSchema or
	// WithOutputSchemaName passed by the caller wins the schema slot (last set
	// wins). The typed Out still drives value extraction below, so structured
	// output keeps working whether or not the caller overrode the schema.
	opts = append([]GenerateOption{WithOutputType(value)}, opts...)

	resp, err := Generate(ctx, r, opts...)
	if err != nil {
		return nil, resp, err
	}

	// A refusal cannot produce the value this helper promises, so it is
	// reported rather than handed back as a zero value that reads as success.
	// [Generate] still returns the response unwrapped.
	if resp.FinishReason == FinishReasonBlocked {
		return nil, resp, blockedError(resp)
	}

	// The remaining abnormal finishes, and a response with no text at all
	// (what a turn holding tool requests, interrupts, or media looks like), have
	// nothing to extract but are not failures. The response goes back unparsed
	// rather than as a schema error naming the wrong cause.
	if resp.FinishReason.isAbnormal() || resp.Text() == "" {
		return nil, resp, nil
	}

	err = resp.Output(&value)
	if err != nil {
		return nil, resp, err
	}

	return &value, resp, nil
}

// StreamValue is either a streamed chunk or the final response of a generate request.
type StreamValue[Out, Stream any] struct {
	Done     bool
	Chunk    Stream         // valid if Done is false
	Output   Out            // valid if Done is true
	Response *ModelResponse // valid if Done is true
}

// ModelStreamValue is a stream value for a model response.
// Out is never set because the output is already available in the Response field.
type ModelStreamValue = StreamValue[struct{}, *ModelResponseChunk]

// errStop is a sentinel error used to signal early termination of streaming.
var errStop = errors.New("stop")

// GenerateStream generates a model response and streams the output.
// It returns an iterator that yields streaming results.
//
// If the yield function is passed a non-nil error, generation has failed with that
// error; the yield function will not be called again. The value beside it is
// still Done and still carries the partial Response, the same pair [Generate]
// returns, so a consumer that streamed a tool loop reads the conversation it
// can send again.
//
// If the yield function's [ModelStreamValue] argument has Done == true, the value's
// Response field contains the final response; the yield function will not be called
// again.
//
// Otherwise the Chunk field of the passed [ModelStreamValue] holds a streamed chunk.
func GenerateStream(ctx context.Context, r api.Registry, opts ...GenerateOption) iter.Seq2[*ModelStreamValue, error] {
	return func(yield func(*ModelStreamValue, error) bool) {
		done := false
		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if done {
				return errStop
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if !yield(&ModelStreamValue{Chunk: chunk}, nil) {
				done = true
				return errStop
			}
			return nil
		}

		// Chain rather than set the callback so a caller-supplied
		// WithStreaming still receives every chunk.
		allOpts := append(slices.Clone(opts), withChainedStreaming(cb))

		resp, err := Generate(ctx, r, allOpts...)
		if done || errors.Is(err, errStop) {
			return
		}
		// A failure yields its partial beside the error, the same pair
		// [Generate] returns, so a consumer that streamed a tool loop can
		// still read the conversation it should send again.
		yield(&ModelStreamValue{Done: true, Response: resp}, err)
	}
}

// GenerateDataStream generates a model response with streaming and returns strongly-typed output.
// It returns an iterator that yields streaming results.
//
// If the yield function is passed a non-nil error, generation has failed with that
// error; the yield function will not be called again. The value beside it is
// still Done and still carries the partial Response (Output stays zero, since
// a failed call produced no value), the same pair [GenerateData] returns.
//
// If the yield function's [StreamValue] argument has Done == true, the value's
// Output and Response fields contain the final typed output and response; the yield function
// will not be called again.
//
// Otherwise the Chunk field of the passed [StreamValue] holds a streamed chunk.
//
// Like [GenerateData], the output format is JSON with a schema inferred from
// Out; overriding the format with a non-JSON [WithOutputFormat] or
// [WithOutputEnums] breaks typed extraction. Also like [GenerateData], a
// blocked response fails with [ErrGenerationBlocked], while a response with no
// text output or one that ended aborted, interrupted, or other yields
// zero-value Output and no error; check Response.FinishReason, Interrupts(),
// and ToolRequests() to handle those.
//
// Chunks are provisional. They are parsed as they arrive, before the finish
// reason exists, so a generation that streams partial output and then ends
// abnormally will have yielded chunks that the final value does not stand
// behind. The Done value is the authoritative one.
func GenerateDataStream[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) iter.Seq2[*StreamValue[Out, Out], error] {
	return func(yield func(*StreamValue[Out, Out], error) bool) {
		done := false
		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if done {
				return errStop
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			var streamValue Out
			if err := chunk.Output(&streamValue); err != nil {
				yield(nil, err)
				done = true
				return err
			}
			// Skip yielding if there's no parseable output yet (e.g., incomplete JSON during streaming).
			if base.IsNil(streamValue) {
				return nil
			}
			if !yield(&StreamValue[Out, Out]{Chunk: streamValue}, nil) {
				done = true
				return errStop
			}
			return nil
		}

		// Prepend WithOutputType so the user can override the output format,
		// and chain the iterator callback so a caller-supplied WithStreaming
		// still receives every chunk.
		var value Out
		allOpts := append([]GenerateOption{WithOutputType(value)}, opts...)
		allOpts = append(allOpts, withChainedStreaming(cb))

		resp, err := Generate(ctx, r, allOpts...)
		if done || errors.Is(err, errStop) {
			return
		}
		if err != nil {
			// The partial rides along, as on [Generate]; Output stays zero,
			// since a failed call produced no value to extract.
			yield(&StreamValue[Out, Out]{Done: true, Response: resp}, err)
			return
		}

		// A refusal cannot produce the value this helper promises, so it is
		// reported rather than handed back as a zero value that reads as success.
		// [Generate] still returns the response unwrapped.
		if resp.FinishReason == FinishReasonBlocked {
			yield(nil, blockedError(resp))
			return
		}

		// The remaining abnormal finishes, and a response with no text at all
		// (what a turn holding tool requests, interrupts, or media looks like), have
		// nothing to extract but are not failures. The response goes back unparsed
		// rather than as a schema error naming the wrong cause.
		if resp.FinishReason.isAbnormal() || resp.Text() == "" {
			yield(&StreamValue[Out, Out]{Done: true, Response: resp}, nil)
			return
		}

		output, err := extractTypedOutput[Out](resp)
		if err != nil {
			yield(&StreamValue[Out, Out]{Done: true, Response: resp}, err)
			return
		}

		yield(&StreamValue[Out, Out]{Done: true, Output: output, Response: resp}, nil)
	}
}

// Generate applies the [Action] to provided request.
func (m *ModelAction) Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	if m == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Model.Generate: generate called on a nil model; check that all models are defined")
	}

	return m.Run(ctx, req, cb)
}

// supportsConstrained returns whether the model supports constrained output.
func (m *ModelAction) supportsConstrained(hasTools bool) bool {
	if m == nil {
		return false
	}

	metadata := m.Desc().Metadata
	if metadata == nil {
		return false
	}

	modelMeta, ok := metadata["model"].(map[string]any)
	if !ok {
		return false
	}

	supportsMeta, ok := modelMeta["supports"].(map[string]any)
	if !ok {
		return false
	}

	constrained, ok := supportsMeta["constrained"].(ConstrainedSupport)
	if !ok {
		return false
	}

	if constrained == "" ||
		constrained == ConstrainedSupportNone ||
		(constrained == ConstrainedSupportNoTools && hasTools) {
		return false
	}

	return true
}

// ensureToolRequestRefs assigns unique refs to tool request parts that don't have one.
// This ensures that when there are multiple calls to the same tool, each can be
// individually matched when resuming with Restart or Respond directives.
func ensureToolRequestRefs(msg *Message) {
	if msg == nil {
		return
	}
	for _, part := range msg.Content {
		// The kind is a string a plugin sets, so the pointer it promises is
		// guarded too rather than dereferenced on trust.
		if part.IsToolRequest() && part.ToolRequest != nil && part.ToolRequest.Ref == "" {
			part.ToolRequest.Ref = uuid.New().String()
		}
	}
}

// clone creates a deep copy of the provided object using JSON marshaling and unmarshaling.
func clone[T any](obj *T) *T {
	if obj == nil {
		return nil
	}

	bytes, err := json.Marshal(obj)
	if err != nil {
		panic(fmt.Sprintf("clone: failed to marshal object: %v", err))
	}

	var newObj T
	if err := json.Unmarshal(bytes, &newObj); err != nil {
		panic(fmt.Sprintf("clone: failed to unmarshal object: %v", err))
	}

	return &newObj
}

// toolFailureError classifies a tool's error for the loop. A tool that failed
// on its own terms is [ErrToolFailed], which the loop reports as a failed
// generation. A tool that stopped because the call's context ended is not a
// tool failure at all: the cancellation is returned with its own status, so
// the partial response carries [FinishReasonAborted] rather than blaming the
// tool for a stop the caller asked for.
func toolFailureError(ctx context.Context, name string, cause error) error {
	if ctx.Err() != nil {
		return status.Errorf(status.ErrCancelled, "tool %q stopped: %w", name, cause)
	}
	return status.Errorf(ErrToolFailed, "tool %q failed: %w", name, cause)
}

// toolStreamer lets the tools of one round, which run concurrently, stream
// through cb with [github.com/firebase/genkit/go/ai/tool.SendPartial] (wrapped
// partial responses) and [github.com/firebase/genkit/go/ai/tool.SendChunk]
// (raw model response chunks). cb, the wrapped stream callback, mutates
// shared role and index state and writes the single stream sink, neither of
// which is safe for concurrent use, so every tool-originated send is
// serialized under one mutex. Streaming is best effort, so a sink error is
// logged and dropped rather than failing the tool's authoritative return
// value. The zero value with a nil cb installs no senders.
type toolStreamer struct {
	cb ModelStreamCallback
	mu sync.Mutex
}

// send streams chunk through cb.
func (s *toolStreamer) send(ctx context.Context, chunk *ModelResponseChunk) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.cb(ctx, chunk); err != nil {
		logger.Debug(ctx, "tool stream callback failed, dropping chunk", "error", err)
	}
}

// context returns ctx with the senders for the call to req installed, or ctx
// itself when there is no stream to send to.
func (s *toolStreamer) context(ctx context.Context, req *ToolRequest) context.Context {
	if s == nil || s.cb == nil {
		return ctx
	}
	ctx = base.ToolPartialSenderKey.NewContext(ctx, func(sendCtx context.Context, output any) {
		s.send(sendCtx, &ModelResponseChunk{
			Role: RoleTool,
			Content: []*Part{NewPartialToolResponsePart(&ToolResponse{
				Name:   req.Name,
				Ref:    req.Ref,
				Output: output,
			})},
		})
	})
	return base.ToolChunkSenderKey.NewContext(ctx, func(sendCtx context.Context, chunk any) {
		if c, ok := chunk.(*ModelResponseChunk); ok {
			s.send(sendCtx, c)
		}
	})
}

// toolRunnerFunc runs a tool through the WrapTool hook chain and returns the
// raw [MultipartToolResponse]. Returned by [buildToolRunner].
type toolRunnerFunc = func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error)

// interruptedPart returns the copy of tool request p that records the
// interrupt raised for it, carrying the interrupt's data (nil for a bare
// interrupt) as typed state; the wire marker is written when the part is
// marshaled. The data is stored as the JSON object it serializes to, the
// shape it has after a wire hop, so a reader such as [InterruptAs] sees one
// shape wherever the part came from. tool.Interrupt already normalized it;
// the conversion here covers an error built with a struct directly. The
// stage that raised the interrupt is recorded with it, for the restart to
// answer (see restartedToolResponse).
func interruptedPart(p *Part, tie *base.ToolInterruptError) (*Part, error) {
	data, err := base.ObjectPayload(tie.Data, "interrupt data")
	if err != nil {
		return nil, err
	}
	newPart := p.typedClone()
	newPart.Interrupt = &ToolInterrupt{Data: bareIfNil(data), RaisedBy: tie.RaisedBy, ReleasedBy: tie.Released}
	return newPart, nil
}

// resolvedPart returns the copy of tool request p that records its interrupt,
// if any, as resolved, for the message history.
func resolvedPart(p *Part) *Part {
	newPart := p.typedClone()
	if newPart.Interrupt != nil {
		newPart.Interrupt.Resolved = true
	}
	return newPart
}

// stripPendingKeys deletes the keys stampPendingToolOutcome writes from m in
// place and returns m, or nil when nothing is left.
func stripPendingKeys(m map[string]any) map[string]any {
	for _, key := range [...]string{"pendingOutput", "pendingMetadata", "pendingContent"} {
		delete(m, key)
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// stampPendingToolOutcome records a resolved tool call's response on its
// request part so a later resume replays it (see handleResumedToolRequest).
// The response's metadata and content ride under their own keys;
// pendingOutput itself stays output-only for cross-SDK parity.
func stampPendingToolOutcome(part *Part, resp *MultipartToolResponse) {
	if part.Metadata == nil {
		part.Metadata = make(map[string]any)
	}
	part.Metadata["pendingOutput"] = resp.Output
	if len(resp.Metadata) > 0 {
		part.Metadata["pendingMetadata"] = resp.Metadata
	}
	if len(resp.Content) > 0 {
		part.Metadata["pendingContent"] = resp.Content
	}
}

// handleToolRequests processes any tool requests in the response. On success
// it returns either a new request to continue the conversation, or, when a
// tool interrupted, a nil request and the revised model message carrying the
// interrupt metadata. On error it returns no message: the caller drops the
// whole round, so the results that did arrive have nowhere to go. The error
// is reported as soon as it arrives; a still-running sibling is left to
// finish detached and its result is discarded.
func handleToolRequests(ctx context.Context, r api.Registry, req *ModelRequest, resp *ModelResponse, cb ModelStreamCallback, messageIndex int, runTool toolRunnerFunc) (*ModelRequest, *Message, error) {
	toolRequests := resp.ToolRequests()
	if len(toolRequests) == 0 {
		return nil, nil, nil
	}

	if logger.FromContext(ctx).Enabled(ctx, slog.LevelDebug) {
		toolNames := make([]string, 0, len(toolRequests))
		for _, p := range toolRequests {
			toolNames = append(toolNames, p.ToolRequest.Name)
		}
		logger.Debug(ctx, "executing tool requests", "tools", toolNames)
	}

	resultChan := make(chan result[*MultipartToolResponse], len(toolRequests))
	toolMsg := &Message{Role: RoleTool}
	revisedMsg := clone(resp.Message)

	stream := &toolStreamer{cb: cb}

	for i, part := range revisedMsg.Content {
		if !part.IsToolRequest() {
			continue
		}

		go func(idx int, p *Part) {
			toolReq := p.ToolRequest
			tool := LookupTool(r, p.ToolRequest.Name)
			if tool == nil {
				resultChan <- result[*MultipartToolResponse]{index: idx, err: status.Errorf(ErrToolNotFound, "tool %q not found", toolReq.Name)}
				return
			}

			toolCtx := stream.context(ctx, toolReq)

			// The part sink spans the whole call, WrapTool hooks included,
			// so a hook can attach parts before or after running the tool.
			// A fresh call answers no restart, including one an enclosing
			// tool call was answering when it called Generate.
			sink := &base.PartSink{}
			defer sink.Close() // On an error or interrupt, too.
			toolCtx = base.ToolPartSinkKey.NewContext(toolCtx, sink)
			toolCtx = base.ToolRestartKey.NewContext(toolCtx, nil)

			multipartResp, err := runTool(toolCtx, tool, toolReq)
			if err != nil {
				var tie *base.ToolInterruptError
				if errors.As(err, &tie) {
					logger.Debug(ctx, "tool triggered an interrupt", "tool", toolReq.Name)
					interrupt, ierr := interruptedPart(p, tie)
					if ierr != nil {
						resultChan <- result[*MultipartToolResponse]{index: idx, err: toolFailureError(ctx, toolReq.Name, ierr)}
						return
					}
					revisedMsg.Content[idx] = interrupt
					resultChan <- result[*MultipartToolResponse]{index: idx, err: tie}
					return
				}

				resultChan <- result[*MultipartToolResponse]{index: idx, err: toolFailureError(ctx, toolReq.Name, err)}
				return
			}
			multipartResp = foldAttachedParts(multipartResp, sink)

			// p is already private to this call (revisedMsg is a deep clone of
			// the model message), and the stamp writes only its metadata, so a
			// shallow copy with its own map is enough.
			newPart := p.typedClone()
			stampPendingToolOutcome(newPart, multipartResp)
			revisedMsg.Content[idx] = newPart

			resultChan <- result[*MultipartToolResponse]{index: idx, value: multipartResp}
		}(i, part)
	}

	// Tools run concurrently, so resultChan delivers responses in completion
	// order. Collect them keyed by the request's position in the model message
	// so they can be re-emitted in request order below.
	toolRespByIndex := make(map[int]*Part, len(toolRequests))
	receivedIndexes := make([]int, 0, len(toolRequests))
	hasInterrupts := false
	var toolErr error
	for len(receivedIndexes) < len(toolRequests) && toolErr == nil {
		res := <-resultChan
		receivedIndexes = append(receivedIndexes, res.index)
		if res.err != nil {
			var tie *base.ToolInterruptError
			if errors.As(res.err, &tie) {
				hasInterrupts = true
				continue
			}
			toolErr = res.err
			continue
		}

		toolReq := revisedMsg.Content[res.index].ToolRequest
		newToolResp := NewToolResponsePart(&ToolResponse{
			Name:    toolReq.Name,
			Ref:     toolReq.Ref,
			Output:  res.value.Output,
			Content: res.value.Content,
		})
		newToolResp.Metadata = res.value.Metadata
		toolRespByIndex[res.index] = newToolResp
	}

	if toolErr != nil {
		// Nothing rides back with the error. The caller drops the whole
		// round, the model message that opened it included, because a
		// conversation ending on a tool request nothing answered is one no
		// provider accepts. A still-running sibling keeps revising its own
		// element of revisedMsg after this returns, which nothing reads, and
		// its send cannot block: resultChan buffers one slot per request.
		return nil, nil, toolErr
	}

	if hasInterrupts {
		return nil, revisedMsg, nil
	}

	// Emit tool responses in the order their requests appear in the model
	// message, not the order the goroutines happened to finish, so the recorded
	// tool message is deterministic across runs.
	toolResps := make([]*Part, 0, len(toolRespByIndex))
	for i := range revisedMsg.Content {
		if part, ok := toolRespByIndex[i]; ok {
			toolResps = append(toolResps, part)
		}
	}

	toolMsg.Content = toolResps

	if cb != nil {
		err := cb(ctx, &ModelResponseChunk{
			Content: toolMsg.Content,
			Role:    RoleTool,
			Index:   messageIndex + 1,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("streaming callback failed: %w", err)
		}
	}

	// The next turn gets its own request value: appending to the current one
	// would retroactively grow the request a WrapModel or WrapGenerate hook is
	// still holding.
	newReq := *req
	newReq.Messages = append(slices.Clone(req.Messages), resp.Message, toolMsg)

	return &newReq, nil, nil
}

// Text returns the contents of the first candidate in a
// [ModelResponse] as a string. It returns an empty string if there
// are no candidates or if the candidate has no message.
func (mr *ModelResponse) Text() string {
	if mr == nil || mr.Message == nil {
		return ""
	}
	return mr.Message.Text()
}

// History returns messages from the request combined with the response message
// to represent the conversation history. The result is always freshly
// allocated, so callers may retain or append to it without disturbing
// Request.Messages.
func (mr *ModelResponse) History() []*Message {
	if mr == nil {
		return nil
	}
	var reqMsgs []*Message
	if mr.Request != nil {
		reqMsgs = mr.Request.Messages
	}
	if mr.Message == nil {
		return slices.Clone(reqMsgs)
	}
	history := make([]*Message, len(reqMsgs)+1)
	copy(history, reqMsgs)
	history[len(reqMsgs)] = mr.Message
	return history
}

// Reasoning concatenates all reasoning parts present in the message
func (mr *ModelResponse) Reasoning() string {
	if mr == nil || mr.Message == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range mr.Message.Content {
		if !p.IsReasoning() {
			continue
		}
		sb.WriteString(p.Text)
	}
	return sb.String()
}

// Output parses the structured output from the response and unmarshals it into v.
// If a format handler is set, it uses the handler's ParseOutput method.
// Otherwise, it falls back to parsing the response text as JSON.
func (mr *ModelResponse) Output(v any) error {
	if mr == nil || mr.Message == nil || len(mr.Message.Content) == 0 {
		return errors.New("no content in response")
	}

	if mr.formatHandler == nil {
		// For backward compatibility, extract JSON from the response text.
		return json.Unmarshal([]byte(base.ExtractJSONFromMarkdown(mr.Message.Text())), v)
	}

	output, err := mr.formatHandler.ParseOutput(mr.Message)
	if err != nil {
		return err
	}

	b, err := json.Marshal(output)
	if err != nil {
		return fmt.Errorf("failed to marshal output: %w", err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("failed to unmarshal output: %w", err)
	}
	return nil
}

// ToolRequests returns the tool requests from the response.
func (mr *ModelResponse) ToolRequests() []*Part {
	var parts []*Part
	if mr == nil || mr.Message == nil {
		return parts
	}
	for _, p := range mr.Message.Content {
		if p.IsToolRequest() {
			parts = append(parts, p)
		}
	}
	return parts
}

// Interrupts returns the interrupted tool request parts from the response.
func (mr *ModelResponse) Interrupts() []*Part {
	var parts []*Part
	if mr == nil || mr.Message == nil {
		return parts
	}
	for _, p := range mr.Message.Content {
		if p.IsInterrupt() {
			parts = append(parts, p)
		}
	}
	return parts
}

// Media returns the media content of the [ModelResponse] as a string.
//
// Only the first media part is returned, and its content type is left behind,
// so a response carrying more than one image, or one whose content type is
// needed to render it, is better read with [ModelResponse.MediaParts].
func (mr *ModelResponse) Media() string {
	if mr == nil || mr.Message == nil {
		return ""
	}
	for _, part := range mr.Message.Content {
		if part.IsMedia() {
			return part.Text
		}
	}
	return ""
}

// MediaParts returns every media part of the [ModelResponse], each carrying its
// content type alongside its data. It returns nil if the response has none.
//
// A model that may answer with media often answers with media alone, so this
// pairs with [ModelResponse.Text] rather than replacing it: the two read
// disjoint halves of a response and either may come back empty.
func (mr *ModelResponse) MediaParts() []*Part {
	if mr == nil {
		return nil
	}
	return mr.Message.MediaParts()
}

// Text returns the text content of the ModelResponseChunk as a string,
// concatenating its text parts and skipping every other kind, as
// [Message.Text] does. It returns an empty string if the chunk has none.
// For the parsed structured output, use [ModelResponseChunk.Output] instead.
func (c *ModelResponseChunk) Text() string {
	if c == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range c.Content {
		// Data parts are deliberately excluded: their payload lives on Data (not
		// Text), so they contribute no text (e.g. A2UI envelope JSON must not
		// leak into Text()).
		if p.IsText() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// Reasoning returns the reasoning content of the ModelResponseChunk as a string.
// It returns an empty string if there is no Content in the response chunk.
func (c *ModelResponseChunk) Reasoning() string {
	if c == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range c.Content {
		if p.IsReasoning() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// Interrupts returns the interrupted tool request parts from the chunk.
func (c *ModelResponseChunk) Interrupts() []*Part {
	var parts []*Part
	if c == nil {
		return parts
	}
	for _, p := range c.Content {
		if p.IsInterrupt() {
			parts = append(parts, p)
		}
	}
	return parts
}

// ToolResponses returns the tool response parts from the chunk.
// Use [Part.IsPartial] to distinguish streaming progress updates
// from final tool results.
func (c *ModelResponseChunk) ToolResponses() []*Part {
	var parts []*Part
	if c == nil {
		return parts
	}
	for _, p := range c.Content {
		if p.IsToolResponse() {
			parts = append(parts, p)
		}
	}
	return parts
}

// Output parses the chunk using the format handler and unmarshals the result into v.
// Returns an error if the format handler is not set or does not support parsing chunks.
func (c *ModelResponseChunk) Output(v any) error {
	if c == nil {
		return errors.New("chunk is nil")
	}
	if c.formatHandler == nil {
		return errors.New("output format chosen does not support parsing chunks")
	}

	output, err := c.formatHandler.ParseChunk(c)
	if err != nil {
		return err
	}

	b, err := json.Marshal(output)
	if err != nil {
		return fmt.Errorf("failed to marshal chunk output: %w", err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("failed to unmarshal output: %w", err)
	}
	return nil
}

// outputer is an interface for types that can unmarshal structured output.
type outputer interface {
	// Text returns the contents of the output as a string.
	Text() string
	// Output parses the structured output from the response and unmarshals it into value.
	Output(value any) error
}

// OutputFrom is a convenience function that parses structured output from a
// [ModelResponse] or [ModelResponseChunk] and returns it as a typed value.
// This is equivalent to calling Output() but returns the value directly instead
// of requiring a pointer argument. If you need to handle the error, use Output() instead.
func OutputFrom[Out any](src outputer) Out {
	output, err := extractTypedOutput[Out](src)
	if err != nil {
		return base.Zero[Out]()
	}
	return output
}

// extractTypedOutput extracts the typed output from a model response.
// It supports string output by calling Text() and returning the result.
func extractTypedOutput[Out any](o outputer) (Out, error) {
	var output Out

	switch any(output).(type) {
	case string:
		text := o.Text()
		// Type assertion to convert string to Out (which we know is string).
		result := any(text).(Out)
		return result, nil
	default:
		if err := o.Output(&output); err != nil {
			return base.Zero[Out](), fmt.Errorf("failed to parse output: %w", err)
		}
		return output, nil
	}
}

// Text returns the textual contents of a [Message] as a string, concatenating
// its text parts and skipping every other kind. It returns an empty string if
// the message has none, which is what a message carrying only an image, only
// raw data, or only a tool request comes back as; read those with
// [Message.MediaParts] and ToolRequests instead.
//
// A data part is deliberately not text: it holds a blob, which the plugins
// send as bytes and [plugins/internal/uri.Data] decodes as a data: URI, so
// concatenating one here would splice base64 into prose.
// If you want to get reasoning from the message, use Reasoning() instead.
func (m *Message) Text() string {
	if m == nil {
		return ""
	}
	if len(m.Content) == 0 {
		return ""
	}
	// Single-part messages are the common case and skip the builder, but they
	// are still filtered: a lone media part is not this message's text.
	if len(m.Content) == 1 {
		// Fast path only applies to a text part; a lone data part carries its
		// payload on Data (not Text) and must not be returned as message text.
		if m.Content[0].IsText() {
			return m.Content[0].Text
		}
	}
	var sb strings.Builder
	for _, p := range m.Content {
		// Data parts are deliberately excluded: their payload lives on Data (not
		// Text), so they contribute no text (e.g. A2UI envelope JSON must not
		// leak into Text()).
		if p.IsText() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// MediaParts returns every media part of a [Message], each carrying its content
// type alongside its data. It returns nil if the message has none.
func (m *Message) MediaParts() []*Part {
	if m == nil {
		return nil
	}
	var parts []*Part
	for _, p := range m.Content {
		if p.IsMedia() {
			parts = append(parts, p)
		}
	}
	return parts
}

// NewResume constructs a [GenerateActionResume] from Part slices.
// This is useful when building [GenerateActionOptions] directly (e.g., from a
// rendered prompt) and need to set the Resume field from [*Part] values
// produced by [InterruptedCall.Restart] and [InterruptedCall.Respond], or
// [Part.ToToolRestart] and [Part.ToToolResponse]. [WithResume] is the
// same for [Generate].
func NewResume(restarts, responds []*Part) *GenerateActionResume {
	return &GenerateActionResume{
		Restart: restarts,
		Respond: responds,
	}
}

// NewModelRef creates a new ModelRef with the given name and configuration.
func NewModelRef(name string, config any) ModelRef {
	return ModelRef{name: name, config: config}
}

// Name returns the name of the model.
func (m ModelRef) Name() string {
	return m.name
}

// Config returns the configuration to use by default for this model.
func (m ModelRef) Config() any {
	return m.config
}

// MarshalJSON implements [json.Marshaler]. ModelRef always marshals as a
// JSON object with "name" and optional "config" fields.
func (m ModelRef) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Name   string `json:"name"`
		Config any    `json:"config,omitempty"`
	}{
		Name:   m.name,
		Config: m.config,
	})
}

// UnmarshalJSON implements [json.Unmarshaler]. It accepts either a JSON
// object with "name" and optional "config" fields, or a plain string
// (interpreted as the model name).
func (m *ModelRef) UnmarshalJSON(data []byte) error {
	// Try string shorthand first.
	var name string
	if err := json.Unmarshal(data, &name); err == nil {
		m.name = name
		m.config = nil
		return nil
	}
	var obj struct {
		Name   string          `json:"name"`
		Config json.RawMessage `json:"config,omitempty"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	m.name = obj.Name
	if len(obj.Config) > 0 {
		var config any
		if err := json.Unmarshal(obj.Config, &config); err != nil {
			return err
		}
		m.config = config
	}
	return nil
}

// JSONSchema implements the invopop/jsonschema customSchemaImpl interface
// so that schema reflection produces the correct object schema instead of
// an empty object (ModelRef has only unexported fields).
func (ModelRef) JSONSchema() *jsonschema.Schema {
	props := jsonschema.NewProperties()
	props.Set("name", &jsonschema.Schema{
		Type:        "string",
		Description: "Model name, e.g. \"googleai/gemini-flash-latest\".",
	})
	props.Set("config", &jsonschema.Schema{
		Description: "Optional model configuration, applied to this model only.",
	})
	return &jsonschema.Schema{
		Type:       "object",
		Properties: props,
		Required:   []string{"name"},
	}
}

// resumeStep is how one pending tool request of a resume resolves, decided
// by planResumedToolRequest before any tool runs.
type resumeStep struct {
	request *Part // The request in history.
	tool    Tool  // Nil when a pending output is replayed.
	respond *Part // The Respond directive that answers the request, if any.
	restart *Part // The Restart directive that re-executes it, if any.
}

// planResumedToolRequest decides how tool request p resolves in a resume:
// by replaying the pending output a previous run left on it, or by the
// Respond or Restart directive in genOpts.Resume that matches it. It runs for
// every pending request before any tool runs, and so does everything that can
// reject the resume without running a tool: a request with no resolution, a
// tool that is not found, a response that does not match the tool's output
// schema, and a restart payload for the tool that does not match its resume
// schema. A rejected resume thus leaves no tool half-run next to it.
func planResumedToolRequest(r api.Registry, genOpts *GenerateActionOptions, p *Part) (*resumeStep, error) {
	if p == nil || !p.IsToolRequest() {
		return nil, status.Errorf(ErrInvalidPart, "handleResumedToolRequest: part is not a tool request")
	}
	// The kind check alone lets through a tool request part with no request,
	// which history assembled by hand can carry; every step below reads it.
	if p.ToolRequest == nil {
		return nil, status.Errorf(ErrInvalidPart, "handleResumedToolRequest: tool request part has no request")
	}
	if _, ok := p.Metadata["pendingOutput"]; ok {
		return &resumeStep{request: p}, nil
	}

	toolReq := p.ToolRequest
	var respondPart, restartPart *Part
	if genOpts.Resume != nil {
		respondPart = resumePartFor(genOpts.Resume.Respond, toolReq, true)
		restartPart = resumePartFor(genOpts.Resume.Restart, toolReq, false)
	}
	if respondPart == nil && restartPart == nil {
		refStr := toolReq.Name
		if toolReq.Ref != "" {
			refStr += "#" + toolReq.Ref
		}
		// A hold a WrapTool hook raised is the one interrupt the tool's
		// typed claim declines, so a claim loop over the tools skips it
		// silently; say so, since that is the likeliest reason it went
		// unanswered.
		hint := ""
		if it := p.interruptState(); it != nil && it.RaisedBy != "" {
			hint = fmt.Sprintf(" The interrupt is a hold by middleware %q, which the tool's Interrupted does not claim; answer it with Part.ToToolRestart or Part.ToToolResponse.", it.RaisedBy)
		}
		return nil, status.Errorf(ErrUnresolvedToolRequest, "unresolved tool request %q was not handled by the Resume argument; you must supply Respond or Restart directives, or ensure there is pending output from a previous tool call.%s", refStr, hint)
	}

	tool := LookupTool(r, toolReq.Name)
	if tool == nil {
		return nil, status.Errorf(ErrToolNotFound, "handleResumedToolRequest: tool %q not found", toolReq.Name)
	}
	if respondPart != nil {
		if schema := tool.Definition().OutputSchema; len(schema) > 0 {
			if err := base.ValidateValue(respondPart.ToolResponse.Output, schema); err != nil {
				return nil, status.Errorf(status.ErrInvalidArgument, "handleResumedToolRequest: tool %q output validation failed: %w", tool.Name(), err)
			}
		}
		return &resumeStep{request: p, tool: tool, respond: respondPart}, nil
	}

	// A payload that reaches the tool, rather than a hook that held the call,
	// is checked by its JSON shape against the schema the tool advertises for
	// its resume type. The tool reads it the same way when it runs; checking
	// here as well is what keeps a bad payload from failing the resume after
	// a sibling ran. A tool that advertises none (one behind a foreign
	// action) is not checked.
	if rs := restartPart.restartState(); rs != nil {
		if it := p.interruptState(); it == nil || it.RaisedBy == "" {
			if schema, ok := tool.Definition().Metadata[toolResumeSchemaKey].(map[string]any); ok {
				payload, _ := resumePayload(rs)
				m, err := base.ObjectPayload(payload, "resume data")
				if err == nil {
					err = base.ValidateValue(m, schema)
				}
				if err != nil {
					return nil, status.Errorf(status.ErrInvalidArgument, "handleResumedToolRequest: tool %q resume data validation failed: %w", tool.Name(), err)
				}
			}
		}
	}
	return &resumeStep{request: p, tool: tool, restart: restartPart}, nil
}

// handleResumedToolRequest carries out step: it replays the pending output,
// builds the response a Respond directive supplies, or re-executes the tool
// for a Restart directive.
func handleResumedToolRequest(ctx context.Context, step *resumeStep, runTool toolRunnerFunc, stream *toolStreamer) (*resumedToolRequestOutput, error) {
	p := step.request
	switch {
	case step.tool == nil:
		// Only the metadata map needs detaching from the caller's part; a
		// deep clone would JSON round-trip the (possibly large) pending
		// payload just to delete it.
		reqPart := *p
		reqPart.Metadata = maps.Clone(p.Metadata)
		delete(reqPart.Metadata, "pendingOutput")
		delete(reqPart.Metadata, "pendingMetadata")
		delete(reqPart.Metadata, "pendingContent")

		newRespPart := NewResponseForToolRequest(p, p.Metadata["pendingOutput"])
		newRespPart.Metadata = map[string]any{"source": "pending"}
		// Restore the response metadata and content parts the original call
		// carried, stashed next to pendingOutput. The content is []*Part in
		// process and generic JSON after a wire or persistence round-trip;
		// ConvertTo decodes both.
		if pm, ok := p.Metadata["pendingMetadata"].(map[string]any); ok {
			maps.Copy(newRespPart.Metadata, pm)
		}
		if content, ok := base.ConvertTo[[]*Part](p.Metadata["pendingContent"]); ok && len(content) > 0 {
			newRespPart.ToolResponse.Content = content
		}
		return &resumedToolRequestOutput{
			toolRequest:  &reqPart,
			toolResponse: newRespPart,
		}, nil

	case step.respond != nil:
		newToolResp := NewToolResponsePart(step.respond.ToolResponse)
		newToolResp.Metadata = step.respond.Metadata
		return &resumedToolRequestOutput{
			toolRequest:  resolvedPart(p),
			toolResponse: newToolResp,
		}, nil
	}

	// History records the request as it ran, so a restart that replaced the
	// input amends the request, whether the tool then completes or
	// interrupts again.
	p = withRestartInput(p, step.restart)
	newToolResp, interrupt, err := restartedToolResponse(ctx, step.tool, p, step.restart, runTool, stream)
	if interrupt != nil {
		return &resumedToolRequestOutput{interrupt: interrupt}, nil
	}
	if err != nil {
		return nil, err
	}
	return &resumedToolRequestOutput{
		toolRequest:  resolvedPart(p),
		toolResponse: newToolResp,
	}, nil
}

// resumePartFor returns the first of parts whose tool name and ref match req:
// the tool response of a Respond directive when respond is set, otherwise the
// tool request of a Restart directive. Nil when none matches.
func resumePartFor(parts []*Part, req *ToolRequest, respond bool) *Part {
	for _, p := range parts {
		if p == nil {
			continue
		}
		if respond {
			if p.ToolResponse != nil && p.ToolResponse.Name == req.Name && p.ToolResponse.Ref == req.Ref {
				return p
			}
		} else if p.ToolRequest != nil && p.ToolRequest.Name == req.Name && p.ToolRequest.Ref == req.Ref {
			return p
		}
	}
	return nil
}

// withRestartInput returns tool request p as restartPart re-executes it: p
// itself, or, when the restart replaced the input, a copy of p carrying the
// new input. The original input is not recorded on the copy: on a tool
// request in history that key would read as a restart.
func withRestartInput(p, restartPart *Part) *Part {
	rs := restartPart.restartState()
	if rs == nil || rs.OriginalInput == nil {
		return p
	}
	revised := p.typedClone()
	req := *p.ToolRequest
	req.Input = restartPart.ToolRequest.Input
	revised.ToolRequest = &req
	return revised
}

// resumePayload returns the payload restart rs delivers: its resume data when
// that is a JSON object, and an empty map for a bare restart, so the call
// still reads as a resumption. ok is false when rs carries a marker that is
// not an object: a peer runtime may mark a restart with any truthy JSON value
// (the JS restartTool passes its resumedMetadata through as given), and Go
// delivers only an object, so such a marker reads as a bare restart.
func resumePayload(rs *ToolRestart) (payload any, ok bool) {
	switch {
	case base.IsNil(rs.Resume):
		return map[string]any{}, true
	case base.IsJSONObject(rs.Resume):
		return rs.Resume, true
	}
	return map[string]any{}, false
}

// restartedToolResponse re-executes tool for a Restart directive and builds
// its tool response. The stage the restart answers, the WrapTool hook that
// held the call or the tool itself, sees it through the context: the resume
// payload it was given (an empty map for a bare restart, so the call still
// reads as a resumption) and, when the caller replaced the input, the
// original one. The payload rides as given, a map or the caller's struct, and
// each reader converts it to what it returns, so a typed restart reaches the
// tool's resume parameter without a conversion. A tool that interrupts again
// returns the interrupted copy of p instead of a response.
func restartedToolResponse(ctx context.Context, tool Tool, p, restartPart *Part, runTool toolRunnerFunc, stream *toolStreamer) (resp, interrupt *Part, err error) {
	name := restartPart.ToolRequest.Name
	resumedCtx := stream.context(ctx, restartPart.ToolRequest)
	var restart *base.ToolRestart
	if rs := restartPart.restartState(); rs != nil {
		resume, ok := resumePayload(rs)
		if !ok {
			logger.Debug(ctx, "resume payload is not a JSON object; restarting with an empty payload", "tool", name, "type", fmt.Sprintf("%T", rs.Resume))
		}
		// A restart answers the stage that raised the interrupt: the
		// WrapTool hook the interrupted request in history names, or the
		// tool itself. The hook chain delivers the payload to that stage
		// alone (see toolStageContext), so a hook's hold never reaches the
		// tool's resume parameter: the tool re-executes as a fresh call
		// that may interrupt on its own.
		var (
			raisedBy string
			released []string
		)
		if it := p.interruptState(); it != nil {
			raisedBy, released = it.RaisedBy, it.ReleasedBy
		}
		// The payload is not checked here: a tool with a resume type knows
		// whether it arrived as that type, and validates it against its
		// resume schema when it did not (see NewResumableTool).
		restart = &base.ToolRestart{Resume: resume, OriginalInput: rs.OriginalInput, RaisedBy: raisedBy, Released: released}
	}
	// Set even when nil, so a restart an enclosing tool call was answering
	// does not reach this one.
	resumedCtx = base.ToolRestartKey.NewContext(resumedCtx, restart)

	sink := &base.PartSink{}
	defer sink.Close() // On an error or interrupt, too.
	resumedCtx = base.ToolPartSinkKey.NewContext(resumedCtx, sink)
	multipartResp, err := runTool(resumedCtx, tool, &ToolRequest{
		Name:  name,
		Ref:   restartPart.ToolRequest.Ref,
		Input: restartPart.ToolRequest.Input,
	})
	if err != nil {
		var tie *base.ToolInterruptError
		if errors.As(err, &tie) {
			logger.Debug(ctx, "restarted tool triggered an interrupt", "tool", name)
			interrupt, ierr := interruptedPart(p, tie)
			if ierr != nil {
				return nil, nil, toolFailureError(ctx, name, ierr)
			}
			return nil, interrupt, nil
		}
		return nil, nil, toolFailureError(ctx, name, err)
	}
	multipartResp = foldAttachedParts(multipartResp, sink)

	newToolResp := NewToolResponsePart(&ToolResponse{
		Name:    name,
		Ref:     restartPart.ToolRequest.Ref,
		Output:  multipartResp.Output,
		Content: multipartResp.Content,
	})
	newToolResp.Metadata = multipartResp.Metadata
	return newToolResp, nil, nil
}

// handleResumeOption amends message history to handle `resume` arguments.
// It returns the amended history.
func handleResumeOption(ctx context.Context, r api.Registry, genOpts *GenerateActionOptions, runTool toolRunnerFunc, cb ModelStreamCallback) (*resumeOptionOutput, error) {
	if genOpts.Resume == nil || (len(genOpts.Resume.Respond) == 0 && len(genOpts.Resume.Restart) == 0) {
		return &resumeOptionOutput{revisedRequest: genOpts}, nil
	}

	// Validate runs first so that a nil part, which a deprecated verb returns
	// for a request it cannot restart, is reported as nil rather than as the
	// wrong kind.
	for _, part := range genOpts.Resume.Respond {
		if err := part.Validate(); err != nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "handleResumeOption: respond part: %w", err)
		}
		if !part.IsToolResponse() {
			return nil, status.Errorf(status.ErrInvalidArgument, "handleResumeOption: respond part is not a tool response")
		}
	}
	for _, part := range genOpts.Resume.Restart {
		if err := part.Validate(); err != nil {
			return nil, status.Errorf(ErrInvalidPart, "handleResumeOption: restart part: %w", err)
		}
		if !part.IsToolRequest() {
			return nil, status.Errorf(ErrInvalidPart, "handleResumeOption: restart part is not a tool request")
		}
	}

	for _, t := range genOpts.Tools {
		if LookupTool(r, t) == nil {
			return nil, status.Errorf(ErrToolNotFound, "handleResumeOption: tool %q not found", t)
		}
	}

	messages := genOpts.Messages
	if len(messages) == 0 {
		return nil, status.Errorf(status.ErrFailedPrecondition, "handleResumeOption: cannot resume generation with no messages")
	}
	lastMessage := messages[len(messages)-1]

	if lastMessage.Role != RoleModel || !slices.ContainsFunc(lastMessage.Content, func(p *Part) bool { return p.IsToolRequest() }) {
		return nil, status.Errorf(status.ErrFailedPrecondition, "handleResumeOption: cannot resume generation unless the last message is by a model with at least one tool request")
	}

	toolReqCount := 0
	for _, part := range lastMessage.Content {
		if part.IsToolRequest() {
			toolReqCount++
		}
	}

	resultChan := make(chan result[*resumedToolRequestOutput], toolReqCount)
	newContent := make([]*Part, len(lastMessage.Content))
	// Restarted tools stream as the tools of a model turn do.
	stream := &toolStreamer{cb: cb}

	// Every request is planned before any tool runs, so a resume that
	// cannot go through fails without having run part of it.
	steps := make(map[int]*resumeStep, toolReqCount)
	for i, part := range lastMessage.Content {
		if !part.IsToolRequest() {
			newContent[i] = part
			continue
		}
		step, err := planResumedToolRequest(r, genOpts, part)
		if err != nil {
			return nil, fmt.Errorf("handleResumeOption: failed to resolve resumed tool request: %w", err)
		}
		steps[i] = step
	}

	for i, step := range steps {
		go func(idx int, step *resumeStep) {
			output, err := handleResumedToolRequest(ctx, step, runTool, stream)
			resultChan <- result[*resumedToolRequestOutput]{
				index: idx,
				value: output,
				err:   err,
			}
		}(i, step)
	}

	respByIndex := make(map[int]*Part, toolReqCount)
	interrupted := false

	for range toolReqCount {
		res := <-resultChan
		if res.err != nil {
			return nil, fmt.Errorf("handleResumeOption: failed to resolve resumed tool request: %w", res.err)
		}

		if res.value.interrupt != nil {
			interrupted = true
			newContent[res.index] = res.value.interrupt
		} else {
			respByIndex[res.index] = res.value.toolResponse
			newContent[res.index] = res.value.toolRequest
		}
	}

	lastMessage.Content = newContent

	if interrupted {
		// Siblings resolved in this resume (restarted runs, supplied
		// responses, replayed pending outputs) are preserved as
		// pendingOutput on their request parts, the way a first-run
		// interrupt preserves completed siblings, so the next resume
		// replays their outcomes instead of demanding new directives.
		for idx, respPart := range respByIndex {
			stampPendingToolOutcome(newContent[idx], &MultipartToolResponse{
				Output:   respPart.ToolResponse.Output,
				Content:  respPart.ToolResponse.Content,
				Metadata: respPart.Metadata,
			})
		}
		return &resumeOptionOutput{
			interruptedResponse: &ModelResponse{
				Message:       lastMessage,
				FinishReason:  "interrupted",
				FinishMessage: "One or more tools triggered interrupts while resuming generation. The model was not called.",
			},
		}, nil
	}

	if len(respByIndex) != toolReqCount {
		return nil, status.Errorf(status.ErrFailedPrecondition, "handleResumeOption: Expected %d tool responses but resolved to %d.", toolReqCount, len(respByIndex))
	}

	// Emit tool responses in the order their requests appear in the model
	// message, matching handleToolRequests, so the resumed tool message is
	// deterministic across runs.
	toolResps := make([]*Part, 0, len(respByIndex))
	for i := range newContent {
		if part, ok := respByIndex[i]; ok {
			toolResps = append(toolResps, part)
		}
	}

	// Message metadata, not part metadata: the typed restart state lives on the
	// individual tool request parts, while this marks the whole tool message as
	// the product of a resumption.
	toolMessage := &Message{
		Role:    RoleTool,
		Content: toolResps,
		Metadata: map[string]any{
			metaResumed: true,
		},
	}
	if genOpts.Resume.Metadata != nil {
		toolMessage.Metadata[metaResumed] = genOpts.Resume.Metadata
	}
	revisedMessages := append(slices.Clone(messages), toolMessage)

	// Copied rather than rebuilt field by field, so that a field added to
	// GenerateActionOptions carries through a resume without being listed here.
	revised := *genOpts
	revised.Messages = revisedMessages
	revised.Resume = nil // These directives have now been applied.

	return &resumeOptionOutput{
		revisedRequest: &revised,
		toolMessage:    toolMessage,
	}, nil
}

// processResources processes messages to replace resource parts with actual content.
func processResources(ctx context.Context, r api.Registry, messages []*Message) ([]*Message, error) {
	processedMessages := make([]*Message, len(messages))
	for i, msg := range messages {
		processedContent := []*Part{}

		for _, part := range msg.Content {
			if part.IsResource() {
				// Find and execute the matching resource
				resourceParts, err := executeResourcePart(ctx, r, part.Resource.Uri)
				if err != nil {
					return nil, fmt.Errorf("failed to process resource %q: %w", part.Resource, err)
				}
				// Replace resource part with content parts
				processedContent = append(processedContent, resourceParts...)
			} else {
				// Keep non-resource parts as-is
				processedContent = append(processedContent, part)
			}
		}

		processedMessages[i] = &Message{
			Role:     msg.Role,
			Content:  processedContent,
			Metadata: msg.Metadata,
		}
	}

	return processedMessages, nil
}

// executeResourcePart finds and executes a resource, returning the content parts.
func executeResourcePart(ctx context.Context, r api.Registry, resourceURI string) ([]*Part, error) {
	resource, input, err := FindMatchingResource(r, resourceURI)
	if err != nil {
		return nil, err
	}

	output, err := resource.Execute(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to execute resource %q: %w", resourceURI, err)
	}

	return output.Content, nil
}
