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
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

// Model represents a model that can generate content based on a request.
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

// ModelStreamCallback is a stream callback of a ModelAction.
type ModelStreamCallback = func(context.Context, *ModelResponseChunk) error

// ModelMiddleware is middleware for model generate requests that takes in a ModelFunc, does something, then returns another ModelFunc.
//
// Deprecated: Use [Middleware] interface with [WithUse] instead, which supports Generate, Model, and Tool hooks.
type ModelMiddleware = core.Middleware[*ModelRequest, *ModelResponse, *ModelResponseChunk]

// model is an action with functions specific to model generation such as Generate().
type model struct {
	core.ActionDef[*ModelRequest, *ModelResponse, *ModelResponseChunk]
}

// generateAction is the type for a utility model generation action that takes in a GenerateActionOptions instead of a ModelRequest.
type generateAction = core.ActionDef[*GenerateActionOptions, *ModelResponse, *ModelResponseChunk]

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
	ConfigSchema map[string]any // JSON schema for the model's config.
	Label        string         // User-friendly name for the model.
	Stage        ModelStage     // Indicates the maturity stage of the model.
	Supports     *ModelSupports // Capabilities of the model.
	Versions     []string       // Available versions of the model.
}

// DefineGenerateAction defines a utility generate action.
func DefineGenerateAction(ctx context.Context, r api.Registry) *generateAction {
	return (*generateAction)(core.DefineStreamingAction(r, "generate", api.ActionTypeUtil, nil, nil,
		func(ctx context.Context, actionOpts *GenerateActionOptions, cb ModelStreamCallback) (resp *ModelResponse, err error) {
			actionOptsBytes, _ := json.Marshal(actionOpts)
			logger.FromContext(ctx).Debug("GenerateAction",
				"input", actionOptsBytes)
			defer func() {
				respBytes, _ := json.Marshal(resp)
				logger.FromContext(ctx).Debug("GenerateAction",
					"output", respBytes,
					"err", err)
			}()

			return GenerateWithRequest(ctx, r, actionOpts, nil, cb)
		}))
}

// NewModel creates a new [Model].
func NewModel(name string, opts *ModelOptions, fn ModelFunc) Model {
	if name == "" {
		panic("ai.NewModel: name is required")
	}

	if opts == nil {
		opts = &ModelOptions{
			Label: name,
		}
	}
	if opts.Supports == nil {
		opts.Supports = &ModelSupports{}
	}

	metadata := map[string]any{
		"type": api.ActionTypeModel,
		"model": map[string]any{
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
			"customOptions": opts.ConfigSchema,
		},
	}

	inputSchema := core.InferSchemaMap(ModelRequest{})
	if inputSchema != nil && opts.ConfigSchema != nil {
		if props, ok := inputSchema["properties"].(map[string]any); ok {
			props["config"] = opts.ConfigSchema
		}
	}

	mws := []ModelMiddleware{
		simulateSystemPrompt(opts, nil),
		augmentWithContext(opts, nil),
		validateSupport(name, opts),
		addAutomaticTelemetry(),
	}
	fn = core.ChainMiddleware(mws...)(fn)

	return &model{*core.NewStreamingAction(name, api.ActionTypeModel, metadata, inputSchema, fn)}
}

// DefineModel creates a new [Model] and registers it.
func DefineModel(r api.Registry, name string, opts *ModelOptions, fn ModelFunc) Model {
	m := NewModel(name, opts, fn)
	m.Register(r)
	return m
}

// LookupModel looks up a [Model] registered by [DefineModel].
// It will try to resolve the model dynamically if the model is not found.
// It returns nil if the model was not resolved.
func LookupModel(r api.Registry, name string) Model {
	action := core.ResolveActionFor[*ModelRequest, *ModelResponse, *ModelResponseChunk](r, api.ActionTypeModel, name)
	if action == nil {
		return nil
	}
	return &model{
		ActionDef: *action,
	}
}

// GenerateWithRequest is the central generation implementation for ai.Generate(), prompt.Execute(), and the GenerateAction direct call.
func GenerateWithRequest(ctx context.Context, r api.Registry, opts *GenerateActionOptions, mmws []ModelMiddleware, cb ModelStreamCallback) (*ModelResponse, error) {
	if opts.Model == "" {
		if defaultModel, ok := r.LookupValue(api.DefaultModelKey).(string); ok && defaultModel != "" {
			opts.Model = defaultModel
		}
		if opts.Model == "" {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: model is required")
		}
	}

	m := LookupModel(r, opts.Model)
	bm := LookupBackgroundModel(r, opts.Model)
	if m == nil && bm == nil {
		return nil, core.NewError(core.NOT_FOUND, "ai.GenerateWithRequest: model %q not found", opts.Model)
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
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: duplicate tool %q", t)
		}

		tool := LookupTool(r, t)
		if tool == nil {
			return nil, core.NewError(core.NOT_FOUND, "ai.GenerateWithRequest: tool %q not found", t)
		}

		toolDefMap[t] = tool.Definition()
	}
	var middlewareTools []Tool
	for _, mw := range mws {
		if mw == nil {
			continue
		}
		for _, t := range mw.Tools {
			if _, ok := toolDefMap[t.Name()]; ok {
				return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: tool %q is contributed by middleware but already declared elsewhere", t.Name())
			}
			toolDefMap[t.Name()] = t.Definition()
			middlewareTools = append(middlewareTools, t)
		}
	}
	if len(middlewareTools) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, t := range middlewareTools {
			t.Register(r)
		}
	}
	toolDefs := make([]*ToolDefinition, 0, len(toolDefMap))
	for _, t := range toolDefMap {
		toolDefs = append(toolDefs, t)
	}

	maxTurns := opts.MaxTurns
	if maxTurns < 0 {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: max turns must be greater than 0, got %d", maxTurns)
	}
	if maxTurns == 0 {
		maxTurns = 5 // Default max turns.
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
			opts.Output.Constrained && outputCfg.Constrained && m != nil && m.(*model).supportsConstrained(len(toolDefs) > 0)

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
			logger.FromContext(ctx).Warn("background model does not support streaming", "model", bm.Name())
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
	var runGenerate func(context.Context, *GenerateParams) (*ModelResponse, error)

	runGenerate = func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
		name := params.Options.StepName
		if name == "" {
			name = "generate"
		}
		spanMetadata := &tracing.SpanMetadata{
			Name:    name,
			Type:    "util",
			Subtype: "util",
		}

		return tracing.RunInNewSpan(ctx, spanMetadata, params.Options, func(ctx context.Context, _ *GenerateActionOptions) (*ModelResponse, error) {
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

			// Handle resume on the first iteration inside this span so that
			// restarted tool execution is both wrapped by WrapGenerate (from
			// the outer middleware chain) and recorded as a child of the
			// current generate span (lifecycle: generate > tool > generate >
			// model > tool).
			if currentTurn == 0 && opts.Resume != nil && (len(opts.Resume.Respond) > 0 || len(opts.Resume.Restart) > 0) {
				resumeOutput, err := handleResumeOption(ctx, r, opts, runTool)
				if err != nil {
					return nil, err
				}

				if resumeOutput.interruptedResponse != nil {
					return nil, core.NewError(core.FAILED_PRECONDITION,
						"One or more tools triggered an interrupt during a restarted execution.")
				}

				opts = resumeOutput.revisedRequest

				if resumeOutput.toolMessage != nil && wrappedCb != nil {
					if err := wrappedCb(ctx, &ModelResponseChunk{
						Content: resumeOutput.toolMessage.Content,
						Role:    RoleTool,
					}); err != nil {
						return nil, fmt.Errorf("streaming callback failed for resumed tool message: %w", err)
					}
				}

				resumeReq := &ModelRequest{
					Messages:   opts.Messages,
					Config:     req.Config,
					Docs:       req.Docs,
					ToolChoice: req.ToolChoice,
					Tools:      req.Tools,
					Output:     req.Output,
				}
				return generate(ctx, resumeReq, currentTurn+1, currentIndex)
			}

			resp, err := fn(ctx, req, wrappedCb)
			if err != nil {
				return nil, err
			}

			// Ensure all tool requests have unique refs for matching during resume.
			ensureToolRequestRefs(resp.Message)

			// If this is a long-running operation response, return it immediately without further processing
			if bm != nil && resp.Operation != nil {
				return resp, nil
			}

			if formatHandler != nil {
				resp.formatHandler = streamingHandler
				// This is legacy behavior. New format handlers should implement ParseMessage as a passthrough.
				resp.Message, err = formatHandler.ParseMessage(resp.Message)
				if err != nil {
					logger.FromContext(ctx).Debug("model failed to generate output matching expected schema", "error", err.Error())
					return nil, core.NewError(core.INTERNAL, "model failed to generate output matching expected schema: %v", err)
				}
			}

			if len(resp.ToolRequests()) == 0 || opts.ReturnToolRequests {
				return resp, nil
			}

			if currentTurn+1 > maxTurns {
				return nil, core.NewError(core.ABORTED, "exceeded maximum tool call iterations (%d)", maxTurns)
			}

			newReq, interruptMsg, err := handleToolRequests(ctx, r, req, resp, wrappedCb, currentIndex, runTool)
			if err != nil {
				return nil, err
			}
			if interruptMsg != nil {
				resp.FinishReason = "interrupted"
				resp.FinishMessage = "One or more tool calls resulted in interrupts."
				resp.Message = interruptMsg
				return resp, nil
			}
			if newReq == nil {
				return resp, nil
			}

			return generate(ctx, newReq, currentTurn+1, currentIndex+1)
		})
	}

	// Compose WrapGenerate hooks once; this chain is invoked for every
	// tool-loop iteration.
	hookedGenerate := buildGenerateChain(mws, runGenerate)

	generate = func(ctx context.Context, req *ModelRequest, currentTurn int, messageIndex int) (*ModelResponse, error) {
		return hookedGenerate(ctx, &GenerateParams{
			Options:      opts,
			Request:      req,
			Iteration:    currentTurn,
			MessageIndex: messageIndex,
			Callback:     middlewareCb,
		})
	}

	return generate(ctx, req, 0, 0)
}

// buildGenerateChain composes the WrapGenerate hooks from mws (outer-to-inner)
// around run. Middleware with a nil WrapGenerate hook is skipped.
func buildGenerateChain(mws []*Hooks, run func(ctx context.Context, params *GenerateParams) (*ModelResponse, error)) func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
	chain := run
	for i := len(mws) - 1; i >= 0; i-- {
		mw := mws[i]
		if mw == nil || mw.WrapGenerate == nil {
			continue
		}
		hook := mw.WrapGenerate
		next := chain
		chain = func(ctx context.Context, params *GenerateParams) (*ModelResponse, error) {
			return hook(ctx, params, next)
		}
	}
	return chain
}

// buildModelChain composes the WrapModel hooks from mws (outer-to-inner)
// around fn. Middleware with a nil WrapModel hook is skipped.
func buildModelChain(mws []*Hooks, fn ModelFunc) ModelFunc {
	chain := fn
	for i := len(mws) - 1; i >= 0; i-- {
		mw := mws[i]
		if mw == nil || mw.WrapModel == nil {
			continue
		}
		hook := mw.WrapModel
		next := chain
		chain = func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return hook(ctx, &ModelParams{Request: req, Callback: cb},
				func(ctx context.Context, params *ModelParams) (*ModelResponse, error) {
					return next(ctx, params.Request, params.Callback)
				})
		}
	}
	return chain
}

// buildToolRunner composes the WrapTool hooks from mws (outer-to-inner) into
// a single function that executes a tool. The returned function is safe to
// invoke from concurrent goroutines; each invocation threads its own params
// through the shared hook chain. When no WrapTool hooks are configured, the
// tool is invoked directly without allocating a ToolParams wrapper.
func buildToolRunner(mws []*Hooks) func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error) {
	hasHook := false
	for _, mw := range mws {
		if mw != nil && mw.WrapTool != nil {
			hasHook = true
			break
		}
	}
	if !hasHook {
		return func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error) {
			return tool.RunRawMultipart(ctx, req.Input)
		}
	}
	chain := func(ctx context.Context, params *ToolParams) (*MultipartToolResponse, error) {
		return params.Tool.RunRawMultipart(ctx, params.Request.Input)
	}
	for i := len(mws) - 1; i >= 0; i-- {
		mw := mws[i]
		if mw == nil || mw.WrapTool == nil {
			continue
		}
		hook := mw.WrapTool
		next := chain
		chain = func(ctx context.Context, params *ToolParams) (*MultipartToolResponse, error) {
			return hook(ctx, params, next)
		}
	}
	return func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error) {
		return chain(ctx, &ToolParams{Request: req, Tool: tool})
	}
}

// Generate generates a model response based on the provided options.
func Generate(ctx context.Context, r api.Registry, opts ...GenerateOption) (*ModelResponse, error) {
	genOpts := &generateOptions{}
	for _, opt := range opts {
		if err := opt.applyGenerate(genOpts); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Generate: error applying options: %v", err)
		}
	}

	if genOpts.OutputSchema != nil {
		resolved, err := core.ResolveSchema(r, genOpts.OutputSchema)
		if err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Generate: invalid output schema: %v", err)
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

	messages := []*Message{}
	if genOpts.SystemFn != nil {
		system, err := genOpts.SystemFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, NewSystemTextMessage(system))
	}
	if genOpts.MessagesFn != nil {
		msgs, err := genOpts.MessagesFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, msgs...)
	}
	if genOpts.PromptFn != nil {
		prompt, err := genOpts.PromptFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, NewUserTextMessage(prompt))
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
		return nil, core.NewError(core.INTERNAL, "ai.Generate: error processing resources: %v", err)
	}
	actionOpts.Messages = processedMessages

	return GenerateWithRequest(ctx, r, actionOpts, genOpts.Middleware, genOpts.Stream)
}

// GenerateText run generate request for this model. Returns generated text only.
func GenerateText(ctx context.Context, r api.Registry, opts ...GenerateOption) (string, error) {
	res, err := Generate(ctx, r, opts...)
	if err != nil {
		return "", err
	}

	return res.Text(), nil
}

// GenerateData runs a generate request and returns strongly-typed output.
// If the response doesn't contain text output (e.g., contains tool requests
// or interrupts instead), the output will be nil and no error is returned.
// Check resp.Interrupts() or resp.ToolRequests() to handle these cases.
func GenerateData[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) (*Out, *ModelResponse, error) {
	var value Out
	opts = append(opts, WithOutputType(value))

	resp, err := Generate(ctx, r, opts...)
	if err != nil {
		return nil, nil, err
	}

	// If there's no text content to parse (e.g., the response contains tool
	// requests or interrupts), return nil output. The caller should check
	// resp.Interrupts() or resp.ToolRequests() to handle these cases.
	if resp.Text() == "" {
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

// errGenerateStop is a sentinel error used to signal early termination of streaming.
var errGenerateStop = errors.New("stop")

// GenerateStream generates a model response and streams the output.
// It returns an iterator that yields streaming results.
//
// If the yield function is passed a non-nil error, generation has failed with that
// error; the yield function will not be called again.
//
// If the yield function's [ModelStreamValue] argument has Done == true, the value's
// Response field contains the final response; the yield function will not be called
// again.
//
// Otherwise the Chunk field of the passed [ModelStreamValue] holds a streamed chunk.
func GenerateStream(ctx context.Context, r api.Registry, opts ...GenerateOption) iter.Seq2[*ModelStreamValue, error] {
	return func(yield func(*ModelStreamValue, error) bool) {
		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if !yield(&ModelStreamValue{Chunk: chunk}, nil) {
				return errGenerateStop
			}
			return nil
		}

		allOpts := append(slices.Clone(opts), WithStreaming(cb))

		resp, err := Generate(ctx, r, allOpts...)
		if err != nil {
			yield(nil, err)
		} else {
			yield(&ModelStreamValue{Done: true, Response: resp}, nil)
		}
	}
}

// GenerateDataStream generates a model response with streaming and returns strongly-typed output.
// It returns an iterator that yields streaming results.
//
// If the yield function is passed a non-nil error, generation has failed with that
// error; the yield function will not be called again.
//
// If the yield function's [StreamValue] argument has Done == true, the value's
// Output and Response fields contain the final typed output and response; the yield function
// will not be called again.
//
// Otherwise the Chunk field of the passed [StreamValue] holds a streamed chunk.
func GenerateDataStream[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) iter.Seq2[*StreamValue[Out, Out], error] {
	return func(yield func(*StreamValue[Out, Out], error) bool) {
		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			var streamValue Out
			if err := chunk.Output(&streamValue); err != nil {
				yield(nil, err)
				return err
			}
			// Skip yielding if there's no parseable output yet (e.g., incomplete JSON during streaming).
			if base.IsNil(streamValue) {
				return nil
			}
			if !yield(&StreamValue[Out, Out]{Chunk: streamValue}, nil) {
				return errGenerateStop
			}
			return nil
		}

		// Prepend WithOutputType so the user can override the output format.
		var value Out
		allOpts := append([]GenerateOption{WithOutputType(value)}, opts...)
		allOpts = append(allOpts, WithStreaming(cb))

		resp, err := Generate(ctx, r, allOpts...)
		if err != nil {
			yield(nil, err)
			return
		}

		// If there's no text content to parse (e.g., the response contains tool
		// requests or interrupts), return zero-value output. The caller should check
		// resp.Interrupts() or resp.ToolRequests() to handle these cases.
		if resp.Text() == "" {
			yield(&StreamValue[Out, Out]{Done: true, Response: resp}, nil)
			return
		}

		output, err := extractTypedOutput[Out](resp)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(&StreamValue[Out, Out]{Done: true, Output: output, Response: resp}, nil)
	}
}

// Generate applies the [Action] to provided request.
func (m *model) Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	if m == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Model.Generate: generate called on a nil model; check that all models are defined")
	}

	return m.ActionDef.Run(ctx, req, cb)
}

// supportsConstrained returns whether the model supports constrained output.
func (m *model) supportsConstrained(hasTools bool) bool {
	if m == nil {
		return false
	}

	metadata := m.ActionDef.Desc().Metadata
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
		if part.IsToolRequest() && part.ToolRequest.Ref == "" {
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

// toolRunnerFunc runs a tool through the WrapTool hook chain and returns the
// raw [MultipartToolResponse]. Returned by [buildToolRunner].
type toolRunnerFunc = func(ctx context.Context, tool Tool, req *ToolRequest) (*MultipartToolResponse, error)

// handleToolRequests processes any tool requests in the response, returning
// either a new request to continue the conversation or nil if no tool requests
// need handling.
func handleToolRequests(ctx context.Context, r api.Registry, req *ModelRequest, resp *ModelResponse, cb ModelStreamCallback, messageIndex int, runTool toolRunnerFunc) (*ModelRequest, *Message, error) {
	toolCount := len(resp.ToolRequests())
	if toolCount == 0 {
		return nil, nil, nil
	}

	resultChan := make(chan result[*MultipartToolResponse])
	toolMsg := &Message{Role: RoleTool}
	revisedMsg := clone(resp.Message)

	for i, part := range revisedMsg.Content {
		if !part.IsToolRequest() {
			continue
		}

		go func(idx int, p *Part) {
			toolReq := p.ToolRequest
			tool := LookupTool(r, p.ToolRequest.Name)
			if tool == nil {
				resultChan <- result[*MultipartToolResponse]{index: idx, err: core.NewError(core.NOT_FOUND, "tool %q not found", toolReq.Name)}
				return
			}

			multipartResp, err := runTool(ctx, tool, toolReq)
			if err != nil {
				var tie *toolInterruptError
				if errors.As(err, &tie) {
					logger.FromContext(ctx).Debug("tool %q triggered an interrupt: %v", toolReq.Name, tie.Metadata)

					newPart := clone(p)
					if newPart.Metadata == nil {
						newPart.Metadata = make(map[string]any)
					}
					if tie.Metadata != nil {
						newPart.Metadata["interrupt"] = tie.Metadata
					} else {
						newPart.Metadata["interrupt"] = true
					}

					revisedMsg.Content[idx] = newPart

					resultChan <- result[*MultipartToolResponse]{index: idx, err: tie}
					return
				}

				resultChan <- result[*MultipartToolResponse]{index: idx, err: core.NewError(core.INTERNAL, "tool %q failed: %v", toolReq.Name, err)}
				return
			}

			newPart := clone(p)
			if newPart.Metadata == nil {
				newPart.Metadata = make(map[string]any)
			}
			newPart.Metadata["pendingOutput"] = multipartResp.Output
			revisedMsg.Content[idx] = newPart

			resultChan <- result[*MultipartToolResponse]{index: idx, value: multipartResp}
		}(i, part)
	}

	var toolResps []*Part
	hasInterrupts := false
	for range toolCount {
		res := <-resultChan
		if res.err != nil {
			var tie *toolInterruptError
			if errors.As(res.err, &tie) {
				hasInterrupts = true
				continue
			}

			return nil, nil, res.err
		}

		toolReq := revisedMsg.Content[res.index].ToolRequest
		toolResps = append(toolResps, NewToolResponsePart(&ToolResponse{
			Name:    toolReq.Name,
			Ref:     toolReq.Ref,
			Output:  res.value.Output,
			Content: res.value.Content,
		}))
	}

	if hasInterrupts {
		return nil, revisedMsg, nil
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

	newReq := req
	newReq.Messages = append(slices.Clone(req.Messages), resp.Message, toolMsg)

	return newReq, nil, nil
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
// to represent the conversation history.
func (mr *ModelResponse) History() []*Message {
	if mr == nil || mr.Message == nil {
		return mr.Request.Messages
	}
	return append(mr.Request.Messages, mr.Message)
}

// Reasoning concatenates all reasoning parts present in the message
func (mr *ModelResponse) Reasoning() string {
	var sb strings.Builder
	if mr == nil || mr.Message == nil {
		return ""
	}

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
	if mr.Message == nil || len(mr.Message.Content) == 0 {
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
func (mr *ModelResponse) ToolRequests() []*ToolRequest {
	toolReqs := []*ToolRequest{}
	if mr == nil || mr.Message == nil {
		return toolReqs
	}
	for _, part := range mr.Message.Content {
		if part.IsToolRequest() {
			toolReqs = append(toolReqs, part.ToolRequest)
		}
	}
	return toolReqs
}

// Interrupts returns the interrupted tool request parts from the response.
func (mr *ModelResponse) Interrupts() []*Part {
	parts := []*Part{}
	if mr == nil || mr.Message == nil {
		return parts
	}
	for _, part := range mr.Message.Content {
		if part.IsInterrupt() {
			parts = append(parts, part)
		}
	}
	return parts
}

// Media returns the media content of the [ModelResponse] as a string.
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

// Text returns the text content of the ModelResponseChunk as a string.
// It returns an empty string if there is no Content in the response chunk.
// For the parsed structured output, use [ModelResponseChunk.Output] instead.
func (c *ModelResponseChunk) Text() string {
	if len(c.Content) == 0 {
		return ""
	}
	if len(c.Content) == 1 {
		return c.Content[0].Text
	}
	var sb strings.Builder
	for _, p := range c.Content {
		if p.IsText() || p.IsData() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// Reasoning returns the reasoning content of the ModelResponseChunk as a string.
// It returns an empty string if there is no Content in the response chunk.
func (c *ModelResponseChunk) Reasoning() string {
	if len(c.Content) == 0 {
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

// Output parses the chunk using the format handler and unmarshals the result into v.
// Returns an error if the format handler is not set or does not support parsing chunks.
func (c *ModelResponseChunk) Output(v any) error {
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

// Text returns the contents of a [Message] as a string. It
// returns an empty string if the message has no content.
// If you want to get reasoning from the message, use Reasoning() instead.
func (m *Message) Text() string {
	if m == nil {
		return ""
	}
	if len(m.Content) == 0 {
		return ""
	}
	if len(m.Content) == 1 {
		return m.Content[0].Text
	}
	var sb strings.Builder
	for _, p := range m.Content {
		if p.IsText() || p.IsData() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
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
		Description: "Model name (e.g. \"googleai/gemini-2.5-flash\")",
	})
	props.Set("config", &jsonschema.Schema{
		Description: "Optional model configuration",
	})
	return &jsonschema.Schema{
		Type:       "object",
		Properties: props,
		Required:   []string{"name"},
	}
}

// handleResumedToolRequest resolves a tool request from a previous, interrupted model turn,
// when generation is being resumed. It determines the outcome of the tool request based on
// pending output, or explicit 'respond' or 'restart' directives in the resume options.
func handleResumedToolRequest(ctx context.Context, r api.Registry, genOpts *GenerateActionOptions, p *Part, runTool toolRunnerFunc) (*resumedToolRequestOutput, error) {
	if p == nil || !p.IsToolRequest() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumedToolRequest: part is not a tool request")
	}

	if pendingOutputVal, ok := p.Metadata["pendingOutput"]; ok {
		newReqPart := clone(p)
		delete(newReqPart.Metadata, "pendingOutput")

		newRespPart := NewResponseForToolRequest(p, pendingOutputVal)
		newRespPart.Metadata = map[string]any{"source": "pending"}

		return &resumedToolRequestOutput{
			toolRequest:  newReqPart,
			toolResponse: newRespPart,
		}, nil
	}

	if genOpts.Resume != nil {
		toolReq := p.ToolRequest

		for _, respondPart := range genOpts.Resume.Respond {
			if respondPart.ToolResponse != nil &&
				respondPart.ToolResponse.Name == toolReq.Name &&
				respondPart.ToolResponse.Ref == toolReq.Ref {
				newToolReq := clone(p)
				if interruptVal, ok := newToolReq.Metadata["interrupt"]; ok {
					delete(newToolReq.Metadata, "interrupt")
					newToolReq.Metadata["resolvedInterrupt"] = interruptVal
				}

				tool := LookupTool(r, toolReq.Name)
				if tool == nil {
					return nil, core.NewError(core.NOT_FOUND, "handleResumedToolRequest: tool %q not found", toolReq.Name)
				}

				toolDef := tool.Definition()
				if len(toolDef.OutputSchema) > 0 {
					outputBytes, err := json.Marshal(respondPart.ToolResponse.Output)
					if err != nil {
						return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumedToolRequest: failed to marshal tool output for validation: %v", err)
					}

					schemaBytes, err := json.Marshal(toolDef.OutputSchema)
					if err != nil {
						return nil, core.NewError(core.INTERNAL, "handleResumedToolRequest: tool %q has invalid output schema: %v", toolReq.Name, err)
					}

					if err := base.ValidateRaw(outputBytes, schemaBytes); err != nil {
						return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumedToolRequest: tool %q output validation failed: %v", toolReq.Name, err)
					}
				}

				newToolResp := NewToolResponsePart(respondPart.ToolResponse)
				newToolResp.Metadata = respondPart.Metadata

				return &resumedToolRequestOutput{
					toolRequest:  newToolReq,
					toolResponse: newToolResp,
				}, nil
			}
		}

		for _, restartPart := range genOpts.Resume.Restart {
			if restartPart.ToolRequest != nil &&
				restartPart.ToolRequest.Name == toolReq.Name &&
				restartPart.ToolRequest.Ref == toolReq.Ref {
				tool := LookupTool(r, restartPart.ToolRequest.Name)
				if tool == nil {
					return nil, core.NewError(core.NOT_FOUND, "handleResumedToolRequest: tool %q not found", restartPart.ToolRequest.Name)
				}

				resumedCtx := ctx
				if resumedVal, ok := restartPart.Metadata["resumed"]; ok {
					// TODO: Better handling here or in tools.go.
					switch resumedVal := resumedVal.(type) {
					case map[string]any:
						resumedCtx = resumedCtxKey.NewContext(resumedCtx, resumedVal)
					case bool:
						if resumedVal {
							resumedCtx = resumedCtxKey.NewContext(resumedCtx, map[string]any{})
						}
					}
				}
				if originalInputVal, ok := restartPart.Metadata["replacedInput"]; ok {
					resumedCtx = origInputCtxKey.NewContext(resumedCtx, originalInputVal)
				}

				restartToolReq := &ToolRequest{
					Name:  restartPart.ToolRequest.Name,
					Ref:   restartPart.ToolRequest.Ref,
					Input: restartPart.ToolRequest.Input,
				}
				multipartResp, err := runTool(resumedCtx, tool, restartToolReq)
				if err != nil {
					var tie *toolInterruptError
					if errors.As(err, &tie) {
						logger.FromContext(ctx).Debug("tool %q triggered an interrupt: %v", restartPart.ToolRequest.Name, tie.Metadata)

						interruptPart := clone(p)
						if interruptPart.Metadata == nil {
							interruptPart.Metadata = make(map[string]any)
						}
						interruptPart.Metadata["interrupt"] = tie.Metadata

						return &resumedToolRequestOutput{
							interrupt: interruptPart,
						}, nil
					}

					return nil, core.NewError(core.INTERNAL, "tool %q failed: %v", restartPart.ToolRequest.Name, err)
				}

				newToolReq := clone(p)
				if interruptVal, ok := newToolReq.Metadata["interrupt"]; ok {
					delete(newToolReq.Metadata, "interrupt")
					newToolReq.Metadata["resolvedInterrupt"] = interruptVal
				}

				newToolResp := NewToolResponsePart(&ToolResponse{
					Name:    restartPart.ToolRequest.Name,
					Ref:     restartPart.ToolRequest.Ref,
					Output:  multipartResp.Output,
					Content: multipartResp.Content,
				})

				return &resumedToolRequestOutput{
					toolRequest:  newToolReq,
					toolResponse: newToolResp,
				}, nil
			}
		}
	}

	refStr := p.ToolRequest.Name
	if p.ToolRequest.Ref != "" {
		refStr = "#" + p.ToolRequest.Ref
	}
	return nil, core.NewError(core.INVALID_ARGUMENT, fmt.Sprintf("unresolved tool request %q was not handled by the Resume argument; you must supply Respond or Restart directives, or ensure there is pending output from a previous tool call", refStr))
}

// handleResumeOption amends message history to handle `resume` arguments.
// It returns the amended history.
func handleResumeOption(ctx context.Context, r api.Registry, genOpts *GenerateActionOptions, runTool toolRunnerFunc) (*resumeOptionOutput, error) {
	if genOpts.Resume == nil || (len(genOpts.Resume.Respond) == 0 && len(genOpts.Resume.Restart) == 0) {
		return &resumeOptionOutput{revisedRequest: genOpts}, nil
	}

	for _, part := range genOpts.Resume.Respond {
		if !part.IsToolResponse() {
			return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumeOption: respond part is not a tool response")
		}
	}
	for _, part := range genOpts.Resume.Restart {
		if !part.IsToolRequest() {
			return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumeOption: restart part is not a tool request")
		}
	}

	toolDefMap := make(map[string]*ToolDefinition)
	for _, t := range genOpts.Tools {
		tool := LookupTool(r, t)
		if tool == nil {
			return nil, core.NewError(core.NOT_FOUND, "handleResumeOption: tool %q not found", t)
		}
		toolDefMap[t] = tool.Definition()
	}

	messages := genOpts.Messages
	if len(messages) == 0 {
		return nil, core.NewError(core.FAILED_PRECONDITION, "handleResumeOption: cannot resume generation with no messages")
	}
	lastMessage := messages[len(messages)-1]

	if lastMessage.Role != RoleModel || !slices.ContainsFunc(lastMessage.Content, func(p *Part) bool { return p.IsToolRequest() }) {
		return nil, core.NewError(core.FAILED_PRECONDITION, "handleResumeOption: cannot resume generation unless the last message is by a model with at least one tool request")
	}

	resultChan := make(chan result[*resumedToolRequestOutput])
	newContent := make([]*Part, len(lastMessage.Content))
	toolReqCount := 0

	for i, part := range lastMessage.Content {
		if !part.IsToolRequest() {
			newContent[i] = part
			continue
		}
		toolReqCount++

		go func(idx int, p *Part) {
			output, err := handleResumedToolRequest(ctx, r, genOpts, p, runTool)
			resultChan <- result[*resumedToolRequestOutput]{
				index: idx,
				value: output,
				err:   err,
			}
		}(i, part)
	}

	var toolResps []*Part
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
			toolResps = append(toolResps, res.value.toolResponse)
			newContent[res.index] = res.value.toolRequest
		}
	}

	lastMessage.Content = newContent

	if interrupted {
		return &resumeOptionOutput{
			interruptedResponse: &ModelResponse{
				Message:       lastMessage,
				FinishReason:  "interrupted",
				FinishMessage: "One or more tools triggered interrupts while resuming generation. The model was not called.",
			},
		}, nil
	}

	if len(toolResps) != toolReqCount {
		return nil, core.NewError(core.FAILED_PRECONDITION, fmt.Sprintf("handleResumeOption: Expected %d tool responses but resolved to %d.", toolReqCount, len(toolResps)))
	}

	toolMessage := &Message{
		Role:    RoleTool,
		Content: toolResps,
		Metadata: map[string]any{
			"resumed": true,
		},
	}
	if genOpts.Resume.Metadata != nil {
		toolMessage.Metadata["resumed"] = genOpts.Resume.Metadata
	}
	revisedMessages := append(slices.Clone(messages), toolMessage)

	return &resumeOptionOutput{
		revisedRequest: &GenerateActionOptions{
			Model:              genOpts.Model,
			Messages:           revisedMessages,
			Tools:              genOpts.Tools,
			MaxTurns:           genOpts.MaxTurns,
			Config:             genOpts.Config,
			ToolChoice:         genOpts.ToolChoice,
			Docs:               genOpts.Docs,
			ReturnToolRequests: genOpts.ReturnToolRequests,
			Output:             genOpts.Output,
			Use:                genOpts.Use,
		},
		toolMessage: toolMessage,
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
