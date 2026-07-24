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
	"fmt"

	"github.com/firebase/genkit/go/core"
)

// Options follow the standard Go functional-options pattern: pass as many as
// you like, in any order, and they merge left to right. Two rules govern how
// repeats combine, so composing a request from several helpers is predictable:
//
//   - Collection options accumulate. Repeating one, or mixing its variants,
//     appends in call order: WithMessages(a), WithMessages(b) sends [a, b], and
//     WithTools, WithResources, WithUse, WithDocs, WithDataset, and the tool
//     resume options behave the same.
//   - Single-value options take the last one set. WithConfig, WithModel /
//     WithModelName, WithSystem, WithPrompt, the output-schema options, and the
//     like each fill one slot, so the final call wins and earlier ones are
//     overwritten rather than rejected.
//
// Applying options therefore never fails on a "set more than once" conflict.
// The only failures are genuinely invalid arguments (for example a type that
// WithInputType cannot turn into a schema), which panic at the call site where
// the mistake is.

// PromptFn is a function that generates a prompt.
type PromptFn = func(context.Context, any) (string, error)

// MessagesFn is a function that generates messages.
type MessagesFn = func(context.Context, any) ([]*Message, error)

// appendMessagesFn composes two message-producing functions so their outputs
// concatenate in call order. It backs the accumulate semantics of
// [WithMessages] and [WithMessagesFn]: passing several of them (in any mix)
// appends their messages instead of overwriting.
func appendMessagesFn(existing, next MessagesFn) MessagesFn {
	if existing == nil {
		return next
	}
	if next == nil {
		return existing
	}
	return func(ctx context.Context, input any) ([]*Message, error) {
		before, err := existing(ctx, input)
		if err != nil {
			return nil, err
		}
		after, err := next(ctx, input)
		if err != nil {
			return nil, err
		}
		return append(before, after...), nil
	}
}

// configOptions holds configuration options.
type configOptions struct {
	Config any // Primitive (model, embedder, etc) configuration.
}

// ConfigOption is an option for model configuration.
type ConfigOption interface {
	applyConfig(*configOptions)
	applyCommonGen(*commonGenOptions)
	applyPrompt(*promptOptions)
	applyGenerate(*generateOptions)
	applyPromptExecute(*promptExecutionOptions)
	applyEmbedder(*embedderOptions)
	applyEvaluator(*evaluatorOptions)
}

// applyConfig applies the option to the config options.
func (o *configOptions) applyConfig(opts *configOptions) {
	if o.Config != nil {
		opts.Config = o.Config
	}
}

// applyCommonGen applies the option to the common options.
func (o *configOptions) applyCommonGen(opts *commonGenOptions) {
	o.applyConfig(&opts.configOptions)
}

// applyPrompt applies the option to the prompt options.
func (o *configOptions) applyPrompt(opts *promptOptions) {
	o.applyConfig(&opts.configOptions)
}

// applyGenerate applies the option to the generate options.
func (o *configOptions) applyGenerate(opts *generateOptions) {
	o.applyConfig(&opts.configOptions)
}

// applyPromptExecute applies the option to the prompt generate options.
func (o *configOptions) applyPromptExecute(opts *promptExecutionOptions) {
	o.applyConfig(&opts.configOptions)
}

// applyEmbedder applies the option to the embed options.
func (o *configOptions) applyEmbedder(opts *embedderOptions) {
	o.applyConfig(&opts.configOptions)
}

// applyEvaluator applies the option to the evaluate options.
func (o *configOptions) applyEvaluator(opts *evaluatorOptions) {
	o.applyConfig(&opts.configOptions)
}

// WithConfig sets the configuration.
func WithConfig(config any) ConfigOption {
	return &configOptions{Config: config}
}

// commonGenOptions are common options for model generation, prompt definition, and prompt execution.
type commonGenOptions struct {
	configOptions
	Model              ModelArg     // Model to use.
	MessagesFn         MessagesFn   // Function to generate messages.
	Tools              []ToolArg    // References to tools to use.
	Resources          []*Resource  // Resources to be temporarily available during generation.
	ToolChoice         ToolChoice   // Whether tool calls are required, disabled, or optional.
	MaxTurns           int          // Maximum number of tool call iterations.
	ReturnToolRequests *bool        // Whether to return tool requests instead of making the tool calls and continuing the generation.
	Use                []Middleware // Middleware to apply to generation (Generate, Model, and Tool hooks).
}

type CommonGenOption interface {
	applyCommonGen(*commonGenOptions)
	applyPrompt(*promptOptions)
	applyGenerate(*generateOptions)
	applyPromptExecute(*promptExecutionOptions)
}

// applyCommonGen applies the option to the common options.
func (o *commonGenOptions) applyCommonGen(opts *commonGenOptions) {
	o.configOptions.applyConfig(&opts.configOptions)

	if o.MessagesFn != nil {
		opts.MessagesFn = appendMessagesFn(opts.MessagesFn, o.MessagesFn)
	}
	if o.Model != nil {
		opts.Model = o.Model
	}
	opts.Tools = append(opts.Tools, o.Tools...)
	opts.Resources = append(opts.Resources, o.Resources...)
	if o.ToolChoice != "" {
		opts.ToolChoice = o.ToolChoice
	}
	if o.MaxTurns > 0 {
		opts.MaxTurns = o.MaxTurns
	}
	if o.ReturnToolRequests != nil {
		opts.ReturnToolRequests = o.ReturnToolRequests
	}
	opts.Use = append(opts.Use, o.Use...)
}

// applyPromptExecute applies the option to the prompt request options.
func (o *commonGenOptions) applyPromptExecute(reqOpts *promptExecutionOptions) {
	o.applyCommonGen(&reqOpts.commonGenOptions)
}

// applyPrompt applies the option to the prompt options.
func (o *commonGenOptions) applyPrompt(pOpts *promptOptions) {
	o.applyCommonGen(&pOpts.commonGenOptions)
}

// applyGenerate applies the option to the generate options.
func (o *commonGenOptions) applyGenerate(genOpts *generateOptions) {
	o.applyCommonGen(&genOpts.commonGenOptions)
}

// WithMessages adds messages to the request, placed between the system and
// user prompts. Repeating this option, or mixing it with [WithMessagesFn],
// appends: messages accumulate in the order the options are passed.
func WithMessages(messages ...*Message) CommonGenOption {
	return &commonGenOptions{
		MessagesFn: func(ctx context.Context, _ any) ([]*Message, error) {
			return messages, nil
		},
	}
}

// WithMessagesFn adds messages produced by fn at request time, placed between
// the system and user prompts. Like [WithMessages], repeating this option (or
// mixing the two) appends the produced messages in call order.
func WithMessagesFn(fn MessagesFn) CommonGenOption {
	return &commonGenOptions{MessagesFn: fn}
}

// WithTools adds tools to use for the generate request. Repeating this option
// appends; duplicate tools (by name) are rejected when the request runs.
// Accepts tool values ([*Tool], [*InterruptibleTool], or any [AnyTool]) or
// [ToolName] references to registered tools.
func WithTools(tools ...ToolArg) CommonGenOption {
	return &commonGenOptions{Tools: tools}
}

// WithModel sets either a [*Model], a [*BackgroundModel], or a [ModelRef] that
// may contain a config.
// Passing [WithConfig] will take precedence over the config in WithModel.
func WithModel(model ModelArg) CommonGenOption {
	return &commonGenOptions{Model: model}
}

// WithModelName sets the model name to call for generation.
// The model name will be resolved to a [Model] and may error if the reference is invalid.
func WithModelName(name string) CommonGenOption {
	return &commonGenOptions{Model: NewModelRef(name, nil)}
}

// WithUse adds middleware to apply to generation. Middleware hooks wrap the
// generate loop, model calls, and tool executions. Repeating this option
// appends to the chain.
//
// Accepts either a middleware config struct (produced by a plugin) or an
// inline adapter via [MiddlewareFunc]. The chain applies outer-to-inner, so
// WithUse(A, B) expands to A { B { ... } }.
func WithUse(middleware ...Middleware) CommonGenOption {
	return &commonGenOptions{Use: middleware}
}

// WithStepName sets a custom name for the generation step in traces.
func WithStepName(name string) GenerateOption {
	return &generateOptions{StepName: name}
}

// WithMaxTurns sets the maximum number of tool call iterations before erroring.
// A tool call happens when tools are provided in the request and a model decides to call one or more as a response.
// Each round trip, including multiple tools in parallel, counts as one turn.
func WithMaxTurns(maxTurns int) CommonGenOption {
	return &commonGenOptions{MaxTurns: maxTurns}
}

// WithReturnToolRequests configures whether to return tool requests instead of making the tool calls and continuing the generation.
func WithReturnToolRequests(returnReqs bool) CommonGenOption {
	return &commonGenOptions{ReturnToolRequests: &returnReqs}
}

// WithToolChoice configures whether by default tool calls are required, disabled, or optional for the prompt.
func WithToolChoice(toolChoice ToolChoice) CommonGenOption {
	return &commonGenOptions{ToolChoice: toolChoice}
}

// WithResources specifies resources to be temporarily available during
// generation. Repeating this option appends. Resources are unregistered
// resources that get attached to a temporary registry during the generation
// request and cleaned up afterward.
func WithResources(resources ...*Resource) CommonGenOption {
	return &commonGenOptions{Resources: resources}
}

// inputOptions are options for the input of a prompt.
type inputOptions struct {
	InputSchema  map[string]any // JSON schema of the input.
	DefaultInput map[string]any // Default input that will be used if no input is provided.
}

// InputOption is an option for the input of a prompt.
// It applies only to DefinePrompt().
type InputOption interface {
	applyInput(*inputOptions)
	applyPrompt(*promptOptions)
	applyTool(*toolOptions)
}

// applyInput applies the option to the input options.
func (o *inputOptions) applyInput(opts *inputOptions) {
	if o.InputSchema != nil {
		opts.InputSchema = o.InputSchema
	}
	if o.DefaultInput != nil {
		opts.DefaultInput = o.DefaultInput
	}
}

// applyPrompt applies the option to the prompt options.
func (o *inputOptions) applyPrompt(pOpts *promptOptions) {
	o.applyInput(&pOpts.inputOptions)
}

// applyTool applies the option to the tool options.
func (o *inputOptions) applyTool(tOpts *toolOptions) {
	o.applyInput(&tOpts.inputOptions)
}

// WithInputType uses the type provided to derive the input schema.
// The inputted value may serve as the default input if no input is given at generation time depending on the action.
// Only supports structs and map[string]any.
func WithInputType(input any) InputOption {
	var defaultInput map[string]any

	switch v := input.(type) {
	case map[string]any:
		defaultInput = v
	default:
		data, err := json.Marshal(input)
		if err != nil {
			panic(fmt.Errorf("failed to marshal default input (WithInputType): %w", err))
		}

		err = json.Unmarshal(data, &defaultInput)
		if err != nil {
			panic(fmt.Errorf("type %T is not supported, only structs and map[string]any are supported (WithInputType)", input))
		}
	}

	return &inputOptions{
		InputSchema:  core.InferSchemaMap(input),
		DefaultInput: defaultInput,
	}
}

// WithInputSchema manually provides a schema map for the prompt's input.
func WithInputSchema(schema map[string]any) InputOption {
	return &inputOptions{InputSchema: schema}
}

// WithInputSchemaName sets a pre-registered schema by name for the input.
// The schema will be resolved lazily at execution time using [DefineSchema].
func WithInputSchemaName(name string) InputOption {
	return &inputOptions{InputSchema: core.SchemaRef(name)}
}

// promptOptions are options for defining a prompt.
type promptOptions struct {
	commonGenOptions
	promptingOptions
	inputOptions
	outputOptions
	Description string         // Description of the prompt.
	Metadata    map[string]any // Arbitrary metadata.
}

// PromptOption is an option for defining a prompt.
// It applies only to DefinePrompt().
type PromptOption interface {
	applyPrompt(*promptOptions)
}

// applyPrompt applies the option to the prompt options.
func (o *promptOptions) applyPrompt(opts *promptOptions) {
	o.commonGenOptions.applyPrompt(opts)
	o.promptingOptions.applyPrompt(opts)
	o.inputOptions.applyPrompt(opts)
	o.outputOptions.applyPrompt(opts)

	if o.Description != "" {
		opts.Description = o.Description
	}
	if o.Metadata != nil {
		opts.Metadata = o.Metadata
	}
}

// WithDescription sets the description of the prompt.
func WithDescription(description string) PromptOption {
	return &promptOptions{Description: description}
}

// WithMetadata sets arbitrary metadata for the prompt.
func WithMetadata(metadata map[string]any) PromptOption {
	return &promptOptions{Metadata: metadata}
}

// promptingOptions are options for the system and user prompts of a prompt or generate request.
type promptingOptions struct {
	SystemFn PromptFn // Function to generate the system prompt.
	PromptFn PromptFn // Function to generate the user prompt.
}

// PromptingOption is an option for the system and user prompts of a prompt or generate request.
// It applies only to DefinePrompt() and Generate().
type PromptingOption interface {
	applyPrompting(*promptingOptions)
	applyPrompt(*promptOptions)
	applyGenerate(*generateOptions)
}

// applyPrompting applies the option to the prompting options.
func (o *promptingOptions) applyPrompting(opts *promptingOptions) {
	if o.SystemFn != nil {
		opts.SystemFn = o.SystemFn
	}
	if o.PromptFn != nil {
		opts.PromptFn = o.PromptFn
	}
}

// applyPrompt applies the option to the prompt options.
func (o *promptingOptions) applyPrompt(opts *promptOptions) {
	o.applyPrompting(&opts.promptingOptions)
}

// applyGenerate applies the option to the generate options.
func (o *promptingOptions) applyGenerate(opts *generateOptions) {
	o.applyPrompting(&opts.promptingOptions)
}

// WithSystem sets the system prompt message.
// The system prompt is always the first message in the list.
func WithSystem(text string, args ...any) PromptingOption {
	return &promptingOptions{
		SystemFn: func(ctx context.Context, _ any) (string, error) {
			// Avoids a compile-time warning about non-constant text.
			t := text
			return fmt.Sprintf(t, args...), nil
		},
	}
}

// WithSystemFn sets the function that generates the system prompt message.
// The system prompt is always the first message in the list.
func WithSystemFn(fn PromptFn) PromptingOption {
	return &promptingOptions{SystemFn: fn}
}

// WithPrompt sets the user prompt message.
// The user prompt is always the last message in the list.
func WithPrompt(text string, args ...any) PromptingOption {
	return &promptingOptions{
		PromptFn: func(ctx context.Context, _ any) (string, error) {
			// Avoids a compile-time warning about non-constant text.
			t := text
			return fmt.Sprintf(t, args...), nil
		},
	}
}

// WithPromptFn sets the function that generates the user prompt message.
// The user prompt is always the last message in the list.
func WithPromptFn(fn PromptFn) PromptingOption {
	return &promptingOptions{PromptFn: fn}
}

// outputOptions are options for the output of a prompt or generate request.
type outputOptions struct {
	OutputSchema       map[string]any // JSON schema of the output.
	OutputFormat       string         // Format of the output. If OutputSchema is set, this is set to OutputFormatJSON.
	OutputInstructions *string        // Instructions to add to conform the output to a schema. If nil, default instructions will be added. If empty string, no instructions will be added.
	CustomConstrained  bool           // Whether generation should use custom constrained output instead of native model constrained output.
}

// OutputOption is an option for the output of a prompt or generate request.
// It applies only to DefinePrompt() and Generate().
type OutputOption interface {
	applyOutput(*outputOptions)
	applyPrompt(*promptOptions)
	applyGenerate(*generateOptions)
}

// applyOutput applies the option to the output options. The schema, format,
// and instructions are independent single-value slots, so the last option to
// set each one wins. This is what lets a caller-supplied [WithOutputSchema]
// override the schema [GenerateData] injects while still using JSON output.
func (o *outputOptions) applyOutput(opts *outputOptions) {
	if o.OutputSchema != nil {
		opts.OutputSchema = o.OutputSchema
	}
	if o.OutputFormat != "" {
		opts.OutputFormat = o.OutputFormat
	}
	if o.OutputInstructions != nil {
		opts.OutputInstructions = o.OutputInstructions
	}
	if o.CustomConstrained {
		opts.CustomConstrained = true
	}
}

// applyPrompt applies the option to the prompt options.
func (o *outputOptions) applyPrompt(pOpts *promptOptions) {
	o.applyOutput(&pOpts.outputOptions)
}

// applyGenerate applies the option to the generate options.
func (o *outputOptions) applyGenerate(genOpts *generateOptions) {
	o.applyOutput(&genOpts.outputOptions)
}

// WithOutputType sets the output format to JSON and the schema derived from the given value.
func WithOutputType(output any) OutputOption {
	return &outputOptions{
		OutputSchema: core.InferSchemaMap(output),
		OutputFormat: OutputFormatJSON,
	}
}

// WithOutputSchema manually provides a schema map for the prompt's output.
// The outputted value will serve as the default output if no output is given at generation time.
func WithOutputSchema(schema map[string]any) OutputOption {
	return &outputOptions{
		OutputSchema: schema,
		OutputFormat: OutputFormatJSON,
	}
}

// WithOutputSchemaName sets the schema name that will be resolved at execution time.
func WithOutputSchemaName(name string) OutputOption {
	return &outputOptions{
		OutputSchema: core.SchemaRef(name),
		OutputFormat: OutputFormatJSON,
	}
}

// WithOutputFormat sets the format of the output.
func WithOutputFormat(format string) OutputOption {
	return &outputOptions{OutputFormat: format}
}

// WithOutputEnums sets the output format to enum and the schema based on the given values.
// Accepts any string-based type (e.g. type MyEnum string).
func WithOutputEnums[T ~string](values ...T) OutputOption {
	enumStrs := make([]string, len(values))
	for i, v := range values {
		enumStrs[i] = string(v)
	}
	return &outputOptions{
		OutputSchema: map[string]any{
			"type": "string",
			"enum": enumStrs,
		},
		OutputFormat: OutputFormatEnum,
	}
}

// WithOutputInstructions sets custom instructions for constraining output format in the prompt.
//
// When [WithOutputType] is used without this option, default instructions will be automatically set.
// If you provide empty instructions, no instructions will be added to the prompt.
//
// This will automatically set [WithCustomConstrainedOutput].
func WithOutputInstructions(instructions string) OutputOption {
	return &outputOptions{
		OutputInstructions: &instructions,
		CustomConstrained:  true,
	}
}

// WithCustomConstrainedOutput opts out of using the model's native constrained output generation.
//
// By default, the system will use the model's native constrained output capabilities when available.
// When this option is set, or when the model doesn't support native constraints, the system will
// use custom implementation to guide the model toward producing properly formatted output.
func WithCustomConstrainedOutput() OutputOption {
	return &outputOptions{CustomConstrained: true}
}

// executionOptions are options for the execution of a prompt or generate request.
type executionOptions struct {
	Stream ModelStreamCallback // Function to call with each chunk of the generated response.
}

// ExecutionOption is an option for the execution of a prompt or generate request. It applies only to Generate() and prompt.Execute().
type ExecutionOption interface {
	applyExecution(*executionOptions)
	applyGenerate(*generateOptions)
	applyPromptExecute(*promptExecutionOptions)
}

// applyExecution applies the option to the runtime options.
func (o *executionOptions) applyExecution(execOpts *executionOptions) {
	if o.Stream != nil {
		execOpts.Stream = o.Stream
	}
}

// applyGenerate applies the option to the generate options.
func (o *executionOptions) applyGenerate(genOpts *generateOptions) {
	o.applyExecution(&genOpts.executionOptions)
}

// applyPromptExecute applies the option to the prompt request options.
func (o *executionOptions) applyPromptExecute(pgOpts *promptExecutionOptions) {
	o.applyExecution(&pgOpts.executionOptions)
}

// WithStreaming sets the stream callback for the generate request.
// A callback is a function that is called with each chunk of the generated response before the final response is returned.
func WithStreaming(callback ModelStreamCallback) ExecutionOption {
	return &executionOptions{Stream: callback}
}

// documentOptions are options for providing context documents to a prompt or generate request or as input to an embedder.
type documentOptions struct {
	Documents []*Document // Docs to pass as context or input.
}

// DocumentOption is an option for providing context or input documents.
// It applies only to [Generate] and [prompt.Execute].
type DocumentOption interface {
	applyDocument(*documentOptions)
	applyGenerate(*generateOptions)
	applyPromptExecute(*promptExecutionOptions)
	applyEmbedder(*embedderOptions)
}

// applyDocument applies the option to the context options.
func (o *documentOptions) applyDocument(docOpts *documentOptions) {
	docOpts.Documents = append(docOpts.Documents, o.Documents...)
}

// applyGenerate applies the option to the generate options.
func (o *documentOptions) applyGenerate(genOpts *generateOptions) {
	o.applyDocument(&genOpts.documentOptions)
}

// applyPromptExecute applies the option to the prompt generate options.
func (o *documentOptions) applyPromptExecute(pgOpts *promptExecutionOptions) {
	o.applyDocument(&pgOpts.documentOptions)
}

// applyEmbedder applies the option to the embed options.
func (o *documentOptions) applyEmbedder(embedOpts *embedderOptions) {
	o.applyDocument(&embedOpts.documentOptions)
}

// WithTextDocs adds text as context documents for generation or as input to an
// embedder. Repeating this option (or mixing it with [WithDocs]) appends.
func WithTextDocs(text ...string) DocumentOption {
	docs := make([]*Document, len(text))
	for i, t := range text {
		docs[i] = DocumentFromText(t, nil)
	}
	return &documentOptions{Documents: docs}
}

// WithDocs adds documents as context for generation or as input to an
// embedder. Repeating this option (or mixing it with [WithTextDocs]) appends.
func WithDocs(docs ...*Document) DocumentOption {
	return &documentOptions{Documents: docs}
}

// evaluatorOptions are options for providing a dataset to evaluate.
type evaluatorOptions struct {
	configOptions
	Dataset   []*Example   // Dataset to evaluate.
	ID        string       // ID of the evaluation.
	Evaluator EvaluatorArg // Evaluator to use.
}

// EvaluatorOption is an option for providing a dataset to evaluate.
// It applies only to [Evaluator.Evaluate].
type EvaluatorOption interface {
	applyEvaluator(*evaluatorOptions)
}

// applyEvaluator applies the option to the evaluator options.
func (o *evaluatorOptions) applyEvaluator(evalOpts *evaluatorOptions) {
	o.applyConfig(&evalOpts.configOptions)

	evalOpts.Dataset = append(evalOpts.Dataset, o.Dataset...)
	if o.ID != "" {
		evalOpts.ID = o.ID
	}
	if o.Evaluator != nil {
		evalOpts.Evaluator = o.Evaluator
	}
}

// WithDataset adds examples to the dataset to evaluate. Repeating this option
// appends.
func WithDataset(examples ...*Example) EvaluatorOption {
	return &evaluatorOptions{Dataset: examples}
}

// WithID sets the ID of the evaluation to uniquely identify it.
func WithID(ID string) EvaluatorOption {
	return &evaluatorOptions{ID: ID}
}

// WithEvaluator sets either an [*Evaluator] or an [EvaluatorRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithEvaluator.
func WithEvaluator(evaluator EvaluatorArg) EvaluatorOption {
	return &evaluatorOptions{Evaluator: evaluator}
}

// WithEvaluatorName sets the evaluator name to call for document evaluation.
// The evaluator name will be resolved to a [Evaluator] and may error if the reference is invalid.
func WithEvaluatorName(name string) EvaluatorOption {
	return &evaluatorOptions{Evaluator: NewEvaluatorRef(name, nil)}
}

// embedderOptions holds configuration and input for an embedder request.
type embedderOptions struct {
	configOptions
	documentOptions
	Embedder EmbedderArg // Embedder to use.
}

// EmbedderOption is an option for configuring an embedder request.
// It applies only to [Embed].
type EmbedderOption interface {
	applyEmbedder(*embedderOptions)
}

// applyEmbedder applies the option to the embed options.
func (o *embedderOptions) applyEmbedder(embedOpts *embedderOptions) {
	o.applyConfig(&embedOpts.configOptions)
	o.applyDocument(&embedOpts.documentOptions)

	if o.Embedder != nil {
		embedOpts.Embedder = o.Embedder
	}
}

// WithEmbedder sets either an [*Embedder] or an [EmbedderRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithEmbedder.
func WithEmbedder(embedder EmbedderArg) EmbedderOption {
	return &embedderOptions{Embedder: embedder}
}

// WithEmbedderName sets the embedder name to call for document embedding.
// The embedder name will be resolved to a [Embedder] and may error if the reference is invalid.
func WithEmbedderName(name string) EmbedderOption {
	return &embedderOptions{Embedder: NewEmbedderRef(name, nil)}
}

// generateOptions are options for generating a model response by calling a model directly.
type generateOptions struct {
	commonGenOptions
	promptingOptions
	outputOptions
	executionOptions
	documentOptions
	RespondParts []*Part // Tool responses to return from interrupted tool calls.
	RestartParts []*Part // Tool requests to restart interrupted tools with.
	StepName     string  // Custom name for the generation step in traces.
}

// GenerateOption is an option for generating a model response. It applies only to Generate().
type GenerateOption interface {
	applyGenerate(*generateOptions)
}

// applyGenerate applies the option to the generate options.
func (o *generateOptions) applyGenerate(genOpts *generateOptions) {
	o.commonGenOptions.applyGenerate(genOpts)
	o.promptingOptions.applyGenerate(genOpts)
	o.outputOptions.applyGenerate(genOpts)
	o.executionOptions.applyGenerate(genOpts)
	o.documentOptions.applyGenerate(genOpts)

	genOpts.RespondParts = append(genOpts.RespondParts, o.RespondParts...)
	genOpts.RestartParts = append(genOpts.RestartParts, o.RestartParts...)
	if o.StepName != "" {
		genOpts.StepName = o.StepName
	}
}

// WithToolResponses provides resolved responses for interrupted tool calls.
// Use this when you already have the result and want to skip re-executing the
// tool. Repeating this option appends.
func WithToolResponses(parts ...*Part) GenerateOption {
	return &generateOptions{RespondParts: parts}
}

// WithToolRestarts re-executes interrupted tool calls with additional metadata.
// Use this when the original call lacked required context (e.g., auth, user
// confirmation) that should now allow the tool to complete successfully.
// Repeating this option appends.
func WithToolRestarts(parts ...*Part) GenerateOption {
	return &generateOptions{RestartParts: parts}
}

// toolOptions holds configuration options for defining tools.
type toolOptions struct {
	inputOptions
	StrictSchema *bool
}

// ToolOption is an option for defining a tool.
type ToolOption interface {
	applyTool(*toolOptions)
}

// applyTool applies the option to the tool options.
func (o *toolOptions) applyTool(opts *toolOptions) {
	if o.StrictSchema != nil {
		opts.StrictSchema = o.StrictSchema
	}
	o.inputOptions.applyTool(opts)
}

// WithStrictSchema controls whether the provider enforces strict JSON schema
// validation on this tool's input. Strict mode requires recursive
// additionalProperties: false and may reject some JSON Schema keywords
// (e.g. minItems/maxItems on Anthropic).
//
// When unset, the provider's default applies. Providers without strict-tool
// support ignore this option.
func WithStrictSchema(strict bool) ToolOption {
	return &toolOptions{StrictSchema: &strict}
}

// restartOptions holds configuration for restarting an interrupted tool.
type restartOptions struct {
	resume   any // Data delivered to the tool function's resume parameter.
	newInput any // Replacement input for the re-executed tool.
}

// RestartOption is an option for restarting an interrupted tool via
// [Part.ToRestart] or [InterruptibleTool.Restart]. Create one with [WithResume]
// or [WithNewInput], or with the compile-time checked equivalents
// [InterruptibleTool.WithResume] and [InterruptibleTool.WithNewInput].
type RestartOption interface {
	applyRestart(*restartOptions)
}

// applyRestart applies the option to the restart options. Both slots are
// single-value, so the last WithResume / WithNewInput set wins.
func (o *restartOptions) applyRestart(opts *restartOptions) {
	if o.resume != nil {
		opts.resume = o.resume
	}
	if o.newInput != nil {
		opts.newInput = o.newInput
	}
}

// WithResume carries data to the tool function's resume parameter when it
// re-executes, e.g. the user's answer to the question the tool interrupted
// with; middleware reads it via [tool.ResumeData]. data must serialize to a
// JSON object (a struct or a map); see [tool.Interrupt] for the rationale.
// [InterruptibleTool.WithResume] is the compile-time checked equivalent.
//
// Omitting it makes a bare restart. The call still counts as a resumption, so
// the tool's resume parameter is non-nil, but it holds the zero value of the
// tool's resume type. Tools that treat a non-nil resume as the approval need
// nothing more; tools that read fields off that type see zero values, so pass
// the data explicitly rather than relying on a bare restart.
//
// Repeating this option takes the last value set.
func WithResume(resume any) RestartOption {
	return &restartOptions{resume: resume}
}

// WithNewInput provides a new input for the tool when it is re-executed on
// restart, for example when the user revised an action before confirming. The
// tool can read the original input via [tool.OriginalInput]. The input is
// validated against the tool's input schema when the tool re-executes.
// [InterruptibleTool.WithNewInput] is the compile-time checked equivalent.
func WithNewInput(input any) RestartOption {
	return &restartOptions{newInput: input}
}

// promptExecutionOptions are options for generating a model response by executing a prompt.
// The prompt input is not an option; it is a typed parameter on
// [Prompt.Execute] and [Prompt.ExecuteStream].
type promptExecutionOptions struct {
	commonGenOptions
	executionOptions
	documentOptions
}

// PromptExecuteOption is an option for executing a prompt. It applies only to [Prompt.Execute].
type PromptExecuteOption interface {
	applyPromptExecute(*promptExecutionOptions)
}

// applyPromptExecute applies the option to the prompt request options.
func (o *promptExecutionOptions) applyPromptExecute(pgOpts *promptExecutionOptions) {
	o.commonGenOptions.applyPromptExecute(pgOpts)
	o.executionOptions.applyPromptExecute(pgOpts)
	o.documentOptions.applyPromptExecute(pgOpts)
}
