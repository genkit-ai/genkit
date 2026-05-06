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

	"github.com/firebase/genkit/go/core"
)

// setOnce assigns src to *dst when srcSet is true. If dstSet is also true,
// returns errors.New(dupErr) without assigning. Used by apply methods to
// collapse the "is-set / already-set / assign" pattern into one line per field.
//
// The src/dst predicates are passed in by the caller because Go's type system
// can't express "nillable" for the union of pointers, interfaces, slices, maps,
// and funcs. Comparable-with-zero would exclude slices/maps/funcs, and a
// reflect-based check would add cost and gotchas.
func setOnce[T any](dst *T, src T, srcSet, dstSet bool, dupErr string) error {
	if !srcSet {
		return nil
	}
	if dstSet {
		return errors.New(dupErr)
	}
	*dst = src
	return nil
}

// PromptFn is a function that generates a prompt.
type PromptFn = func(context.Context, any) (string, error)

// MessagesFn is a function that generates messages.
type MessagesFn = func(context.Context, any) ([]*Message, error)

// configOptions holds configuration options.
type configOptions struct {
	Config any // Primitive (model, embedder, retriever, etc) configuration.
}

// ConfigOption is an option for primitive configuration. It applies wherever
// a Config is meaningful: generation, prompt definition, prompt execution,
// embedding, retrieval, and evaluation.
type ConfigOption interface {
	CommonGenOption
	EmbedderOption
	RetrieverOption
	EvaluatorOption
}

// applyConfig applies the option to the config options.
func (o *configOptions) applyConfig(opts *configOptions) error {
	return setOnce(&opts.Config, o.Config, o.Config != nil, opts.Config != nil,
		"cannot set config more than once (WithConfig)")
}

// applyCommonGen applies the option to the common options.
func (o *configOptions) applyCommonGen(opts *commonGenOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// applyPrompt applies the option to the prompt options.
func (o *configOptions) applyPrompt(opts *promptOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// applyGenerate applies the option to the generate options.
func (o *configOptions) applyGenerate(opts *generateOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// applyPromptExecute applies the option to the prompt generate options.
func (o *configOptions) applyPromptExecute(opts *promptExecutionOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// applyEmbedder applies the option to the embed options.
func (o *configOptions) applyEmbedder(opts *embedderOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// applyRetriever applies the option to the retrieve options.
func (o *configOptions) applyRetriever(opts *retrieverOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// applyEvaluator applies the option to the evaluate options.
func (o *configOptions) applyEvaluator(opts *evaluatorOptions) error {
	return o.applyConfig(&opts.configOptions)
}

// WithConfig sets the configuration.
func WithConfig(config any) ConfigOption {
	return &configOptions{Config: config}
}

// commonGenOptions are common options for model generation, prompt definition, and prompt execution.
type commonGenOptions struct {
	configOptions
	Model              ModelArg          // Model to use.
	MessagesFn         MessagesFn        // Function to generate messages.
	Tools              []ToolRef         // References to tools to use.
	Resources          []Resource        // Resources to be temporarily available during generation.
	ToolChoice         ToolChoice        // Whether tool calls are required, disabled, or optional.
	MaxTurns           int               // Maximum number of tool call iterations.
	ReturnToolRequests *bool             // Whether to return tool requests instead of making the tool calls and continuing the generation.
	Middleware         []ModelMiddleware // Deprecated: Use WithUse instead. Middleware to apply to the model request and model response.
	Use                []Middleware      // Middleware to apply to generation (Generate, Model, and Tool hooks).
}

// CommonGenOption is an option common to model generation, prompt definition,
// and prompt execution.
type CommonGenOption interface {
	GenerateOption
	PromptOption
	PromptExecuteOption
}

// applyCommonGen applies the option to the common options.
func (o *commonGenOptions) applyCommonGen(opts *commonGenOptions) error {
	if err := o.configOptions.applyConfig(&opts.configOptions); err != nil {
		return err
	}
	if err := setOnce(&opts.MessagesFn, o.MessagesFn, o.MessagesFn != nil, opts.MessagesFn != nil,
		"cannot set messages more than once (either WithMessages or WithMessagesFn)"); err != nil {
		return err
	}
	if err := setOnce(&opts.Model, o.Model, o.Model != nil, opts.Model != nil,
		"cannot set model more than once (either WithModel or WithModelName)"); err != nil {
		return err
	}
	if err := setOnce(&opts.Tools, o.Tools, o.Tools != nil, opts.Tools != nil,
		"cannot set tools more than once (WithTools)"); err != nil {
		return err
	}
	if err := setOnce(&opts.Resources, o.Resources, o.Resources != nil, opts.Resources != nil,
		"cannot set resources more than once (WithResources)"); err != nil {
		return err
	}
	if err := setOnce(&opts.ToolChoice, o.ToolChoice, o.ToolChoice != "", opts.ToolChoice != "",
		"cannot set tool choice more than once (WithToolChoice)"); err != nil {
		return err
	}
	if err := setOnce(&opts.MaxTurns, o.MaxTurns, o.MaxTurns > 0, opts.MaxTurns > 0,
		"cannot set max turns more than once (WithMaxTurns)"); err != nil {
		return err
	}
	if err := setOnce(&opts.ReturnToolRequests, o.ReturnToolRequests, o.ReturnToolRequests != nil, opts.ReturnToolRequests != nil,
		"cannot configure returning tool requests more than once (WithReturnToolRequests)"); err != nil {
		return err
	}
	if err := setOnce(&opts.Middleware, o.Middleware, o.Middleware != nil, opts.Middleware != nil,
		"cannot set middleware more than once (WithMiddleware)"); err != nil {
		return err
	}
	if err := setOnce(&opts.Use, o.Use, o.Use != nil, opts.Use != nil,
		"cannot set middleware more than once (WithUse)"); err != nil {
		return err
	}
	return nil
}

// applyPromptExecute applies the option to the prompt request options.
func (o *commonGenOptions) applyPromptExecute(reqOpts *promptExecutionOptions) error {
	return o.applyCommonGen(&reqOpts.commonGenOptions)
}

// applyPrompt applies the option to the prompt options.
func (o *commonGenOptions) applyPrompt(pOpts *promptOptions) error {
	return o.applyCommonGen(&pOpts.commonGenOptions)
}

// applyGenerate applies the option to the generate options.
func (o *commonGenOptions) applyGenerate(genOpts *generateOptions) error {
	return o.applyCommonGen(&genOpts.commonGenOptions)
}

// WithMessages sets the messages.
// These messages will be sandwiched between the system and user prompts.
func WithMessages(messages ...*Message) CommonGenOption {
	return &commonGenOptions{
		MessagesFn: func(ctx context.Context, _ any) ([]*Message, error) {
			return messages, nil
		},
	}
}

// WithMessagesFn sets the request messages to the result of the function.
// These messages will be sandwiched between the system and user messages.
func WithMessagesFn(fn MessagesFn) CommonGenOption {
	return &commonGenOptions{MessagesFn: fn}
}

// WithTools sets the tools to use for the generate request.
func WithTools(tools ...ToolRef) CommonGenOption {
	return &commonGenOptions{Tools: tools}
}

// WithModel sets either a [Model] or a [ModelRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithModel.
func WithModel(model ModelArg) CommonGenOption {
	return &commonGenOptions{Model: model}
}

// WithModelName sets the model name to call for generation.
// The model name will be resolved to a [Model] and may error if the reference is invalid.
func WithModelName(name string) CommonGenOption {
	return &commonGenOptions{Model: NewModelRef(name, nil)}
}

// WithMiddleware sets middleware to apply to the model request.
//
// Deprecated: Use [WithUse] instead, which supports Generate, Model, and Tool hooks.
func WithMiddleware(middleware ...ModelMiddleware) CommonGenOption {
	return &commonGenOptions{Middleware: middleware}
}

// WithUse sets middleware to apply to generation. Middleware hooks wrap
// the generate loop, model calls, and tool executions.
//
// Accepts either a middleware config struct (produced by a plugin) or an
// inline adapter via [MiddlewareFunc]. The chain applies outer-to-inner, so
// WithUse(A, B) expands to A { B { ... } }.
func WithUse(middleware ...Middleware) CommonGenOption {
	return &commonGenOptions{Use: middleware}
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

// WithResources specifies resources to be temporarily available during generation.
// Resources are unregistered resources that get attached to a temporary registry
// during the generation request and cleaned up afterward.
func WithResources(resources ...Resource) CommonGenOption {
	return &commonGenOptions{Resources: resources}
}

// inputOptions are options for the input of a prompt.
type inputOptions struct {
	InputSchema  map[string]any // JSON schema of the input.
	DefaultInput map[string]any // Default input that will be used if no input is provided.
}

// InputOption is an option for the input of a prompt or tool.
// It applies to DefinePrompt() and DefineTool().
type InputOption interface {
	PromptOption
	ToolOption
}

// applyInput applies the option to the input options.
func (o *inputOptions) applyInput(opts *inputOptions) error {
	if err := setOnce(&opts.InputSchema, o.InputSchema, o.InputSchema != nil, opts.InputSchema != nil,
		"cannot set input schema more than once (WithInputType, WithInputSchema, or WithInputSchemaName)"); err != nil {
		return err
	}
	if err := setOnce(&opts.DefaultInput, o.DefaultInput, o.DefaultInput != nil, opts.DefaultInput != nil,
		"cannot set default input more than once (WithInputType)"); err != nil {
		return err
	}
	return nil
}

// applyPrompt applies the option to the prompt options.
func (o *inputOptions) applyPrompt(pOpts *promptOptions) error {
	return o.applyInput(&pOpts.inputOptions)
}

// applyTool applies the option to the tool options.
func (o *inputOptions) applyTool(tOpts *toolOptions) error {
	return o.applyInput(&tOpts.inputOptions)
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
	applyPrompt(*promptOptions) error
}

// applyPrompt applies the option to the prompt options.
func (o *promptOptions) applyPrompt(opts *promptOptions) error {
	if err := o.commonGenOptions.applyPrompt(opts); err != nil {
		return err
	}
	if err := o.promptingOptions.applyPrompt(opts); err != nil {
		return err
	}
	if err := o.inputOptions.applyPrompt(opts); err != nil {
		return err
	}
	if err := o.outputOptions.applyPrompt(opts); err != nil {
		return err
	}
	if err := setOnce(&opts.Description, o.Description, o.Description != "", opts.Description != "",
		"cannot set description more than once (WithDescription)"); err != nil {
		return err
	}
	if err := setOnce(&opts.Metadata, o.Metadata, o.Metadata != nil, opts.Metadata != nil,
		"cannot set metadata more than once (WithMetadata)"); err != nil {
		return err
	}
	return nil
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
	GenerateOption
	PromptOption
}

// applyPrompting applies the option to the prompting options.
func (o *promptingOptions) applyPrompting(opts *promptingOptions) error {
	if err := setOnce(&opts.SystemFn, o.SystemFn, o.SystemFn != nil, opts.SystemFn != nil,
		"cannot set system text more than once (either WithSystem or WithSystemFn)"); err != nil {
		return err
	}
	if err := setOnce(&opts.PromptFn, o.PromptFn, o.PromptFn != nil, opts.PromptFn != nil,
		"cannot set prompt text more than once (either WithPrompt or WithPromptFn)"); err != nil {
		return err
	}
	return nil
}

// applyPrompt applies the option to the prompt options.
func (o *promptingOptions) applyPrompt(opts *promptOptions) error {
	return o.applyPrompting(&opts.promptingOptions)
}

// applyGenerate applies the option to the generate options.
func (o *promptingOptions) applyGenerate(opts *generateOptions) error {
	return o.applyPrompting(&opts.promptingOptions)
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
	GenerateOption
	PromptOption
}

// applyOutput applies the option to the output options.
func (o *outputOptions) applyOutput(opts *outputOptions) error {
	if err := setOnce(&opts.OutputSchema, o.OutputSchema, o.OutputSchema != nil, opts.OutputSchema != nil,
		"cannot set output schema more than once (WithOutputType, WithOutputSchema, or WithOutputSchemaName)"); err != nil {
		return err
	}
	if err := setOnce(&opts.OutputInstructions, o.OutputInstructions, o.OutputInstructions != nil, opts.OutputInstructions != nil,
		"cannot set output instructions more than once (WithOutputFormat)"); err != nil {
		return err
	}
	// OutputFormat and CustomConstrained are override (no duplicate check):
	// WithOutputType sets format alongside the schema, and we want both to land.
	if o.OutputFormat != "" {
		opts.OutputFormat = o.OutputFormat
	}
	if o.CustomConstrained {
		opts.CustomConstrained = o.CustomConstrained
	}
	return nil
}

// applyPrompt applies the option to the prompt options.
func (o *outputOptions) applyPrompt(pOpts *promptOptions) error {
	return o.applyOutput(&pOpts.outputOptions)
}

// applyGenerate applies the option to the generate options.
func (o *outputOptions) applyGenerate(genOpts *generateOptions) error {
	return o.applyOutput(&genOpts.outputOptions)
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
	GenerateOption
	PromptExecuteOption
}

// applyExecution applies the option to the runtime options.
func (o *executionOptions) applyExecution(execOpts *executionOptions) error {
	return setOnce(&execOpts.Stream, o.Stream, o.Stream != nil, execOpts.Stream != nil,
		"cannot set stream callback more than once (WithStream)")
}

// applyGenerate applies the option to the generate options.
func (o *executionOptions) applyGenerate(genOpts *generateOptions) error {
	return o.applyExecution(&genOpts.executionOptions)
}

// applyPromptExecute applies the option to the prompt request options.
func (o *executionOptions) applyPromptExecute(pgOpts *promptExecutionOptions) error {
	return o.applyExecution(&pgOpts.executionOptions)
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
// It applies to [Generate], [prompt.Execute], [Embed], and [Retrieve].
type DocumentOption interface {
	GenerateOption
	PromptExecuteOption
	EmbedderOption
	RetrieverOption
}

// applyDocument applies the option to the context options.
func (o *documentOptions) applyDocument(docOpts *documentOptions) error {
	return setOnce(&docOpts.Documents, o.Documents, o.Documents != nil, docOpts.Documents != nil,
		"cannot set documents more than once (WithDocs)")
}

// applyGenerate applies the option to the generate options.
func (o *documentOptions) applyGenerate(genOpts *generateOptions) error {
	return o.applyDocument(&genOpts.documentOptions)
}

// applyPromptExecute applies the option to the prompt generate options.
func (o *documentOptions) applyPromptExecute(pgOpts *promptExecutionOptions) error {
	return o.applyDocument(&pgOpts.documentOptions)
}

// applyEmbedder applies the option to the embed options.
func (o *documentOptions) applyEmbedder(embedOpts *embedderOptions) error {
	return o.applyDocument(&embedOpts.documentOptions)
}

// applyRetriever applies the option to the retrieve options.
func (o *documentOptions) applyRetriever(retOpts *retrieverOptions) error {
	return o.applyDocument(&retOpts.documentOptions)
}

// WithTextDocs sets the text to be used as context documents for generation or as input to an embedder.
func WithTextDocs(text ...string) DocumentOption {
	docs := make([]*Document, len(text))
	for i, t := range text {
		docs[i] = DocumentFromText(t, nil)
	}
	return &documentOptions{Documents: docs}
}

// WithDocs sets the documents to be used as context for generation or as input to an embedder.
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
	applyEvaluator(*evaluatorOptions) error
}

// applyEvaluator applies the option to the evaluator options.
func (o *evaluatorOptions) applyEvaluator(evalOpts *evaluatorOptions) error {
	if err := o.applyConfig(&evalOpts.configOptions); err != nil {
		return err
	}
	if err := setOnce(&evalOpts.Dataset, o.Dataset, o.Dataset != nil, evalOpts.Dataset != nil,
		"cannot set dataset more than once (WithDataset)"); err != nil {
		return err
	}
	if err := setOnce(&evalOpts.ID, o.ID, o.ID != "", evalOpts.ID != "",
		"cannot set ID more than once (WithID)"); err != nil {
		return err
	}
	if err := setOnce(&evalOpts.Evaluator, o.Evaluator, o.Evaluator != nil, evalOpts.Evaluator != nil,
		"cannot set evaluator more than once (WithEvaluator or WithEvaluatorName)"); err != nil {
		return err
	}
	return nil
}

// WithDataset sets the dataset to do evaluation on.
func WithDataset(examples ...*Example) EvaluatorOption {
	return &evaluatorOptions{Dataset: examples}
}

// WithID sets the ID of the evaluation to uniquely identify it.
func WithID(ID string) EvaluatorOption {
	return &evaluatorOptions{ID: ID}
}

// WithEvaluator sets either a [Evaluator] or a [EvaluatorRef] that may contain a config.
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
	applyEmbedder(*embedderOptions) error
}

// applyEmbedder applies the option to the embed options.
func (o *embedderOptions) applyEmbedder(embedOpts *embedderOptions) error {
	if err := o.applyConfig(&embedOpts.configOptions); err != nil {
		return err
	}
	if err := o.applyDocument(&embedOpts.documentOptions); err != nil {
		return err
	}
	return setOnce(&embedOpts.Embedder, o.Embedder, o.Embedder != nil, embedOpts.Embedder != nil,
		"cannot set embedder more than once (WithEmbedder or WithEmbedderName)")
}

// WithEmbedder sets either a [Embedder] or a [EmbedderRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithEmbedder.
func WithEmbedder(embedder EmbedderArg) EmbedderOption {
	return &embedderOptions{Embedder: embedder}
}

// WithEmbedderName sets the embedder name to call for document embedding.
// The embedder name will be resolved to a [Embedder] and may error if the reference is invalid.
func WithEmbedderName(name string) EmbedderOption {
	return &embedderOptions{Embedder: NewEmbedderRef(name, nil)}
}

// retrieverOptions holds configuration and input for a retriever request.
type retrieverOptions struct {
	configOptions
	documentOptions
	Retriever RetrieverArg // Retriever to use.
}

// RetrieverOption is an option for configuring a retriever request.
// It applies only to [Retriever.Retrieve].
type RetrieverOption interface {
	applyRetriever(*retrieverOptions) error
}

// applyRetriever applies the option to the retrieve options.
func (o *retrieverOptions) applyRetriever(retOpts *retrieverOptions) error {
	if err := o.applyConfig(&retOpts.configOptions); err != nil {
		return err
	}
	if err := o.applyDocument(&retOpts.documentOptions); err != nil {
		return err
	}
	return setOnce(&retOpts.Retriever, o.Retriever, o.Retriever != nil, retOpts.Retriever != nil,
		"cannot set retriever more than once (WithRetriever or WithRetrieverName)")
}

// WithRetriever sets either a [Retriever] or a [RetrieverRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithRetriever.
func WithRetriever(retriever RetrieverArg) RetrieverOption {
	return &retrieverOptions{Retriever: retriever}
}

// WithRetrieverName sets the retriever name to call for document retrieval.
// The retriever name will be resolved to a [Retriever] and may error if the reference is invalid.
func WithRetrieverName(name string) RetrieverOption {
	return &retrieverOptions{Retriever: NewRetrieverRef(name, nil)}
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
}

// GenerateOption is an option for generating a model response. It applies only to Generate().
type GenerateOption interface {
	applyGenerate(*generateOptions) error
}

// applyGenerate applies the option to the generate options.
func (o *generateOptions) applyGenerate(genOpts *generateOptions) error {
	if err := o.commonGenOptions.applyGenerate(genOpts); err != nil {
		return err
	}
	if err := o.promptingOptions.applyGenerate(genOpts); err != nil {
		return err
	}
	if err := o.outputOptions.applyGenerate(genOpts); err != nil {
		return err
	}
	if err := o.executionOptions.applyGenerate(genOpts); err != nil {
		return err
	}
	if err := o.documentOptions.applyGenerate(genOpts); err != nil {
		return err
	}
	if err := setOnce(&genOpts.RespondParts, o.RespondParts, o.RespondParts != nil, genOpts.RespondParts != nil,
		"cannot set respond parts more than once (WithToolResponses)"); err != nil {
		return err
	}
	if err := setOnce(&genOpts.RestartParts, o.RestartParts, o.RestartParts != nil, genOpts.RestartParts != nil,
		"cannot set restart parts more than once (WithToolRestarts)"); err != nil {
		return err
	}
	return nil
}

// WithToolResponses provides resolved responses for interrupted tool calls.
// Use this when you already have the result and want to skip re-executing the tool.
func WithToolResponses(parts ...*Part) GenerateOption {
	return &generateOptions{RespondParts: parts}
}

// WithToolRestarts re-executes interrupted tool calls with additional metadata.
// Use this when the original call lacked required context (e.g., auth, user confirmation)
// that should now allow the tool to complete successfully.
func WithToolRestarts(parts ...*Part) GenerateOption {
	return &generateOptions{RestartParts: parts}
}

// toolOptions holds configuration options for defining tools.
type toolOptions struct {
	inputOptions
}

// ToolOption is an option for defining a tool.
type ToolOption interface {
	applyTool(*toolOptions) error
}

// applyTool applies the option to the tool options.
func (o *toolOptions) applyTool(opts *toolOptions) error {
	return o.inputOptions.applyTool(opts)
}

// promptExecutionOptions are options for generating a model response by executing a prompt.
type promptExecutionOptions struct {
	commonGenOptions
	executionOptions
	documentOptions
	Input any // Input fields for the prompt. If not nil this should be a struct that matches the prompt's input schema.
}

// PromptExecuteOption is an option for executing a prompt. It applies only to [prompt.Execute].
type PromptExecuteOption interface {
	applyPromptExecute(*promptExecutionOptions) error
}

// applyPromptExecute applies the option to the prompt request options.
func (o *promptExecutionOptions) applyPromptExecute(pgOpts *promptExecutionOptions) error {
	if err := o.commonGenOptions.applyPromptExecute(pgOpts); err != nil {
		return err
	}
	if err := o.executionOptions.applyPromptExecute(pgOpts); err != nil {
		return err
	}
	if err := o.documentOptions.applyPromptExecute(pgOpts); err != nil {
		return err
	}
	return setOnce(&pgOpts.Input, o.Input, o.Input != nil, pgOpts.Input != nil,
		"cannot set input more than once (WithInput)")
}

// WithInput sets the input for the prompt request. Input must conform to the
// prompt's input schema and can either be a map[string]any or a struct of the same api.
func WithInput(input any) PromptExecuteOption {
	return &promptExecutionOptions{Input: input}
}
