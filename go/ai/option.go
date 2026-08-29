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
	"fmt"
	"slices"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// PromptFn is a function that generates a prompt from a prompt's input.
//
// The input is untyped. [WithPromptFn] and [WithSystemFn] take a function with
// a concrete input type and convert for you.
type PromptFn = func(context.Context, any) (string, error)

// MessagesFn is a function that generates messages from a prompt's input.
//
// The input is untyped. [WithMessagesFn] takes a function with a concrete input
// type and converts for you.
type MessagesFn = func(context.Context, any) ([]*Message, error)

// appendFn composes two content-producing functions so their outputs
// concatenate in call order. It backs the accumulate semantics of the
// collection options: passing several of them, in any mix, appends instead of
// overwriting. Either side may be nil, in which case the other is returned
// unwrapped.
//
// [MessagesFn] and [DocsFn] are aliases for this shape, so one helper serves
// both.
func appendFn[T any](existing, next func(context.Context, any) ([]T, error)) func(context.Context, any) ([]T, error) {
	if existing == nil {
		return next
	}
	if next == nil {
		return existing
	}
	return func(ctx context.Context, input any) ([]T, error) {
		before, err := existing(ctx, input)
		if err != nil {
			return nil, err
		}
		after, err := next(ctx, input)
		if err != nil {
			return nil, err
		}
		// Concat rather than append onto before: WithMessages(history...)
		// hands the caller's slice straight through, so appending in place
		// would write into the spare capacity of the array backing their
		// history. Concat always allocates a fresh slice.
		return slices.Concat(before, after), nil
	}
}

// PartsFn is a function that generates message content from a prompt's input.
//
// The input is untyped. [WithPromptPartsFn] and [WithSystemPartsFn] take a
// function with a concrete input type and convert for you.
type PartsFn = func(context.Context, any) ([]*Part, error)

// DocsFn is a function that generates context documents from a prompt's input.
//
// The input is untyped. [WithDocsFn] takes a function with a concrete input
// type and converts for you.
type DocsFn = func(context.Context, any) ([]*Document, error)

// coerceFn adapts a typed content function to the untyped signature the option
// structs store. option names the option in the error, since a prompt can fill
// several slots from functions and only one of them rejected the input.
//
// The raw value's Go type depends on how the prompt was invoked: an in-process
// call with [WithInput] passes the value through as-is, while the reflection
// API (Dev UI, flows over HTTP) and the default recorded by [WithInputType]
// both arrive as map[string]any. Converting here means content functions see
// the same type whichever way they were reached. A nil raw value yields the
// zero value of In, which is what [Generate] passes.
func coerceFn[In, Out any](option string, fn func(context.Context, In) (Out, error)) func(context.Context, any) (Out, error) {
	return func(ctx context.Context, raw any) (Out, error) {
		in, err := base.ConvertToExact[In](raw)
		if err != nil {
			var zero Out
			// base cannot classify this itself: the same failure is only a
			// caller's bad input once a content function disagrees with it.
			return zero, status.Errorf(ErrInputTypeMismatch, "%s input: %w", option, err)
		}
		return fn(ctx, in)
	}
}

// staticPartsFn adapts fixed parts to a [PartsFn]. It clones per render, so a
// later append to the message's content cannot reach the parts stored on the
// prompt.
func staticPartsFn(parts []*Part) PartsFn {
	return func(context.Context, any) ([]*Part, error) {
		return slices.Clone(parts), nil
	}
}

// textPartsFn adapts a string-returning content function to a [PartsFn]. An
// empty string yields no parts and so no message, matching empty template text.
func textPartsFn[In any](option string, fn func(context.Context, In) (string, error)) PartsFn {
	coerced := coerceFn(option, fn)
	return func(ctx context.Context, raw any) ([]*Part, error) {
		text, err := coerced(ctx, raw)
		if err != nil {
			return nil, err
		}
		if text == "" {
			return nil, nil
		}
		return []*Part{NewTextPart(text)}, nil
	}
}

// configOptions holds configuration options.
type configOptions struct {
	Config any // Primitive (model, embedder, retriever, etc) configuration.
}

// ConfigOption is an option for model configuration. It is accepted anywhere
// a primitive takes a config: generation, prompt definition and execution,
// embedding, retrieval, and evaluation.
type ConfigOption interface {
	CommonGenOption
	EmbedderOption
	RetrieverOption
	EvaluatorOption
	applyConfig(*configOptions)
}

func (o *configOptions) applyConfig(opts *configOptions) {
	if o.Config != nil {
		opts.Config = o.Config
	}
}

func (o *configOptions) applyCommonGen(opts *commonGenOptions) { o.applyConfig(&opts.configOptions) }
func (o *configOptions) applyPrompt(opts *promptOptions)       { o.applyConfig(&opts.configOptions) }
func (o *configOptions) applyGenerate(opts *generateOptions)   { o.applyConfig(&opts.configOptions) }
func (o *configOptions) applyEmbedder(opts *embedderOptions)   { o.applyConfig(&opts.configOptions) }
func (o *configOptions) applyRetriever(opts *retrieverOptions) { o.applyConfig(&opts.configOptions) }
func (o *configOptions) applyEvaluator(opts *evaluatorOptions) { o.applyConfig(&opts.configOptions) }

func (o *configOptions) applyPromptExecute(opts *promptExecutionOptions) {
	o.applyConfig(&opts.configOptions)
}

// WithConfig sets the configuration. Repeating this option takes the last
// config set.
func WithConfig(config any) ConfigOption {
	return &configOptions{Config: config}
}

// commonGenOptions are common options for model generation, prompt definition, and prompt execution.
type commonGenOptions struct {
	configOptions
	Model              ModelArg          // Model to use.
	MessagesFn         MessagesFn        // Function to generate messages. Used verbatim.
	MessagesText       *string           // Template text for the conversation, rendered as a dotprompt template.
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
	PromptOption
	GenerateOption
	PromptExecuteOption
	applyCommonGen(*commonGenOptions)
}

func (o *commonGenOptions) applyCommonGen(opts *commonGenOptions) {
	o.configOptions.applyConfig(&opts.configOptions)

	opts.MessagesFn = appendFn(opts.MessagesFn, o.MessagesFn)
	if o.MessagesText != nil {
		opts.MessagesText = o.MessagesText
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
	opts.Middleware = append(opts.Middleware, o.Middleware...)
	opts.Use = append(opts.Use, o.Use...)
}

func (o *commonGenOptions) applyPrompt(opts *promptOptions) { o.applyCommonGen(&opts.commonGenOptions) }
func (o *commonGenOptions) applyGenerate(opts *generateOptions) {
	o.applyCommonGen(&opts.commonGenOptions)
}

func (o *commonGenOptions) applyPromptExecute(opts *promptExecutionOptions) {
	o.applyCommonGen(&opts.commonGenOptions)
}

// WithMessages adds messages to the request, placed between the system and
// user prompts. Repeating this option, or mixing it with [WithMessagesFn],
// appends: messages accumulate in the order the options are passed.
//
// Message text is used verbatim, never compiled as a dotprompt template, so
// history containing literal braces passes through untouched. To build message
// text from a prompt's input, use [WithMessagesFn], which receives the typed
// input, or [WithMessagesTemplate] for a multi-turn template.
//
// Declaring the conversation makes the prompt responsible for the messages
// passed to [Prompt.Execute]: they are not spliced in automatically, because
// only the prompt knows where its examples end and a real conversation begins.
// Place them with {{history}} in a [WithMessagesTemplate] or
// [HistoryFromContext] in a [WithMessagesFn].
func WithMessages(messages ...*Message) CommonGenOption {
	return WithMessagesFn(func(context.Context, any) ([]*Message, error) {
		return messages, nil
	})
}

// WithMessagesTemplate sets the conversation to a dotprompt template.
// Optional args are applied to text with [fmt.Sprintf].
//
// This is the multi-turn form: each {{role "..."}} block starts a new message,
// so one template can express a system preamble, few-shot examples, and the
// user turn. It is what a .prompt file's body compiles to.
//
// Messages passed to [Prompt.Execute] reach the template at {{history}}, or,
// without an explicit {{history}}, are inserted before its final user message.
//
// Unlike [WithMessages], the text here is compiled, so literal braces in it are
// template syntax. Content from a user or a remote source belongs in
// [WithMessages] or [WithMessagesFn], which never compile it.
//
// The template is the whole conversation, so nothing else can contribute to it:
// passing this in the same [DefinePrompt] call as [WithMessages] or
// [WithMessagesFn] panics. Write those messages as {{role}} blocks instead.
// Repeating this option alone fills one slot, so the last one set wins.
//
// Compiling needs a prompt, so this is a [PromptOption]: passing it to
// [Generate] or [Prompt.Execute] does not compile. Use [WithMessages] there.
func WithMessagesTemplate(text string, args ...any) PromptOption {
	text = sprintfText(text, args)
	return &promptOptions{commonGenOptions: commonGenOptions{MessagesText: &text}}
}

// WithMessagesFn adds messages produced by fn at request time, placed between
// the system and user prompts. Like [WithMessages], repeating this option (or
// mixing the two) appends the produced messages in call order.
//
// fn receives the prompt's input converted to In, or the zero value of In when
// there is none, as at [Generate]. Its messages are used verbatim, as with
// [WithMessages].
//
// Those messages are the conversation the prompt declares, so messages supplied
// at execution time are not appended on top of them. fn reads them from
// [HistoryFromContext] instead, and can summarize or truncate them rather than
// only prepend to them.
func WithMessagesFn[In any](fn func(context.Context, In) ([]*Message, error)) CommonGenOption {
	return &commonGenOptions{MessagesFn: coerceFn("WithMessagesFn", fn)}
}

// WithTools adds tools to use for the generate request. Repeating this option
// appends; duplicate tools (by name) are rejected when the request runs.
func WithTools(tools ...ToolRef) CommonGenOption {
	return &commonGenOptions{Tools: tools}
}

// WithModel sets either a [Model] or a [ModelRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithModel.
// Repeating this option, or mixing it with [WithModelName], takes the last
// model set.
func WithModel(model ModelArg) CommonGenOption {
	return &commonGenOptions{Model: model}
}

// WithModelName sets the model name to call for generation.
// The model name will be resolved to a [Model] and may error if the reference is invalid.
// Repeating this option, or mixing it with [WithModel], takes the last model set.
func WithModelName(name string) CommonGenOption {
	return WithModel(NewModelRef(name, nil))
}

// WithMiddleware adds middleware to apply to the model request. Repeating this
// option appends to the chain.
//
// Deprecated: Use [WithUse] instead, which supports Generate, Model, and Tool hooks.
func WithMiddleware(middleware ...ModelMiddleware) CommonGenOption {
	return &commonGenOptions{Middleware: middleware}
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
// Defaults to 50.
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
func WithResources(resources ...Resource) CommonGenOption {
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
	PromptOption
	applyInput(*inputOptions)
}

// InputSchemaOption is an [InputOption] that also applies to the tool
// constructors, where the input schema it supplies stands in for an In type
// parameter of 'any'. A tool whose input shape a Go type can express should
// use the type parameter instead.
type InputSchemaOption interface {
	InputOption
	ToolOption
}

// applyInput applies the option to the input options. The input configuration
// is one slot: the last option to set it replaces both the schema and the
// default input together, so overriding [WithInputType] with [WithInputSchema]
// or [WithInputSchemaName] does not leave the old type's defaults behind to be
// rendered against the new schema.
func (o *inputOptions) applyInput(opts *inputOptions) {
	if o.InputSchema != nil || o.DefaultInput != nil {
		opts.InputSchema = o.InputSchema
		opts.DefaultInput = o.DefaultInput
	}
}

func (o *inputOptions) applyPrompt(opts *promptOptions) { o.applyInput(&opts.inputOptions) }
func (o *inputOptions) applyTool(opts *toolOptions)     { o.applyInput(&opts.inputOptions) }

// WithInputType uses the type provided to derive the input schema.
// The inputted value may serve as the default input if no input is given at generation time depending on the action.
// Only supports structs and map[string]any.
//
// It applies to the tool constructors too, where the derived schema stands in
// for an In type parameter of 'any'. Prefer the type parameter there: it is
// the same schema plus a typed function signature.
func WithInputType(input any) InputSchemaOption {
	// Converting rather than marshaling by hand gives the default input the
	// same Go types it would have arriving over the wire.
	defaultInput, err := base.ConvertToExact[map[string]any](input)
	if err != nil {
		panic(fmt.Errorf("type %T is not supported, only structs and map[string]any are supported (WithInputType)", input))
	}

	return &inputOptions{
		InputSchema:  core.InferSchemaMap(input),
		DefaultInput: defaultInput,
	}
}

// WithInputSchema manually provides a schema map for the input.
func WithInputSchema(schema map[string]any) InputSchemaOption {
	return &inputOptions{InputSchema: schema}
}

// WithInputSchemaName sets a pre-registered schema by name for the input.
// The schema is resolved from the registry at execution time; register it with
// [github.com/firebase/genkit/go/genkit.DefineSchema].
func WithInputSchemaName(name string) InputSchemaOption {
	return &inputOptions{InputSchema: core.SchemaRef(name)}
}

// promptOptions are options for defining a prompt.
type promptOptions struct {
	commonGenOptions
	promptingOptions
	inputOptions
	outputOptions
	documentOptions
	Description string         // Description of the prompt.
	Metadata    map[string]any // Arbitrary metadata.
}

// PromptOption is an option for defining a prompt.
// It applies only to DefinePrompt().
type PromptOption interface {
	applyPrompt(*promptOptions)
}

func (o *promptOptions) applyPrompt(opts *promptOptions) {
	o.commonGenOptions.applyPrompt(opts)
	o.promptingOptions.applyPrompt(opts)
	o.inputOptions.applyPrompt(opts)
	o.outputOptions.applyPrompt(opts)

	o.documentOptions.applyPrompt(opts)

	if o.Description != "" {
		opts.Description = o.Description
	}
	if o.Metadata != nil {
		opts.Metadata = o.Metadata
	}
}

// WithDescription sets the description of the prompt. Repeating this option
// takes the last description set.
func WithDescription(description string) PromptOption {
	return &promptOptions{Description: description}
}

// WithMetadata sets arbitrary metadata for the prompt. Repeating this option
// replaces the metadata rather than merging it.
func WithMetadata(metadata map[string]any) PromptOption {
	return &promptOptions{Metadata: metadata}
}

// promptingOptions are options for the system and user prompts of a prompt or generate request.
//
// Each slot holds either template text or a content function, never both. Text
// is compiled against the prompt's input; a function's result is used verbatim,
// since only text the prompt author wrote is compiled.
type promptingOptions struct {
	SystemText *string // Template text for the system prompt.
	SystemFn   PartsFn // Function returning system prompt content.
	PromptText *string // Template text for the user prompt.
	PromptFn   PartsFn // Function returning user prompt content.
}

// PromptingOption is an option for the system and user prompts of a prompt or generate request.
// It applies only to DefinePrompt() and Generate().
type PromptingOption interface {
	PromptOption
	GenerateOption
	applyPrompting(*promptingOptions)
}

// applyPrompting merges the two single-message slots, each shared by the four
// options that fill it, so the last one set wins.
//
// Both fields of a slot are assigned together to clear the loser: the renderer
// consults them in a fixed order, and a leftover value would let that order
// pick the winner instead of the caller.
func (o *promptingOptions) applyPrompting(opts *promptingOptions) {
	if o.SystemText != nil || o.SystemFn != nil {
		opts.SystemText, opts.SystemFn = o.SystemText, o.SystemFn
	}
	if o.PromptText != nil || o.PromptFn != nil {
		opts.PromptText, opts.PromptFn = o.PromptText, o.PromptFn
	}
}

func (o *promptingOptions) applyPrompt(opts *promptOptions) { o.applyPrompting(&opts.promptingOptions) }
func (o *promptingOptions) applyGenerate(opts *generateOptions) {
	o.applyPrompting(&opts.promptingOptions)
}

// sprintfText applies args to text, leaving it alone when none were given.
func sprintfText(text string, args []any) string {
	if len(args) == 0 {
		return text
	}
	// Assigning avoids a compile-time warning about non-constant text.
	t := text
	return fmt.Sprintf(t, args...)
}

// WithSystem sets the system prompt message.
// The system prompt is always the first message in the list.
//
// With [DefinePrompt], the text is compiled as a dotprompt template against the
// prompt's input, so it may reference input fields such as {{name}}. args, if
// given, are applied with [fmt.Sprintf] first.
//
// A {{role}} marker in the text is an error: this slot is one system message.
// [WithMessagesTemplate] is where turns with their own roles belong.
func WithSystem(text string, args ...any) PromptingOption {
	text = sprintfText(text, args)
	return &promptingOptions{SystemText: &text}
}

// WithSystemFn sets the function that generates the system prompt message.
// The system prompt is always the first message in the list.
//
// fn receives the prompt's input converted to In, or the zero value of In when
// there is none, as at [Generate]. Its string is used verbatim, never compiled
// as a template, so it may safely hold user content and literal braces. Use
// [WithSystemPartsFn] to return non-text content.
//
// It shares one slot with [WithSystem], [WithSystemParts], and
// [WithSystemPartsFn]: the last one set wins.
func WithSystemFn[In any](fn func(context.Context, In) (string, error)) PromptingOption {
	return &promptingOptions{SystemFn: textPartsFn("WithSystemFn", fn)}
}

// WithSystemParts sets the content of the system prompt message.
// The system prompt is always the first message in the list.
//
// It is the multi-part form of [WithSystem], for instructions that mix text
// with media or other non-text parts. The parts are used verbatim, never
// compiled as a template. Use [WithSystemPartsFn] when the content depends on
// the prompt's input.
//
// It shares one slot with [WithSystem], [WithSystemFn], and
// [WithSystemPartsFn]: the last one set wins.
func WithSystemParts(parts ...*Part) PromptingOption {
	return &promptingOptions{SystemFn: staticPartsFn(parts)}
}

// WithSystemPartsFn sets the function that generates the content of the system
// prompt message. The system prompt is always the first message in the list.
//
// It is the multi-part form of [WithSystemFn]. The returned parts are used
// verbatim.
//
// It shares one slot with [WithSystem], [WithSystemParts], and [WithSystemFn]:
// the last one set wins.
func WithSystemPartsFn[In any](fn func(context.Context, In) ([]*Part, error)) PromptingOption {
	return &promptingOptions{SystemFn: coerceFn("WithSystemPartsFn", fn)}
}

// WithPrompt sets the user prompt message.
// The user prompt is always the last message in the list.
//
// With [DefinePrompt], the text is compiled as a dotprompt template against the
// prompt's input, so it may reference input fields such as {{name}}. args, if
// given, are applied with [fmt.Sprintf] first.
//
// A {{role}} marker in the text is an error: this slot is one user message.
// [WithMessagesTemplate] is where turns with their own roles belong.
func WithPrompt(text string, args ...any) PromptingOption {
	text = sprintfText(text, args)
	return &promptingOptions{PromptText: &text}
}

// WithPromptFn sets the function that generates the user prompt message.
// The user prompt is always the last message in the list.
//
// fn receives the prompt's input converted to In, or the zero value of In when
// there is none, as at [Generate]. Its string is used verbatim, never compiled
// as a template, so it may safely hold user content and literal braces. Use
// [WithPromptPartsFn] to return non-text content.
//
// It shares one slot with [WithPrompt], [WithPromptParts], and
// [WithPromptPartsFn]: the last one set wins.
func WithPromptFn[In any](fn func(context.Context, In) (string, error)) PromptingOption {
	return &promptingOptions{PromptFn: textPartsFn("WithPromptFn", fn)}
}

// WithPromptParts sets the content of the user prompt message.
// The user prompt is always the last message in the list.
//
// It is the multi-part form of [WithPrompt], for prompts that mix text with
// media or other non-text parts. The parts are used verbatim, never compiled as
// a template. Use [WithPromptPartsFn] when the content depends on the prompt's
// input.
//
//	genkit.Generate(ctx, g,
//		ai.WithModelName("googleai/gemini-flash-latest"),
//		ai.WithPromptParts(
//			ai.NewTextPart("What is in this image?"),
//			ai.NewMediaPart("image/png", imageURL),
//		),
//	)
//
// It shares one slot with [WithPrompt], [WithPromptFn], and
// [WithPromptPartsFn]: the last one set wins.
func WithPromptParts(parts ...*Part) PromptingOption {
	return &promptingOptions{PromptFn: staticPartsFn(parts)}
}

// WithPromptPartsFn sets the function that generates the content of the user
// prompt message. The user prompt is always the last message in the list.
//
// It is the multi-part form of [WithPromptFn]. The returned parts are used
// verbatim.
//
// It shares one slot with [WithPrompt], [WithPromptParts], and [WithPromptFn]:
// the last one set wins.
func WithPromptPartsFn[In any](fn func(context.Context, In) ([]*Part, error)) PromptingOption {
	return &promptingOptions{PromptFn: coerceFn("WithPromptPartsFn", fn)}
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
	PromptOption
	GenerateOption
	applyOutput(*outputOptions)
}

// OutputSchemaOption is an [OutputOption] that provides an explicit output
// schema, inline or by name. Unlike the schema-inference and
// generation-steering options, it also applies to the tool constructors: a
// tool's output shape is decided by its function, so the only output options
// that make sense there are explicit schemas standing in for an Out type
// parameter of 'any'; a tool whose output a Go type can express should use
// the type parameter instead.
type OutputSchemaOption interface {
	OutputOption
	ToolOption
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

func (o *outputOptions) applyPrompt(opts *promptOptions)     { o.applyOutput(&opts.outputOptions) }
func (o *outputOptions) applyGenerate(opts *generateOptions) { o.applyOutput(&opts.outputOptions) }

// applyTool applies the option to the tool options. Only the explicit-schema
// output options reach here: [OutputSchemaOption] is the sole output option
// type that satisfies [ToolOption]. The schema is a single-value slot, so the
// last option to set it wins.
func (o *outputOptions) applyTool(tOpts *toolOptions) {
	if o.OutputSchema != nil {
		tOpts.OutputSchema = o.OutputSchema
	}
}

// WithOutputType sets the output format to JSON and the schema derived from the given value.
func WithOutputType(output any) OutputOption {
	return &outputOptions{
		OutputSchema: core.InferSchemaMap(output),
		OutputFormat: OutputFormatJSON,
	}
}

// WithOutputSchema manually provides a schema map for the output.
func WithOutputSchema(schema map[string]any) OutputSchemaOption {
	return &outputOptions{
		OutputSchema: schema,
		OutputFormat: OutputFormatJSON,
	}
}

// WithOutputSchemaName sets the schema name that will be resolved at execution time.
// Register the schema with [github.com/firebase/genkit/go/genkit.DefineSchema].
func WithOutputSchemaName(name string) OutputSchemaOption {
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
	Stream      ModelStreamCallback // Function to call with each chunk of the generated response.
	chainStream bool                // Chain Stream after an existing callback instead of replacing it.
}

// ExecutionOption is an option for the execution of a prompt or generate request. It applies only to Generate() and prompt.Execute().
type ExecutionOption interface {
	GenerateOption
	PromptExecuteOption
	applyExecution(*executionOptions)
}

// applyExecution fills the stream slot, honoring chainStream: a chained
// callback runs after any existing one instead of replacing it.
func (o *executionOptions) applyExecution(execOpts *executionOptions) {
	if o.Stream == nil {
		return
	}
	if prev, next := execOpts.Stream, o.Stream; o.chainStream && prev != nil {
		execOpts.Stream = func(ctx context.Context, chunk *ModelResponseChunk) error {
			if err := prev(ctx, chunk); err != nil {
				return err
			}
			return next(ctx, chunk)
		}
		return
	}
	execOpts.Stream = o.Stream
}

func (o *executionOptions) applyGenerate(opts *generateOptions) {
	o.applyExecution(&opts.executionOptions)
}

func (o *executionOptions) applyPromptExecute(opts *promptExecutionOptions) {
	o.applyExecution(&opts.executionOptions)
}

// WithStreaming sets the stream callback for the generate request.
// A callback is a function that is called with each chunk of the generated response before the final response is returned.
// Repeating this option takes the last callback set. The stream-returning
// APIs ([GenerateStream], [Prompt.ExecuteStream], and their typed variants)
// attach their own iterator callback without displacing one set here; both
// receive every chunk.
func WithStreaming(callback ModelStreamCallback) ExecutionOption {
	return &executionOptions{Stream: callback}
}

// withChainedStreaming installs a stream callback without displacing one the
// caller already set: any existing callback runs first, then this one. The
// stream-returning wrappers use it to attach their iterator callback while
// keeping a caller-supplied [WithStreaming] observable; a plain WithStreaming
// appended after the caller's options would win the last-win slot and
// silently drop theirs.
func withChainedStreaming(callback ModelStreamCallback) ExecutionOption {
	return &executionOptions{Stream: callback, chainStream: true}
}

// documentOptions are options for providing context documents to a prompt or generate request or as input to an embedder.
type documentOptions struct {
	Documents []*Document // Docs to pass as context or input.
	DocsFn    DocsFn      // Function to generate docs from a prompt's input.
}

// DocumentOption is an option for providing context or input documents.
// It applies to [DefinePrompt], [Generate], [Prompt.Execute], [Embed], and
// [Retrieve].
type DocumentOption interface {
	PromptOption
	GenerateOption
	PromptExecuteOption
	EmbedderOption
	RetrieverOption
	applyDocument(*documentOptions)
}

// applyDocument accumulates documents: the fixed ones append in call order and
// the [WithDocsFn] functions compose the same way.
func (o *documentOptions) applyDocument(opts *documentOptions) {
	opts.Documents = append(opts.Documents, o.Documents...)
	opts.DocsFn = appendFn(opts.DocsFn, o.DocsFn)
}

func (o *documentOptions) applyPrompt(opts *promptOptions) {
	o.applyDocument(&opts.documentOptions)
}

func (o *documentOptions) applyGenerate(opts *generateOptions) {
	o.applyDocument(&opts.documentOptions)
}
func (o *documentOptions) applyEmbedder(opts *embedderOptions) {
	o.applyDocument(&opts.documentOptions)
}
func (o *documentOptions) applyRetriever(opts *retrieverOptions) {
	o.applyDocument(&opts.documentOptions)
}

func (o *documentOptions) applyPromptExecute(opts *promptExecutionOptions) {
	o.applyDocument(&opts.documentOptions)
}

// WithTextDocs adds text as context documents for generation or as input to an
// embedder. Repeating this option (or mixing it with [WithDocs]) appends.
func WithTextDocs(text ...string) DocumentOption {
	docs := make([]*Document, len(text))
	for i, t := range text {
		docs[i] = DocumentFromText(t, nil)
	}
	return WithDocs(docs...)
}

// WithDocs adds documents as context for generation or as input to an
// embedder. Repeating this option (or mixing it with [WithTextDocs]) appends.
func WithDocs(docs ...*Document) DocumentOption {
	return &documentOptions{Documents: docs}
}

// WithDocsFn sets the function that selects the context documents for a prompt,
// such as by querying a retriever. It applies only to [DefinePrompt], since
// only a prompt has input to give it.
//
// fn receives the prompt's input converted to In.
//
// Documents accumulate: repeating this option, or combining it with [WithDocs]
// or [WithTextDocs], adds to the set rather than replacing it. The fixed
// documents resolve first, then the computed ones, each in call order.
func WithDocsFn[In any](fn func(context.Context, In) ([]*Document, error)) PromptOption {
	return &promptOptions{documentOptions: documentOptions{DocsFn: coerceFn("WithDocsFn", fn)}}
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
// Repeating this option takes the last ID set.
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
	return WithEvaluator(NewEvaluatorRef(name, nil))
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

func (o *embedderOptions) applyEmbedder(embedOpts *embedderOptions) {
	o.applyConfig(&embedOpts.configOptions)
	o.applyDocument(&embedOpts.documentOptions)

	if o.Embedder != nil {
		embedOpts.Embedder = o.Embedder
	}
}

// WithEmbedder sets either a [Embedder] or a [EmbedderRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithEmbedder.
func WithEmbedder(embedder EmbedderArg) EmbedderOption {
	return &embedderOptions{Embedder: embedder}
}

// WithEmbedderName sets the embedder name to call for document embedding.
// The embedder name will be resolved to a [Embedder] and may error if the reference is invalid.
func WithEmbedderName(name string) EmbedderOption {
	return WithEmbedder(NewEmbedderRef(name, nil))
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
	applyRetriever(*retrieverOptions)
}

func (o *retrieverOptions) applyRetriever(retOpts *retrieverOptions) {
	o.applyConfig(&retOpts.configOptions)
	o.applyDocument(&retOpts.documentOptions)

	if o.Retriever != nil {
		retOpts.Retriever = o.Retriever
	}
}

// WithRetriever sets either a [Retriever] or a [RetrieverRef] that may contain a config.
// Passing [WithConfig] will take precedence over the config in WithRetriever.
func WithRetriever(retriever RetrieverArg) RetrieverOption {
	return &retrieverOptions{Retriever: retriever}
}

// WithRetrieverName sets the retriever name to call for document retrieval.
// The retriever name will be resolved to a [Retriever] and may error if the reference is invalid.
func WithRetrieverName(name string) RetrieverOption {
	return WithRetriever(NewRetrieverRef(name, nil))
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
	OutputSchema map[string]any // JSON schema of the tool's output.
	StrictSchema *bool
}

// ToolOption is an option for defining a tool.
type ToolOption interface {
	applyTool(*toolOptions)
}

func (o *toolOptions) applyTool(opts *toolOptions) {
	if o.OutputSchema != nil {
		opts.OutputSchema = o.OutputSchema
	}
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

// promptExecutionOptions are options for generating a model response by executing a prompt.
type promptExecutionOptions struct {
	commonGenOptions
	executionOptions
	documentOptions
	Input any // Input fields for the prompt. If not nil this should be a struct that matches the prompt's input schema.
}

// PromptExecuteOption is an option for executing a prompt. It applies only to [prompt.Execute].
type PromptExecuteOption interface {
	applyPromptExecute(*promptExecutionOptions)
}

func (o *promptExecutionOptions) applyPromptExecute(pgOpts *promptExecutionOptions) {
	o.commonGenOptions.applyPromptExecute(pgOpts)
	o.executionOptions.applyPromptExecute(pgOpts)
	o.documentOptions.applyPromptExecute(pgOpts)

	if o.Input != nil {
		pgOpts.Input = o.Input
	}
}

// WithInput sets the input for the prompt request. Input must conform to the
// prompt's input schema and can either be a map[string]any or a struct of the same api.
// Repeating this option takes the last input set. APIs that take the input as
// a typed argument ([DataPrompt.Execute], [DataPrompt.ExecuteStream]) apply
// that argument after these options, so the typed argument wins.
func WithInput(input any) PromptExecuteOption {
	return &promptExecutionOptions{Input: input}
}
