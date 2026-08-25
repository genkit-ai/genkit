// Copyright 2024 Google LLC
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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"log/slog"
	"maps"
	"os"
	"path"
	"reflect"
	"regexp"
	"slices"
	"strings"

	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// Prompt is the interface for a prompt that can be executed and rendered.
type Prompt interface {
	// Name returns the name of the prompt.
	Name() string
	// Execute executes the prompt with the given options and returns a [ModelResponse].
	//
	// # Options
	//
	// Input:
	//
	//   - [WithInput]: Supply the prompt's input, overriding the default from [WithInputType]
	//
	// Conversation:
	//
	//   - [WithMessages]: Supply the conversation this execution continues
	//   - [WithMessagesFn]: As above, computed from the input
	//
	// A prompt that declares a conversation of its own decides where these go,
	// with {{history}} or [HistoryFromContext]. One that declares none uses
	// them directly, between the system message and the user prompt.
	//
	// Overrides, each replacing what the prompt was defined with:
	//
	//   - [WithModel], [WithModelName]: Call a different model
	//   - [WithConfig]: Replace the generation config
	//   - [WithDocs], [WithTextDocs]: Replace the context documents, skipping any [WithDocsFn]
	//   - [WithTools], [WithToolChoice], [WithMaxTurns], [WithReturnToolRequests]: Change tool behavior
	//   - [WithMiddleware], [WithUse]: Add middleware for this execution
	//   - [WithStreaming]: Receive streamed chunks
	Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error)
	// ExecuteStream executes the prompt with streaming and returns an iterator.
	// It accepts the same options as Execute.
	ExecuteStream(ctx context.Context, opts ...PromptExecuteOption) iter.Seq2[*ModelStreamValue, error]
	// Render renders the prompt with the given input and returns a [GenerateActionOptions] to be used with [GenerateWithRequest].
	Render(ctx context.Context, input any) (*GenerateActionOptions, error)
}

// prompt is a prompt template that can be executed to generate a model response.
type prompt struct {
	core.Action[any, *GenerateActionOptions, struct{}]
	promptOptions
	registry api.Registry
}

// DataPrompt is a prompt with strongly-typed input and output.
// It wraps an underlying [Prompt] and provides type-safe Execute and Render methods.
// The Out type parameter can be string for text outputs or any struct type for JSON outputs.
type DataPrompt[In, Out any] struct {
	prompt
}

// DefinePrompt creates a new [Prompt] and registers it.
func DefinePrompt(r api.Registry, name string, opts ...PromptOption) Prompt {
	if name == "" {
		panic("ai.DefinePrompt: name is required")
	}

	pOpts := &promptOptions{}
	for _, opt := range opts {
		opt.applyPrompt(pOpts)
	}
	// Panic at definition rather than at render: this is a wiring mistake, and
	// this is the call that has to change.
	if pOpts.MessagesText != nil && pOpts.MessagesFn != nil {
		panic(fmt.Sprintf("ai.DefinePrompt: %q sets both WithMessagesTemplate and WithMessages/WithMessagesFn, which have no meaningful combination: the template is the whole conversation. Write the messages as {{role}} blocks in the template, or drop the template and build the conversation from WithMessages and WithMessagesFn.", name))
	}

	p := &prompt{
		registry:      r,
		promptOptions: *pOpts,
	}

	var modelName string
	if pOpts.Model != nil {
		modelName = pOpts.Model.Name()
	}

	if modelRef, ok := pOpts.Model.(ModelRef); ok && pOpts.Config == nil {
		if cfg := modelRef.Config(); !base.IsNil(cfg) {
			pOpts.Config = cfg
		}
	}

	var tools []string
	for _, value := range pOpts.commonGenOptions.Tools {
		tools = append(tools, value.Name())
	}

	metadata := p.Metadata
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["type"] = api.ActionTypeExecutablePrompt

	baseName, variant, _ := strings.Cut(name, ".")

	use, err := configsToRefs(pOpts.commonGenOptions.Use)
	if err != nil {
		panic(fmt.Errorf("ai.DefinePrompt: error processing middleware: %w", err))
	}

	promptMetadata := map[string]any{
		"name":         baseName,
		"description":  p.Description,
		"model":        modelName,
		"config":       p.Config,
		"input":        map[string]any{"schema": p.InputSchema},
		"output":       map[string]any{"schema": p.OutputSchema},
		"defaultInput": p.DefaultInput,
		"tools":        tools,
		"toolChoice":   pOpts.ToolChoice,
		"maxTurns":     p.MaxTurns,
	}
	if len(use) > 0 {
		promptMetadata["use"] = use
	}
	if variant != "" {
		promptMetadata["variant"] = variant
	}
	if m, ok := metadata["prompt"].(map[string]any); ok {
		maps.Copy(m, promptMetadata)
	} else {
		metadata["prompt"] = promptMetadata
	}

	a := core.NewActionOf(api.ActionTypeExecutablePrompt, name, &core.ActionOptions{Metadata: metadata, InputSchema: p.InputSchema}, p.buildRequest)
	a.Register(r)
	p.Action = *a

	return p
}

// LookupPrompt looks up a [Prompt] registered by [DefinePrompt].
// It returns nil if the prompt was not defined.
func LookupPrompt(r api.Registry, name string) Prompt {
	action := core.ResolveActionFor[any, *GenerateActionOptions, struct{}](r, api.ActionTypeExecutablePrompt, name)
	if action == nil {
		return nil
	}
	return &prompt{
		Action:   *action,
		registry: r,
	}
}

// Execute renders a prompt, does variable substitution and
// passes the rendered template to the AI model specified by the prompt.
func (p *prompt) Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error) {
	if p == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Prompt.Execute: prompt is nil")
	}

	execOpts := &promptExecutionOptions{}
	for _, opt := range opts {
		opt.applyPromptExecute(execOpts)
	}
	// Messages passed at execution time reach Render through a context scoped
	// to that call, so the prompt decides where they land. Generation below
	// runs on the original ctx: otherwise they would ride along into every
	// tool call and nested prompt in the generate loop.
	renderCtx := ctx
	if execOpts.MessagesFn != nil {
		history, err := execOpts.MessagesFn(ctx, execOpts.Input)
		if err != nil {
			return nil, err
		}
		renderCtx = withPromptHistory(renderCtx, history)
	}
	if len(execOpts.Documents) > 0 {
		// Documents supplied here replace the prompt's own, so tell Render not
		// to run a WithDocsFn whose result would be discarded.
		renderCtx = withPromptDocsOverride(renderCtx)
	}

	// Render() should populate all data from the prompt. Prompt fields should
	// *not* be referenced in this function as it may have been loaded from
	// the registry and is missing the options passed in at definition.
	actionOpts, err := p.Render(renderCtx, execOpts.Input)
	if err != nil {
		return nil, err
	}

	if modelRef, ok := execOpts.Model.(ModelRef); ok && execOpts.Config == nil {
		if cfg := modelRef.Config(); !base.IsNil(cfg) {
			execOpts.Config = cfg
		}
	}

	if execOpts.Config != nil {
		actionOpts.Config = execOpts.Config
	}

	if len(execOpts.Documents) > 0 {
		actionOpts.Docs = execOpts.Documents
	}

	if execOpts.ToolChoice != "" {
		actionOpts.ToolChoice = execOpts.ToolChoice
	}

	if execOpts.Model != nil {
		actionOpts.Model = execOpts.Model.Name()
	}

	if execOpts.MaxTurns != 0 {
		actionOpts.MaxTurns = execOpts.MaxTurns
	}

	if execOpts.ReturnToolRequests != nil {
		actionOpts.ReturnToolRequests = *execOpts.ReturnToolRequests
	}

	toolRefs := execOpts.Tools
	if len(toolRefs) == 0 {
		toolRefs = make([]ToolRef, 0, len(actionOpts.Tools))
		for _, toolName := range actionOpts.Tools {
			toolRefs = append(toolRefs, ToolName(toolName))
		}
	}

	toolNames, newTools, err := resolveUniqueTools(p.registry, toolRefs)
	if err != nil {
		return nil, err
	}
	actionOpts.Tools = toolNames

	r := p.registry
	if len(newTools) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, t := range newTools {
			t.Register(r)
		}
	}

	refs, err := configsToRefs(execOpts.Use)
	if err != nil {
		return nil, fmt.Errorf("Prompt.Execute: %w", err)
	}
	if len(refs) > 0 {
		actionOpts.Use = refs
	}

	return GenerateWithRequest(ctx, r, actionOpts, execOpts.Middleware, execOpts.Stream)
}

// ExecuteStream executes the prompt with streaming and returns an iterator.
//
// If the yield function is passed a non-nil error, execution has failed with that
// error; the yield function will not be called again.
//
// If the yield function's [ModelStreamValue] argument has Done == true, the value's
// Response field contains the final response; the yield function will not be called again.
//
// Otherwise the Chunk field of the passed [ModelStreamValue] holds a streamed chunk.
func (p *prompt) ExecuteStream(ctx context.Context, opts ...PromptExecuteOption) iter.Seq2[*ModelStreamValue, error] {
	return func(yield func(*ModelStreamValue, error) bool) {
		if p == nil {
			yield(nil, status.Errorf(status.ErrInvalidArgument, "Prompt.ExecuteStream: prompt is nil"))
			return
		}

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
		resp, err := p.Execute(ctx, allOpts...)
		if done || errors.Is(err, errStop) {
			return
		}
		if err != nil {
			yield(nil, err)
			return
		}

		yield(&ModelStreamValue{Done: true, Response: resp}, nil)
	}
}

// Render renders the prompt template based on user input.
func (p *prompt) Render(ctx context.Context, input any) (*GenerateActionOptions, error) {
	if p == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Prompt.Render: prompt is nil")
	}

	if len(p.Middleware) > 0 {
		logger.Warn(ctx, "middleware set on prompt is ignored during Prompt.Render, use Prompt.Execute to apply it", "prompt", p.Name())
	}

	// TODO: This is hacky; we should have a helper that fetches the metadata.
	if input == nil {
		input = p.Desc().Metadata["prompt"].(map[string]any)["defaultInput"]
	}

	return p.Run(ctx, input, nil)
}

// Desc returns a descriptor of the prompt with resolved schema references.
func (p *prompt) Desc() api.ActionDesc {
	desc := p.Action.Desc()
	descMeta := maps.Clone(desc.Metadata)
	if promptMeta, ok := descMeta["prompt"].(map[string]any); ok {
		promptMeta = maps.Clone(promptMeta)
		if inputMeta, ok := promptMeta["input"].(map[string]any); ok {
			inputMeta = maps.Clone(inputMeta)
			if inputSchema, ok := inputMeta["schema"].(map[string]any); ok {
				if resolved, err := core.ResolveSchema(p.registry, inputSchema); err == nil {
					inputMeta["schema"] = resolved
				}
			}
			promptMeta["input"] = inputMeta
		}
		if outputMeta, ok := promptMeta["output"].(map[string]any); ok {
			outputMeta = maps.Clone(outputMeta)
			if outputSchema, ok := outputMeta["schema"].(map[string]any); ok {
				if resolved, err := core.ResolveSchema(p.registry, outputSchema); err == nil {
					outputMeta["schema"] = resolved
				}
			}
			promptMeta["output"] = outputMeta
		}
		descMeta["prompt"] = promptMeta
	}
	desc.Metadata = descMeta
	return desc
}

// buildVariables returns a map holding prompt field values based
// on a struct or a pointer to a struct. The struct value should have
// JSON tags that correspond to the Prompt's input schema.
// Only exported fields of the struct will be used.
func buildVariables(variables any) (map[string]any, error) {
	if variables == nil {
		return nil, nil
	}

	v := reflect.Indirect(reflect.ValueOf(variables))
	if v.Kind() == reflect.Map {
		// ensure JSON tags are taken in consideration (allowing snake case fields)
		jsonData, err := json.Marshal(variables)
		if err != nil {
			return nil, fmt.Errorf("unable to marshal prompt field values: %w", err)
		}
		var resultVariables map[string]any
		if err := json.Unmarshal(jsonData, &resultVariables); err != nil {
			return nil, fmt.Errorf("unable to unmarshal prompt field values: %w", err)
		}
		return resultVariables, nil
	}
	if v.Kind() != reflect.Struct {
		return nil, status.Errorf(status.ErrInvalidArgument, "prompt input must be a struct, a pointer to one, or a map")
	}
	vt := v.Type()

	// TODO: Verify the struct with p.Config.InputSchema.

	m := make(map[string]any)

fieldLoop:
	for i := range vt.NumField() {
		ft := vt.Field(i)
		if ft.PkgPath != "" {
			continue
		}

		jsonTag := ft.Tag.Get("json")
		jsonName, rest, _ := strings.Cut(jsonTag, ",")
		if jsonName == "" {
			jsonName = ft.Name
		}

		vf := v.Field(i)

		// If the field is the zero value, and omitempty is set,
		// don't pass it as a prompt input variable.
		if vf.IsZero() {
			for rest != "" {
				var key string
				key, rest, _ = strings.Cut(rest, ",")
				if key == "omitempty" {
					continue fieldLoop
				}
			}
		}

		m[jsonName] = vf.Interface()
	}

	return m, nil
}

// buildRequest prepares a [GenerateActionOptions] based on the prompt,
// using the input variables and other information in the [prompt].
func (p *prompt) buildRequest(ctx context.Context, input any) (*GenerateActionOptions, error) {
	// Only the text options need template variables; content functions receive
	// the raw input.
	var m map[string]any
	var err error
	if p.SystemText != nil || p.PromptText != nil || p.MessagesText != nil {
		m, err = buildVariables(input)
		if err != nil {
			return nil, err
		}
	}

	dp := p.registry.Dotprompt()

	messages := []*Message{}
	messages, err = renderSystemPrompt(ctx, p.promptOptions, messages, m, input, dp)
	if err != nil {
		return nil, err
	}
	messages, err = renderMessages(ctx, p.promptOptions, messages, m, input, dp)
	if err != nil {
		return nil, err
	}
	messages, err = renderUserPrompt(ctx, p.promptOptions, messages, m, input, dp)
	if err != nil {
		return nil, err
	}

	var tools []string
	for _, t := range p.Tools {
		tools = append(tools, t.Name())
	}

	config := p.Config
	if modelRef, ok := p.Model.(ModelRef); ok && config == nil {
		config = modelRef.Config()
	}

	var modelName string
	if p.Model != nil {
		modelName = p.Model.Name()
	}

	outputSchema, err := core.ResolveSchema(p.registry, p.OutputSchema)
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "invalid output schema for prompt %q: %w", p.Name(), err)
	}

	useRefs, err := configsToRefs(p.Use)
	if err != nil {
		return nil, fmt.Errorf("prompt %q: %w", p.Name(), err)
	}

	docs := p.Documents
	// Skipped when the execution supplies its own documents, which replace
	// these: running a retrieval query for a discarded result is waste.
	if p.DocsFn != nil && !promptDocsOverridden(ctx) {
		computed, err := p.DocsFn(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("prompt %q: resolving docs: %w", p.Name(), err)
		}
		// Concat rather than append: p.Documents is reused by every execution,
		// so appending would write into the spare capacity behind it.
		docs = slices.Concat(docs, computed)
	}

	return &GenerateActionOptions{
		Model:              modelName,
		Config:             config,
		ToolChoice:         p.ToolChoice,
		MaxTurns:           p.MaxTurns,
		ReturnToolRequests: p.ReturnToolRequests != nil && *p.ReturnToolRequests,
		Messages:           messages,
		Docs:               docs,
		Tools:              tools,
		Use:                useRefs,
		Output: &GenerateActionOutputConfig{
			Format:       p.OutputFormat,
			JsonSchema:   outputSchema,
			Instructions: p.OutputInstructions,
			Constrained:  !p.CustomConstrained,
		},
	}, nil
}

// renderSystemPrompt renders a system prompt message. Text from [WithSystem]
// is compiled against the input; content from a function is used verbatim,
// since compiling it would reinterpret computed text as a template.
func renderSystemPrompt(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.SystemFn != nil {
		parts, err := opts.SystemFn(ctx, raw)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 {
			return messages, nil
		}
		return append(messages, &Message{Role: RoleSystem, Content: parts}), nil
	}

	if opts.SystemText == nil {
		return messages, nil
	}

	rendered, err := renderSingleMessage(ctx, opts, *opts.SystemText, input, dp, "WithSystem", RoleSystem)
	if err != nil {
		return nil, err
	}

	return append(messages, rendered), nil
}

// roleMarkerPattern matches a {{role ...}} helper call in template source.
// Checking the source rather than the rendered messages is what makes the check
// exact: dotprompt starts every render as a user message, so a rendered user
// role is indistinguishable from an explicit {{role "user"}}.
var roleMarkerPattern = regexp.MustCompile(`\{\{~?\s*role\s`)

// renderSingleMessage renders template text that fills exactly one message of
// role want, naming option in the error when the template asks for anything
// else. The role is the slot's, not the template's: the rest of the
// conversation is positioned against these two messages.
func renderSingleMessage(ctx context.Context, opts promptOptions, text string, input map[string]any, dp *dotprompt.Dotprompt, option string, want Role) (*Message, error) {
	if roleMarkerPattern.MatchString(text) {
		return nil, status.Errorf(status.ErrInvalidArgument,
			"%s contains a {{role}} marker: it fills a single %s message whose role is fixed, so use WithMessagesTemplate to write turns",
			option, want)
	}

	rendered, err := renderPrompt(ctx, opts, text, input, nil, dp)
	if err != nil {
		return nil, err
	}
	if len(rendered) != 1 {
		return nil, status.Errorf(status.ErrInvalidArgument,
			"%s produced %d messages, want 1: its template fills a single message, so use WithMessagesTemplate for a multi-turn template",
			option, len(rendered))
	}
	if got := rendered[0].Role; got != want && got != RoleUser {
		// Belt and braces for a marker the pattern did not catch: RoleUser is
		// dotprompt's default and so carries no intent.
		return nil, status.Errorf(status.ErrInvalidArgument,
			"%s produced a %s message, want %s: use WithMessagesTemplate to write turns with other roles",
			option, got, want)
	}

	rendered[0].Role = want
	return rendered[0], nil
}

// renderUserPrompt renders a user prompt message, following the same rules as
// [renderSystemPrompt].
func renderUserPrompt(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.PromptFn != nil {
		parts, err := opts.PromptFn(ctx, raw)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 {
			return messages, nil
		}
		return append(messages, &Message{Role: RoleUser, Content: parts}), nil
	}

	if opts.PromptText == nil {
		return messages, nil
	}

	rendered, err := renderSingleMessage(ctx, opts, *opts.PromptText, input, dp, "WithPrompt", RoleUser)
	if err != nil {
		return nil, err
	}

	return append(messages, rendered), nil
}

// renderMessages appends the messages that sit between the system and user
// prompts.
//
// A prompt that declares its own conversation owns the messages supplied at
// execution time, because only it knows where its few-shot examples end and a
// real conversation begins, so it must place them itself. A prompt that
// declares none gets them here, in the middle.
//
// Only [WithMessagesTemplate] text is compiled. Messages are used verbatim, so
// history containing literal braces passes through untouched.
func renderMessages(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	history := HistoryFromContext(ctx)

	if opts.MessagesFn == nil && opts.MessagesText == nil {
		return appendMessageClones(messages, history), nil
	}

	// The template and the verbatim messages are mutually exclusive, so at most
	// one of these runs. Only the template places the caller's history, at
	// {{history}}; a function reaches it through HistoryFromContext instead.
	if opts.MessagesText != nil {
		rendered, err := renderPrompt(ctx, opts, *opts.MessagesText, input, history, dp)
		if err != nil {
			return nil, err
		}
		// The renderer built these fresh, and cloned the history it spliced in.
		return append(messages, rendered...), nil
	}
	msgs, err := opts.MessagesFn(ctx, raw)
	if err != nil {
		return nil, err
	}
	messages = appendMessageClones(messages, compactMessages(msgs))

	return messages, nil
}

// compactMessages returns src without its nil entries, or src itself when it
// has none. A conversation arrives from a caller-owned slice or a user-written
// function, and a nil in one has no representation downstream: the template
// renderer dereferences it, and every other path carries it as far as the
// action's output schema, which rejects it as a null message.
//
// Drop them where the conversation enters, never later. [renderPrompt] hands
// dotprompt one placeholder per message and restores the originals by position,
// so a filter applied to only one of those two lists shifts the rest.
func compactMessages(src []*Message) []*Message {
	if !slices.Contains(src, nil) {
		return src
	}
	out := make([]*Message, 0, len(src))
	for _, m := range src {
		if m != nil {
			out = append(out, m)
		}
	}
	return out
}

// appendMessageClones appends a clone of each message in src to dst.
//
// Everything spliced into a request is cloned, because later stages mutate
// messages in place, as middleware does when it stamps metadata. The originals
// belong to someone else: [WithMessages] messages are stored on the prompt and
// reused by every execution, and history typically aliases a session or the
// caller's own slice. [Message.Clone] copies the Content slice and Metadata map
// too, since appending to a shared array or writing a shared map races even
// when the [Message] itself is fresh.
func appendMessageClones(dst []*Message, src []*Message) []*Message {
	for _, msg := range src {
		dst = append(dst, msg.Clone())
	}
	return dst
}

// renderPrompt renders a prompt template using dotprompt functionalities.
//
// history is the conversation the template may place, at {{history}} or, absent
// that, before the template's final user message. Only the conversation slot
// passes it; the single-message slots have nowhere to put it.
func renderPrompt(ctx context.Context, opts promptOptions, templateText string, input map[string]any, history []*Message, dp *dotprompt.Dotprompt) ([]*Message, error) {
	renderedFunc, err := dp.Compile(templateText, &dotprompt.PromptMetadata{})
	if err != nil {
		return nil, err
	}

	return renderDotpromptToMessages(ctx, renderedFunc, input, history, &dotprompt.PromptMetadata{
		Input: dotprompt.PromptMetadataInput{
			Default: opts.DefaultInput,
		},
	})
}

// renderDotpromptToMessages executes a dotprompt prompt function and converts the result to a slice of messages
func renderDotpromptToMessages(ctx context.Context, promptFn dotprompt.PromptFunction, input map[string]any, history []*Message, additionalMetadata *dotprompt.PromptMetadata) ([]*Message, error) {
	// Prepare the context for rendering
	templateContext := map[string]any{}
	actionCtx := core.FromContext(ctx)
	maps.Copy(templateContext, actionCtx)

	// Inject session state if available (accessible via {{@state.field}} in templates)
	if state := base.PromptStateFromContext(ctx); state != nil {
		templateContext["state"] = state
	}

	// Call the prompt function with the input and context
	rendered, err := promptFn(&dotprompt.DataArgument{
		Input:    input,
		Messages: toDotpromptMessages(history),
		Context:  templateContext,
	}, additionalMetadata)
	if err != nil {
		return nil, fmt.Errorf("failed to render prompt: %w", err)
	}

	convertedMessages := []*Message{}
	historyIdx := 0
	for _, message := range rendered.Messages {
		// Restore the original rather than converting the stand-in back: a
		// message carries part kinds dotprompt cannot represent, such as tool
		// requests and resources, which a round trip would drop.
		if message.Metadata["purpose"] == "history" && historyIdx < len(history) {
			convertedMessages = append(convertedMessages, history[historyIdx].Clone())
			historyIdx++
			continue
		}
		parts, err := convertToPartPointers(message.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to convert parts: %w", err)
		}
		role := Role(message.Role)
		convertedMessages = append(convertedMessages, &Message{
			Role:    role,
			Content: parts,
		})
	}

	return convertedMessages, nil
}

// toDotpromptMessages converts history into the messages dotprompt places for
// {{history}}.
//
// Only the roles have to be faithful: the content is a stand-in, since
// renderDotpromptToMessages restores each original once dotprompt has decided
// where it goes. They are marked as history on the way in because dotprompt
// marks them itself only at {{history}}, not when it inserts them before the
// final user message, and the restore has to recognize them either way.
func toDotpromptMessages(history []*Message) []dotprompt.Message {
	if len(history) == 0 {
		return nil
	}
	msgs := make([]dotprompt.Message, 0, len(history))
	for _, m := range history {
		content := make([]dotprompt.Part, 0, len(m.Content))
		for _, p := range m.Content {
			text := ""
			if p.IsText() {
				text = p.Text
			}
			content = append(content, &dotprompt.TextPart{Text: text})
		}
		msgs = append(msgs, dotprompt.Message{
			Role:        dotprompt.Role(m.Role),
			Content:     content,
			HasMetadata: dotprompt.HasMetadata{Metadata: map[string]any{"purpose": "history"}},
		})
	}
	return msgs
}

// convertToPartPointers converts []dotprompt.Part to []*Part
func convertToPartPointers(parts []dotprompt.Part) ([]*Part, error) {
	result := make([]*Part, len(parts))
	for i, part := range parts {
		switch p := part.(type) {
		case *dotprompt.TextPart:
			if p.Text != "" {
				result[i] = NewTextPart(p.Text)
			}
		case *dotprompt.MediaPart:
			ct, data, err := contentType(p.Media.ContentType, p.Media.URL)
			if err != nil {
				return nil, err
			}
			result[i] = NewMediaPart(ct, string(data))
		}
	}
	return result, nil
}

// LoadPromptDirFromFS loads prompts and partials from a filesystem for the given namespace.
// The fsys parameter should be an fs.FS implementation (e.g., embed.FS or os.DirFS).
// The dir parameter specifies the directory within the filesystem where prompts are located.
func LoadPromptDirFromFS(r api.Registry, fsys fs.FS, dir, namespace string) {
	if fsys == nil {
		panic("ai.LoadPrompt: no prompt filesystem provided")
	}

	if _, err := fs.Stat(fsys, dir); err != nil {
		panic(fmt.Errorf("failed to access prompt directory %q in filesystem: %w", dir, err))
	}

	entries, err := fs.ReadDir(fsys, dir)
	if err != nil {
		panic(fmt.Errorf("failed to read prompt directory structure: %w", err))
	}

	for _, entry := range entries {
		filename := entry.Name()
		filePath := path.Join(dir, filename)
		if entry.IsDir() {
			LoadPromptDirFromFS(r, fsys, filePath, namespace)
		} else if strings.HasSuffix(filename, ".prompt") {
			if strings.HasPrefix(filename, "_") {
				partialName := strings.TrimSuffix(filename[1:], ".prompt")
				source, err := fs.ReadFile(fsys, filePath)
				if err != nil {
					slog.Error("failed to read prompt partial file, skipping it", "file", filePath, "error", err)
					continue
				}
				r.RegisterPartial(partialName, string(source))
				slog.Debug("registered dotprompt partial", "partial", partialName, "file", filePath)
			} else {
				LoadPromptFromFS(r, fsys, dir, filename, namespace)
			}
		}
	}
}

// LoadPromptFromFS loads a single prompt from a filesystem into the registry.
// The fsys parameter should be an fs.FS implementation (e.g., embed.FS or os.DirFS).
// The dir parameter specifies the directory within the filesystem where the prompt is located.
func LoadPromptFromFS(r api.Registry, fsys fs.FS, dir, filename, namespace string) Prompt {
	name := strings.TrimSuffix(filename, ".prompt")

	sourceFile := path.Join(dir, filename)
	source, err := fs.ReadFile(fsys, sourceFile)
	if err != nil {
		slog.Error("failed to read prompt file, skipping it", "file", sourceFile, "error", err)
		return nil
	}

	p, err := LoadPromptFromSource(r, string(source), name, namespace)
	if err != nil {
		slog.Error("failed to load prompt file, skipping it", "file", sourceFile, "error", err)
		return nil
	}

	slog.Debug("registered dotprompt", "prompt", p.Name(), "file", sourceFile)
	return p
}

// LoadPromptFromSource loads a prompt from raw .prompt file content.
// The source parameter should contain the complete .prompt file text (frontmatter + template).
// The name parameter is the prompt name (may include variant suffix like "myPrompt.variant").
func LoadPromptFromSource(r api.Registry, source, name, namespace string) (Prompt, error) {
	name, variant, _ := strings.Cut(name, ".")

	dp := r.Dotprompt()

	parsedPrompt, err := dp.Parse(source)
	if err != nil {
		return nil, fmt.Errorf("failed to parse dotprompt: %w", err)
	}

	metadata, err := dp.RenderMetadata(source, &parsedPrompt.PromptMetadata)
	if err != nil {
		return nil, fmt.Errorf("failed to render dotprompt metadata: %w", err)
	}

	toolRefs := make([]ToolRef, len(metadata.Tools))
	for i, tool := range metadata.Tools {
		toolRefs[i] = ToolName(tool)
	}

	promptOptMetadata := metadata.Metadata
	if promptOptMetadata == nil {
		promptOptMetadata = make(map[string]any)
	}

	var promptMetadata map[string]any
	if m, ok := promptOptMetadata["prompt"].(map[string]any); ok {
		promptMetadata = m
	} else {
		promptMetadata = make(map[string]any)
	}
	promptMetadata["template"] = parsedPrompt.Template
	if variant != "" {
		promptMetadata["variant"] = variant
	}
	promptOptMetadata["prompt"] = promptMetadata
	promptOptMetadata["type"] = api.ActionTypeExecutablePrompt

	opts := &promptOptions{
		commonGenOptions: commonGenOptions{
			configOptions: configOptions{
				Config: (map[string]any)(metadata.Config),
			},
			Model: NewModelRef(metadata.Model, nil),
			Tools: toolRefs,
		},
		inputOptions: inputOptions{
			DefaultInput: metadata.Input.Default,
		},
		Metadata:    promptOptMetadata,
		Description: metadata.Description,
	}

	if toolChoice, ok := metadata.Raw["toolChoice"].(ToolChoice); ok {
		opts.ToolChoice = toolChoice
	}

	if maxTurns, ok := metadata.Raw["maxTurns"].(uint64); ok {
		opts.MaxTurns = int(maxTurns)
	}

	if returnToolRequests, ok := metadata.Raw["returnToolRequests"].(bool); ok {
		opts.ReturnToolRequests = &returnToolRequests
	}

	if uses, err := parseDotpromptUse(metadata.Raw["use"]); err != nil {
		return nil, fmt.Errorf("prompt %q: %w", name, err)
	} else if len(uses) > 0 {
		opts.Use = uses
	}

	if inputSchema, ok := metadata.Input.Schema.(*jsonschema.Schema); ok {
		if inputSchema.Ref != "" {
			opts.InputSchema = core.SchemaRef(inputSchema.Ref)
		} else {
			opts.InputSchema = base.SchemaAsMap(inputSchema)
		}
	}

	if inputSchema, ok := metadata.Input.Schema.(map[string]any); ok {
		if ref, ok := inputSchema["$ref"].(string); ok {
			opts.InputSchema = core.SchemaRef(ref)
		} else {
			opts.InputSchema = inputSchema
		}
	}

	if metadata.Output.Format != "" {
		opts.OutputFormat = metadata.Output.Format
	}

	if outputSchema, ok := metadata.Output.Schema.(*jsonschema.Schema); ok {
		if outputSchema.Ref != "" {
			opts.OutputSchema = core.SchemaRef(outputSchema.Ref)
		} else {
			opts.OutputSchema = base.SchemaAsMap(outputSchema)
		}
		if opts.OutputFormat == "" {
			opts.OutputFormat = OutputFormatJSON
		}
	}

	if outputSchema, ok := metadata.Output.Schema.(map[string]any); ok {
		if ref, ok := outputSchema["$ref"].(string); ok {
			opts.OutputSchema = core.SchemaRef(ref)
		} else {
			opts.OutputSchema = outputSchema
		}
		if opts.OutputFormat == "" {
			opts.OutputFormat = OutputFormatJSON
		}
	}

	key := promptKey(name, variant, namespace)

	prompt := DefinePrompt(r, key, opts, WithMessagesTemplate(parsedPrompt.Template))

	return prompt, nil
}

// parseDotpromptUse converts the value of the dotprompt `use:` frontmatter
// field into a slice of lazy [Middleware] references. Each entry may be a
// bare string (interpreted as a registered middleware name) or a map with
// `name` and optional `config`, the two shapes the frontmatter accepts.
// Returns nil if the input is nil or an empty slice.
func parseDotpromptUse(raw any) ([]Middleware, error) {
	if raw == nil {
		return nil, nil
	}
	entries, ok := raw.([]any)
	if !ok {
		return nil, status.Errorf(status.ErrInvalidArgument, "`use` must be a list, got %T", raw)
	}
	uses := make([]Middleware, 0, len(entries))
	for i, entry := range entries {
		switch v := entry.(type) {
		case string:
			if v == "" {
				return nil, status.Errorf(status.ErrInvalidArgument, "`use[%d]` is an empty string", i)
			}
			uses = append(uses, middlewareRefArg{name: v})
		case map[string]any:
			name, _ := v["name"].(string)
			if name == "" {
				return nil, status.Errorf(status.ErrInvalidArgument, "`use[%d]` is missing required `name` field", i)
			}
			uses = append(uses, middlewareRefArg{name: name, config: v["config"]})
		default:
			return nil, status.Errorf(status.ErrInvalidArgument, "`use[%d]` must be a string or map, got %T", i, entry)
		}
	}
	return uses, nil
}

// LoadPromptDir loads prompts and partials from a directory on the local filesystem.
func LoadPromptDir(r api.Registry, dir string, namespace string) {
	LoadPromptDirFromFS(r, os.DirFS(dir), ".", namespace)
}

// LoadPrompt loads a single prompt from a directory on the local filesystem into the registry.
func LoadPrompt(r api.Registry, dir, filename, namespace string) Prompt {
	return LoadPromptFromFS(r, os.DirFS(dir), ".", filename, namespace)
}

// promptKey generates a unique key for the prompt in the registry.
func promptKey(name string, variant string, namespace string) string {
	if namespace != "" {
		return fmt.Sprintf("%s/%s%s", namespace, name, variantKey(variant))
	}
	return fmt.Sprintf("%s%s", name, variantKey(variant))
}

// variantKey formats the variant part of the key.
func variantKey(variant string) string {
	if variant != "" {
		return fmt.Sprintf(".%s", variant)
	}
	return ""
}

// contentType determines the MIME content type of the given data URI
func contentType(ct, uri string) (string, []byte, error) {
	if uri == "" {
		return "", nil, status.Errorf(ErrInvalidPart, "found empty URI in part")
	}

	if strings.HasPrefix(uri, "gs://") || strings.HasPrefix(uri, "http") {
		// The content type may be unknown at render time for URL-based media.
		// For http(s) URLs the download middleware fetches the resource and
		// fills in the content type; for gs:// and other natively-supported
		// URLs (e.g. YouTube) the model resolves it. Defer content-type
		// validation to the model/plugin layer instead of failing to render
		// the prompt.
		return ct, []byte(uri), nil
	}
	if contents, isData := strings.CutPrefix(uri, "data:"); isData {
		prefix, _, found := strings.Cut(contents, ",")
		if !found {
			return "", nil, status.Errorf(ErrInvalidPart, "failed to parse data URI: missing comma")
		}

		if p, isBase64 := strings.CutSuffix(prefix, ";base64"); isBase64 {
			if ct == "" {
				ct = p
			}
			return ct, []byte(uri), nil
		}
	}

	return "", nil, status.Errorf(ErrInvalidPart, "uri content type not found")
}

// DefineDataPrompt creates a new data prompt and registers it.
// It automatically infers input schema from the In type parameter and configures
// output schema and JSON format from the Out type parameter (unless Out is string).
func DefineDataPrompt[In, Out any](r api.Registry, name string, opts ...PromptOption) *DataPrompt[In, Out] {
	if name == "" {
		panic("ai.DefineDataPrompt: name is required")
	}

	var in In
	allOpts := []PromptOption{WithInputType(in)}

	var out Out
	switch any(out).(type) {
	case string:
		// String output - no schema needed
	default:
		// Prepend WithOutputType so the user can override the output format.
		allOpts = append(allOpts, WithOutputType(out))
	}

	allOpts = append(allOpts, opts...)
	p := DefinePrompt(r, name, allOpts...)

	return &DataPrompt[In, Out]{prompt: *p.(*prompt)}
}

// LookupDataPrompt looks up a prompt by name and wraps it with type information.
// This is useful for wrapping prompts loaded from .prompt files with strong types.
// It returns nil if the prompt was not found.
func LookupDataPrompt[In, Out any](r api.Registry, name string) *DataPrompt[In, Out] {
	return AsDataPrompt[In, Out](LookupPrompt(r, name))
}

// AsDataPrompt wraps an existing Prompt with type information, returning a DataPrompt.
// This is useful for adding strong typing to a dynamically obtained prompt.
func AsDataPrompt[In, Out any](p Prompt) *DataPrompt[In, Out] {
	if p == nil {
		return nil
	}

	return &DataPrompt[In, Out]{prompt: *p.(*prompt)}
}

// Execute executes the typed prompt and returns the strongly-typed output along with the full model response.
// For structured output types (non-string Out), the prompt must be configured with the appropriate
// output schema, either through [DefineDataPrompt] or by using [WithOutputType] when defining the prompt.
// The typed input argument fills the input slot last, so it wins over any
// [WithInput] passed in opts.
func (dp *DataPrompt[In, Out]) Execute(ctx context.Context, input In, opts ...PromptExecuteOption) (Out, *ModelResponse, error) {
	if dp == nil {
		return base.Zero[Out](), nil, status.Errorf(status.ErrInvalidArgument, "DataPrompt.Execute: prompt is nil")
	}

	allOpts := append(slices.Clone(opts), WithInput(input))
	resp, err := dp.prompt.Execute(ctx, allOpts...)
	if err != nil {
		return base.Zero[Out](), nil, err
	}

	output, err := extractTypedOutput[Out](resp)
	if err != nil {
		return base.Zero[Out](), resp, err
	}

	return output, resp, nil
}

// ExecuteStream executes the typed prompt with streaming and returns an iterator.
//
// If the yield function is passed a non-nil error, execution has failed with that
// error; the yield function will not be called again.
//
// If the yield function's StreamValue argument has Done == true, the value's
// Output and Response fields contain the final typed output and response; the yield function
// will not be called again.
//
// Otherwise the Chunk field of the passed StreamValue holds a streamed chunk.
//
// For structured output types (non-string Out), the prompt must be configured with the appropriate
// output schema, either through [DefineDataPrompt] or by using [WithOutputType] when defining the prompt.
// The typed input argument fills the input slot last, so it wins over any
// [WithInput] passed in opts.
func (dp *DataPrompt[In, Out]) ExecuteStream(ctx context.Context, input In, opts ...PromptExecuteOption) iter.Seq2[*StreamValue[Out, Out], error] {
	return func(yield func(*StreamValue[Out, Out], error) bool) {
		if dp == nil {
			yield(nil, status.Errorf(status.ErrInvalidArgument, "DataPrompt.ExecuteStream: prompt is nil"))
			return
		}

		done := false
		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if done {
				return errStop
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			streamValue, err := extractTypedOutput[Out](chunk)
			if err != nil {
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

		// The typed input is applied last so it wins the input slot; the
		// iterator callback is chained so a caller-supplied WithStreaming
		// still receives every chunk.
		allOpts := append(slices.Clone(opts), WithInput(input), withChainedStreaming(cb))
		resp, err := dp.prompt.Execute(ctx, allOpts...)
		if done || errors.Is(err, errStop) {
			return
		}
		if err != nil {
			yield(nil, err)
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

// Render renders the typed prompt template with the given input.
func (dp *DataPrompt[In, Out]) Render(ctx context.Context, input In) (*GenerateActionOptions, error) {
	if dp == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "DataPrompt.Render: prompt is nil")
	}

	return dp.prompt.Render(ctx, input)
}
