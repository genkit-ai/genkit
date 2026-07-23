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
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
)

// Prompt is an executable prompt with typed input and output.
//
// In is the input type of the prompt, Out is the type Execute returns (string
// for text prompts), and Stream is the type of streamed chunks. Users never
// write Stream themselves; the constructor determines it. Use [DefinePrompt]
// for text output (Stream is [*ModelResponseChunk]) or [DefineDataPrompt] for
// structured output (Stream is Out). The [TextPrompt] and [DataPrompt]
// aliases name those two shapes.
type Prompt[In, Out, Stream any] struct {
	action[any, *GenerateActionOptions, struct{}]
	promptOptions
	registry api.Registry
}

// TextPrompt is a prompt that produces text output. Execute returns the
// response text and ExecuteStream yields raw [*ModelResponseChunk] values.
// It is the type returned by [DefinePrompt] and [LookupPrompt].
type TextPrompt[In any] = Prompt[In, string, *ModelResponseChunk]

// DataPrompt is a prompt that produces structured output of type Out.
// Execute returns the parsed output and ExecuteStream yields cumulative
// parsed Out values. It is the type returned by [DefineDataPrompt] and
// [LookupDataPrompt].
type DataPrompt[In, Out any] = Prompt[In, Out, Out]

// DefinePrompt creates a prompt with typed input and text output and
// registers it. The input schema is inferred from In unless an input schema
// option is provided (or In is an interface type like any, in which case the
// input is dynamically typed).
//
// The naming mirrors [Generate]: DefinePrompt produces text, [DefineDataPrompt]
// produces structured output.
func DefinePrompt[In any](r api.Registry, name string, opts ...PromptOption) *TextPrompt[In] {
	return definePrompt[In, string, *ModelResponseChunk](r, name, opts)
}

// DefineDataPrompt creates a prompt with typed input and structured output
// and registers it. The input schema is inferred from In and the output
// schema and JSON format from Out, unless explicit schema options are
// provided (or the type parameter is an interface type like any).
//
// The naming mirrors [GenerateData]: DefineDataPrompt produces structured
// output, [DefinePrompt] produces text.
func DefineDataPrompt[In, Out any](r api.Registry, name string, opts ...PromptOption) *DataPrompt[In, Out] {
	return definePrompt[In, Out, Out](r, name, opts)
}

// definePrompt is the shared implementation of [DefinePrompt] and
// [DefineDataPrompt].
func definePrompt[In, Out, Stream any](r api.Registry, name string, opts []PromptOption) *Prompt[In, Out, Stream] {
	if name == "" {
		panic("ai.DefinePrompt: name is required")
	}

	pOpts := &promptOptions{}
	for _, opt := range opts {
		if err := opt.applyPrompt(pOpts); err != nil {
			panic(fmt.Errorf("ai.DefinePrompt: error applying options: %w", err))
		}
	}

	if pOpts.InputSchema == nil {
		if t := reflect.TypeFor[In](); t.Kind() != reflect.Interface {
			pOpts.InputSchema = core.InferSchemaMap(base.Zero[In]())
		}
	}

	if pOpts.OutputSchema == nil {
		if t := reflect.TypeFor[Out](); t.Kind() != reflect.Interface && t != reflect.TypeFor[string]() {
			pOpts.OutputSchema = core.InferSchemaMap(base.Zero[Out]())
			if pOpts.OutputFormat == "" {
				pOpts.OutputFormat = OutputFormatJSON
			}
		}
	}

	p := &Prompt[In, Out, Stream]{
		registry:      r,
		promptOptions: *pOpts,
	}

	var modelName string
	if pOpts.Model != nil {
		modelName = pOpts.Model.Name()
	}

	if modelRef, ok := pOpts.Model.(ModelRef); ok && pOpts.Config == nil {
		pOpts.Config = modelRef.Config()
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

	p.action = *core.DefineAction(r, name, api.ActionTypeExecutablePrompt, &core.ActionOptions{
		Metadata:    metadata,
		InputSchema: p.InputSchema,
	}, p.buildRequest)

	return p
}

// LookupPrompt looks up a prompt registered by [DefinePrompt],
// [DefineDataPrompt], or loaded from a .prompt file. The returned prompt is
// dynamically typed (its input is any and its output is text); use
// [LookupDataPrompt] to attach static types instead.
// It returns nil if the prompt was not defined.
func LookupPrompt(r api.Registry, name string) *TextPrompt[any] {
	action := core.ResolveActionFor[any, *GenerateActionOptions, struct{}](r, api.ActionTypeExecutablePrompt, name)
	if action == nil {
		return nil
	}
	return &TextPrompt[any]{
		action:   *action,
		registry: r,
	}
}

// LookupDataPrompt looks up a prompt by name and attaches static input and
// output types to it. This is useful for accessing prompts loaded from
// .prompt files with strong types. The types are not verified against the
// prompt's declared schemas; input is validated at execution time.
// It returns nil if the prompt was not defined.
func LookupDataPrompt[In, Out any](r api.Registry, name string) *DataPrompt[In, Out] {
	return (*DataPrompt[In, Out])(LookupPrompt(r, name))
}

// Execute renders the prompt with the given input, does variable
// substitution, and passes the rendered template to the AI model specified by
// the prompt. It returns the typed output (for a [TextPrompt], the response
// text) along with the full [ModelResponse].
//
// For structured output, if the response contains no text to parse (e.g., it
// contains tool requests or interrupts instead), the output is the zero value
// and no error is returned; check resp.Interrupts() or resp.ToolRequests()
// to handle these cases.
func (p *Prompt[In, Out, Stream]) Execute(ctx context.Context, input In, opts ...PromptExecuteOption) (Out, *ModelResponse, error) {
	resp, err := p.execute(ctx, input, opts)
	if err != nil {
		return base.Zero[Out](), nil, err
	}

	output, err := outputFromResponse[Out](resp)
	if err != nil {
		return base.Zero[Out](), resp, err
	}

	return output, resp, nil
}

// outputFromResponse extracts the typed output from a final response,
// returning the zero value without error when there is no text to parse
// (e.g., the response holds tool requests or interrupts).
func outputFromResponse[Out any](resp *ModelResponse) (Out, error) {
	if _, isString := any(base.Zero[Out]()).(string); !isString && resp.Text() == "" {
		return base.Zero[Out](), nil
	}
	return extractTypedOutput[Out](resp)
}

// execute renders the prompt and runs the generate request, returning the raw
// model response.
func (p *Prompt[In, Out, Stream]) execute(ctx context.Context, input In, opts []PromptExecuteOption) (*ModelResponse, error) {
	if p == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.Execute: prompt is nil")
	}

	// With a dynamically typed input (In = any), an option mistakenly passed
	// in the input position would compile; catch it here instead of rendering
	// garbage.
	if _, ok := any(input).(PromptExecuteOption); ok {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.Execute: an option (%T) was passed as the prompt input; input is the argument before any options (pass nil if the prompt takes no input)", input)
	}

	execOpts := &promptExecutionOptions{}
	for _, opt := range opts {
		if err := opt.applyPromptExecute(execOpts); err != nil {
			return nil, fmt.Errorf("Prompt.Execute: error applying options: %w", err)
		}
	}
	// Render() should populate all data from the prompt. Prompt fields should
	// *not* be referenced in this function as it may have been loaded from
	// the registry and is missing the options passed in at definition.
	actionOpts, err := p.Render(ctx, input)
	if err != nil {
		return nil, err
	}

	if modelRef, ok := execOpts.Model.(ModelRef); ok && execOpts.Config == nil {
		execOpts.Config = modelRef.Config()
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

	if execOpts.MessagesFn != nil {
		m, err := buildVariables(input)
		if err != nil {
			return nil, err
		}

		tempOpts := promptOptions{
			commonGenOptions: commonGenOptions{
				MessagesFn: execOpts.MessagesFn,
			},
		}

		execMsgs, err := renderMessages(ctx, tempOpts, []*Message{}, m, input, p.registry.Dotprompt())
		if err != nil {
			return nil, err
		}

		var systemMsgs []*Message
		var msgs []*Message
		foundNonSystem := false

		for _, msg := range actionOpts.Messages {
			if msg.Role == RoleSystem && !foundNonSystem {
				systemMsgs = append(systemMsgs, msg)
			} else {
				foundNonSystem = true
				msgs = append(msgs, msg)
			}
		}

		actionOpts.Messages = append(systemMsgs, execMsgs...)
		actionOpts.Messages = append(actionOpts.Messages, msgs...)
	}

	toolArgs := execOpts.Tools
	if len(toolArgs) == 0 {
		toolArgs = make([]ToolArg, 0, len(actionOpts.Tools))
		for _, toolName := range actionOpts.Tools {
			toolArgs = append(toolArgs, ToolName(toolName))
		}
	}

	toolNames, newTools, err := resolveUniqueTools(p.registry, toolArgs)
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

	return GenerateWithRequest(ctx, r, actionOpts, execOpts.Stream)
}

// ExecuteStream executes the prompt with streaming and returns an iterator.
//
// If the yield function is passed a non-nil error, execution has failed with
// that error; the yield function will not be called again.
//
// If the yield function's [StreamValue] argument has Done == true, the
// value's Output and Response fields contain the final typed output and
// response; the yield function will not be called again.
//
// Otherwise the Chunk field of the passed [StreamValue] holds a streamed
// chunk: a raw [*ModelResponseChunk] for a [TextPrompt], or the cumulative
// parsed output so far for a [DataPrompt] (chunks that don't yet parse are
// skipped).
func (p *Prompt[In, Out, Stream]) ExecuteStream(ctx context.Context, input In, opts ...PromptExecuteOption) iter.Seq2[*StreamValue[Out, Stream], error] {
	return func(yield func(*StreamValue[Out, Stream], error) bool) {
		if p == nil {
			yield(nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.ExecuteStream: prompt is nil"))
			return
		}

		rawChunks := reflect.TypeFor[Stream]() == reflect.TypeFor[*ModelResponseChunk]()

		done := false
		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if done {
				return errStop
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			var value Stream
			if rawChunks {
				value = any(chunk).(Stream)
			} else {
				parsed, err := extractTypedOutput[Stream](chunk)
				if err != nil {
					yield(nil, err)
					done = true
					return err
				}
				// Skip yielding if there's no parseable output yet (e.g., incomplete JSON during streaming).
				if base.IsNil(parsed) {
					return nil
				}
				value = parsed
			}
			if !yield(&StreamValue[Out, Stream]{Chunk: value}, nil) {
				done = true
				return errStop
			}
			return nil
		}

		allOpts := append(slices.Clone(opts), WithStreaming(cb))
		resp, err := p.execute(ctx, input, allOpts)
		if done || errors.Is(err, errStop) {
			return
		}
		if err != nil {
			yield(nil, err)
			return
		}

		output, err := outputFromResponse[Out](resp)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(&StreamValue[Out, Stream]{Done: true, Output: output, Response: resp}, nil)
	}
}

// Render renders the prompt template based on user input, returning a
// [GenerateActionOptions] to be used with [GenerateWithRequest].
func (p *Prompt[In, Out, Stream]) Render(ctx context.Context, input In) (*GenerateActionOptions, error) {
	if p == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.Render: prompt is nil")
	}

	in := any(input)
	// TODO: This is hacky; we should have a helper that fetches the metadata.
	if in == nil {
		in = p.Desc().Metadata["prompt"].(map[string]any)["defaultInput"]
	}

	return p.Run(ctx, in, nil)
}

// Desc returns a descriptor of the prompt with resolved schema references.
func (p *Prompt[In, Out, Stream]) Desc() api.ActionDesc {
	desc := p.action.Desc()
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
		return nil, errors.New("prompt.buildVariables: fields not a struct or pointer to a struct or a map")
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
// using the input variables and other information in the [Prompt].
func (p *Prompt[In, Out, Stream]) buildRequest(ctx context.Context, input any) (*GenerateActionOptions, error) {
	m, err := buildVariables(input)
	if err != nil {
		return nil, err
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
		return nil, core.NewError(core.INVALID_ARGUMENT, "invalid output schema for prompt %q: %v", p.Name(), err)
	}

	useRefs, err := configsToRefs(p.Use)
	if err != nil {
		return nil, fmt.Errorf("prompt %q: %w", p.Name(), err)
	}

	return &GenerateActionOptions{
		Model:              modelName,
		Config:             config,
		ToolChoice:         p.ToolChoice,
		MaxTurns:           p.MaxTurns,
		ReturnToolRequests: p.ReturnToolRequests != nil && *p.ReturnToolRequests,
		Messages:           messages,
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

// renderSystemPrompt renders a system prompt message.
func renderSystemPrompt(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.SystemFn == nil {
		return messages, nil
	}

	templateText, err := opts.SystemFn(ctx, raw)
	if err != nil {
		return nil, err
	}

	renderedMessages, err := renderPrompt(ctx, opts, templateText, input, dp)
	if err != nil {
		return nil, err
	}

	for _, m := range renderedMessages {
		if m.Role == "" || (len(renderedMessages) == 1 && m.Role == RoleUser) {
			m.Role = RoleSystem
		}
		messages = append(messages, m)
	}

	return messages, nil
}

// renderUserPrompt renders a user prompt message.
func renderUserPrompt(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.PromptFn == nil {
		return messages, nil
	}

	templateText, err := opts.PromptFn(ctx, raw)
	if err != nil {
		return nil, err
	}

	renderedMessages, err := renderPrompt(ctx, opts, templateText, input, dp)
	if err != nil {
		return nil, err
	}

	for _, m := range renderedMessages {
		if m.Role == "" || (len(renderedMessages) == 1 && m.Role != RoleUser) {
			m.Role = RoleUser
		}
		messages = append(messages, m)
	}

	return messages, nil
}

// renderMessages renders a slice of messages.
func renderMessages(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.MessagesFn == nil {
		return messages, nil
	}

	msgs, err := opts.MessagesFn(ctx, raw)
	if err != nil {
		return nil, err
	}

	// Create new message copies to avoid mutating shared messages during concurrent execution
	renderedMsgs := make([]*Message, 0, len(msgs))
	for _, msg := range msgs {
		hasTextPart := slices.ContainsFunc(msg.Content, (*Part).IsText)

		if !hasTextPart {
			// Create a new message with non-text content instead of mutating the original
			renderedMsg := &Message{
				Role:     msg.Role,
				Content:  msg.Content,
				Metadata: msg.Metadata,
			}
			renderedMsgs = append(renderedMsgs, renderedMsg)
			continue
		}

		for _, part := range msg.Content {
			if part.IsText() {
				messagesFromText, err := renderPrompt(ctx, opts, part.Text, input, dp)
				if err != nil {
					return nil, err
				}
				for _, m := range messagesFromText {
					// If the rendered message has no role, or it is a single message with default role,
					// use the original message's role.
					role := m.Role
					if role == "" || (len(messagesFromText) == 1 && role == RoleUser) {
						role = msg.Role
					}
					renderedMsgs = append(renderedMsgs, &Message{
						Role:     role,
						Content:  m.Content,
						Metadata: msg.Metadata,
					})
				}
			} else {
				// Preserve non-text parts as-is in the current last message if possible, or create a new one
				if len(renderedMsgs) > 0 && renderedMsgs[len(renderedMsgs)-1].Role == msg.Role {
					renderedMsgs[len(renderedMsgs)-1].Content = append(renderedMsgs[len(renderedMsgs)-1].Content, part)
				} else {
					renderedMsgs = append(renderedMsgs, &Message{
						Role:     msg.Role,
						Content:  []*Part{part},
						Metadata: msg.Metadata,
					})
				}
			}
		}
	}

	return append(messages, renderedMsgs...), nil
}

// renderPrompt renders a prompt template using dotprompt functionalities
func renderPrompt(ctx context.Context, opts promptOptions, templateText string, input map[string]any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	renderedFunc, err := dp.Compile(templateText, &dotprompt.PromptMetadata{})
	if err != nil {
		return nil, err
	}

	return renderDotpromptToMessages(ctx, renderedFunc, input, &dotprompt.PromptMetadata{
		Input: dotprompt.PromptMetadataInput{
			Default: opts.DefaultInput,
		},
	})
}

// renderDotpromptToMessages executes a dotprompt prompt function and converts the result to a slice of messages
func renderDotpromptToMessages(ctx context.Context, promptFn dotprompt.PromptFunction, input map[string]any, additionalMetadata *dotprompt.PromptMetadata) ([]*Message, error) {
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
		Input:   input,
		Context: templateContext,
	}, additionalMetadata)
	if err != nil {
		return nil, fmt.Errorf("failed to render prompt: %w", err)
	}

	convertedMessages := []*Message{}
	for _, message := range rendered.Messages {
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
func LoadPromptDirFromFS(r api.Registry, fsys fs.FS, dir, namespace string) error {
	if fsys == nil {
		return errors.New("no prompt filesystem provided")
	}

	if _, err := fs.Stat(fsys, dir); err != nil {
		return fmt.Errorf("failed to access prompt directory %q in filesystem: %w", dir, err)
	}

	entries, err := fs.ReadDir(fsys, dir)
	if err != nil {
		return fmt.Errorf("failed to read prompt directory structure: %w", err)
	}

	for _, entry := range entries {
		filename := entry.Name()
		filePath := path.Join(dir, filename)
		if entry.IsDir() {
			if err := LoadPromptDirFromFS(r, fsys, filePath, namespace); err != nil {
				return err
			}
		} else if strings.HasSuffix(filename, ".prompt") {
			if strings.HasPrefix(filename, "_") {
				partialName := strings.TrimSuffix(filename[1:], ".prompt")
				source, err := fs.ReadFile(fsys, filePath)
				if err != nil {
					return fmt.Errorf("failed to read partial file %q: %w", filePath, err)
				}
				r.RegisterPartial(partialName, string(source))
				slog.Debug("Registered Dotprompt partial", "name", partialName, "file", filePath)
			} else {
				if _, err := LoadPromptFromFS(r, fsys, dir, filename, namespace); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// LoadPromptFromFS loads a single prompt from a filesystem into the registry.
// The fsys parameter should be an fs.FS implementation (e.g., embed.FS or os.DirFS).
// The dir parameter specifies the directory within the filesystem where the prompt is located.
func LoadPromptFromFS(r api.Registry, fsys fs.FS, dir, filename, namespace string) (*TextPrompt[any], error) {
	name := strings.TrimSuffix(filename, ".prompt")

	sourceFile := path.Join(dir, filename)
	source, err := fs.ReadFile(fsys, sourceFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read prompt file %q: %w", sourceFile, err)
	}

	p, err := LoadPromptFromSource(r, string(source), name, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to load prompt %q: %w", sourceFile, err)
	}

	slog.Debug("Registered Dotprompt", "name", p.Name(), "file", sourceFile)
	return p, nil
}

// LoadPromptFromSource loads a prompt from raw .prompt file content.
// The source parameter should contain the complete .prompt file text (frontmatter + template).
// The name parameter is the prompt name (may include variant suffix like "myPrompt.variant").
func LoadPromptFromSource(r api.Registry, source, name, namespace string) (*TextPrompt[any], error) {
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

	toolArgs := make([]ToolArg, len(metadata.Tools))
	for i, tool := range metadata.Tools {
		toolArgs[i] = ToolName(tool)
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
			Tools: toolArgs,
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

	prompt := DefinePrompt[any](r, key, opts, WithPrompt(parsedPrompt.Template))

	return prompt, nil
}

// parseDotpromptUse converts the value of the dotprompt `use:` frontmatter
// field into a slice of lazy [Middleware] references. Each entry may be a
// bare string (interpreted as a registered middleware name) or a map with
// `name` and optional `config`, mirroring the TypeScript MiddlewareRef shape.
// Returns nil if the input is nil or an empty slice.
func parseDotpromptUse(raw any) ([]Middleware, error) {
	if raw == nil {
		return nil, nil
	}
	entries, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("`use` must be a list, got %T", raw)
	}
	uses := make([]Middleware, 0, len(entries))
	for i, entry := range entries {
		switch v := entry.(type) {
		case string:
			if v == "" {
				return nil, fmt.Errorf("`use[%d]` is an empty string", i)
			}
			uses = append(uses, middlewareRefArg{name: v})
		case map[string]any:
			name, _ := v["name"].(string)
			if name == "" {
				return nil, fmt.Errorf("`use[%d]` is missing required `name` field", i)
			}
			uses = append(uses, middlewareRefArg{name: name, config: v["config"]})
		default:
			return nil, fmt.Errorf("`use[%d]` must be a string or map, got %T", i, entry)
		}
	}
	return uses, nil
}

// LoadPromptDir loads prompts and partials from a directory on the local filesystem.
func LoadPromptDir(r api.Registry, dir string, namespace string) error {
	return LoadPromptDirFromFS(r, os.DirFS(dir), ".", namespace)
}

// LoadPrompt loads a single prompt from a directory on the local filesystem into the registry.
func LoadPrompt(r api.Registry, dir, filename, namespace string) (*TextPrompt[any], error) {
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
		return "", nil, errors.New("found empty URI in part")
	}

	if strings.HasPrefix(uri, "gs://") || strings.HasPrefix(uri, "http") {
		if ct == "" {
			return "", nil, errors.New("must supply contentType when using media from gs:// or http(s):// URLs")
		}
		return ct, []byte(uri), nil
	}
	if contents, isData := strings.CutPrefix(uri, "data:"); isData {
		prefix, _, found := strings.Cut(contents, ",")
		if !found {
			return "", nil, errors.New("failed to parse data URI: missing comma")
		}

		if p, isBase64 := strings.CutSuffix(prefix, ";base64"); isBase64 {
			if ct == "" {
				ct = p
			}
			return ct, []byte(uri), nil
		}
	}

	return "", nil, errors.New("uri content type not found")
}
