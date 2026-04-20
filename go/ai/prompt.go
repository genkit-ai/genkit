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
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
)

// Prompt is the interface for a prompt that can be executed and rendered.
type Prompt interface {
	// Name returns the name of the prompt.
	Name() string
	// Execute executes the prompt with the given options and returns a [ModelResponse].
	Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error)
	// ExecuteStream executes the prompt with streaming and returns an iterator.
	ExecuteStream(ctx context.Context, opts ...PromptExecuteOption) iter.Seq2[*ModelStreamValue, error]
	// Render renders the prompt with the given input and returns a [GenerateActionOptions] to be used with [GenerateWithRequest].
	Render(ctx context.Context, input any) (*GenerateActionOptions, error)
}

// prompt is a prompt template that can be executed to generate a model response.
type prompt struct {
	core.ActionDef[any, *GenerateActionOptions, struct{}]
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
		if err := opt.applyPrompt(pOpts); err != nil {
			panic(fmt.Errorf("ai.DefinePrompt: error applying options: %w", err))
		}
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

	promptMetadata := map[string]any{
		"name":         baseName,
		"description":  p.Description,
		"model":        modelName,
		"config":       p.Config,
		"input":        map[string]any{"schema": p.InputSchema},
		"output":       map[string]any{"schema": p.OutputSchema},
		"defaultInput": p.DefaultInput,
		"tools":        tools,
		"maxTurns":     p.MaxTurns,
	}
	if variant != "" {
		promptMetadata["variant"] = variant
	}
	if m, ok := metadata["prompt"].(map[string]any); ok {
		maps.Copy(m, promptMetadata)
	} else {
		metadata["prompt"] = promptMetadata
	}

	p.ActionDef = *core.DefineAction(r, name, api.ActionTypeExecutablePrompt, metadata, p.InputSchema, p.buildRequest)

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
		ActionDef: *action,
		registry:  r,
	}
}

// Execute renders a prompt, does variable substitution and
// passes the rendered template to the AI model specified by the prompt.
func (p *prompt) Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error) {
	if p == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.Execute: prompt is nil")
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
	actionOpts, err := p.Render(ctx, execOpts.Input)
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
		m, err := buildVariables(execOpts.Input)
		if err != nil {
			return nil, err
		}

		tempOpts := promptOptions{
			commonGenOptions: commonGenOptions{
				MessagesFn: execOpts.MessagesFn,
			},
		}

		execMsgs, err := renderMessages(ctx, tempOpts, []*Message{}, m, execOpts.Input, p.registry.Dotprompt())
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
			yield(nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.ExecuteStream: prompt is nil"))
			return
		}

		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if !yield(&ModelStreamValue{Chunk: chunk}, nil) {
				return errPromptStop
			}
			return nil
		}

		allOpts := append(slices.Clone(opts), WithStreaming(cb))
		resp, err := p.Execute(ctx, allOpts...)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(&ModelStreamValue{Done: true, Response: resp}, nil)
	}
}

// errPromptStop is a sentinel error used to signal early termination of streaming.
var errPromptStop = errors.New("stop")

// Render renders the prompt template based on user input.
func (p *prompt) Render(ctx context.Context, input any) (*GenerateActionOptions, error) {
	if p == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Prompt.Render: prompt is nil")
	}

	if len(p.Middleware) > 0 {
		logger.FromContext(ctx).Warn(fmt.Sprintf("middleware set on prompt %q will be ignored during Prompt.Render", p.Name()))
	}

	// TODO: This is hacky; we should have a helper that fetches the metadata.
	if input == nil {
		input = p.Desc().Metadata["prompt"].(map[string]any)["defaultInput"]
	}

	return p.Run(ctx, input, nil)
}

// Desc returns a descriptor of the prompt with resolved schema references.
func (p *prompt) Desc() api.ActionDesc {
	desc := p.ActionDef.Desc()
	promptMeta := desc.Metadata["prompt"].(map[string]any)
	if inputMeta, ok := promptMeta["input"].(map[string]any); ok {
		if inputSchema, ok := inputMeta["schema"].(map[string]any); ok {
			if resolved, err := core.ResolveSchema(p.registry, inputSchema); err == nil {
				inputMeta["schema"] = resolved
			}
		}
	}
	if outputMeta, ok := promptMeta["output"].(map[string]any); ok {
		if outputSchema, ok := outputMeta["schema"].(map[string]any); ok {
			if resolved, err := core.ResolveSchema(p.registry, outputSchema); err == nil {
				outputMeta["schema"] = resolved
			}
		}
	}
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
// using the input variables and other information in the [prompt].
func (p *prompt) buildRequest(ctx context.Context, input any) (*GenerateActionOptions, error) {
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

	return &GenerateActionOptions{
		Model:              modelName,
		Config:             config,
		ToolChoice:         p.ToolChoice,
		MaxTurns:           p.MaxTurns,
		ReturnToolRequests: p.ReturnToolRequests != nil && *p.ReturnToolRequests,
		Messages:           messages,
		Tools:              tools,
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
	if state := session.StateFromContext(ctx); state != nil {
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
func LoadPromptDirFromFS(r api.Registry, fsys fs.FS, dir, namespace string) {
	if fsys == nil {
		panic(errors.New("no prompt filesystem provided"))
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
					slog.Error("Failed to read partial file", "error", err)
					continue
				}
				r.RegisterPartial(partialName, string(source))
				slog.Debug("Registered Dotprompt partial", "name", partialName, "file", filePath)
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
		slog.Error("Failed to read prompt file", "file", sourceFile, "error", err)
		return nil
	}

	p, err := LoadPromptFromSource(r, string(source), name, namespace)
	if err != nil {
		slog.Error("Failed to load prompt", "file", sourceFile, "error", err)
		return nil
	}

	slog.Debug("Registered Dotprompt", "name", p.Name(), "file", sourceFile)
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

	prompt := DefinePrompt(r, key, opts, WithPrompt(parsedPrompt.Template))

	return prompt, nil
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
func (dp *DataPrompt[In, Out]) Execute(ctx context.Context, input In, opts ...PromptExecuteOption) (Out, *ModelResponse, error) {
	if dp == nil {
		return base.Zero[Out](), nil, core.NewError(core.INVALID_ARGUMENT, "DataPrompt.Execute: prompt is nil")
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
func (dp *DataPrompt[In, Out]) ExecuteStream(ctx context.Context, input In, opts ...PromptExecuteOption) iter.Seq2[*StreamValue[Out, Out], error] {
	return func(yield func(*StreamValue[Out, Out], error) bool) {
		if dp == nil {
			yield(nil, core.NewError(core.INVALID_ARGUMENT, "DataPrompt.ExecuteStream: prompt is nil"))
			return
		}

		cb := func(ctx context.Context, chunk *ModelResponseChunk) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			streamValue, err := extractTypedOutput[Out](chunk)
			if err != nil {
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

		allOpts := append(slices.Clone(opts), WithInput(input), WithStreaming(cb))
		resp, err := dp.prompt.Execute(ctx, allOpts...)
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
		return nil, errors.New("DataPrompt.Render: prompt is nil")
	}

	return dp.prompt.Render(ctx, input)
}
