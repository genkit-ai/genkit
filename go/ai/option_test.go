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
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// applyGen builds a generateOptions from the given options, mirroring what
// Generate does internally.
func applyGen(opts ...GenerateOption) *generateOptions {
	g := &generateOptions{}
	for _, o := range opts {
		o.applyGenerate(g)
	}
	return g
}

// messageText returns the text of each message a MessagesFn produces, for
// asserting on accumulation order.
func messageText(t *testing.T, fn MessagesFn) []string {
	t.Helper()
	if fn == nil {
		t.Fatal("MessagesFn is nil")
	}
	msgs, err := fn(context.Background(), nil)
	if err != nil {
		t.Fatalf("MessagesFn error: %v", err)
	}
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.Text()
	}
	return out
}

// TestCollectionOptionsAccumulate verifies that options carrying multiple items
// append across repeated calls (and across their variants) rather than
// erroring or overwriting.
func TestCollectionOptionsAccumulate(t *testing.T) {
	t.Run("messages append across WithMessages and WithMessagesFn", func(t *testing.T) {
		g := applyGen(
			WithMessages(NewUserTextMessage("a")),
			WithMessages(NewUserTextMessage("b"), NewUserTextMessage("c")),
			WithMessagesFn(func(context.Context, any) ([]*Message, error) {
				return []*Message{NewUserTextMessage("d")}, nil
			}),
		)
		assertEqual(t, messageText(t, g.MessagesFn), []string{"a", "b", "c", "d"})
	})

	t.Run("messages do not alias the caller's slice", func(t *testing.T) {
		// WithMessages(history...) hands the caller's slice straight through.
		// Appending onto it would land in the spare capacity of the array
		// backing history, which is invisible until the caller appends to
		// their own slice and silently rewrites the messages we produced.
		history := make([]*Message, 1, 4)
		history[0] = NewUserTextMessage("original")

		g := applyGen(
			WithMessages(history...),
			WithMessages(NewUserTextMessage("appended")),
		)
		msgs, err := g.MessagesFn(context.Background(), nil)
		if err != nil {
			t.Fatalf("MessagesFn error: %v", err)
		}
		if len(msgs) != 2 {
			t.Fatalf("len(msgs) = %d, want 2", len(msgs))
		}

		// The caller keeps building their own history, reusing index 1 of the
		// shared array. Our already-produced messages must not change.
		history = append(history, NewUserTextMessage("caller's next turn"))

		if msgs[1].Text() != "appended" {
			t.Errorf("produced messages aliased the caller's array: msgs[1] = %q, want %q",
				msgs[1].Text(), "appended")
		}
		if history[0].Text() != "original" {
			t.Errorf("caller history was mutated: got %q, want %q", history[0].Text(), "original")
		}
	})

	t.Run("tools append", func(t *testing.T) {
		t1 := &mockTool{name: "t/1"}
		t2 := &mockTool{name: "t/2"}
		t3 := &mockTool{name: "t/3"}
		g := applyGen(WithTools(t1, t2), WithTools(t3))
		assertEqual(t, g.Tools, []ToolRef{t1, t2, t3}, cmpopts.IgnoreUnexported(mockTool{}))
	})

	t.Run("docs append across WithDocs and WithTextDocs", func(t *testing.T) {
		g := applyGen(WithDocs(DocumentFromText("doc", nil)), WithTextDocs("text"))
		if len(g.Documents) != 2 {
			t.Fatalf("len(Documents) = %d, want 2", len(g.Documents))
		}
	})

	t.Run("resources append", func(t *testing.T) {
		res := func(name string) Resource {
			return NewResource(name, &ResourceOptions{URI: "res://" + name},
				func(context.Context, *ResourceInput) (*ResourceOutput, error) { return nil, nil })
		}
		g := applyGen(WithResources(res("a")), WithResources(res("b"), res("c")))
		if len(g.Resources) != 3 {
			t.Errorf("len(Resources) = %d, want 3", len(g.Resources))
		}
	})

	t.Run("middleware appends in order", func(t *testing.T) {
		g := applyGen(
			WithMiddleware(func(next ModelFunc) ModelFunc { return next }),
			WithMiddleware(
				func(next ModelFunc) ModelFunc { return next },
				func(next ModelFunc) ModelFunc { return next },
			),
		)
		if len(g.Middleware) != 3 {
			t.Errorf("len(Middleware) = %d, want 3", len(g.Middleware))
		}
	})

	t.Run("tool responses and restarts append", func(t *testing.T) {
		g := applyGen(
			WithToolResponses(NewTextPart("r1")),
			WithToolResponses(NewTextPart("r2")),
			WithToolRestarts(NewTextPart("s1")),
			WithToolRestarts(NewTextPart("s2")),
		)
		if len(g.RespondParts) != 2 {
			t.Errorf("len(RespondParts) = %d, want 2", len(g.RespondParts))
		}
		if len(g.RestartParts) != 2 {
			t.Errorf("len(RestartParts) = %d, want 2", len(g.RestartParts))
		}
	})

	t.Run("dataset appends", func(t *testing.T) {
		e := &evaluatorOptions{}
		for _, o := range []EvaluatorOption{
			WithDataset(&Example{}),
			WithDataset(&Example{}, &Example{}),
		} {
			o.applyEvaluator(e)
		}
		if len(e.Dataset) != 3 {
			t.Errorf("len(Dataset) = %d, want 3", len(e.Dataset))
		}
	})
}

// TestSingleValueOptionsLastWins verifies that options filling a single slot
// take the last value set instead of erroring on repeats.
func TestSingleValueOptionsLastWins(t *testing.T) {
	t.Run("model: WithModel then WithModelName", func(t *testing.T) {
		g := applyGen(WithModel(&mockModel{name: "first/model"}), WithModelName("second/model"))
		if g.Model == nil || g.Model.Name() != "second/model" {
			t.Errorf("Model = %v, want name second/model", g.Model)
		}
	})

	t.Run("config: last wins", func(t *testing.T) {
		last := &GenerationCommonConfig{Temperature: 0.9}
		g := applyGen(WithConfig(&GenerationCommonConfig{Temperature: 0.1}), WithConfig(last))
		if g.Config != last {
			t.Errorf("Config = %v, want %v", g.Config, last)
		}
	})

	t.Run("tool choice, max turns, return tool requests: last wins", func(t *testing.T) {
		g := applyGen(
			WithToolChoice(ToolChoiceAuto), WithToolChoice(ToolChoiceRequired),
			WithMaxTurns(2), WithMaxTurns(7),
			WithReturnToolRequests(true), WithReturnToolRequests(false),
		)
		if g.ToolChoice != ToolChoiceRequired {
			t.Errorf("ToolChoice = %q, want %q", g.ToolChoice, ToolChoiceRequired)
		}
		if g.MaxTurns != 7 {
			t.Errorf("MaxTurns = %d, want 7", g.MaxTurns)
		}
		if g.ReturnToolRequests == nil || *g.ReturnToolRequests {
			t.Errorf("ReturnToolRequests = %v, want false", g.ReturnToolRequests)
		}
	})

	t.Run("system and prompt: last wins across text and fn", func(t *testing.T) {
		g := applyGen(
			WithSystem("sys one"),
			WithSystemFn(func(context.Context, any) (string, error) { return "sys two", nil }),
			WithPrompt("usr one"),
			WithPrompt("usr two"),
		)
		parts, err := g.SystemFn(context.Background(), nil)
		if err != nil {
			t.Fatalf("SystemFn: %v", err)
		}
		if len(parts) != 1 || parts[0].Text != "sys two" {
			t.Errorf("system = %+v, want one part %q", parts, "sys two")
		}
		// The later function has to clear the earlier text, or the renderer's
		// fixed order would decide the winner instead of the call order.
		if g.SystemText != nil {
			t.Errorf("SystemText = %q, want it cleared by the later WithSystemFn", *g.SystemText)
		}
		if g.PromptFn != nil {
			t.Error("PromptFn set, want it cleared by the later WithPrompt")
		}
		if g.PromptText == nil || *g.PromptText != "usr two" {
			t.Errorf("prompt = %v, want %q", g.PromptText, "usr two")
		}
	})

	t.Run("streaming: last wins, no error on repeat", func(t *testing.T) {
		g := applyGen(
			WithStreaming(func(context.Context, *ModelResponseChunk) error { return nil }),
			WithStreaming(func(context.Context, *ModelResponseChunk) error { return nil }),
		)
		if g.Stream == nil {
			t.Error("Stream is nil, want non-nil")
		}
	})

	t.Run("prompt input: last wins", func(t *testing.T) {
		opts := &promptExecutionOptions{}
		for _, o := range []PromptExecuteOption{WithInput("input1"), WithInput("input2")} {
			o.applyPromptExecute(opts)
		}
		if opts.Input != "input2" {
			t.Errorf("Input = %v, want input2", opts.Input)
		}
	})

	t.Run("input config: last wins as one slot, stale defaults cleared", func(t *testing.T) {
		opts := &promptOptions{}
		for _, o := range []InputOption{
			WithInputType(struct {
				Test string `json:"test"`
			}{Test: "stale"}),
			WithInputSchemaName("Override"),
		} {
			o.applyPrompt(opts)
		}
		if ref, _ := opts.InputSchema["$ref"].(string); ref != "genkit:Override" {
			t.Errorf("InputSchema.$ref = %v, want genkit:Override", opts.InputSchema["$ref"])
		}
		// The schema override replaces the whole input config: the default
		// input inferred from the overridden type must not survive to be
		// rendered against the new schema.
		if opts.DefaultInput != nil {
			t.Errorf("DefaultInput = %v, want nil after schema override", opts.DefaultInput)
		}
	})

	t.Run("output schema: last wins, keeping JSON format", func(t *testing.T) {
		custom := map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "string"}}}

		// Mirrors GenerateData's prepend: the inferred type is applied first,
		// the caller's explicit schema second.
		g := applyGen(
			WithOutputType(struct {
				Value int `json:"value"`
			}{}),
			WithOutputSchema(custom),
		)
		assertEqual(t, g.OutputSchema, custom)
		if g.OutputFormat != OutputFormatJSON {
			t.Errorf("OutputFormat = %q, want %q", g.OutputFormat, OutputFormatJSON)
		}
	})
}

func TestPromptOptions(t *testing.T) {
	opts := &promptOptions{}
	for _, o := range []PromptOption{
		WithDescription("test description"),
		WithMetadata(map[string]any{"key": "value"}),
		WithInputType(struct {
			Test string `json:"test"`
		}{}),
	} {
		o.applyPrompt(opts)
	}
	if opts.Description != "test description" {
		t.Errorf("Description = %q, want %q", opts.Description, "test description")
	}
	if opts.InputSchema == nil {
		t.Error("InputSchema is nil")
	}
}

func TestGenerateOptionsComplete(t *testing.T) {
	opts := &generateOptions{}

	mw := func(next ModelFunc) ModelFunc { return next }
	model := &mockModel{name: "test/model"}
	tool := &mockTool{name: "test/tool"}
	streamFunc := func(context.Context, *ModelResponseChunk) error { return nil }
	doc := DocumentFromText("doc", nil)
	options := []GenerateOption{
		WithModel(model),
		WithMessages(NewUserTextMessage("message")),
		WithConfig(&GenerationCommonConfig{Temperature: 0.7}),
		WithTools(tool),
		WithToolChoice(ToolChoiceAuto),
		WithMaxTurns(3),
		WithReturnToolRequests(true),
		WithMiddleware(mw),
		WithSystem("system prompt"),
		WithPrompt("user prompt"),
		WithDocs(doc),
		WithOutputType(map[string]string{"key": "value"}),
		WithOutputInstructions(""),
		WithCustomConstrainedOutput(),
		WithStreaming(streamFunc),
	}

	for _, opt := range options {
		opt.applyGenerate(opts)
	}

	returnToolRequests := true
	expected := &generateOptions{
		commonGenOptions: commonGenOptions{
			configOptions: configOptions{
				Config: &GenerationCommonConfig{Temperature: 0.7},
			},
			Model:              model,
			Tools:              []ToolRef{tool},
			ToolChoice:         ToolChoiceAuto,
			MaxTurns:           3,
			ReturnToolRequests: &returnToolRequests,
			Middleware:         []ModelMiddleware{mw},
		},
		promptingOptions: promptingOptions{
			SystemText: opts.SystemText,
			PromptText: opts.PromptText,
		},
		outputOptions: outputOptions{
			OutputFormat: OutputFormatJSON,
			OutputSchema: opts.OutputSchema,
			OutputInstructions: func() *string {
				s := ""
				return &s
			}(),
			CustomConstrained: true,
		},
		executionOptions: executionOptions{
			Stream: streamFunc,
		},
		documentOptions: documentOptions{
			Documents: []*Document{doc},
		},
	}

	if diff := cmp.Diff(expected, opts,
		cmpopts.IgnoreFields(commonGenOptions{}, "MessagesFn", "Middleware"),
		cmpopts.IgnoreFields(promptingOptions{}, "SystemFn", "PromptFn"),
		cmpopts.IgnoreFields(executionOptions{}, "Stream"),
		cmpopts.IgnoreUnexported(mockModel{}, mockTool{}),
		cmp.AllowUnexported(generateOptions{}, commonGenOptions{}, promptingOptions{},
			outputOptions{}, executionOptions{}, documentOptions{})); diff != "" {
		t.Errorf("Options not applied correctly, diff (-want +got):\n%s", diff)
	}

	if opts.MessagesFn == nil {
		t.Errorf("MessagesFn should not be nil")
	}
	if len(opts.Middleware) == 0 {
		t.Errorf("Middleware should not be empty")
	}
	// WithSystem and WithPrompt fill their slot with template text; the
	// parts and function forms fill the same slot with a function instead.
	if opts.SystemText == nil && opts.SystemFn == nil {
		t.Errorf("the system slot should be filled")
	}
	if opts.PromptText == nil && opts.PromptFn == nil {
		t.Errorf("the prompt slot should be filled")
	}
	if opts.Stream == nil {
		t.Errorf("Stream should not be nil")
	}
}
func TestPromptOptionsComplete(t *testing.T) {
	opts := &promptOptions{}

	mw := func(next ModelFunc) ModelFunc { return next }
	model := &mockModel{name: "test/model"}
	tool := &mockTool{name: "test/tool"}
	input := struct {
		Test string `json:"test"`
	}{
		Test: "value",
	}

	options := []PromptOption{
		WithModel(model),
		WithMessages(NewUserTextMessage("message")),
		WithConfig(&GenerationCommonConfig{Temperature: 0.7}),
		WithTools(tool),
		WithToolChoice(ToolChoiceAuto),
		WithMaxTurns(3),
		WithReturnToolRequests(true),
		WithMiddleware(mw),
		WithSystem("system prompt"),
		WithPrompt("user prompt"),
		WithDescription("test description"),
		WithMetadata(map[string]any{"key": "value"}),
		WithOutputType(map[string]string{"key": "value"}),
		WithOutputInstructions(""),
		WithCustomConstrainedOutput(),
		WithInputType(input),
	}

	for _, opt := range options {
		opt.applyPrompt(opts)
	}

	returnToolRequests := true
	expected := &promptOptions{
		commonGenOptions: commonGenOptions{
			configOptions: configOptions{
				Config: &GenerationCommonConfig{Temperature: 0.7},
			},
			Model:              model,
			Tools:              []ToolRef{tool},
			ToolChoice:         ToolChoiceAuto,
			MaxTurns:           3,
			ReturnToolRequests: &returnToolRequests,
			Middleware:         []ModelMiddleware{mw},
		},
		promptingOptions: promptingOptions{
			SystemText: opts.SystemText,
			PromptText: opts.PromptText,
		},
		inputOptions: inputOptions{
			InputSchema:  opts.InputSchema,
			DefaultInput: map[string]any{"test": "value"},
		},
		outputOptions: outputOptions{
			OutputFormat: OutputFormatJSON,
			OutputSchema: opts.OutputSchema,
			OutputInstructions: func() *string {
				s := ""
				return &s
			}(),
			CustomConstrained: true,
		},
		Description: "test description",
		Metadata:    map[string]any{"key": "value"},
	}

	if diff := cmp.Diff(expected, opts,
		cmpopts.IgnoreFields(commonGenOptions{}, "MessagesFn", "Middleware"),
		cmpopts.IgnoreFields(promptingOptions{}, "SystemFn", "PromptFn"),
		cmpopts.IgnoreFields(outputOptions{}, "OutputSchema"),
		cmpopts.IgnoreFields(inputOptions{}, "InputSchema"),
		cmpopts.IgnoreUnexported(mockModel{}, mockTool{}),
		cmp.AllowUnexported(promptOptions{}, commonGenOptions{}, promptingOptions{},
			inputOptions{}, outputOptions{})); diff != "" {
		t.Errorf("Options not applied correctly, diff (-want +got):\n%s", diff)
	}

	if opts.MessagesFn == nil {
		t.Errorf("MessagesFn should not be nil")
	}
	if len(opts.Middleware) == 0 {
		t.Errorf("Middleware should not be empty")
	}
	// WithSystem and WithPrompt fill their slot with template text; the
	// parts and function forms fill the same slot with a function instead.
	if opts.SystemText == nil && opts.SystemFn == nil {
		t.Errorf("the system slot should be filled")
	}
	if opts.PromptText == nil && opts.PromptFn == nil {
		t.Errorf("the prompt slot should be filled")
	}
	if opts.OutputSchema == nil {
		t.Errorf("OutputSchema should not be nil")
	}
	if opts.InputSchema == nil {
		t.Errorf("InputSchema should not be nil")
	}
}

func TestPromptExecuteOptionsComplete(t *testing.T) {
	opts := &promptExecutionOptions{}

	mw := func(next ModelFunc) ModelFunc { return next }
	model := &mockModel{name: "test/model"}
	tool := &mockTool{name: "test/tool"}
	streamFunc := func(context.Context, *ModelResponseChunk) error { return nil }
	input := map[string]string{"key": "value"}
	doc := DocumentFromText("doc", nil)

	options := []PromptExecuteOption{
		WithModel(model),
		WithMessages(NewUserTextMessage("message")),
		WithConfig(&GenerationCommonConfig{Temperature: 0.7}),
		WithTools(tool),
		WithToolChoice(ToolChoiceAuto),
		WithMaxTurns(3),
		WithReturnToolRequests(true),
		WithMiddleware(mw),
		WithDocs(doc),
		WithStreaming(streamFunc),
		WithInput(input),
	}

	for _, opt := range options {
		opt.applyPromptExecute(opts)
	}

	returnToolRequests := true
	expected := &promptExecutionOptions{
		commonGenOptions: commonGenOptions{
			configOptions: configOptions{
				Config: &GenerationCommonConfig{Temperature: 0.7},
			},
			Model:              model,
			Tools:              []ToolRef{tool},
			ToolChoice:         ToolChoiceAuto,
			MaxTurns:           3,
			ReturnToolRequests: &returnToolRequests,
			Middleware:         []ModelMiddleware{mw},
		},
		executionOptions: executionOptions{
			Stream: streamFunc,
		},
		documentOptions: documentOptions{
			Documents: []*Document{doc},
		},
		Input: input,
	}

	if diff := cmp.Diff(expected, opts,
		cmpopts.IgnoreFields(commonGenOptions{}, "MessagesFn", "Middleware"),
		cmpopts.IgnoreFields(executionOptions{}, "Stream"),
		cmpopts.IgnoreUnexported(mockModel{}, mockTool{}),
		cmp.AllowUnexported(promptExecutionOptions{}, commonGenOptions{},
			executionOptions{})); diff != "" {
		t.Errorf("Options not applied correctly, diff (-want +got):\n%s", diff)
	}

	if opts.MessagesFn == nil {
		t.Errorf("MessagesFn should not be nil")
	}
	if opts.Middleware == nil {
		t.Errorf("Middleware should not be nil")
	}
	if opts.Stream == nil {
		t.Errorf("Stream should not be nil")
	}
}

type mockModel struct {
	name string
}

func (m *mockModel) Name() string {
	return m.name
}

func (m *mockModel) Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	return nil, nil
}

type mockTool struct {
	name string
}

func (t *mockTool) Name() string {
	return t.name
}

func (t *mockTool) Definition() *ToolDefinition {
	return &ToolDefinition{Name: t.name}
}

func (t *mockTool) RunRaw(ctx context.Context, input any) (any, error) {
	return nil, nil
}

func (t *mockTool) RunRawMultipart(ctx context.Context, input any) (*MultipartToolResponse, error) {
	return nil, nil
}

func (t *mockTool) Register(r api.Registry) {
}

func TestWithInputSchemaName(t *testing.T) {
	t.Run("creates input option with schema reference", func(t *testing.T) {
		opt := WithInputSchemaName("MyInputType")
		opts := &promptOptions{}
		opt.applyPrompt(opts)

		if opts.InputSchema == nil {
			t.Fatal("InputSchema is nil")
		}

		ref, ok := opts.InputSchema["$ref"].(string)
		if !ok {
			t.Fatal("InputSchema.$ref is not a string")
		}
		if ref != "genkit:MyInputType" {
			t.Errorf("InputSchema.$ref = %q, want %q", ref, "genkit:MyInputType")
		}
	})
}

func TestWithOutputSchema(t *testing.T) {
	t.Run("creates output option with direct schema", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		opt := WithOutputSchema(schema)
		opts := &generateOptions{}
		opt.applyGenerate(opts)

		if opts.OutputSchema == nil {
			t.Fatal("OutputSchema is nil")
		}
		if opts.OutputFormat != OutputFormatJSON {
			t.Errorf("OutputFormat = %q, want %q", opts.OutputFormat, OutputFormatJSON)
		}
	})
}

func TestWithOutputEnums(t *testing.T) {
	t.Run("creates enum output with string values", func(t *testing.T) {
		opt := WithOutputEnums("red", "green", "blue")
		opts := &generateOptions{}
		opt.applyGenerate(opts)

		if opts.OutputSchema == nil {
			t.Fatal("OutputSchema is nil")
		}
		if opts.OutputFormat != OutputFormatEnum {
			t.Errorf("OutputFormat = %q, want %q", opts.OutputFormat, OutputFormatEnum)
		}

		enumType, ok := opts.OutputSchema["type"].(string)
		if !ok || enumType != "string" {
			t.Errorf("OutputSchema.type = %v, want %q", opts.OutputSchema["type"], "string")
		}

		enumVals, ok := opts.OutputSchema["enum"].([]string)
		if !ok {
			t.Fatalf("OutputSchema.enum is not []string: %T", opts.OutputSchema["enum"])
		}
		if len(enumVals) != 3 {
			t.Errorf("len(enum) = %d, want 3", len(enumVals))
		}
	})

	t.Run("works with custom string type", func(t *testing.T) {
		type Color string
		opt := WithOutputEnums(Color("red"), Color("green"))
		opts := &generateOptions{}
		opt.applyGenerate(opts)

		enumVals := opts.OutputSchema["enum"].([]string)
		if enumVals[0] != "red" || enumVals[1] != "green" {
			t.Errorf("enum values = %v, want [red, green]", enumVals)
		}
	})
}

func TestWithEvaluatorName(t *testing.T) {
	t.Run("creates evaluator option with reference", func(t *testing.T) {
		opt := WithEvaluatorName("test/myEvaluator")
		opts := &evaluatorOptions{}
		opt.applyEvaluator(opts)

		if opts.Evaluator == nil {
			t.Fatal("Evaluator is nil")
		}
		if opts.Evaluator.Name() != "test/myEvaluator" {
			t.Errorf("Evaluator.Name() = %q, want %q", opts.Evaluator.Name(), "test/myEvaluator")
		}
	})
}

type withInputTypeQuery struct {
	City string `json:"city"`
}

// TestWithInputTypeAppliesToTools pins that WithInputType reaches the tool
// constructors, not just prompts. The tool constructors have accepted it since
// before go/v1.9.0, so narrowing it to an option prompts alone can take would
// break existing callers at compile time with no deprecation.
func TestWithInputTypeAppliesToTools(t *testing.T) {
	var _ ToolOption = WithInputType(withInputTypeQuery{})
	var _ PromptOption = WithInputType(withInputTypeQuery{})

	tl := NewTool("withInputTypeTool", "takes a typed input",
		func(ctx *ToolContext, in any) (string, error) { return "ok", nil },
		WithInputType(withInputTypeQuery{}))

	props, ok := tl.Definition().InputSchema["properties"].(map[string]any)
	if !ok || props["city"] == nil {
		t.Errorf("InputSchema = %v, want the schema derived from withInputTypeQuery", tl.Definition().InputSchema)
	}
}
