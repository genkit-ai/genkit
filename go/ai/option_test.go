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

// messageText renders the concatenated text of a MessagesFn's output, for
// asserting on accumulation order.
func messageText(t *testing.T, fn MessagesFn) []string {
	t.Helper()
	if fn == nil {
		return nil
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
		got := messageText(t, g.MessagesFn)
		want := []string{"a", "b", "c", "d"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("messages diff (-want +got):\n%s", diff)
		}
	})

	t.Run("tools append", func(t *testing.T) {
		t1 := &mockTool{name: "t/1"}
		t2 := &mockTool{name: "t/2"}
		t3 := &mockTool{name: "t/3"}
		g := applyGen(WithTools(t1, t2), WithTools(t3))
		if diff := cmp.Diff([]ToolArg{t1, t2, t3}, g.Tools,
			cmpopts.IgnoreUnexported(mockTool{})); diff != "" {
			t.Errorf("tools diff (-want +got):\n%s", diff)
		}
	})

	t.Run("docs append across WithDocs and WithTextDocs", func(t *testing.T) {
		d := DocumentFromText("doc", nil)
		g := applyGen(WithDocs(d), WithTextDocs("text"))
		if len(g.Documents) != 2 {
			t.Fatalf("len(Documents) = %d, want 2", len(g.Documents))
		}
	})

	t.Run("resources append", func(t *testing.T) {
		g := applyGen(
			WithResources(&Resource{}),
			WithResources(&Resource{}, &Resource{}),
		)
		if len(g.Resources) != 3 {
			t.Errorf("len(Resources) = %d, want 3", len(g.Resources))
		}
	})

	t.Run("middleware append in order", func(t *testing.T) {
		a := MiddlewareFunc(func(context.Context) (*Hooks, error) { return &Hooks{}, nil })
		b := MiddlewareFunc(func(context.Context) (*Hooks, error) { return &Hooks{}, nil })
		g := applyGen(WithUse(a), WithUse(b))
		if len(g.Use) != 2 {
			t.Errorf("len(Use) = %d, want 2", len(g.Use))
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
		g := applyGen(
			WithConfig(&GenerationCommonConfig{Temperature: 0.1}),
			WithConfig(last),
		)
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
		if sys, _ := g.SystemFn(context.Background(), nil); sys != "sys two" {
			t.Errorf("system = %q, want %q", sys, "sys two")
		}
		if usr, _ := g.PromptFn(context.Background(), nil); usr != "usr two" {
			t.Errorf("prompt = %q, want %q", usr, "usr two")
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
}

// TestOutputSchemaLastWins verifies the output-schema slot behaves as last-wins,
// which is what lets GenerateData inject a schema that a caller can override.
func TestOutputSchemaLastWins(t *testing.T) {
	custom := map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "string"}}}

	// Simulate GenerateData's prepend: the inferred type is applied first, the
	// caller's explicit schema second.
	g := applyGen(
		WithOutputType(struct {
			Value int `json:"value"`
		}{}),
		WithOutputSchema(custom),
	)
	if diff := cmp.Diff(custom, g.OutputSchema); diff != "" {
		t.Errorf("OutputSchema not overridden by caller (-want +got):\n%s", diff)
	}
	if g.OutputFormat != OutputFormatJSON {
		t.Errorf("OutputFormat = %q, want %q", g.OutputFormat, OutputFormatJSON)
	}
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

	mw := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) { return &Hooks{}, nil })
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
		WithUse(mw),
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
			Tools:              []ToolArg{tool},
			ToolChoice:         ToolChoiceAuto,
			MaxTurns:           3,
			ReturnToolRequests: &returnToolRequests,
			Use:                []Middleware{mw},
		},
		promptingOptions: promptingOptions{
			SystemFn: opts.SystemFn,
			PromptFn: opts.PromptFn,
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
		cmpopts.IgnoreFields(commonGenOptions{}, "MessagesFn", "Use"),
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
	if len(opts.Use) == 0 {
		t.Errorf("Use should not be empty")
	}
	if opts.SystemFn == nil {
		t.Errorf("SystemFn should not be nil")
	}
	if opts.PromptFn == nil {
		t.Errorf("PromptFn should not be nil")
	}
	if opts.Stream == nil {
		t.Errorf("Stream should not be nil")
	}
}

func TestPromptOptionsComplete(t *testing.T) {
	opts := &promptOptions{}

	mw := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) { return &Hooks{}, nil })
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
		WithUse(mw),
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
			Tools:              []ToolArg{tool},
			ToolChoice:         ToolChoiceAuto,
			MaxTurns:           3,
			ReturnToolRequests: &returnToolRequests,
			Use:                []Middleware{mw},
		},
		promptingOptions: promptingOptions{
			SystemFn: opts.SystemFn,
			PromptFn: opts.PromptFn,
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
		cmpopts.IgnoreFields(commonGenOptions{}, "MessagesFn", "Use"),
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
	if len(opts.Use) == 0 {
		t.Errorf("Use should not be empty")
	}
	if opts.SystemFn == nil {
		t.Errorf("SystemFn should not be nil")
	}
	if opts.PromptFn == nil {
		t.Errorf("PromptFn should not be nil")
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

	mw := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) { return &Hooks{}, nil })
	model := &mockModel{name: "test/model"}
	tool := &mockTool{name: "test/tool"}
	streamFunc := func(context.Context, *ModelResponseChunk) error { return nil }
	doc := DocumentFromText("doc", nil)

	options := []PromptExecuteOption{
		WithModel(model),
		WithMessages(NewUserTextMessage("message")),
		WithConfig(&GenerationCommonConfig{Temperature: 0.7}),
		WithTools(tool),
		WithToolChoice(ToolChoiceAuto),
		WithMaxTurns(3),
		WithReturnToolRequests(true),
		WithUse(mw),
		WithDocs(doc),
		WithStreaming(streamFunc),
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
			Tools:              []ToolArg{tool},
			ToolChoice:         ToolChoiceAuto,
			MaxTurns:           3,
			ReturnToolRequests: &returnToolRequests,
			Use:                []Middleware{mw},
		},
		executionOptions: executionOptions{
			Stream: streamFunc,
		},
		documentOptions: documentOptions{
			Documents: []*Document{doc},
		},
	}

	if diff := cmp.Diff(expected, opts,
		cmpopts.IgnoreFields(commonGenOptions{}, "MessagesFn", "Use"),
		cmpopts.IgnoreFields(executionOptions{}, "Stream"),
		cmpopts.IgnoreUnexported(mockModel{}, mockTool{}),
		cmp.AllowUnexported(promptExecutionOptions{}, commonGenOptions{},
			executionOptions{})); diff != "" {
		t.Errorf("Options not applied correctly, diff (-want +got):\n%s", diff)
	}

	if opts.MessagesFn == nil {
		t.Errorf("MessagesFn should not be nil")
	}
	if opts.Use == nil {
		t.Errorf("Use should not be nil")
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

func (m *mockModel) modelArg() {}

func (m *mockModel) Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	return nil, nil
}

type mockTool struct {
	name string
}

func (t *mockTool) Name() string {
	return t.name
}

func (t *mockTool) toolArg() {}

func (t *mockTool) Definition() *ToolDefinition {
	return &ToolDefinition{Name: t.name}
}

func (t *mockTool) RunRaw(ctx context.Context, input any) (*MultipartToolResponse, error) {
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
