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

func TestCommonOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []CommonGenOption
		wantErr bool
	}{
		{
			name: "valid options",
			opts: []CommonGenOption{
				WithMessages(NewUserTextMessage("test")),
				WithConfig(&GenerationCommonConfig{Temperature: 0.7}),
				WithModel(&mockModel{name: "test/model"}),
				WithTools(&mockTool{name: "test/tool"}),
				WithToolChoice(ToolChoiceAuto),
				WithMaxTurns(3),
				WithReturnToolRequests(true),
				WithMiddleware(func(next ModelFunc) ModelFunc { return next }),
			},
			wantErr: false,
		},
		{
			name: "mutually exclusive - messages",
			opts: []CommonGenOption{
				WithMessages(NewUserTextMessage("test")),
				WithMessagesFn(func(context.Context, any) ([]*Message, error) { return nil, nil }),
			},
			wantErr: true,
		},
		{
			name: "mutually exclusive - model",
			opts: []CommonGenOption{
				WithModel(&mockModel{name: "test/model"}),
				WithModelName("test/model"),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			genOpts := &generateOptions{}
			promptOpts := &promptOptions{}
			pgOpts := &promptExecutionOptions{}

			var err error
			for _, opt := range tt.opts {
				err = opt.applyGenerate(genOpts)
				if err != nil {
					break
				}
			}

			if (err != nil) != tt.wantErr {
				t.Errorf("applyGenerate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			for _, opt := range tt.opts {
				if err = opt.applyPrompt(promptOpts); err != nil {
					t.Errorf("applyPrompt() unexpected error = %v", err)
					return
				}
			}

			for _, opt := range tt.opts {
				if err = opt.applyPromptExecute(pgOpts); err != nil {
					t.Errorf("applyPromptExecute() unexpected error = %v", err)
					return
				}
			}
		})
	}
}

func TestPromptOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []PromptOption
		wantErr bool
	}{
		{
			name: "valid options",
			opts: []PromptOption{
				WithDescription("test description"),
				WithMetadata(map[string]any{"key": "value"}),
				WithInputType(struct {
					Test string `json:"test"`
				}{}),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := &promptOptions{}
			var err error
			for _, opt := range tt.opts {
				err = opt.applyPrompt(opts)
				if err != nil {
					break
				}
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("applyPrompt() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPromptingOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []PromptingOption
		wantErr bool
	}{
		{
			name: "valid options",
			opts: []PromptingOption{
				WithSystem("system instruction"),
				WithPrompt("user prompt"),
			},
			wantErr: false,
		},
		{
			name: "mutually exclusive - system",
			opts: []PromptingOption{
				WithSystem("system instruction"),
				WithSystemFn(func(context.Context, any) (string, error) { return "system", nil }),
			},
			wantErr: true,
		},
		{
			name: "mutually exclusive - prompt",
			opts: []PromptingOption{
				WithPrompt("user prompt"),
				WithPromptFn(func(context.Context, any) (string, error) { return "prompt", nil }),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			genOpts := &generateOptions{}
			promptOpts := &promptOptions{}

			var err error
			for _, opt := range tt.opts {
				err = opt.applyGenerate(genOpts)
				if err != nil {
					break
				}
			}

			if (err != nil) != tt.wantErr {
				t.Errorf("applyGenerate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			for _, opt := range tt.opts {
				if err = opt.applyPrompt(promptOpts); err != nil {
					t.Errorf("applyPrompt() unexpected error = %v", err)
					return
				}
			}
		})
	}
}

func TestOutputOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []OutputOption
		wantErr bool
	}{
		{
			name: "valid - output type",
			opts: []OutputOption{
				WithOutputType(struct {
					Test string `json:"test"`
				}{}),
			},
			wantErr: false,
		},
		{
			name: "valid - output format",
			opts: []OutputOption{
				WithOutputFormat(OutputFormatText),
			},
			wantErr: false,
		},
		{
			name: "valid - output instruction",
			opts: []OutputOption{
				WithOutputInstructions(""),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			genOpts := &generateOptions{}
			promptOpts := &promptOptions{}

			var err error
			for _, opt := range tt.opts {
				err = opt.applyGenerate(genOpts)
				if err != nil {
					break
				}
			}

			if (err != nil) != tt.wantErr {
				t.Errorf("applyGenerate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			for _, opt := range tt.opts {
				if err = opt.applyPrompt(promptOpts); err != nil {
					t.Errorf("applyPrompt() unexpected error = %v", err)
					return
				}
			}
		})
	}
}

func TestExecutionOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []ExecutionOption
		wantErr bool
	}{
		{
			name: "valid options",
			opts: []ExecutionOption{
				WithStreaming(func(context.Context, *ModelResponseChunk) error { return nil }),
			},
			wantErr: false,
		},
		{
			name: "duplicate - streaming",
			opts: []ExecutionOption{
				WithStreaming(func(context.Context, *ModelResponseChunk) error { return nil }),
				WithStreaming(func(context.Context, *ModelResponseChunk) error { return nil }),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			genOpts := &generateOptions{}
			pgOpts := &promptExecutionOptions{}

			var err error
			for _, opt := range tt.opts {
				err = opt.applyGenerate(genOpts)
				if err != nil {
					break
				}
			}

			if (err != nil) != tt.wantErr {
				t.Errorf("applyGenerate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			for _, opt := range tt.opts {
				if err = opt.applyPromptExecute(pgOpts); err != nil {
					t.Errorf("applyPromptExecute() unexpected error = %v", err)
					return
				}
			}
		})
	}
}

func TestPromptGenerateOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []PromptExecuteOption
		wantErr bool
	}{
		{
			name: "valid options",
			opts: []PromptExecuteOption{
				WithInput(map[string]string{"key": "value"}),
			},
			wantErr: false,
		},
		{
			name: "duplicate - input",
			opts: []PromptExecuteOption{
				WithInput("input1"),
				WithInput("input2"),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := &promptExecutionOptions{}
			var err error
			for _, opt := range tt.opts {
				err = opt.applyPromptExecute(opts)
				if err != nil {
					break
				}
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("applyPromptExecute() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
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
		if err := opt.applyGenerate(opts); err != nil {
			t.Fatalf("Failed to apply option: %v", err)
		}
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
		if err := opt.applyPrompt(opts); err != nil {
			t.Fatalf("Failed to apply option: %v", err)
		}
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
		if err := opt.applyPromptExecute(opts); err != nil {
			t.Fatalf("Failed to apply option: %v", err)
		}
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

		if err := opt.applyPrompt(opts); err != nil {
			t.Fatalf("applyPrompt() error: %v", err)
		}

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

		if err := opt.applyGenerate(opts); err != nil {
			t.Fatalf("applyGenerate() error: %v", err)
		}

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

		if err := opt.applyGenerate(opts); err != nil {
			t.Fatalf("applyGenerate() error: %v", err)
		}

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

		if err := opt.applyGenerate(opts); err != nil {
			t.Fatalf("applyGenerate() error: %v", err)
		}

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

		if err := opt.applyEvaluator(opts); err != nil {
			t.Fatalf("applyEvaluator() error: %v", err)
		}

		if opts.Evaluator == nil {
			t.Fatal("Evaluator is nil")
		}
		if opts.Evaluator.Name() != "test/myEvaluator" {
			t.Errorf("Evaluator.Name() = %q, want %q", opts.Evaluator.Name(), "test/myEvaluator")
		}
	})
}
