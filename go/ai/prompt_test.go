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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

type InputOutput struct {
	Text string `json:"text"`
}

func testTool(reg api.Registry, name string) Tool {
	return DefineTool(reg, name, "use when need to execute a test",
		func(ctx *ToolContext, input struct {
			Test string
		},
		) (string, error) {
			return input.Test, nil
		},
	)
}

func TestOutputFormat(t *testing.T) {
	tests := []struct {
		name   string
		output any
		format string
		err    bool
	}{
		{
			name:   "mismatch",
			output: InputOutput{},
			format: OutputFormatText,
			err:    true,
		},
		{
			name:   "json",
			output: InputOutput{},
			format: OutputFormatJSON,
			err:    false,
		},
		{
			name:   "text",
			format: OutputFormatText,
			err:    false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reg := registry.New()

			if test.output == nil {
				DefinePrompt(
					reg, "aModel",
					WithInputType(InputOutput{Text: "test"}),
					WithOutputFormat(test.format),
				)
			} else {
				DefinePrompt(
					reg, "bModel",
					WithInputType(InputOutput{Text: "test"}),
					WithOutputType(test.output),
					WithOutputFormat(test.format),
				)
			}
		})
	}
}

func TestInputFormat(t *testing.T) {
	reg := registry.New()

	type hello struct {
		Name string `json:"name"`
	}

	tests := []struct {
		name         string
		templateText string
		inputType    any
		input        map[string]any
		render       string
	}{
		{
			name:         "structInput",
			templateText: "hello {{name}}",
			inputType:    hello{},
			input:        map[string]any{"name": "world"},
			render:       "hello world",
		},
		{
			name:         "mapInput",
			templateText: "hello {{name}}",
			inputType:    map[string]any{"name": "world"},
			input:        map[string]any{"name": "world"},
			render:       "hello world",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var err error
			var p Prompt

			if test.inputType != nil {
				p = DefinePrompt(reg, test.name, WithPrompt(test.templateText), WithInputType(test.inputType))
			} else {
				p = DefinePrompt(reg, test.name, WithPrompt(test.templateText))
			}

			req, err := p.Render(context.Background(), test.input)
			if err != nil {
				t.Fatal(err)
			}

			if req.Messages[0].Content[0].Text != test.render {
				t.Errorf("got %q want %q", req.Messages[0].Content[0].Text, test.render)
			}
		})
	}
}

type HelloPromptInput struct {
	Name string
}

func definePromptModel(reg api.Registry) Model {
	return DefineModel(reg, "test/chat",
		&ModelOptions{Supports: &ModelSupports{
			Tools:      true,
			Multiturn:  true,
			ToolChoice: true,
			SystemRole: true,
		}},
		func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
			toolCalled := false
			for _, msg := range gr.Messages {
				if msg.Content[0].IsToolResponse() {
					toolCalled = true
				}
			}

			if !toolCalled && len(gr.Tools) == 1 {
				part := NewToolRequestPart(&ToolRequest{
					Name:  "testTool",
					Input: map[string]any{"Test": "Bar"},
				})

				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role:    RoleModel,
						Content: []*Part{part},
					},
				}, nil
			}

			if msc != nil {
				msc(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("3!")},
				})
				msc(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("2!")},
				})
				msc(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("1!")},
				})
			}

			textResponse := ""
			var contentTexts []string
			for _, m := range gr.Messages {
				if m.Role != RoleUser && m.Role != RoleModel {
					textResponse += fmt.Sprintf("%s: ", m.Role)
				}

				if m.Role == RoleTool {
					contentTexts = append(contentTexts, m.Content[0].ToolResponse.Output.(string))
				}

				for _, c := range m.Content {
					contentTexts = append(contentTexts, c.Text)
				}
			}

			textResponse += strings.Join(contentTexts, "; ")
			textResponse += "; config: " + base.PrettyJSONString(gr.Config)
			textResponse += "; context: " + base.PrettyJSONString(gr.Docs)

			return &ModelResponse{
				Request: gr,
				Message: NewModelTextMessage(fmt.Sprintf("Echo: %s", textResponse)),
			}, nil
		})
}

func TestValidPrompt(t *testing.T) {
	reg := registry.New()

	ConfigureFormats(reg)

	model := definePromptModel(reg)

	tests := []struct {
		name           string
		model          Model
		systemText     string
		systemFn       PromptFn
		promptText     string
		promptFn       PromptFn
		messages       []*Message
		messagesFn     MessagesFn
		tools          []ToolRef
		config         *GenerationCommonConfig
		inputType      any
		input          any
		executeOptions []PromptExecuteOption
		wantTextOutput string
		wantGenerated  *ModelRequest
		state          any
		only           bool
	}{
		{
			name:       "user and system prompt, basic",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is foo",
			input:      HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
			},
			wantTextOutput: "Echo: system: say hello; my name is foo; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name:      "user and system prompt, functions",
			model:     model,
			config:    &GenerationCommonConfig{Temperature: 11},
			inputType: HelloPromptInput{},
			systemFn: func(ctx context.Context, input any) (string, error) {
				return "say hello to {{Name}}", nil
			},
			promptFn: func(ctx context.Context, input any) (string, error) {
				return "my name is {{Name}}", nil
			},
			input: HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
			},
			wantTextOutput: "Echo: system: say hello to foo; my name is foo; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello to foo")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name:       "messages prompt, basic",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is foo",
			messages: []*Message{
				{
					Role:    RoleUser,
					Content: []*Part{NewTextPart("you're history")},
				},
			},
			input: HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
			},
			wantTextOutput: "Echo: system: say hello; you're history; my name is foo; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("you're history")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name:       "messages prompt, function",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is foo",
			messagesFn: func(ctx context.Context, input any) ([]*Message, error) {
				return []*Message{
					{
						Role:    RoleModel,
						Content: []*Part{NewTextPart("your name is {{Name}}")},
					},
				}, nil
			},
			input: HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
			},
			wantTextOutput: "Echo: system: say hello; your name is foo; my name is foo; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleModel,
						Content: []*Part{NewTextPart("your name is foo")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name:       "messages prompt, input struct",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is foo",
			messagesFn: func(ctx context.Context, input any) ([]*Message, error) {
				var p HelloPromptInput
				switch param := input.(type) {
				case HelloPromptInput:
					p = param
				}
				return []*Message{
					{
						Role:    RoleModel,
						Content: []*Part{NewTextPart(fmt.Sprintf("your name is %s", p.Name))},
					},
				}, nil
			},
			input: HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
			},
			wantTextOutput: "Echo: system: say hello; your name is foo; my name is foo; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleModel,
						Content: []*Part{NewTextPart("your name is foo")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name:       "prompt with tools",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is foo",
			tools:      []ToolRef{testTool(reg, "testTool")},
			input:      HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
			},
			wantTextOutput: "Echo: system: tool: say hello; my name is foo; ; Bar; ; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
					{
						Role:    RoleModel,
						Content: []*Part{NewToolRequestPart(&ToolRequest{Name: "testTool", Input: map[string]any{"Test": "Bar"}})},
					},
					{
						Role:    RoleTool,
						Content: []*Part{NewToolResponsePart(&ToolResponse{Output: "Bar"})},
					},
				},
				Tools: []*ToolDefinition{
					{
						Name:        "testTool",
						Description: "use when need to execute a test",
						InputSchema: map[string]any{
							"additionalProperties": bool(false),
							"properties":           map[string]any{"Test": map[string]any{"type": string("string")}},
							"required":             []any{string("Test")},
							"type":                 string("object"),
						},
						OutputSchema: map[string]any{"type": string("string")},
						Metadata: map[string]any{
							"multipart": false,
						},
					},
				},
			},
		},
		{
			name:       "execute with MessagesFn option",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is {{Name}}",
			input:      HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
				WithMessages(NewModelTextMessage("I remember you said your name is {{Name}}")),
			},
			wantTextOutput: "Echo: system: say hello; I remember you said your name is foo; my name is foo; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleModel,
						Content: []*Part{NewTextPart("I remember you said your name is foo")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name:       "execute with tools overriding prompt-level tools",
			model:      model,
			config:     &GenerationCommonConfig{Temperature: 11},
			inputType:  HelloPromptInput{},
			systemText: "say hello",
			promptText: "my name is foo",
			tools:      []ToolRef{testTool(reg, "promptTool")},
			input:      HelloPromptInput{Name: "foo"},
			executeOptions: []PromptExecuteOption{
				WithInput(HelloPromptInput{Name: "foo"}),
				WithTools(testTool(reg, "executeOverrideTool")),
			},
			wantTextOutput: "Echo: system: tool: say hello; my name is foo; ; Bar; ; config: {\n  \"temperature\": 11\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 11,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				ToolChoice: "required",
				Messages: []*Message{
					{
						Role:    RoleSystem,
						Content: []*Part{NewTextPart("say hello")},
					},
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
					{
						Role:    RoleModel,
						Content: []*Part{NewToolRequestPart(&ToolRequest{Name: "executeOverrideTool", Input: map[string]any{"Test": "Bar"}})},
					},
					{
						Role:    RoleTool,
						Content: []*Part{NewToolResponsePart(&ToolResponse{Output: "Bar"})},
					},
				},
				Tools: []*ToolDefinition{
					{
						Name:        "executeOverrideTool",
						Description: "use when need to execute a test",
						InputSchema: map[string]any{
							"additionalProperties": bool(false),
							"properties":           map[string]any{"Test": map[string]any{"type": string("string")}},
							"required":             []any{string("Test")},
							"type":                 string("object"),
						},
						OutputSchema: map[string]any{"type": string("string")},
						Metadata: map[string]any{
							"multipart": false,
						},
					},
				},
			},
		},
	}

	cmpPart := func(a, b *Part) bool {
		if a.IsText() != b.IsText() {
			return false
		}
		if a.Text != b.Text {
			return false
		}
		if a.ContentType != b.ContentType {
			return false
		}
		return true
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var opts []PromptOption
			opts = append(opts, WithInputType(test.inputType))
			opts = append(opts, WithModel(test.model))
			opts = append(opts, WithConfig(test.config))
			opts = append(opts, WithToolChoice(ToolChoiceRequired))
			opts = append(opts, WithTools(test.tools...))
			opts = append(opts, WithMaxTurns(1))

			if test.systemText != "" {
				opts = append(opts, WithSystem(test.systemText))
			}
			if test.systemFn != nil {
				opts = append(opts, WithSystemFn(test.systemFn))
			}
			if test.messages != nil {
				opts = append(opts, WithMessages(test.messages...))
			}
			if test.messagesFn != nil {
				opts = append(opts, WithMessagesFn(test.messagesFn))
			}
			if test.promptText != "" {
				opts = append(opts, WithPrompt(test.promptText))
			}
			if test.promptFn != nil {
				opts = append(opts, WithPromptFn(test.promptFn))
			}

			p := DefinePrompt(reg, test.name, opts...)

			output, err := p.Execute(context.Background(), test.executeOptions...)
			if err != nil {
				t.Fatal(err)
			}

			if output.Text() != test.wantTextOutput {
				t.Errorf("got %q want %q", output.Text(), test.wantTextOutput)
			}

			if diff := cmp.Diff(test.wantGenerated, output.Request, cmp.Comparer(cmpPart), cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("mismatch (-want, +got):\n%s", diff)
			}
		})
	}
}

func testGenerate(ctx context.Context, req *ModelRequest, cb func(context.Context, *ModelResponseChunk) error) (*ModelResponse, error) {
	input := req.Messages[0].Content[0].Text
	output := fmt.Sprintf("AI reply to %q", input)

	if req.Output.Format == "json" {
		output = `{"text": "AI reply to JSON"}`
	}

	if cb != nil {
		cb(ctx, &ModelResponseChunk{
			Content: []*Part{NewTextPart("stream!")},
		})
	}

	r := &ModelResponse{
		Message: &Message{
			Content: []*Part{
				NewTextPart(output),
			},
		},
		Request: req,
	}
	return r, nil
}

func TestOptionsPatternExecute(t *testing.T) {
	reg := registry.New()

	ConfigureFormats(reg)

	testModel := DefineModel(reg, "options/test", nil, testGenerate)

	t.Run("Streaming", func(t *testing.T) {
		p := DefinePrompt(reg, "TestExecute", WithInputType(InputOutput{}), WithPrompt("TestExecute"))

		streamText := ""
		resp, err := p.Execute(
			context.Background(),
			WithInput(InputOutput{
				Text: "TestExecute",
			}),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
			WithModel(testModel),
			WithDocs(&Document{Content: []*Part{NewTextPart("Banana")}}),
		)
		if err != nil {
			t.Fatal(err)
		}

		assertResponse(t, resp, `AI reply to "TestExecute"`)
		if diff := cmp.Diff(streamText, "stream!"); diff != "" {
			t.Errorf("Text() diff (+got -want):\n%s", diff)
		}
	})

	t.Run("WithModelName", func(t *testing.T) {
		p := DefinePrompt(reg, "TestModelname", WithInputType(InputOutput{}), WithPrompt("TestModelname"))

		resp, err := p.Execute(
			context.Background(),
			WithInput(InputOutput{
				Text: "testing",
			}),
			WithModelName("options/test"),
		)
		if err != nil {
			t.Fatal(err)
		}

		assertResponse(t, resp, `AI reply to "TestModelname"`)
	})
}

func TestDefaultsOverride(t *testing.T) {
	reg := registry.New()

	// Set up default formats
	ConfigureFormats(reg)

	testModel := DefineModel(reg, "defineoptions/test", nil, testGenerate)
	model := definePromptModel(reg)

	tests := []struct {
		name           string
		define         []PromptOption
		execute        []PromptExecuteOption
		wantTextOutput string
		wantGenerated  *ModelRequest
	}{
		{
			name: "Config",
			define: []PromptOption{
				WithPrompt("my name is foo"),
				WithConfig(&GenerationCommonConfig{Temperature: 11}),
				WithModel(model),
			},
			execute: []PromptExecuteOption{
				WithConfig(&GenerationCommonConfig{Temperature: 12}),
			},
			wantTextOutput: "Echo: my name is foo; config: {\n  \"temperature\": 12\n}; context: null",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 12,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				Messages: []*Message{
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name: "Model",
			define: []PromptOption{
				WithPrompt("my name is foo"),
				WithModel(model),
			},
			execute: []PromptExecuteOption{
				WithConfig(&GenerationCommonConfig{Temperature: 12}),
				WithModel(testModel),
			},
			wantTextOutput: "AI reply to \"my name is foo\"",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 12,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				Messages: []*Message{
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
		{
			name: "ModelName",
			define: []PromptOption{
				WithPrompt("my name is foo"),
				WithModelName("test/chat"),
			},
			execute: []PromptExecuteOption{
				WithConfig(&GenerationCommonConfig{Temperature: 12}),
				WithModelName("defineoptions/test"),
			},
			wantTextOutput: "AI reply to \"my name is foo\"",
			wantGenerated: &ModelRequest{
				Config: &GenerationCommonConfig{
					Temperature: 12,
				},
				Output: &ModelOutputConfig{
					ContentType: "text/plain",
				},
				Messages: []*Message{
					{
						Role:    RoleUser,
						Content: []*Part{NewTextPart("my name is foo")},
					},
				},
			},
		},
	}

	cmpPart := func(a, b *Part) bool {
		if a.IsText() != b.IsText() {
			return false
		}
		if a.Text != b.Text {
			return false
		}
		if a.ContentType != b.ContentType {
			return false
		}
		return true
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := DefinePrompt(reg, test.name, test.define...)

			output, err := p.Execute(
				context.Background(),
				test.execute...,
			)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(test.wantGenerated, output.Request, cmp.Comparer(cmpPart), cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("mismatch (-want, +got):\n%s", diff)
			}

			if output.Text() != test.wantTextOutput {
				t.Errorf("got %q want %q", output.Text(), test.wantTextOutput)
			}
		})
	}
}

func assertResponse(t *testing.T, resp *ModelResponse, want string) {
	if resp.Message == nil {
		t.Fatal("response has candidate with no message")
	}
	if len(resp.Message.Content) != 1 {
		t.Errorf("got %d message parts, want 1", len(resp.Message.Content))
		if len(resp.Message.Content) < 1 {
			t.FailNow()
		}
	}
	got := resp.Message.Content[0].Text
	if got != want {
		t.Errorf("fake model replied with %q, want %q", got, want)
	}
}

func TestLoadPromptSnakeCase(t *testing.T) {
	tempDir := t.TempDir()
	mockPromptFile := filepath.Join(tempDir, "snake.prompt")
	mockPromptContent := `---
model: googleai/gemini-2.5-flash
input:
  schema:
    items(array):
      teamColor: string
      team_name: string
---
{{#each items as |it|}}
{{ it.teamColor }},{{ it.team_name }}
{{/each}}
`
	err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0o644)
	if err != nil {
		t.Fatalf("Failed to create mock prompt file: %v", err)
	}

	reg := registry.New()
	LoadPromptFromFS(reg, os.DirFS(tempDir), ".", "snake.prompt", "snake-namespace")

	prompt := LookupPrompt(reg, "snake-namespace/snake")
	if prompt == nil {
		t.Fatalf("prompt was not registered")
	}

	type SnakeInput struct {
		TeamColor string `json:"teamColor"` // intentionally leaving camel case to test snake + camel support
		TeamName  string `json:"team_name"`
	}

	input := map[string]any{"items": []SnakeInput{
		{TeamColor: "RED", TeamName: "Firebase"},
		{TeamColor: "BLUE", TeamName: "Gophers"},
		{TeamColor: "GREEN", TeamName: "Google"},
	}}

	actionOpts, err := prompt.Render(context.Background(), input)
	if err != nil {
		t.Fatalf("error rendering prompt: %v", err)
	}
	if actionOpts.Messages == nil {
		t.Fatal("expecting messages to be rendered")
	}
	renderedPrompt := actionOpts.Messages[0].Text()
	for line := range strings.SplitSeq(renderedPrompt, "\n") {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "RED") {
			if !strings.Contains(trimmedLine, "Firebase") {
				t.Fatalf("wrong template render, want: RED,Firebase, got: %s", trimmedLine)
			}
		} else if strings.HasPrefix(trimmedLine, "BLUE") {
			if !strings.Contains(trimmedLine, "Gophers") {
				t.Fatalf("wrong template render, want: BLUE,Gophers, got: %s", trimmedLine)
			}
		} else if strings.HasPrefix(trimmedLine, "GREEN") {
			if !strings.Contains(trimmedLine, "Google") {
				t.Fatalf("wrong template render, want: GREEN,Google, got: %s", trimmedLine)
			}
		}
	}
}

func TestLoadPrompt_FileNotFound(t *testing.T) {
	// Initialize a mock registry
	reg := registry.New()

	// Call loadPrompt with a non-existent file in a valid temp directory
	tempDir := t.TempDir()
	LoadPromptFromFS(reg, os.DirFS(tempDir), ".", "missing.prompt", "test-namespace")

	// Verify that the prompt was not registered
	prompt := LookupPrompt(reg, "missing")
	if prompt != nil {
		t.Fatalf("Prompt should not have been registered for a missing file")
	}
}

func TestLoadPrompt_InvalidPromptFile(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Create an invalid .prompt file
	invalidPromptFile := filepath.Join(tempDir, "invalid.prompt")
	invalidPromptContent := `invalid json content`
	err := os.WriteFile(invalidPromptFile, []byte(invalidPromptContent), 0o644)
	if err != nil {
		t.Fatalf("Failed to create invalid prompt file: %v", err)
	}

	// Initialize a mock registry
	reg := registry.New()

	// Call loadPrompt
	LoadPromptFromFS(reg, os.DirFS(tempDir), ".", "invalid.prompt", "test-namespace")

	// Verify that the prompt was not registered
	prompt := LookupPrompt(reg, "invalid")
	if prompt != nil {
		t.Fatalf("Prompt should not have been registered for an invalid file")
	}
}

func TestLoadPrompt_WithVariant(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Create a mock .prompt file with a variant
	mockPromptFile := filepath.Join(tempDir, "example.variant.prompt")
	mockPromptContent := `---
model: test-model
description: A test prompt
---

Hello, {{name}}!
`
	err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0o644)
	if err != nil {
		t.Fatalf("Failed to create mock prompt file: %v", err)
	}

	// Initialize a mock registry
	reg := registry.New()

	// Call loadPrompt
	LoadPromptFromFS(reg, os.DirFS(tempDir), ".", "example.variant.prompt", "test-namespace")

	// Verify that the prompt was registered correctly
	prompt := LookupPrompt(reg, "test-namespace/example.variant")
	if prompt == nil {
		t.Fatalf("Prompt was not registered")
	}

	// Verify that the metadata name does NOT include the variant
	promptMetadata, ok := prompt.(api.Action).Desc().Metadata["prompt"].(map[string]any)
	if !ok {
		t.Fatalf("Expected Metadata['prompt'] to be a map")
	}
	if promptMetadata["name"] != "test-namespace/example" {
		t.Errorf("Expected metadata name 'test-namespace/example', got '%s'", promptMetadata["name"])
	}
	if promptMetadata["variant"] != "variant" {
		t.Errorf("Expected variant 'variant', got '%s'", promptMetadata["variant"])
	}
}

func TestDefinePrompt_WithVariant(t *testing.T) {
	reg := registry.New()

	DefinePrompt(reg, "example.code", WithPrompt("Hello, {{name}}!"))

	prompt := LookupPrompt(reg, "example.code")
	if prompt == nil {
		t.Fatalf("Prompt was not registered")
	}

	promptMetadata, ok := prompt.(api.Action).Desc().Metadata["prompt"].(map[string]any)
	if !ok {
		t.Fatalf("Expected Metadata['prompt'] to be a map")
	}
	if promptMetadata["name"] != "example" {
		t.Errorf("Expected metadata name 'example', got '%s'", promptMetadata["name"])
	}
	if promptMetadata["variant"] != "code" {
		t.Errorf("Expected variant 'code', got '%v'", promptMetadata["variant"])
	}
}

func TestDefinePrompt_WithoutVariant(t *testing.T) {
	reg := registry.New()

	DefinePrompt(reg, "simple", WithPrompt("Hello, world!"))

	prompt := LookupPrompt(reg, "simple")
	if prompt == nil {
		t.Fatalf("Prompt was not registered")
	}

	promptMetadata, ok := prompt.(api.Action).Desc().Metadata["prompt"].(map[string]any)
	if !ok {
		t.Fatalf("Expected Metadata['prompt'] to be a map")
	}
	if promptMetadata["name"] != "simple" {
		t.Errorf("Expected metadata name 'simple', got '%s'", promptMetadata["name"])
	}
	if _, exists := promptMetadata["variant"]; exists {
		t.Errorf("Expected no variant for prompt without dot, got '%v'", promptMetadata["variant"])
	}
}

func TestLoadPromptFolder(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Create mock prompt and partial files
	mockPromptFile := filepath.Join(tempDir, "example.prompt")
	mockSubDir := filepath.Join(tempDir, "subdir")
	err := os.Mkdir(mockSubDir, 0o755)
	if err != nil {
		t.Fatalf("Failed to create subdirectory: %v", err)
	}

	mockPromptContent := `---
model: test-model
description: A test prompt
input:
  schema:
    type: object
    properties:
      name:
        type: string
output:
  format: text
  schema:
    type: string
---

Hello, {{name}}!
`

	err = os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0o644)
	if err != nil {
		t.Fatalf("Failed to create mock prompt file: %v", err)
	}

	// Create a mock prompt file in the subdirectory
	mockSubPromptFile := filepath.Join(mockSubDir, "sub_example.prompt")
	err = os.WriteFile(mockSubPromptFile, []byte(mockPromptContent), 0o644)
	if err != nil {
		t.Fatalf("Failed to create mock prompt file in subdirectory: %v", err)
	}

	// Initialize a mock registry
	reg := registry.New()

	// Call LoadPromptFolder
	LoadPromptDirFromFS(reg, os.DirFS(tempDir), ".", "test-namespace")

	// Verify that the prompt was registered correctly
	prompt := LookupPrompt(reg, "test-namespace/example")
	if prompt == nil {
		t.Fatalf("Prompt was not registered")
	}

	// Verify the prompt in the subdirectory was registered correctly
	subPrompt := LookupPrompt(reg, "test-namespace/sub_example")
	if subPrompt == nil {
		t.Fatalf("Prompt in subdirectory was not registered")
	}
}

func TestLoadPromptFolder_EmptyDirectory(t *testing.T) {
	// Initialize a mock registry
	reg := registry.New()

	// Create an empty temp directory
	tempDir := t.TempDir()

	// Call LoadPromptFolder with an empty directory
	LoadPromptDirFromFS(reg, os.DirFS(tempDir), ".", "test-namespace")

	// Verify that no prompts were registered
	if prompt := LookupPrompt(reg, "example"); prompt != nil {
		t.Fatalf("Prompt should not have been registered for an empty directory")
	}
}

func TestLoadPromptFS(t *testing.T) {
	mockPromptContent := `---
model: test/chat
description: A test prompt
input:
  schema:
    type: object
    properties:
      name:
        type: string
output:
  format: text
  schema:
    type: string
---

Hello, {{name}}!
`
	mockPartialContent := `Welcome {{name}}!`

	fsys := fstest.MapFS{
		"prompts/example.prompt":    &fstest.MapFile{Data: []byte(mockPromptContent)},
		"prompts/sub/nested.prompt": &fstest.MapFile{Data: []byte(mockPromptContent)},
		"prompts/_greeting.prompt":  &fstest.MapFile{Data: []byte(mockPartialContent)},
	}

	reg := registry.New()

	LoadPromptDirFromFS(reg, fsys, "prompts", "test-namespace")

	prompt := LookupPrompt(reg, "test-namespace/example")
	if prompt == nil {
		t.Fatalf("Prompt 'test-namespace/example' was not registered")
	}

	nestedPrompt := LookupPrompt(reg, "test-namespace/nested")
	if nestedPrompt == nil {
		t.Fatalf("Nested prompt 'test-namespace/nested' was not registered")
	}
}

func TestLoadPromptFS_WithVariant(t *testing.T) {
	mockPromptContent := `---
model: test/chat
description: A test prompt with variant
---

Hello from variant!
`

	fsys := fstest.MapFS{
		"prompts/greeting.experimental.prompt": &fstest.MapFile{Data: []byte(mockPromptContent)},
	}

	reg := registry.New()

	LoadPromptDirFromFS(reg, fsys, "prompts", "")

	prompt := LookupPrompt(reg, "greeting.experimental")
	if prompt == nil {
		t.Fatalf("Prompt with variant 'greeting.experimental' was not registered")
	}
}

func TestLoadPromptFS_NilFS(t *testing.T) {
	reg := registry.New()

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for nil filesystem")
		}
	}()

	LoadPromptDirFromFS(reg, nil, "prompts", "test-namespace")
}

func TestLoadPromptFS_InvalidRoot(t *testing.T) {
	fsys := fstest.MapFS{
		"other/example.prompt": &fstest.MapFile{Data: []byte("test")},
	}

	reg := registry.New()

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for invalid root directory")
		}
	}()

	LoadPromptDirFromFS(reg, fsys, "nonexistent", "test-namespace")
}

func TestLoadPromptFromFS(t *testing.T) {
	mockPromptContent := `---
model: test/chat
description: A single prompt test
---

Test content
`

	fsys := fstest.MapFS{
		"prompts/single.prompt": &fstest.MapFile{Data: []byte(mockPromptContent)},
	}

	reg := registry.New()

	prompt := LoadPromptFromFS(reg, fsys, "prompts", "single.prompt", "ns")
	if prompt == nil {
		t.Fatalf("LoadPromptFromFS failed to load prompt")
	}

	lookedUp := LookupPrompt(reg, "ns/single")
	if lookedUp == nil {
		t.Fatalf("Prompt 'ns/single' was not registered")
	}
}

func TestLoadPromptFromRaw(t *testing.T) {
	t.Run("basic prompt", func(t *testing.T) {
		reg := registry.New()

		source := `---
model: test/chat
description: A raw prompt test
input:
  schema:
    name: string
---
Hello, {{name}}!
`
		prompt, err := LoadPromptFromSource(reg, source, "rawPrompt", "test-ns")
		if err != nil {
			t.Fatalf("LoadPromptFromRaw failed: %v", err)
		}
		if prompt == nil {
			t.Fatal("LoadPromptFromRaw returned nil prompt")
		}

		lookedUp := LookupPrompt(reg, "test-ns/rawPrompt")
		if lookedUp == nil {
			t.Fatal("Prompt 'test-ns/rawPrompt' was not registered")
		}

		actionOpts, err := prompt.Render(context.Background(), map[string]any{"name": "World"})
		if err != nil {
			t.Fatalf("Render failed: %v", err)
		}
		if len(actionOpts.Messages) == 0 {
			t.Fatal("Expected messages to be rendered")
		}
		renderedText := actionOpts.Messages[0].Text()
		if renderedText != "Hello, World!" {
			t.Errorf("Expected 'Hello, World!', got %q", renderedText)
		}
	})

	t.Run("prompt with variant", func(t *testing.T) {
		reg := registry.New()

		source := `---
model: test/chat
description: A variant prompt
---
Formal greeting
`
		prompt, err := LoadPromptFromSource(reg, source, "greeting.formal", "")
		if err != nil {
			t.Fatalf("LoadPromptFromRaw failed: %v", err)
		}
		if prompt == nil {
			t.Fatal("LoadPromptFromRaw returned nil prompt")
		}

		lookedUp := LookupPrompt(reg, "greeting.formal")
		if lookedUp == nil {
			t.Fatal("Prompt 'greeting.formal' was not registered")
		}

		promptMetadata, ok := lookedUp.(api.Action).Desc().Metadata["prompt"].(map[string]any)
		if !ok {
			t.Fatal("Expected Metadata['prompt'] to be a map")
		}
		if promptMetadata["name"] != "greeting" {
			t.Errorf("Expected metadata name 'greeting', got '%s'", promptMetadata["name"])
		}
		if promptMetadata["variant"] != "formal" {
			t.Errorf("Expected variant 'formal', got '%v'", promptMetadata["variant"])
		}
	})

	t.Run("prompt without namespace", func(t *testing.T) {
		reg := registry.New()

		source := `---
model: test/chat
---
Simple prompt
`
		prompt, err := LoadPromptFromSource(reg, source, "simple", "")
		if err != nil {
			t.Fatalf("LoadPromptFromRaw failed: %v", err)
		}
		if prompt == nil {
			t.Fatal("LoadPromptFromRaw returned nil prompt")
		}

		lookedUp := LookupPrompt(reg, "simple")
		if lookedUp == nil {
			t.Fatal("Prompt 'simple' was not registered")
		}
	})

	t.Run("prompt with inline output schema", func(t *testing.T) {
		reg := registry.New()
		ConfigureFormats(reg)

		source := `---
model: test/chat
output:
  format: json
  schema:
    type: object
    properties:
      title:
        type: string
      description:
        type: string
    required:
      - title
      - description
---
Generate something
`
		prompt, err := LoadPromptFromSource(reg, source, "outputSchemaPrompt", "")
		if err != nil {
			t.Fatalf("LoadPromptFromRaw failed: %v", err)
		}
		if prompt == nil {
			t.Fatal("LoadPromptFromRaw returned nil prompt")
		}

		actionOpts, err := prompt.Render(context.Background(), nil)
		if err != nil {
			t.Fatalf("Render failed: %v", err)
		}

		// Verify that the output config is set correctly
		if actionOpts.Output == nil {
			t.Fatal("Expected Output config to be set")
		}
		if actionOpts.Output.Format != OutputFormatJSON {
			t.Errorf("Expected output format 'json', got %q", actionOpts.Output.Format)
		}
		if actionOpts.Output.JsonSchema == nil {
			t.Fatal("Expected output JsonSchema to be set for inline schema")
		}

		// Verify the schema structure
		schema := actionOpts.Output.JsonSchema
		if schema["type"] != "object" {
			t.Errorf("Expected schema type 'object', got %v", schema["type"])
		}
		properties, ok := schema["properties"].(map[string]any)
		if !ok {
			t.Fatal("Expected schema properties to be a map")
		}
		if _, ok := properties["title"]; !ok {
			t.Error("Expected schema to have 'title' property")
		}
		if _, ok := properties["description"]; !ok {
			t.Error("Expected schema to have 'description' property")
		}
	})
}

// TestDefinePartialAndHelperJourney demonstrates a complete user journey for defining
// and using both partials and helpers.
func TestDefinePartialAndHelper(t *testing.T) {
	// Initialize a mock registry
	reg := registry.New()
	ConfigureFormats(reg)

	model := definePromptModel(reg)

	reg.RegisterPartial("header", "Welcome {{name}}!")
	reg.RegisterHelper("uppercase", func(s string) string {
		return strings.ToUpper(s)
	})

	p := DefinePrompt(
		reg,
		"test",
		WithPrompt(`{{> header}} {{uppercase greeting}}`),
		WithModel(model))

	result, err := p.Execute(context.Background(), WithInput(map[string]any{
		"name":     "User",
		"greeting": "hello",
	}))
	if err != nil {
		t.Fatalf("Failed to execute prompt: %v", err)
	}

	testOutput := "Welcome User!HELLO"
	if result.Request.Messages[0].Content[0].Text != testOutput {
		t.Errorf("got %q want %q", result.Request.Messages[0].Content[0].Text, testOutput)
	}
}

func TestMultiMessagesPrompt(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	mockPromptFile := filepath.Join(tempDir, "example.prompt")
	mockPromptContent := `---
model: test/chat
description: A test prompt
---
{{ role "system" }}

You are a pirate!

{{ role "user" }}
Hello!
`
	err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0o644)
	if err != nil {
		t.Fatalf("Failed to create mock prompt file: %v", err)
	}

	// Initialize a mock registry
	reg := registry.New()
	ConfigureFormats(reg)
	definePromptModel(reg)

	prompt := LoadPromptFromFS(reg, os.DirFS(tempDir), ".", "example.prompt", "multi-namespace")

	result, err := prompt.Execute(context.Background())
	if err != nil {
		t.Fatalf("Failed to execute prompt: %v", err)
	}

	// Check that we have exactly 2 messages (system and user)
	if len(result.Request.Messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(result.Request.Messages))
	}

	// Check first message (system role)
	systemMsg := result.Request.Messages[0]
	if systemMsg.Role != RoleSystem {
		t.Errorf("Expected first message role to be 'system', got '%s'", systemMsg.Role)
	}
	if strings.TrimSpace(systemMsg.Text()) != "You are a pirate!" {
		t.Errorf("Expected system message text to be 'You are a pirate!', got '%s'", systemMsg.Text())
	}

	// Check second message (user role)
	userMsg := result.Request.Messages[1]
	if userMsg.Role != RoleUser {
		t.Errorf("Expected second message role to be 'user', got '%s'", userMsg.Role)
	}
	if strings.TrimSpace(userMsg.Text()) != "Hello!" {
		t.Errorf("Expected user message text to be 'Hello!', got '%s'", userMsg.Text())
	}
}

func TestDeferredSchemaResolution(t *testing.T) {
	tempDir := t.TempDir()

	// prompt file that references a schema "Recipe"
	mockPromptFile := filepath.Join(tempDir, "deferred.prompt")
	mockPromptContent := `---
model: test-model
output:
  schema: Recipe
---
Generate a recipe for {{food}}.
`
	if err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0o644); err != nil {
		t.Fatalf("Failed to create mock prompt file: %v", err)
	}

	reg := registry.New()
	ConfigureFormats(reg)

	DefineModel(reg, "test-model", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		// Mock response that matches the expected schema structure
		return &ModelResponse{
			Message: NewModelTextMessage(`{"title": "Tacos", "ingredients": [{"name": "Tortilla", "quantity": "3"}]}`),
			Request: req,
		}, nil
	})

	// this should succeed and create a placeholder schema in the registry
	prompt := LoadPrompt(reg, tempDir, "deferred.prompt", "test")
	if prompt == nil {
		t.Fatal("Failed to load prompt with undefined schema")
	}

	// verify the prompt is loaded with a schema reference
	// the internal representation stores the schema with $ref for lazy resolution
	actionDef := prompt.(api.Action).Desc()
	outputSchema := actionDef.Metadata["prompt"].(map[string]any)["output"].(map[string]any)["schema"]
	if outputSchema == nil {
		t.Fatal("Output schema should not be nil")
	}
	schemaMap, ok := outputSchema.(map[string]any)
	if !ok {
		t.Fatalf("Expected output schema to be a map, got: %T", outputSchema)
	}
	ref, ok := schemaMap["$ref"].(string)
	if !ok {
		t.Fatalf("Expected output schema to have $ref, got: %v", schemaMap)
	}
	if ref != "genkit:Recipe" {
		t.Fatalf("Expected output schema $ref to be 'genkit:Recipe', got: %v", ref)
	}

	// define the "Recipe" schema (deferred resolution)
	core.DefineSchema(reg, "Recipe", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{"type": "string"},
			"ingredients": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name":     map[string]any{"type": "string"},
						"quantity": map[string]any{"type": "string"},
					},
					"required": []string{"name", "quantity"},
				},
			},
		},
		"required": []string{"title", "ingredients"},
	})

	// we should now resolve "Recipe" correctly
	resp, err := prompt.Execute(context.Background(), WithInput(map[string]any{"food": "tacos"}))
	if err != nil {
		t.Fatalf("Failed to execute prompt with deferred schema: %v", err)
	}

	if resp.Request.Output.Schema == nil {
		t.Fatal("Expected request to have a resolved output schema")
	}

	schema := resp.Request.Output.Schema
	if ref, ok := schema["$ref"].(string); ok {
		// Schema is a reference, resolve it from $defs
		defs, ok := schema["$defs"].(map[string]any)
		if !ok {
			t.Fatalf("Schema has $ref %q but no $defs", ref)
		}
		// Assuming ref is like "#/$defs/Recipe"
		parts := strings.Split(ref, "/")
		defName := parts[len(parts)-1]
		if def, ok := defs[defName].(map[string]any); ok {
			schema = def
		} else {
			t.Fatalf("Could not resolve definition for %q", defName)
		}
	}

	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("Resolved schema should have properties, got: %v", schema)
	}
	if _, ok := props["ingredients"]; !ok {
		t.Error("Resolved schema should have 'ingredients' property")
	}
}

func TestDeferredSchemaResolution_Missing(t *testing.T) {
	tempDir := t.TempDir()

	// prompt file that references a schema "MissingRecipe"
	mockPromptFile := filepath.Join(tempDir, "deferred_missing.prompt")
	mockPromptContent := `---
model: test-model
output:
  schema: MissingRecipe
---
Generate a recipe.
`
	if err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0o644); err != nil {
		t.Fatalf("Failed to create mock prompt file: %v", err)
	}

	reg := registry.New()
	ConfigureFormats(reg)

	DefineModel(reg, "test-model", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Message: NewModelTextMessage(`{}`),
			Request: req,
		}, nil
	})

	prompt := LoadPrompt(reg, tempDir, "deferred_missing.prompt", "test")
	if prompt == nil {
		t.Fatal("Failed to load prompt")
	}

	_, err := prompt.Execute(context.Background())
	if err == nil {
		t.Fatal("Expected error when executing prompt with missing schema")
	}
	// "schema \"MissingRecipe\" not found"
	if !strings.Contains(err.Error(), "schema \"MissingRecipe\" not found") {
		t.Errorf("Expected error 'schema \"MissingRecipe\" not found', got: %v", err)
	}
}

func TestWithOutputSchemaName_DefinePrompt(t *testing.T) {
	reg := registry.New()
	ConfigureFormats(reg)

	DefineModel(reg, "test-model", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Message: NewModelTextMessage(`{"foo": "bar"}`),
			Request: req,
		}, nil
	})

	// Define schema
	core.DefineSchema(reg, "FooSchema", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"foo": map[string]any{"type": "string"},
		},
	})

	// Define prompt using WithOutputSchemaName
	prompt := DefinePrompt(reg, "testPrompt",
		WithModelName("test-model"),
		WithPrompt("test"),
		WithOutputSchemaName("FooSchema"),
	)

	resp, err := prompt.Execute(context.Background())
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if resp.Request.Output.Schema == nil {
		t.Fatal("Expected output schema to be set")
	}
}

func TestWithOutputSchemaName_DefinePrompt_Missing(t *testing.T) {
	reg := registry.New()
	ConfigureFormats(reg)

	DefineModel(reg, "test-model", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Message: NewModelTextMessage(`{}`),
			Request: req,
		}, nil
	})

	// Define prompt using WithOutputSchemaName, but Schema is missing
	prompt := DefinePrompt(reg, "testPromptMissing",
		WithModelName("test-model"),
		WithPrompt("test"),
		WithOutputSchemaName("MissingSchema"),
	)

	_, err := prompt.Execute(context.Background())
	if err == nil {
		t.Fatal("Expected error when executing prompt with missing schema")
	}
	if !strings.Contains(err.Error(), "schema \"MissingSchema\" not found") {
		t.Errorf("Expected error 'schema \"MissingSchema\" not found', got: %v", err)
	}
}

func TestDataPromptExecute(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	type GreetingInput struct {
		Name string `json:"name"`
	}

	type GreetingOutput struct {
		Message string `json:"message"`
		Count   int    `json:"count"`
	}

	t.Run("typed input and output", func(t *testing.T) {
		var capturedInput any

		testModel := DefineModel(r, "test/dataPromptModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedInput = req.Messages[0].Text()
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"message":"Hello, Alice!","count":1}`)},
				},
			}, nil
		})

		dp := DefineDataPrompt[GreetingInput, GreetingOutput](r, "greetingPrompt",
			WithModel(testModel),
			WithPrompt("Greet {{name}}"),
		)

		output, resp, err := dp.Execute(context.Background(), GreetingInput{Name: "Alice"})
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}

		if capturedInput != "Greet Alice" {
			t.Errorf("expected input %q, got %q", "Greet Alice", capturedInput)
		}

		if output.Message != "Hello, Alice!" {
			t.Errorf("expected message %q, got %q", "Hello, Alice!", output.Message)
		}
		if output.Count != 1 {
			t.Errorf("expected count 1, got %d", output.Count)
		}
		if resp == nil {
			t.Error("expected response to be returned")
		}
	})

	t.Run("string output type", func(t *testing.T) {
		testModel := DefineModel(r, "test/stringDataPromptModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("Hello, World!"),
			}, nil
		})

		dp := DefineDataPrompt[GreetingInput, string](r, "stringOutputPrompt",
			WithModel(testModel),
			WithPrompt("Say hello to {{name}}"),
		)

		output, resp, err := dp.Execute(context.Background(), GreetingInput{Name: "World"})
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}

		if output != "Hello, World!" {
			t.Errorf("expected output %q, got %q", "Hello, World!", output)
		}
		if resp == nil {
			t.Error("expected response to be returned")
		}
	})

	t.Run("nil prompt returns error", func(t *testing.T) {
		var dp *DataPrompt[GreetingInput, GreetingOutput]

		_, _, err := dp.Execute(context.Background(), GreetingInput{Name: "test"})
		if err == nil {
			t.Error("expected error for nil prompt")
		}
	})

	t.Run("additional options passed through", func(t *testing.T) {
		var capturedConfig any

		testModel := DefineModel(r, "test/optionsDataPromptModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedConfig = req.Config
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"message":"test","count":0}`)},
				},
			}, nil
		})

		dp := DefineDataPrompt[GreetingInput, GreetingOutput](r, "optionsPrompt",
			WithModel(testModel),
			WithPrompt("Test {{name}}"),
		)

		_, _, err := dp.Execute(context.Background(), GreetingInput{Name: "test"},
			WithConfig(&GenerationCommonConfig{Temperature: 0.5}),
		)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}

		config, ok := capturedConfig.(*GenerationCommonConfig)
		if !ok {
			t.Fatalf("expected *GenerationCommonConfig, got %T", capturedConfig)
		}
		if config.Temperature != 0.5 {
			t.Errorf("expected temperature 0.5, got %v", config.Temperature)
		}
	})

	t.Run("returns error for invalid output parsing", func(t *testing.T) {
		testModel := DefineModel(r, "test/parseFailDataPromptModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("not valid json"),
			}, nil
		})

		dp := DefineDataPrompt[GreetingInput, GreetingOutput](r, "parseFailPrompt",
			WithModel(testModel),
			WithPrompt("Test {{name}}"),
		)

		_, _, err := dp.Execute(context.Background(), GreetingInput{Name: "test"})
		if err == nil {
			t.Error("expected error for invalid JSON output")
		}
	})
}

func TestDataPromptExecuteStream(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	type StreamInput struct {
		Topic string `json:"topic"`
	}

	type StreamOutput struct {
		Text  string `json:"text"`
		Index int    `json:"index"`
	}

	t.Run("typed streaming with struct output", func(t *testing.T) {
		testModel := DefineModel(r, "test/streamDataPromptModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewJSONPart(`{"text":"chunk1","index":1}`)},
				})
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewJSONPart(`{"text":"final","index":99}`)},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"text":"final","index":99}`)},
				},
			}, nil
		})

		dp := DefineDataPrompt[StreamInput, StreamOutput](r, "streamPrompt",
			WithModel(testModel),
			WithPrompt("Stream about {{topic}}"),
		)

		var chunks []StreamOutput
		var finalOutput StreamOutput
		var finalResponse *ModelResponse

		for val, err := range dp.ExecuteStream(context.Background(), StreamInput{Topic: "testing"}) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalOutput = val.Output
				finalResponse = val.Response
			} else {
				chunks = append(chunks, val.Chunk)
			}
		}

		if len(chunks) < 1 {
			t.Errorf("expected at least 1 chunk, got %d", len(chunks))
		}

		if finalOutput.Text != "final" || finalOutput.Index != 99 {
			t.Errorf("expected final {final, 99}, got %+v", finalOutput)
		}
		if finalResponse == nil {
			t.Error("expected final response")
		}
	})

	t.Run("string output streaming", func(t *testing.T) {
		testModel := DefineModel(r, "test/stringStreamDataPromptModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("First ")},
				})
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("Second")},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("First Second"),
			}, nil
		})

		dp := DefineDataPrompt[StreamInput, string](r, "stringStreamPrompt",
			WithModel(testModel),
			WithPrompt("Generate text about {{topic}}"),
		)

		var chunks []string
		var finalOutput string

		for val, err := range dp.ExecuteStream(context.Background(), StreamInput{Topic: "strings"}) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalOutput = val.Output
			} else {
				chunks = append(chunks, val.Chunk)
			}
		}

		if len(chunks) != 2 {
			t.Errorf("expected 2 chunks, got %d", len(chunks))
		}
		if chunks[0] != "First " {
			t.Errorf("chunk 0: expected %q, got %q", "First ", chunks[0])
		}
		if chunks[1] != "Second" {
			t.Errorf("chunk 1: expected %q, got %q", "Second", chunks[1])
		}

		if finalOutput != "First Second" {
			t.Errorf("expected final %q, got %q", "First Second", finalOutput)
		}
	})

	t.Run("nil prompt returns error", func(t *testing.T) {
		var dp *DataPrompt[StreamInput, StreamOutput]

		var receivedErr error
		for _, err := range dp.ExecuteStream(context.Background(), StreamInput{Topic: "test"}) {
			if err != nil {
				receivedErr = err
				break
			}
		}

		if receivedErr == nil {
			t.Error("expected error for nil prompt")
		}
	})

	t.Run("handles options passed at execute time", func(t *testing.T) {
		var capturedConfig any

		testModel := DefineModel(r, "test/optionsStreamModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedConfig = req.Config
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewJSONPart(`{"text":"chunk","index":1}`)},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"text":"done","index":2}`)},
				},
			}, nil
		})

		dp := DefineDataPrompt[StreamInput, StreamOutput](r, "optionsStreamPrompt",
			WithModel(testModel),
			WithPrompt("Test {{topic}}"),
		)

		for range dp.ExecuteStream(context.Background(), StreamInput{Topic: "options"},
			WithConfig(&GenerationCommonConfig{Temperature: 0.7}),
		) {
		}

		config, ok := capturedConfig.(*GenerationCommonConfig)
		if !ok {
			t.Fatalf("expected *GenerationCommonConfig, got %T", capturedConfig)
		}
		if config.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %v", config.Temperature)
		}
	})

	t.Run("propagates errors", func(t *testing.T) {
		expectedErr := errors.New("stream failed")

		testModel := DefineModel(r, "test/errorStreamDataPromptModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return nil, expectedErr
		})

		dp := DefineDataPrompt[StreamInput, StreamOutput](r, "errorStreamPrompt",
			WithModel(testModel),
			WithPrompt("Test {{topic}}"),
		)

		var receivedErr error
		for _, err := range dp.ExecuteStream(context.Background(), StreamInput{Topic: "error"}) {
			if err != nil {
				receivedErr = err
				break
			}
		}

		if receivedErr == nil {
			t.Error("expected error to be propagated")
		}
		if !errors.Is(receivedErr, expectedErr) {
			t.Errorf("expected error %v, got %v", expectedErr, receivedErr)
		}
	})
}

func TestPromptExecuteStream(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	t.Run("yields chunks then final response", func(t *testing.T) {
		chunkTexts := []string{"A", "B", "C"}

		testModel := DefineModel(r, "test/promptStreamModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				for _, text := range chunkTexts {
					cb(ctx, &ModelResponseChunk{
						Content: []*Part{NewTextPart(text)},
					})
				}
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("ABC"),
			}, nil
		})

		p := DefinePrompt(r, "streamTestPrompt",
			WithModel(testModel),
			WithPrompt("Test"),
		)

		var chunks []*ModelResponseChunk
		var finalResponse *ModelResponse

		for val, err := range p.ExecuteStream(context.Background()) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalResponse = val.Response
			} else {
				chunks = append(chunks, val.Chunk)
			}
		}

		if len(chunks) != 3 {
			t.Errorf("expected 3 chunks, got %d", len(chunks))
		}
		for i, chunk := range chunks {
			if chunk.Text() != chunkTexts[i] {
				t.Errorf("chunk %d: expected %q, got %q", i, chunkTexts[i], chunk.Text())
			}
		}

		if finalResponse == nil {
			t.Fatal("expected final response")
		}
		if finalResponse.Text() != "ABC" {
			t.Errorf("expected final text %q, got %q", "ABC", finalResponse.Text())
		}
	})

	t.Run("nil prompt returns error", func(t *testing.T) {
		var p *prompt

		var receivedErr error
		for _, err := range p.ExecuteStream(context.Background()) {
			if err != nil {
				receivedErr = err
				break
			}
		}

		if receivedErr == nil {
			t.Error("expected error for nil prompt")
		}
	})

	t.Run("handles execution options", func(t *testing.T) {
		var capturedConfig any

		testModel := DefineModel(r, "test/optionsPromptExecModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedConfig = req.Config
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("chunk")},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("done"),
			}, nil
		})

		p := DefinePrompt(r, "execOptionsTestPrompt",
			WithModel(testModel),
			WithPrompt("Test"),
		)

		for range p.ExecuteStream(context.Background(),
			WithConfig(&GenerationCommonConfig{Temperature: 0.9}),
		) {
		}

		config, ok := capturedConfig.(*GenerationCommonConfig)
		if !ok {
			t.Fatalf("expected *GenerationCommonConfig, got %T", capturedConfig)
		}
		if config.Temperature != 0.9 {
			t.Errorf("expected temperature 0.9, got %v", config.Temperature)
		}
	})
}

// TestSessionStateInjection tests that session state is automatically injected
// into prompt templates and accessible via {{@state.field}} syntax.
func TestSessionStateInjection(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)

	// Define a test state type
	type UserState struct {
		Name        string            `json:"name"`
		Preferences map[string]string `json:"preferences"`
	}

	t.Run("session state accessible in prompt template", func(t *testing.T) {
		var capturedPrompt string

		testModel := DefineModel(r, "test/sessionStateModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedPrompt = req.Messages[0].Text()
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("response"),
			}, nil
		})

		// Create a prompt that uses {{@state.name}} syntax
		p := DefinePrompt(r, "sessionStatePrompt",
			WithModel(testModel),
			WithPrompt("Hello {{@state.name}}, your theme is {{@state.preferences.theme}}"),
		)

		// Create a session with state
		ctx := context.Background()
		sess, err := session.New(ctx, session.WithInitialState(UserState{
			Name:        "Alice",
			Preferences: map[string]string{"theme": "dark"},
		}))
		if err != nil {
			t.Fatalf("Failed to create session: %v", err)
		}

		// Attach session to context
		ctx = session.NewContext(ctx, sess)

		// Execute prompt with session in context
		_, err = p.Execute(ctx)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}

		// Verify the session state was injected into the template
		expected := "Hello Alice, your theme is dark"
		if capturedPrompt != expected {
			t.Errorf("Expected prompt %q, got %q", expected, capturedPrompt)
		}
	})

	t.Run("prompt works without session in context", func(t *testing.T) {
		var capturedPrompt string

		testModel := DefineModel(r, "test/noSessionModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedPrompt = req.Messages[0].Text()
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("response"),
			}, nil
		})

		// Create a prompt that uses regular input variables (not session state)
		p := DefinePrompt(r, "noSessionPrompt",
			WithModel(testModel),
			WithPrompt("Hello {{name}}"),
			WithInputType(struct {
				Name string `json:"name"`
			}{}),
		)

		// Execute without session in context
		ctx := context.Background()
		_, err := p.Execute(ctx, WithInput(map[string]any{"name": "Bob"}))
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}

		expected := "Hello Bob"
		if capturedPrompt != expected {
			t.Errorf("Expected prompt %q, got %q", expected, capturedPrompt)
		}
	})

	t.Run("session state and input variables can be used together", func(t *testing.T) {
		var capturedPrompt string

		testModel := DefineModel(r, "test/mixedModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedPrompt = req.Messages[0].Text()
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("response"),
			}, nil
		})

		// Create a prompt that uses both input and session state
		p := DefinePrompt(r, "mixedPrompt",
			WithModel(testModel),
			WithPrompt("User {{@state.name}} asks: {{question}}"),
			WithInputType(struct {
				Question string `json:"question"`
			}{}),
		)

		// Create session
		ctx := context.Background()
		sess, err := session.New(ctx, session.WithInitialState(UserState{
			Name: "Charlie",
		}))
		if err != nil {
			t.Fatalf("Failed to create session: %v", err)
		}
		ctx = session.NewContext(ctx, sess)

		// Execute with both session and input
		_, err = p.Execute(ctx, WithInput(map[string]any{"question": "What is the weather?"}))
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}

		expected := "User Charlie asks: What is the weather?"
		if capturedPrompt != expected {
			t.Errorf("Expected prompt %q, got %q", expected, capturedPrompt)
		}
	})
}

// TestDefineExecuteOptionInteractions tests the complex interactions between
// options set at DefinePrompt time vs Execute time.
func TestDefineExecuteOptionInteractions(t *testing.T) {
	t.Run("ToolChoice override", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/toolChoiceModel",
			handler: capturingModelHandler(&captured),
		})

		tool := defineFakeTool(t, r, "testTool", "a test tool")

		// Define with ToolChoiceAuto
		p := DefinePrompt(r, "toolChoicePrompt",
			WithModel(model),
			WithPrompt("test"),
			WithTools(tool),
			WithToolChoice(ToolChoiceAuto),
			WithMaxTurns(1),
		)

		// Execute with ToolChoiceRequired - should override
		_, err := p.Execute(context.Background(),
			WithToolChoice(ToolChoiceRequired),
		)
		assertNoError(t, err)

		if captured.ToolChoice != ToolChoiceRequired {
			t.Errorf("ToolChoice = %q, want %q", captured.ToolChoice, ToolChoiceRequired)
		}
	})

	t.Run("ToolChoice no override when not specified at execute", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/toolChoiceNoOverride",
			handler: capturingModelHandler(&captured),
		})

		tool := defineFakeTool(t, r, "testTool2", "a test tool")

		// Define with ToolChoiceRequired
		p := DefinePrompt(r, "toolChoiceNoOverridePrompt",
			WithModel(model),
			WithPrompt("test"),
			WithTools(tool),
			WithToolChoice(ToolChoiceRequired),
			WithMaxTurns(1),
		)

		// Execute without specifying ToolChoice - should use define-time value
		_, err := p.Execute(context.Background())
		assertNoError(t, err)

		if captured.ToolChoice != ToolChoiceRequired {
			t.Errorf("ToolChoice = %q, want %q", captured.ToolChoice, ToolChoiceRequired)
		}
	})

	t.Run("MaxTurns override", func(t *testing.T) {
		r := newTestRegistry(t)
		callCount := 0

		model := defineFakeModel(t, r, fakeModelConfig{
			name: "test/maxTurnsModel",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				callCount++
				// Always request tool call to test max turns
				if callCount < 10 {
					return &ModelResponse{
						Request: req,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{NewToolRequestPart(&ToolRequest{
								Name:  "maxTurnsTool",
								Input: map[string]any{"value": "test"},
							})},
						},
					}, nil
				}
				return &ModelResponse{
					Request: req,
					Message: NewModelTextMessage("done"),
				}, nil
			},
		})

		tool := defineFakeTool(t, r, "maxTurnsTool", "a tool for max turns test")

		// Define with MaxTurns 5
		p := DefinePrompt(r, "maxTurnsPrompt",
			WithModel(model),
			WithPrompt("test"),
			WithTools(tool),
			WithMaxTurns(5),
		)

		// Execute with MaxTurns 2 - should override and stop after 2 turns
		_, err := p.Execute(context.Background(),
			WithMaxTurns(2),
		)

		// Should error due to max turns exceeded
		if err == nil {
			t.Error("expected max turns error, got nil")
		}
		// Call count should be limited by execute-time MaxTurns (2) + 1 for initial
		if callCount > 3 {
			t.Errorf("callCount = %d, expected <= 3 (limited by execute MaxTurns)", callCount)
		}
	})

	t.Run("ReturnToolRequests override", func(t *testing.T) {
		r := newTestRegistry(t)

		model := defineFakeModel(t, r, fakeModelConfig{
			name: "test/returnToolReqsModel",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: req,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{NewToolRequestPart(&ToolRequest{
							Name:  "returnToolReqsTool",
							Input: map[string]any{"value": "test"},
						})},
					},
				}, nil
			},
		})

		tool := defineFakeTool(t, r, "returnToolReqsTool", "tool for return requests test")

		// Define with ReturnToolRequests false (default)
		p := DefinePrompt(r, "returnToolReqsPrompt",
			WithModel(model),
			WithPrompt("test"),
			WithTools(tool),
			WithReturnToolRequests(false),
			WithMaxTurns(1),
		)

		// Execute with ReturnToolRequests true - should override and return tool requests
		resp, err := p.Execute(context.Background(),
			WithReturnToolRequests(true),
		)
		assertNoError(t, err)

		// Should have tool request in response
		hasToolRequest := false
		for _, part := range resp.Message.Content {
			if part.IsToolRequest() {
				hasToolRequest = true
				break
			}
		}
		if !hasToolRequest {
			t.Error("expected tool request in response when ReturnToolRequests=true")
		}
	})

	t.Run("Tools complete replacement", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/toolsReplaceModel",
			handler: capturingModelHandler(&captured),
		})

		toolA := defineFakeTool(t, r, "toolA", "tool A")
		toolB := defineFakeTool(t, r, "toolB", "tool B")
		toolC := defineFakeTool(t, r, "toolC", "tool C")

		// Define with tools A and B
		p := DefinePrompt(r, "toolsReplacePrompt",
			WithModel(model),
			WithPrompt("test"),
			WithTools(toolA, toolB),
			WithMaxTurns(1),
		)

		// Execute with tool C - should REPLACE (not merge) define-time tools
		_, err := p.Execute(context.Background(),
			WithTools(toolC),
		)
		assertNoError(t, err)

		// Should only have tool C
		if len(captured.Tools) != 1 {
			t.Errorf("len(Tools) = %d, want 1", len(captured.Tools))
		}
		if len(captured.Tools) > 0 && captured.Tools[0].Name != "toolC" {
			t.Errorf("Tool name = %q, want %q", captured.Tools[0].Name, "toolC")
		}
	})

	t.Run("Tools inherit when not specified at execute", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/toolsInheritModel",
			handler: capturingModelHandler(&captured),
		})

		toolA := defineFakeTool(t, r, "toolInheritA", "tool A")
		toolB := defineFakeTool(t, r, "toolInheritB", "tool B")

		// Define with tools A and B
		p := DefinePrompt(r, "toolsInheritPrompt",
			WithModel(model),
			WithPrompt("test"),
			WithTools(toolA, toolB),
			WithMaxTurns(1),
		)

		// Execute without specifying tools - should inherit define-time tools
		_, err := p.Execute(context.Background())
		assertNoError(t, err)

		if len(captured.Tools) != 2 {
			t.Errorf("len(Tools) = %d, want 2", len(captured.Tools))
		}
	})

	t.Run("Docs at execute time", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/docsModel",
			handler: capturingModelHandler(&captured),
		})

		// Define without docs
		p := DefinePrompt(r, "docsPrompt",
			WithModel(model),
			WithPrompt("test"),
		)

		// Execute with docs
		doc := DocumentFromText("context document", nil)
		_, err := p.Execute(context.Background(),
			WithDocs(doc),
		)
		assertNoError(t, err)

		if len(captured.Docs) != 1 {
			t.Errorf("len(Docs) = %d, want 1", len(captured.Docs))
		}
	})

	t.Run("Config replacement not merge", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/configReplaceModel",
			handler: capturingModelHandler(&captured),
		})

		// Define with Temperature and TopK
		p := DefinePrompt(r, "configReplacePrompt",
			WithModel(model),
			WithPrompt("test"),
			WithConfig(&GenerationCommonConfig{Temperature: 0.5, TopK: 10}),
		)

		// Execute with only Temperature - config is REPLACED, not merged
		_, err := p.Execute(context.Background(),
			WithConfig(&GenerationCommonConfig{Temperature: 0.9}),
		)
		assertNoError(t, err)

		config, ok := captured.Config.(*GenerationCommonConfig)
		if !ok {
			t.Fatalf("Config type = %T, want *GenerationCommonConfig", captured.Config)
		}
		if config.Temperature != 0.9 {
			t.Errorf("Temperature = %v, want 0.9", config.Temperature)
		}
		// TopK should be zero (default) since config was replaced
		if config.TopK != 0 {
			t.Errorf("TopK = %v, want 0 (config replaced, not merged)", config.TopK)
		}
	})

	t.Run("Model override at execute time", func(t *testing.T) {
		r := newTestRegistry(t)

		defineModel := defineFakeModel(t, r, fakeModelConfig{
			name: "test/defineModel",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: req,
					Message: NewModelTextMessage("from define model"),
				}, nil
			},
		})

		executeModel := defineFakeModel(t, r, fakeModelConfig{
			name: "test/executeModel",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: req,
					Message: NewModelTextMessage("from execute model"),
				}, nil
			},
		})

		// Define with defineModel
		p := DefinePrompt(r, "modelOverridePrompt",
			WithModel(defineModel),
			WithPrompt("test"),
		)

		// Execute with executeModel - should use execute model
		resp, err := p.Execute(context.Background(),
			WithModel(executeModel),
		)
		assertNoError(t, err)

		if resp.Text() != "from execute model" {
			t.Errorf("response = %q, want %q", resp.Text(), "from execute model")
		}
	})

	t.Run("MessagesFn at execute time inserts between system and user", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		model := defineFakeModel(t, r, fakeModelConfig{
			name:    "test/messagesFnModel",
			handler: capturingModelHandler(&captured),
		})

		// Define with system and user prompt
		p := DefinePrompt(r, "messagesFnPrompt",
			WithModel(model),
			WithSystem("system instruction"),
			WithPrompt("user question"),
		)

		// Execute with MessagesFn - messages should be inserted between system and user
		_, err := p.Execute(context.Background(),
			WithMessages(NewModelTextMessage("conversation history")),
		)
		assertNoError(t, err)

		// Expected order: system, MessagesFn content, user
		if len(captured.Messages) != 3 {
			t.Fatalf("len(Messages) = %d, want 3", len(captured.Messages))
		}
		if captured.Messages[0].Role != RoleSystem {
			t.Errorf("Messages[0].Role = %q, want %q", captured.Messages[0].Role, RoleSystem)
		}
		if captured.Messages[1].Role != RoleModel {
			t.Errorf("Messages[1].Role = %q, want %q", captured.Messages[1].Role, RoleModel)
		}
		if captured.Messages[2].Role != RoleUser {
			t.Errorf("Messages[2].Role = %q, want %q", captured.Messages[2].Role, RoleUser)
		}
	})

	t.Run("ModelRef config used when no explicit config", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		// Define model first
		defineFakeModel(t, r, fakeModelConfig{
			name:    "test/modelRefConfigModel",
			handler: capturingModelHandler(&captured),
		})

		// Create ModelRef with embedded config
		modelRef := NewModelRef("test/modelRefConfigModel", &GenerationCommonConfig{Temperature: 0.7})

		p := DefinePrompt(r, "modelRefConfigPrompt",
			WithModel(modelRef),
			WithPrompt("test"),
		)

		// Execute without config - should use ModelRef's config
		_, err := p.Execute(context.Background())
		assertNoError(t, err)

		config, ok := captured.Config.(*GenerationCommonConfig)
		if !ok {
			t.Fatalf("Config type = %T, want *GenerationCommonConfig", captured.Config)
		}
		if config.Temperature != 0.7 {
			t.Errorf("Temperature = %v, want 0.7", config.Temperature)
		}
	})

	t.Run("Explicit config overrides ModelRef config", func(t *testing.T) {
		r := newTestRegistry(t)
		var captured *ModelRequest

		defineFakeModel(t, r, fakeModelConfig{
			name:    "test/modelRefOverrideModel",
			handler: capturingModelHandler(&captured),
		})

		modelRef := NewModelRef("test/modelRefOverrideModel", &GenerationCommonConfig{Temperature: 0.7})

		p := DefinePrompt(r, "modelRefOverridePrompt",
			WithModel(modelRef),
			WithPrompt("test"),
		)

		// Execute with explicit config - should override ModelRef's config
		_, err := p.Execute(context.Background(),
			WithConfig(&GenerationCommonConfig{Temperature: 0.3}),
		)
		assertNoError(t, err)

		config, ok := captured.Config.(*GenerationCommonConfig)
		if !ok {
			t.Fatalf("Config type = %T, want *GenerationCommonConfig", captured.Config)
		}
		if config.Temperature != 0.3 {
			t.Errorf("Temperature = %v, want 0.3", config.Temperature)
		}
	})
}

// TestPromptErrorPaths tests error handling in prompt operations.
func TestPromptErrorPaths(t *testing.T) {
	t.Run("DefinePrompt with empty name panics", func(t *testing.T) {
		r := newTestRegistry(t)
		assertPanic(t, func() {
			DefinePrompt(r, "")
		}, "name is required")
	})

	t.Run("Execute on nil prompt returns error", func(t *testing.T) {
		var p *prompt
		_, err := p.Execute(context.Background())
		assertError(t, err, "prompt is nil")
	})

	t.Run("Render on nil prompt returns error", func(t *testing.T) {
		var p *prompt
		_, err := p.Render(context.Background(), nil)
		assertError(t, err, "prompt is nil")
	})

	t.Run("ExecuteStream on nil prompt yields error", func(t *testing.T) {
		var p *prompt
		var gotErr error
		for _, err := range p.ExecuteStream(context.Background()) {
			if err != nil {
				gotErr = err
				break
			}
		}
		assertError(t, gotErr, "prompt is nil")
	})

	t.Run("buildVariables with invalid type returns error", func(t *testing.T) {
		// buildVariables expects struct, pointer to struct, or map
		_, err := buildVariables(42) // int is not valid
		if err == nil {
			t.Error("expected error for invalid type, got nil")
		}
	})
}

// TestLookupPromptCoverage tests LookupPrompt edge cases.
func TestLookupPromptCoverage(t *testing.T) {
	t.Run("returns nil for non-existent prompt", func(t *testing.T) {
		r := newTestRegistry(t)
		p := LookupPrompt(r, "nonexistent")
		if p != nil {
			t.Error("expected nil for non-existent prompt")
		}
	})

	t.Run("returns prompt for existing prompt", func(t *testing.T) {
		r := newTestRegistry(t)
		DefinePrompt(r, "existingPrompt", WithPrompt("hello"))
		p := LookupPrompt(r, "existingPrompt")
		if p == nil {
			t.Error("expected prompt, got nil")
		}
		if p.Name() != "existingPrompt" {
			t.Errorf("Name() = %q, want %q", p.Name(), "existingPrompt")
		}
	})
}

// TestDataPromptRender tests DataPrompt.Render method.
func TestDataPromptRender(t *testing.T) {
	r := newTestRegistry(t)

	type RenderInput struct {
		Name string `json:"name"`
	}

	type RenderOutput struct {
		Greeting string `json:"greeting"`
	}

	model := defineFakeModel(t, r, fakeModelConfig{
		name: "test/renderModel",
	})

	dp := DefineDataPrompt[RenderInput, RenderOutput](r, "renderPrompt",
		WithModel(model),
		WithPrompt("Hello {{name}}"),
	)

	t.Run("renders with typed input", func(t *testing.T) {
		opts, err := dp.Render(context.Background(), RenderInput{Name: "World"})
		assertNoError(t, err)

		if len(opts.Messages) == 0 {
			t.Fatal("expected messages")
		}
		if opts.Messages[0].Text() != "Hello World" {
			t.Errorf("rendered text = %q, want %q", opts.Messages[0].Text(), "Hello World")
		}
	})

	t.Run("nil DataPrompt returns error", func(t *testing.T) {
		var nilDP *DataPrompt[RenderInput, RenderOutput]
		_, err := nilDP.Render(context.Background(), RenderInput{})
		if err == nil {
			t.Error("expected error for nil DataPrompt")
		}
	})
}

// TestLookupDataPrompt tests LookupDataPrompt function.
func TestLookupDataPrompt(t *testing.T) {
	r := newTestRegistry(t)

	model := defineFakeModel(t, r, fakeModelConfig{
		name: "test/lookupDataModel",
	})

	DefinePrompt(r, "lookupDataPrompt",
		WithModel(model),
		WithPrompt("test"),
	)

	t.Run("returns DataPrompt for existing prompt", func(t *testing.T) {
		dp := LookupDataPrompt[map[string]any, string](r, "lookupDataPrompt")
		if dp == nil {
			t.Error("expected DataPrompt, got nil")
		}
	})

	t.Run("returns nil for non-existent prompt", func(t *testing.T) {
		dp := LookupDataPrompt[map[string]any, string](r, "nonexistent")
		if dp != nil {
			t.Error("expected nil for non-existent prompt")
		}
	})
}

// TestAsDataPrompt tests AsDataPrompt function.
func TestAsDataPrompt(t *testing.T) {
	r := newTestRegistry(t)

	model := defineFakeModel(t, r, fakeModelConfig{
		name: "test/asDataModel",
	})

	p := DefinePrompt(r, "asDataPrompt",
		WithModel(model),
		WithPrompt("test"),
	)

	t.Run("wraps existing prompt", func(t *testing.T) {
		dp := AsDataPrompt[map[string]any, string](p)
		if dp == nil {
			t.Error("expected DataPrompt, got nil")
		}
	})

	t.Run("returns nil for nil prompt", func(t *testing.T) {
		dp := AsDataPrompt[map[string]any, string](nil)
		if dp != nil {
			t.Error("expected nil for nil prompt")
		}
	})
}

// TestPromptKeyVariantKey tests the prompt key generation helpers.
func TestPromptKeyVariantKey(t *testing.T) {
	tests := []struct {
		name       string
		promptName string
		variant    string
		namespace  string
		want       string
	}{
		{
			name:       "simple name",
			promptName: "greeting",
			want:       "greeting",
		},
		{
			name:       "with variant",
			promptName: "greeting",
			variant:    "formal",
			want:       "greeting.formal",
		},
		{
			name:       "with namespace",
			promptName: "greeting",
			namespace:  "myapp",
			want:       "myapp/greeting",
		},
		{
			name:       "with namespace and variant",
			promptName: "greeting",
			variant:    "formal",
			namespace:  "myapp",
			want:       "myapp/greeting.formal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := promptKey(tt.promptName, tt.variant, tt.namespace)
			if got != tt.want {
				t.Errorf("promptKey(%q, %q, %q) = %q, want %q",
					tt.promptName, tt.variant, tt.namespace, got, tt.want)
			}
		})
	}
}

// TestContentType tests the contentType helper function.
func TestContentType(t *testing.T) {
	tests := []struct {
		name        string
		ct          string
		uri         string
		wantCT      string
		wantData    string
		wantErr     bool
		errContains string
	}{
		{
			name:     "gs:// URL with content type",
			ct:       "image/png",
			uri:      "gs://bucket/image.png",
			wantCT:   "image/png",
			wantData: "gs://bucket/image.png",
		},
		{
			name:        "gs:// URL without content type",
			ct:          "",
			uri:         "gs://bucket/image.png",
			wantErr:     true,
			errContains: "must supply contentType",
		},
		{
			name:     "http URL with content type",
			ct:       "image/jpeg",
			uri:      "https://example.com/image.jpg",
			wantCT:   "image/jpeg",
			wantData: "https://example.com/image.jpg",
		},
		{
			name:        "http URL without content type",
			ct:          "",
			uri:         "https://example.com/image.jpg",
			wantErr:     true,
			errContains: "must supply contentType",
		},
		{
			name:     "data URI with base64",
			ct:       "",
			uri:      "data:image/png;base64,iVBORw0KGgo=",
			wantCT:   "image/png",
			wantData: "data:image/png;base64,iVBORw0KGgo=",
		},
		{
			name:     "data URI with explicit content type override",
			ct:       "image/jpeg",
			uri:      "data:image/png;base64,iVBORw0KGgo=",
			wantCT:   "image/jpeg",
			wantData: "data:image/png;base64,iVBORw0KGgo=",
		},
		{
			name:        "empty URI",
			ct:          "image/png",
			uri:         "",
			wantErr:     true,
			errContains: "found empty URI",
		},
		{
			name:        "malformed data URI",
			ct:          "",
			uri:         "data:image/png",
			wantErr:     true,
			errContains: "missing comma",
		},
		{
			name:        "unknown URI scheme",
			ct:          "",
			uri:         "file:///path/to/file",
			wantErr:     true,
			errContains: "uri content type not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCT, gotData, err := contentType(tt.ct, tt.uri)

			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error containing %q, got nil", tt.errContains)
				} else if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error = %q, want containing %q", err.Error(), tt.errContains)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if gotCT != tt.wantCT {
				t.Errorf("contentType = %q, want %q", gotCT, tt.wantCT)
			}
			if string(gotData) != tt.wantData {
				t.Errorf("data = %q, want %q", string(gotData), tt.wantData)
			}
		})
	}
}

// TestDefineDataPromptPanics tests panic conditions in DefineDataPrompt.
func TestDefineDataPromptPanics(t *testing.T) {
	t.Run("empty name panics", func(t *testing.T) {
		r := newTestRegistry(t)
		assertPanic(t, func() {
			DefineDataPrompt[map[string]any, string](r, "")
		}, "name is required")
	})
}

// TestLoadPromptTemplateVariableSubstitution tests that template variables are
// properly substituted with actual input values at execution time.
// This is a regression test for https://github.com/genkit-ai/genkit/issues/3924
func TestLoadPromptTemplateVariableSubstitution(t *testing.T) {
	t.Run("single role", func(t *testing.T) {
		tempDir := t.TempDir()

		mockPromptFile := filepath.Join(tempDir, "greeting.prompt")
		mockPromptContent := `---
model: test/chat
description: A greeting prompt with variables
---
Hello {{name}}, welcome to {{place}}!
`

		if err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0644); err != nil {
			t.Fatalf("Failed to create mock prompt file: %v", err)
		}

		prompt := LoadPrompt(registry.New(), tempDir, "greeting.prompt", "template-var-test")

		// Test with first set of input values
		actionOpts1, err := prompt.Render(context.Background(), map[string]any{
			"name":  "Alice",
			"place": "Wonderland",
		})
		if err != nil {
			t.Fatalf("Failed to render prompt with first input: %v", err)
		}

		if len(actionOpts1.Messages) != 1 {
			t.Fatalf("Expected 1 message, got %d", len(actionOpts1.Messages))
		}

		text1 := actionOpts1.Messages[0].Content[0].Text
		if !strings.Contains(text1, "Alice") {
			t.Errorf("Expected message to contain 'Alice', got: %s", text1)
		}
		if !strings.Contains(text1, "Wonderland") {
			t.Errorf("Expected message to contain 'Wonderland', got: %s", text1)
		}

		// Test with second set of input values (different from first)
		actionOpts2, err := prompt.Render(context.Background(), map[string]any{
			"name":  "Bob",
			"place": "Paradise",
		})
		if err != nil {
			t.Fatalf("Failed to render prompt with second input: %v", err)
		}

		if len(actionOpts2.Messages) != 1 {
			t.Fatalf("Expected 1 message, got %d", len(actionOpts2.Messages))
		}

		text2 := actionOpts2.Messages[0].Content[0].Text
		if !strings.Contains(text2, "Bob") {
			t.Errorf("Expected message to contain 'Bob', got: %s", text2)
		}
		if !strings.Contains(text2, "Paradise") {
			t.Errorf("Expected message to contain 'Paradise', got: %s", text2)
		}

		// Critical: Ensure the second render did NOT use the first input values
		if strings.Contains(text2, "Alice") {
			t.Errorf("BUG: Second render contains 'Alice' from first input! Got: %s", text2)
		}
		if strings.Contains(text2, "Wonderland") {
			t.Errorf("BUG: Second render contains 'Wonderland' from first input! Got: %s", text2)
		}
	})

	t.Run("multi role", func(t *testing.T) {
		tempDir := t.TempDir()

		mockPromptFile := filepath.Join(tempDir, "handlebars_role.prompt")
		mockPromptContent := `---
model: test/chat
description: A multi-role prompt using Handlebars role syntax
---
{{role "system"}}
You are a {{personality}} assistant.

{{role "user"}}
Hello {{name}}, please help me with {{task}}.
`

		if err := os.WriteFile(mockPromptFile, []byte(mockPromptContent), 0644); err != nil {
			t.Fatalf("Failed to create mock prompt file: %v", err)
		}

		prompt := LoadPrompt(registry.New(), tempDir, "handlebars_role.prompt", "handlebars-role-test")

		// Test with first set of input values
		actionOpts1, err := prompt.Render(context.Background(), map[string]any{
			"personality": "helpful",
			"name":        "Alice",
			"task":        "coding",
		})
		if err != nil {
			t.Fatalf("Failed to render prompt with first input: %v", err)
		}

		if len(actionOpts1.Messages) != 2 {
			t.Fatalf("Expected 2 messages, got %d", len(actionOpts1.Messages))
		}

		// Check system message
		systemMsg := actionOpts1.Messages[0]
		if systemMsg.Role != RoleSystem {
			t.Errorf("Expected first message role to be 'system', got '%s'", systemMsg.Role)
		}
		systemText := systemMsg.Content[0].Text
		if !strings.Contains(systemText, "helpful") {
			t.Errorf("Expected system message to contain 'helpful', got: %s", systemText)
		}

		// Check user message
		userMsg := actionOpts1.Messages[1]
		if userMsg.Role != RoleUser {
			t.Errorf("Expected second message role to be 'user', got '%s'", userMsg.Role)
		}
		userText := userMsg.Content[0].Text
		if !strings.Contains(userText, "Alice") {
			t.Errorf("Expected user message to contain 'Alice', got: %s", userText)
		}
		if !strings.Contains(userText, "coding") {
			t.Errorf("Expected user message to contain 'coding', got: %s", userText)
		}

		// Test with second set of input values (different from first)
		actionOpts2, err := prompt.Render(context.Background(), map[string]any{
			"personality": "professional",
			"name":        "Bob",
			"task":        "writing",
		})
		if err != nil {
			t.Fatalf("Failed to render prompt with second input: %v", err)
		}

		if len(actionOpts2.Messages) != 2 {
			t.Fatalf("Expected 2 messages, got %d", len(actionOpts2.Messages))
		}

		// Check system message with new values
		systemMsg2 := actionOpts2.Messages[0]
		systemText2 := systemMsg2.Content[0].Text
		if !strings.Contains(systemText2, "professional") {
			t.Errorf("Expected system message to contain 'professional', got: %s", systemText2)
		}
		if strings.Contains(systemText2, "helpful") {
			t.Errorf("BUG: Second render system message contains 'helpful' from first input! Got: %s", systemText2)
		}

		// Check user message with new values
		userMsg2 := actionOpts2.Messages[1]
		userText2 := userMsg2.Content[0].Text
		if !strings.Contains(userText2, "Bob") {
			t.Errorf("Expected user message to contain 'Bob', got: %s", userText2)
		}
		if !strings.Contains(userText2, "writing") {
			t.Errorf("Expected user message to contain 'writing', got: %s", userText2)
		}
		if strings.Contains(userText2, "Alice") {
			t.Errorf("BUG: Second render user message contains 'Alice' from first input! Got: %s", userText2)
		}
		if strings.Contains(userText2, "coding") {
			t.Errorf("BUG: Second render user message contains 'coding' from first input! Got: %s", userText2)
		}
	})
}

func TestParseDotpromptUse(t *testing.T) {
	tests := []struct {
		name    string
		raw     any
		want    []Middleware
		wantErr bool
	}{
		{name: "nil", raw: nil, want: nil},
		{name: "empty list", raw: []any{}, want: []Middleware{}},
		{
			name: "bare names",
			raw:  []any{"a", "b"},
			want: []Middleware{middlewareRefArg{name: "a"}, middlewareRefArg{name: "b"}},
		},
		{
			name: "mixed names and refs with config",
			raw: []any{
				"a",
				map[string]any{"name": "b", "config": map[string]any{"k": 1}},
			},
			want: []Middleware{
				middlewareRefArg{name: "a"},
				middlewareRefArg{name: "b", config: map[string]any{"k": 1}},
			},
		},
		{
			name: "ref without config",
			raw:  []any{map[string]any{"name": "x"}},
			want: []Middleware{middlewareRefArg{name: "x"}},
		},
		{name: "not a list", raw: "single", wantErr: true},
		{name: "empty string entry", raw: []any{""}, wantErr: true},
		{name: "map missing name", raw: []any{map[string]any{"config": "x"}}, wantErr: true},
		{name: "unsupported entry type", raw: []any{42}, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseDotpromptUse(tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(middlewareRefArg{})); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLoadPrompt_WithUseMiddleware(t *testing.T) {
	const promptSource = `---
model: test/fakeModel
use:
  - test/counter
  - name: test/counter
    config:
      label: from-yaml
---
hello
`
	reg := newTestRegistry(t)
	defineFakeModel(t, reg, fakeModelConfig{})

	var modelCalls int32
	NewMiddleware("counter", counterConfig{sharedModelCalls: &modelCalls}).Register(reg)

	p, err := LoadPromptFromSource(reg, promptSource, "withuse", "")
	if err != nil {
		t.Fatalf("LoadPromptFromSource: %v", err)
	}

	if _, err := p.Execute(context.Background()); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	// Both `use:` entries fire on the single model call: 2 hooks layered.
	if got := modelCalls; got != 2 {
		t.Errorf("model hook calls = %d, want 2 (one per use entry)", got)
	}
}

func TestLoadPrompt_WithUseMiddleware_NotRegistered(t *testing.T) {
	const promptSource = `---
model: test/fakeModel
use:
  - test/missing
---
hi
`
	reg := newTestRegistry(t)
	defineFakeModel(t, reg, fakeModelConfig{})

	p, err := LoadPromptFromSource(reg, promptSource, "missingmw", "")
	if err != nil {
		t.Fatalf("LoadPromptFromSource: %v", err)
	}

	if _, err := p.Execute(context.Background()); err == nil {
		t.Fatal("expected NOT_FOUND error for unregistered middleware referenced from `use:`")
	}
}

func TestLoadPrompt_WithUseMiddleware_InvalidShape(t *testing.T) {
	const promptSource = `---
model: test/fakeModel
use: "not-a-list"
---
hi
`
	reg := newTestRegistry(t)
	if _, err := LoadPromptFromSource(reg, promptSource, "badshape", ""); err == nil {
		t.Fatal("expected error for non-list `use:`")
	}
}
