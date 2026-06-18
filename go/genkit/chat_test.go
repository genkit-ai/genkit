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

package genkit

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit/session"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/invopop/jsonschema"
)

type HelloPromptInput struct {
	Name string
}

var chatGenkit, _ = Init(context.Background())
var chatModel = getChatModel(chatGenkit)
var chatTool = getNameTool(chatGenkit)
var chatPrompt = getChatPrompt(chatGenkit)

func getNameTool(g *Genkit) *ai.ToolDef[struct{ Name string }, string] {
	return DefineTool(g, "updateName", "use this tool to update the name of the user",
		func(ctx *ai.ToolContext, input struct {
			Name string
		}) (string, error) {
			// Set name in state
			session, err := session.FromContext(ctx)
			if err != nil {
				return "", err
			}

			err = session.UpdateState(input)
			if err != nil {
				return "", err
			}

			return "Hello, my name is " + input.Name, nil
		},
	)
}

func getChatModel(g *Genkit) ai.Model {
	return ai.DefineModel(g.reg, "test", "chat", &ai.ModelInfo{
		Supports: &ai.ModelInfoSupports{
			Tools:      true,
			Multiturn:  true,
			SystemRole: true,
		},
	}, func(ctx context.Context, gr *ai.ModelRequest, msc ai.ModelStreamingCallback) (*ai.ModelResponse, error) {
		toolCalled := false
		for _, msg := range gr.Messages {
			if msg.Content[0].IsToolResponse() {
				toolCalled = true
			}
		}

		if !toolCalled && len(gr.Tools) == 1 {
			part := ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  "updateName",
				Input: map[string]any{"Name": "Earl"},
			})

			return &ai.ModelResponse{
				Request: gr,
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: []*ai.Part{part},
				},
			}, nil
		}

		if msc != nil {
			msc(ctx, &ai.ModelResponseChunk{
				Content: []*ai.Part{ai.NewTextPart("3!")},
			})
			msc(ctx, &ai.ModelResponseChunk{
				Content: []*ai.Part{ai.NewTextPart("2!")},
			})
			msc(ctx, &ai.ModelResponseChunk{
				Content: []*ai.Part{ai.NewTextPart("1!")},
			})
		}

		textResponse := ""
		var contentTexts []string
		for _, m := range gr.Messages {
			if m.Role != ai.RoleUser && m.Role != ai.RoleModel {
				textResponse += fmt.Sprintf("%s: ", m.Role)
			}

			if m.Role == ai.RoleTool {
				contentTexts = append(contentTexts, m.Content[0].ToolResponse.Output.(string))
			}

			for _, c := range m.Content {
				contentTexts = append(contentTexts, c.Text)
			}
		}

		textResponse += strings.Join(contentTexts, "; ")
		textResponse += "; config: " + base.PrettyJSONString(gr.Config)
		textResponse += "; context: " + base.PrettyJSONString(gr.Context)

		return &ai.ModelResponse{
			Request: gr,
			Message: ai.NewModelTextMessage(fmt.Sprintf("Echo: %s", textResponse)),
		}, nil
	})
}

func getChatPrompt(g *Genkit) *ai.Prompt {
	return DefinePrompt(
		g,
		"prompts",
		"helloPrompt",
		nil, // Additional model config
		jsonschema.Reflect(&HelloPromptInput{}),
		func(ctx context.Context, input any) (*ai.ModelRequest, error) {
			params, ok := input.(HelloPromptInput)
			if !ok {
				return nil, errors.New("input doesn't satisfy schema")
			}
			prompt := fmt.Sprintf(
				"Say hello to %s",
				params.Name)
			return &ai.ModelRequest{Messages: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart(prompt)}},
			}}, nil
		},
	)
}

func TestChat(t *testing.T) {
	ctx := context.Background()

	chat, err := NewChat(ctx, chatGenkit, WithModel(chatModel))
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err := chat.Send(ctx, "Tell me a joke")
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Echo: Tell me a joke; config: null; context: null"
	if resp.Text() != want {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	resp, err = chat.Send(ctx, "Bye")
	if err != nil {
		t.Fatal(err.Error())
	}

	want = "Echo: Tell me a joke; Echo: Tell me a joke; config: null; context: null; Bye; config: null; context: null"
	if resp.Text() != want {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	// Should this be 4?
	if len(resp.Request.Messages) != 3 {
		t.Errorf("no history, got %d message want %d", len(resp.Request.Messages), 3)
	}
}

func TestChatWithStreaming(t *testing.T) {
	ctx := context.Background()

	streamText := ""
	chat, err := NewChat(
		ctx,
		chatGenkit,
		WithModel(chatModel),
		WithStreaming(func(ctx context.Context, grc *ai.ModelResponseChunk) error {
			streamText += grc.Text()
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err := chat.Send(ctx, "Hello")
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Echo: Hello; config: null; context: null"
	if resp.Text() != want {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	want = "3!2!1!"
	if streamText != want {
		t.Errorf("got %q want %q", streamText, want)
	}
}

func TestChatWithOptions(t *testing.T) {
	ctx := context.Background()

	session, err := session.New(ctx)
	if err != nil {
		t.Fatal(err.Error())
	}

	chat, err := NewChat(
		ctx,
		chatGenkit,
		WithModel(chatModel),
		WithSession(session),
		WithThreadName("test"),
		WithSystemText("you are a helpful assistant"),
		WithConfig(ai.GenerationCommonConfig{Temperature: 1}),
		WithContext(&ai.Document{Content: []*ai.Part{ai.NewTextPart("Banana")}}),
		WithTools(chatTool),
		WithOutputSchema("hello world"),
		WithOutputFormat(ai.OutputFormatText),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	if chat.Session == nil || chat.Session.GetID() != session.GetID() {
		t.Errorf("session is not set")
	}

	if chat.ThreadName != "test" {
		t.Errorf("thread name is not set")
	}

	if !strings.Contains(chat.SystemText, "you are a helpful assistant") {
		t.Errorf("system text is not set")
	}

	if chat.Request.Schema == nil {
		t.Errorf("output schema not set")
	}

	if chat.Request.Format != ai.OutputFormatText {
		t.Errorf("output format is not set")
	}

	resp, err := chat.Send(ctx, "Hello")
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Echo: system: tool: you are a helpful assistant; Hello; ; Hello, my name is Earl; ; config: {\n  \"temperature\": 1\n}; context: [\n  {\n    \"content\": [\n      {\n        \"text\": \"Banana\"\n      }\n    ]\n  }\n]"
	if resp.Text() != want {
		t.Errorf("got %q want %q", resp.Text(), want)
	}
}

func TestChatWithOptionsErrorHandling(t *testing.T) {
	ctx := context.Background()

	session, err := session.New(ctx)
	if err != nil {
		t.Fatal(err.Error())
	}

	var tests = []struct {
		name string
		with ChatOption
	}{
		{
			name: "WithModel",
			with: WithModel(chatModel),
		},
		{
			name: "WithSession",
			with: WithSession(session),
		},
		{
			name: "WithThreadName",
			with: WithThreadName("test"),
		},
		{
			name: "WithStreaming",
			with: WithStreaming(func(ctx context.Context, grc *ai.ModelResponseChunk) error {
				return nil
			}),
		},
		{
			name: "WithSystemText",
			with: WithSystemText("system text"),
		},
		{
			name: "WithConfig",
			with: WithConfig(ai.GenerationCommonConfig{}),
		},
		{
			name: "WithContext",
			with: WithContext(&ai.Document{Content: []*ai.Part{ai.NewTextPart("Banana")}}),
		},
		{
			name: "WithTools",
			with: WithTools(chatTool),
		},
		{
			name: "WithOutputSchema",
			with: WithOutputSchema(map[string]any{"name": "world"}),
		},
		{
			name: "WithOutputFormat",
			with: WithOutputFormat(ai.OutputFormatJSON),
		},
		{
			name: "WithInput",
			with: WithInput("Hello"),
		},
		{
			name: "WithPrompt",
			with: WithPrompt(chatPrompt),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewChat(
				context.Background(),
				chatGenkit,
				test.with,
				test.with,
			)

			if err == nil {
				t.Errorf("%s could be set twice", test.name)
			}
		})
	}
}

func TestGetChatMessages(t *testing.T) {
	var tests = []struct {
		name   string
		input  any
		output []*ai.Message
	}{
		{
			name:  "string",
			input: "hello",
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
			},
		},
		{
			name:  "integer",
			input: 1,
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("1")}},
			},
		},
		{
			name:  "float",
			input: 3.14,
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("3.140000")}},
			},
		},
		{
			name: "ai.message",
			input: ai.Message{
				Content: []*ai.Part{ai.NewTextPart("hello")},
			},
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
			},
		},
		{
			name: "&ai.message",
			input: &ai.Message{
				Content: []*ai.Part{ai.NewTextPart("hello")},
			},
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
			},
		},
		{
			name: "ai.messages",
			input: []ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
				{Content: []*ai.Part{ai.NewTextPart("world")}},
			},
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
				{Content: []*ai.Part{ai.NewTextPart("world")}},
			},
		},
		{
			name: "&ai.messages",
			input: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
				{Content: []*ai.Part{ai.NewTextPart("world")}},
			},
			output: []*ai.Message{
				{Content: []*ai.Part{ai.NewTextPart("hello")}},
				{Content: []*ai.Part{ai.NewTextPart("world")}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			messages, err := getChatMessages(test.input)
			if err != nil {
				t.Fatal(err)
			}

			if len(messages) != len(test.output) {
				t.Errorf("got %d messages want %d", len(messages), len(test.output))
			}

			for i, msg := range messages {
				if msg.Content[0].Text != test.output[i].Content[0].Text {
					t.Errorf("got %q want %q", msg.Content[0].Text, test.output[i].Content[0].Text)
				}
			}
		})
	}
}

func TestMultiChatSession(t *testing.T) {
	ctx := context.Background()

	session, err := session.New(ctx)
	if err != nil {
		t.Fatal(err.Error())
	}

	lawyerChat, err := NewChat(
		ctx,
		chatGenkit,
		WithModel(chatModel),
		WithThreadName("lawyer thread"),
		WithSystemText("talk like a lawyer"),
		WithSession(session),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err := lawyerChat.Send(ctx, "Hello lawyer")
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Echo: system: talk like a lawyer; Hello lawyer; config: null; context: null"
	if resp.Text() != want {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	pirateChat, err := NewChat(
		ctx,
		chatGenkit,
		WithModel(chatModel),
		WithThreadName("pirate thread"),
		WithSystemText("talk like a pirate"),
		WithSession(session),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err = pirateChat.Send(ctx, "Hello pirate")
	if err != nil {
		t.Fatal(err.Error())
	}

	want = "Echo: system: talk like a pirate; Hello pirate; config: null; context: null"
	if resp.Text() != want {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	data, err := session.GetData()
	if err != nil {
		t.Fatal(err.Error())
	}
	if len(data.Threads) != 2 {
		t.Errorf("session should have 2 threads")
	}
}

func TestStateUpdate(t *testing.T) {
	ctx := context.Background()

	session, err := session.New(ctx,
		session.WithStateType(HelloPromptInput{}),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	chat, err := NewChat(
		ctx,
		chatGenkit,
		WithModel(chatModel),
		WithSystemText("update state"),
		WithTools(chatTool),
		WithSession(session),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err := chat.Send(ctx, "What's your name?")
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Hello, my name is Earl"
	if !strings.Contains(resp.Text(), want) {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	data, err := session.GetData()
	if err != nil {
		t.Fatal(err.Error())
	}
	if data.State["Name"] != "Earl" {
		t.Error("session state not set")
	}
}

func TestChatWithPrompt(t *testing.T) {
	ctx := context.Background()

	// Chat from prompt with input
	chat, err := NewChat(
		ctx,
		chatGenkit,
		WithModel(chatModel),
		WithPrompt(chatPrompt),
		WithInput(HelloPromptInput{Name: "Earl"}),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err := chat.Send(ctx, "Send prompt")
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Say hello to Earl"
	if !strings.Contains(resp.Text(), want) {
		t.Errorf("got %q want %q", resp.Text(), want)
	}

	// Send text instead of messages
	text, err := chat.SendText(ctx, "Send prompt")
	if err != nil {
		t.Fatal(err.Error())
	}

	if !strings.Contains(text, want) {
		t.Errorf("got %q want %q", text, want)
	}

	// Rendered prompt to chat
	mr, err := chatPrompt.Render(ctx, HelloPromptInput{Name: "someone else"})
	if err != nil {
		t.Fatal(err.Error())
	}

	resp, err = chat.Send(ctx, mr.Messages)
	if err != nil {
		t.Fatal(err.Error())
	}

	want = "Say hello to someone else"
	if !strings.Contains(resp.Text(), want) {
		t.Errorf("got %q want %q", resp.Text(), want)
	}
}
