// Copyright 2026 Google LLC
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

package mcp

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/mcp"
)

func TestMCPServerListsAndRendersGenkitPrompts(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefinePrompt(g, "greet",
		ai.WithDescription("Greet a person"),
		ai.WithInputSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string", "description": "Person to greet"},
			},
			"required": []string{"name"},
		}),
		ai.WithSystem("Be friendly."),
		ai.WithPrompt("Hello {{name}}"),
	)

	s := NewMCPServer(g, MCPServerOptions{Name: "prompts"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}
	if names := s.ListRegisteredPrompts(); len(names) != 1 || names[0] != "greet" {
		t.Fatalf("registered prompts = %v, want [greet]", names)
	}

	listed := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/list"}`))
	listResponse, ok := listed.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/list response = %T, want JSONRPCResponse", listed)
	}
	list, ok := listResponse.Result.(mcp.ListPromptsResult)
	if !ok || len(list.Prompts) != 1 {
		t.Fatalf("prompts/list result = %v, want one prompt", listResponse.Result)
	}
	got := list.Prompts[0]
	if got.Name != "greet" || got.Description != "Greet a person" || len(got.Arguments) != 1 || got.Arguments[0].Name != "name" || !got.Arguments[0].Required {
		t.Fatalf("prompt descriptor = %+v, want greet with required name", got)
	}

	rendered := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":2,"method":"prompts/get","params":{"name":"greet","arguments":{"name":"Ada"}}}`))
	getResponse, ok := rendered.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/get response = %v, want JSONRPCResponse", rendered)
	}
	result, ok := getResponse.Result.(mcp.GetPromptResult)
	if !ok || len(result.Messages) != 2 {
		t.Fatalf("prompts/get result = %v, want system and user messages", getResponse.Result)
	}
	if text, ok := result.Messages[0].Content.(mcp.TextContent); !ok || result.Messages[0].Role != mcp.RoleUser || text.Text != "System instructions:\nBe friendly." {
		t.Errorf("first prompt message = %+v, want system instructions", result.Messages[0])
	}
	if text, ok := result.Messages[1].Content.(mcp.TextContent); !ok || result.Messages[1].Role != mcp.RoleUser || text.Text != "Hello Ada" {
		t.Errorf("second prompt message = %+v, want Hello Ada", result.Messages[1])
	}

	missing := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":3,"method":"prompts/get","params":{"name":"greet"}}`))
	if _, ok := missing.(mcp.JSONRPCError); !ok {
		t.Fatalf("missing required argument response = %v, want JSONRPCError", missing)
	}
}

func TestGenkitPromptMediaConversion(t *testing.T) {
	content, err := genkitPartToMCP(ai.NewMediaPart("image/png", "data:image/png;base64,aGVsbG8="))
	if err != nil {
		t.Fatal(err)
	}
	image, ok := content.(mcp.ImageContent)
	if !ok || image.MIMEType != "image/png" || image.Data != "aGVsbG8=" {
		t.Fatalf("media conversion = %v, want base64 image", content)
	}
}

func TestMCPServerDiscoversDotPromptFiles(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPromptFS(fstest.MapFS{
		"prompts/welcome.prompt": &fstest.MapFile{Data: []byte("---\ndescription: Welcome a user\ninput:\n  schema:\n    name: string\n---\nWelcome {{name}}")},
	}))
	s := NewMCPServer(g, MCPServerOptions{Name: "dotprompt"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}

	response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/get","params":{"name":"welcome","arguments":{"name":"Ada"}}}`))
	got, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/get response = %v, want JSONRPCResponse", response)
	}
	result, ok := got.Result.(mcp.GetPromptResult)
	if !ok || len(result.Messages) != 1 {
		t.Fatalf("prompts/get result = %v, want one message", got.Result)
	}
	if text, ok := result.Messages[0].Content.(mcp.TextContent); !ok || text.Text != "Welcome Ada" {
		t.Fatalf("rendered DotPrompt message = %v, want Welcome Ada", result.Messages[0].Content)
	}
}

func TestMCPServerConvertsTypedPromptArguments(t *testing.T) {
	type countInput struct {
		Count int `json:"count"`
	}
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefineSchema(g, "CountInput", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{"type": "integer"},
		},
		"required": []string{"count"},
	})
	genkit.DefinePrompt(g, "count",
		ai.WithInputSchemaName("CountInput"),
		ai.WithPromptFn(func(_ context.Context, input countInput) (string, error) {
			return fmt.Sprintf("Count %d", input.Count), nil
		}),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "typed-prompts"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}

	response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/get","params":{"name":"count","arguments":{"count":"3"}}}`))
	got, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/get response = %v, want JSONRPCResponse", response)
	}
	result, ok := got.Result.(mcp.GetPromptResult)
	if !ok || len(result.Messages) != 1 {
		t.Fatalf("prompts/get result = %v, want one message", got.Result)
	}
	if text, ok := result.Messages[0].Content.(mcp.TextContent); !ok || text.Text != "Count 3" {
		t.Fatalf("typed prompt message = %v, want Count 3", result.Messages[0].Content)
	}

	invalid := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":2,"method":"prompts/get","params":{"name":"count","arguments":{"count":"three"}}}`))
	if _, ok := invalid.(mcp.JSONRPCError); !ok {
		t.Fatalf("invalid integer response = %v, want JSONRPCError", invalid)
	}
}

func TestMCPServerUsesPromptDefaults(t *testing.T) {
	type greetInput struct {
		Greeting string `json:"greeting"`
		Name     string `json:"name"`
	}
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefinePrompt(g, "default-greet",
		ai.WithInputType(greetInput{Greeting: "Hello", Name: "Ada"}),
		ai.WithPrompt("{{greeting}} {{name}}"),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "defaults"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}
	listed := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/list"}`)).(mcp.JSONRPCResponse).Result.(mcp.ListPromptsResult)
	for _, arg := range listed.Prompts[0].Arguments {
		if arg.Required {
			t.Fatalf("defaulted argument %q marked required", arg.Name)
		}
	}

	for _, test := range []struct{ args, want string }{
		{"", "Hello Ada"},
		{`,"arguments":{}`, "Hello Ada"},
		{`,"arguments":{"name":"Grace"}`, "Hello Grace"},
	} {
		response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/get","params":{"name":"default-greet"`+test.args+`}}`))
		got, ok := response.(mcp.JSONRPCResponse)
		if !ok {
			t.Fatalf("prompts/get with arguments %q = %v, want success", test.args, response)
		}
		result := got.Result.(mcp.GetPromptResult)
		if text := result.Messages[0].Content.(mcp.TextContent).Text; text != test.want {
			t.Fatalf("prompts/get with arguments %q = %q, want %q", test.args, text, test.want)
		}
	}
}

func TestMCPServerRendersScalarPrompt(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefinePrompt(g, "scalar-greet",
		ai.WithInputSchema(map[string]any{"type": "string", "description": "Person to greet"}),
		ai.WithPromptFn(func(_ context.Context, name string) (string, error) {
			return "Hello " + name, nil
		}),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "scalar"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}

	listed := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/list"}`)).(mcp.JSONRPCResponse).Result.(mcp.ListPromptsResult)
	if args := listed.Prompts[0].Arguments; len(args) != 1 || args[0].Name != "input" || !args[0].Required {
		t.Fatalf("scalar prompt arguments = %v, want required input", args)
	}
	response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":2,"method":"prompts/get","params":{"name":"scalar-greet","arguments":{"input":"Ada"}}}`))
	got, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("scalar prompts/get response = %v, want success", response)
	}
	result := got.Result.(mcp.GetPromptResult)
	if text := result.Messages[0].Content.(mcp.TextContent).Text; text != "Hello Ada" {
		t.Fatalf("scalar prompt message = %q, want Hello Ada", text)
	}
}

func TestMCPPromptInputPreservesNestedInteger(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"settings": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"count": map[string]any{"type": "integer"},
				},
			},
		},
	}
	input, err := mcpPromptInput(schema, map[string]string{"settings": `{"count":9007199254740993}`})
	if err != nil {
		t.Fatal(err)
	}
	settings := input.(map[string]any)["settings"].(map[string]any)
	if got, ok := settings["count"].(int64); !ok || got != 9007199254740993 {
		t.Fatalf("nested integer = %v (%T), want exact int64", settings["count"], settings["count"])
	}
}

func TestMCPServerIncludesDocumentsAndOutputInstructions(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefinePrompt(g, "answer-with-context",
		ai.WithTextDocs("The answer is 42."),
		ai.WithOutputSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "integer"},
			},
		}),
		ai.WithPrompt("What is the answer?"),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "context"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}
	response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/get","params":{"name":"answer-with-context"}}`))
	got, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/get response = %v, want success", response)
	}
	result := got.Result.(mcp.GetPromptResult)
	var combined string
	for _, message := range result.Messages {
		if text, ok := message.Content.(mcp.TextContent); ok {
			combined += text.Text + "\n"
		}
	}
	for _, want := range []string{"The answer is 42.", "What is the answer?", "Output should be in JSON format", `"answer"`} {
		if !strings.Contains(combined, want) {
			t.Fatalf("MCP prompt lacks %q: %s", want, combined)
		}
	}
}

func TestMCPServerAcceptsUnsignedInteger(t *testing.T) {
	type input struct {
		Count uint64 `json:"count"`
	}
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefinePrompt(g, "unsigned",
		ai.WithInputSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"count": map[string]any{"type": "integer"},
			},
			"required": []string{"count"},
		}),
		ai.WithPromptFn(func(_ context.Context, in input) (string, error) {
			return fmt.Sprintf("Count %d", in.Count), nil
		}),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "unsigned"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}
	response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/get","params":{"name":"unsigned","arguments":{"count":"18446744073709551615"}}}`))
	got, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/get response = %v, want success", response)
	}
	result := got.Result.(mcp.GetPromptResult)
	if text := result.Messages[0].Content.(mcp.TextContent).Text; text != "Count 18446744073709551615" {
		t.Fatalf("unsigned prompt message = %q", text)
	}

	schema := map[string]any{"type": "object", "properties": map[string]any{
		"settings": map[string]any{"type": "object", "properties": map[string]any{
			"count": map[string]any{"type": "integer"},
		}},
	}}
	nested, err := mcpPromptInput(schema, map[string]string{"settings": `{"count":18446744073709551615}`})
	if err != nil {
		t.Fatal(err)
	}
	count := nested.(map[string]any)["settings"].(map[string]any)["count"]
	if value, ok := count.(uint64); !ok || value != ^uint64(0) {
		t.Fatalf("nested unsigned integer = %v (%T), want uint64 max", count, count)
	}
}

func TestMCPClientPreservesPromptMedia(t *testing.T) {
	client := &GenkitMCPClient{}
	messages := client.convertMCPMessages([]mcp.PromptMessage{
		{Role: mcp.RoleUser, Content: mcp.NewImageContent("aGVsbG8=", "image/png")},
		{Role: mcp.RoleAssistant, Content: mcp.NewAudioContent("d29ybGQ=", "audio/wav")},
	})
	if len(messages) != 2 {
		t.Fatalf("converted messages = %d, want 2", len(messages))
	}
	if part := messages[0].Content[0]; part.Kind != ai.PartMedia || part.ContentType != "image/png" || part.Text != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image part = %+v", part)
	}
	if part := messages[1].Content[0]; part.Kind != ai.PartMedia || part.ContentType != "audio/wav" || part.Text != "data:audio/wav;base64,d29ybGQ=" {
		t.Fatalf("audio part = %+v", part)
	}
}

func TestOutputInstructionsKeepConversationOrder(t *testing.T) {
	original := []*ai.Message{
		ai.NewUserTextMessage("Question"),
		ai.NewModelTextMessage("Earlier answer"),
	}
	updated := withGenkitOutputInstructions(original, "Answer in JSON.")
	if original[0].Content[0].Text != "Question" {
		t.Fatalf("original prompt was mutated: %q", original[0].Content[0].Text)
	}
	if len(updated) != 2 || updated[1].Role != ai.RoleModel {
		t.Fatalf("conversation order changed: %+v", updated)
	}
	if got := updated[0].Content[0].Text; !strings.Contains(got, "Answer in JSON.") {
		t.Fatalf("output instructions missing from user turn: %q", got)
	}
}

func TestMCPPromptInputUsesAdditionalProperties(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"additionalProperties": map[string]any{"type": "integer"},
	}
	input, err := mcpPromptInput(schema, map[string]string{"count": "3"})
	if err != nil {
		t.Fatal(err)
	}
	if got := input.(map[string]any)["count"]; got != int64(3) {
		t.Fatalf("dynamic integer = %v (%T), want int64(3)", got, got)
	}

	nestedSchema := map[string]any{"type": "object", "properties": map[string]any{
		"settings": map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "integer"}},
	}}
	nested, err := mcpPromptInput(nestedSchema, map[string]string{"settings": `{"count":1e3}`})
	if err != nil {
		t.Fatal(err)
	}
	if got := nested.(map[string]any)["settings"].(map[string]any)["count"]; got != int64(1000) {
		t.Fatalf("nested dynamic integer = %v (%T), want int64(1000)", got, got)
	}
}

func TestMCPPromptInputAcceptsIntegralJSONNumbers(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{
		"values": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
	}}
	input, err := mcpPromptInput(schema, map[string]string{"values": `[1.0,1e3,18446744073709551615]`})
	if err != nil {
		t.Fatal(err)
	}
	values := input.(map[string]any)["values"].([]any)
	for index, want := range []any{int64(1), int64(1000), uint64(^uint64(0))} {
		if values[index] != want {
			t.Fatalf("value %d = %v (%T), want %v (%T)", index, values[index], values[index], want, want)
		}
	}
	if _, err := mcpPromptInput(schema, map[string]string{"values": `[1.5]`}); err == nil {
		t.Fatal("fractional value accepted as integer")
	}
}

func TestMCPServerExposesSchemaLessPromptInput(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.DefinePrompt(g, "schema-less", ai.WithPrompt("Hello {{name}}"))
	s := NewMCPServer(g, MCPServerOptions{Name: "schema-less"})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}
	listed := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/list"}`)).(mcp.JSONRPCResponse).Result.(mcp.ListPromptsResult)
	if args := listed.Prompts[0].Arguments; len(args) != 1 || args[0].Name != "inputJson" {
		t.Fatalf("schema-less prompt arguments = %v, want inputJson", args)
	}
	response := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":2,"method":"prompts/get","params":{"name":"schema-less","arguments":{"inputJson":"{\"name\":\"Ada\"}"}}}`))
	got, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/get response = %v, want success", response)
	}
	result := got.Result.(mcp.GetPromptResult)
	if text := result.Messages[0].Content.(mcp.TextContent).Text; text != "Hello Ada" {
		t.Fatalf("schema-less prompt message = %q, want Hello Ada", text)
	}
}

func TestMCPPromptConvertsToolTurnsAndRejectsRemoteMedia(t *testing.T) {
	messages, err := genkitMessagesToMCP([]*ai.Message{
		ai.NewModelMessage(ai.NewToolRequestPart(&ai.ToolRequest{Name: "lookup", Input: map[string]any{"id": 3}})),
		ai.NewMessage(ai.RoleTool, nil, ai.NewToolResponsePart(&ai.ToolResponse{Name: "lookup", Output: map[string]any{"name": "Ada"}})),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 || messages[0].Role != mcp.RoleAssistant || messages[1].Role != mcp.RoleUser {
		t.Fatalf("converted tool turns = %+v", messages)
	}
	if text := messages[1].Content.(mcp.TextContent).Text; !strings.Contains(text, `"name":"Ada"`) {
		t.Fatalf("tool result content missing: %q", text)
	}
	if _, err := genkitPartToMCP(ai.NewMediaPart("image/png", "https://example.com/image.png")); err == nil {
		t.Fatal("remote media was silently converted to text")
	}
	if _, err := genkitPartToMCP(ai.NewMediaPart("image/png", "HTTPS://example.com/image.png")); err == nil || !strings.Contains(err.Error(), "remote prompt media") {
		t.Fatalf("uppercase remote media error = %v, want descriptive remote media error", err)
	}
}
