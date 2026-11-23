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
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/mark3labs/mcp-go/mcp"
)

func listPrompts(t *testing.T, s *GenkitMCPServer) []mcp.Prompt {
	t.Helper()
	response := s.GetServer().HandleMessage(context.Background(), []byte(`{"jsonrpc":"2.0","id":1,"method":"prompts/list"}`))
	result, ok := response.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("prompts/list = %+v, want a result", response)
	}
	return result.Result.(mcp.ListPromptsResult).Prompts
}

// getPrompt renders a prompt with prompts/get. It returns the text of each
// message, or the error message when the request fails.
func getPrompt(t *testing.T, s *GenkitMCPServer, name string, arguments map[string]string) ([]string, string) {
	t.Helper()
	params, err := json.Marshal(map[string]any{"name": name, "arguments": arguments})
	if err != nil {
		t.Fatal(err)
	}
	request := fmt.Sprintf(`{"jsonrpc":"2.0","id":2,"method":"prompts/get","params":%s}`, params)
	switch response := s.GetServer().HandleMessage(context.Background(), []byte(request)).(type) {
	case mcp.JSONRPCResponse:
		var texts []string
		for _, message := range response.Result.(mcp.GetPromptResult).Messages {
			texts = append(texts, message.Content.(mcp.TextContent).Text)
		}
		return texts, ""
	case mcp.JSONRPCError:
		return nil, response.Error.Message
	default:
		t.Fatalf("prompts/get %s = %+v, want a result or an error", name, response)
		return nil, ""
	}
}

func TestMCPServerListsAndRendersGenkitPrompts(t *testing.T) {
	g := genkit.Init(context.Background(), genkit.WithPromptFS(fstest.MapFS{
		"prompts/welcome.prompt": &fstest.MapFile{Data: []byte("---\ndescription: Welcome a user\ninput:\n  schema:\n    name: string\n---\nWelcome {{name}}")},
	}))
	genkit.DefinePrompt(g, "greet",
		ai.WithDescription("Greet a person"),
		ai.WithInputSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":     map[string]any{"type": "string", "description": "Person to greet"},
				"greeting": map[string]any{"type": "string"},
			},
			"required": []string{"name"},
		}),
		ai.WithSystem("Be friendly."),
		ai.WithPrompt("{{#if greeting}}{{greeting}}{{else}}Hello{{/if}} {{name}}"),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "prompts"})

	want := []mcp.Prompt{
		{Name: "greet", Description: "Greet a person", Arguments: []mcp.PromptArgument{
			{Name: "greeting"},
			{Name: "name", Description: "Person to greet", Required: true},
		}},
		{Name: "welcome", Description: "Welcome a user", Arguments: []mcp.PromptArgument{{Name: "name", Required: true}}},
	}
	if diff := cmp.Diff(want, listPrompts(t, s)); diff != "" {
		t.Errorf("prompts/list mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"greet", "welcome"}, s.ListRegisteredPrompts()); diff != "" {
		t.Errorf("ListRegisteredPrompts mismatch (-want +got):\n%s", diff)
	}

	for _, test := range []struct {
		name, prompt string
		arguments    map[string]string
		want         []string
		wantErr      string
	}{
		{"system message", "greet", map[string]string{"name": "Ada"}, []string{"System instructions:\nBe friendly.", "Hello Ada"}, ""},
		{"optional argument", "greet", map[string]string{"name": "Ada", "greeting": "Hi"}, []string{"System instructions:\nBe friendly.", "Hi Ada"}, ""},
		{"missing required argument", "greet", nil, nil, `requires argument "name"`},
		{"dotprompt file", "welcome", map[string]string{"name": "Ada"}, []string{"Welcome Ada"}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := getPrompt(t, s, test.prompt, test.arguments)
			if !strings.Contains(err, test.wantErr) || (test.wantErr == "") != (err == "") {
				t.Fatalf("prompts/get error = %q, want %q", err, test.wantErr)
			}
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("prompts/get mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertPromptArgument(t *testing.T) {
	for _, test := range []struct {
		schemaType any
		value      string
		want       any
		wantErr    bool
	}{
		{"integer", "3", int64(3), false},
		{"integer", "three", nil, true},
		{"number", "1.5", 1.5, false},
		{"boolean", "true", true, false},
		{"boolean", "yes", nil, true},
		{"object", `{"id":7}`, map[string]any{"id": float64(7)}, false},
		{"array", `[1,"x"]`, []any{float64(1), "x"}, false},
		{"object", "{", nil, true},
		{"object", "{} {}", nil, true},
		{"string", "3", "3", false},
		{nil, "3", "3", false},
		{[]any{"integer", "null"}, "3", "3", false},
	} {
		got, err := convertPromptArgument(map[string]any{"type": test.schemaType}, test.value)
		if (err != nil) != test.wantErr {
			t.Errorf("convertPromptArgument(%v, %q) error = %v, want error %t", test.schemaType, test.value, err, test.wantErr)
			continue
		}
		if !test.wantErr && !cmp.Equal(got, test.want) {
			t.Errorf("convertPromptArgument(%v, %q) = %#v, want %#v", test.schemaType, test.value, got, test.want)
		}
	}
}

// Schemas reach the server as Go values: resolved from a named schema, or built
// with Go map and slice types that are not map[string]any and []any.
func TestMCPServerConvertsTypedPromptArguments(t *testing.T) {
	type countInput struct {
		Count int `json:"count"`
	}
	for _, test := range []struct {
		name   string
		option ai.PromptOption
	}{
		{"named schema", ai.WithInputSchemaName("CountInput")},
		{"typed Go schema maps", ai.WithInputSchema(map[string]any{
			"type": "object",
			"properties": map[string]map[string]any{
				"count": {"type": "integer", "description": "Item count"},
			},
			"required": []string{"count"},
		})},
	} {
		t.Run(test.name, func(t *testing.T) {
			g := genkit.Init(context.Background())
			genkit.DefineSchema(g, "CountInput", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"count": map[string]any{"type": "integer", "description": "Item count"},
				},
				"required": []string{"count"},
			})
			genkit.DefinePrompt(g, "count", test.option,
				ai.WithPromptFn(func(_ context.Context, input countInput) (string, error) {
					return fmt.Sprintf("Count %d", input.Count), nil
				}),
			)
			s := NewMCPServer(g, MCPServerOptions{Name: "typed"})

			want := []mcp.PromptArgument{{Name: "count", Description: "Item count", Required: true}}
			if diff := cmp.Diff(want, listPrompts(t, s)[0].Arguments); diff != "" {
				t.Errorf("prompt arguments mismatch (-want +got):\n%s", diff)
			}
			if got, err := getPrompt(t, s, "count", map[string]string{"count": "7"}); err != "" || !cmp.Equal(got, []string{"Count 7"}) {
				t.Errorf("prompts/get count=7 = %q, %q, want [Count 7]", got, err)
			}
			if _, err := getPrompt(t, s, "count", map[string]string{"count": "seven"}); !strings.Contains(err, `argument "count"`) {
				t.Errorf("prompts/get count=seven error = %q, want invalid argument error", err)
			}
		})
	}
}

func TestMCPServerUsesPromptDefaults(t *testing.T) {
	type greetInput struct {
		Greeting string `json:"greeting,omitempty"`
		Name     string `json:"name"`
	}
	g := genkit.Init(context.Background())
	genkit.DefinePrompt(g, "default-greet",
		ai.WithInputType(greetInput{Greeting: "Hello", Name: "Ada"}),
		ai.WithPrompt("{{greeting}} {{name}}"),
	)
	genkit.DefinePrompt(g, "zero-greet",
		ai.WithInputType(greetInput{}),
		ai.WithPrompt("{{greeting}} {{name}}"),
	)
	s := NewMCPServer(g, MCPServerOptions{Name: "defaults"})
	// Required follows the input schema, whatever the default input holds.
	for _, prompt := range listPrompts(t, s) {
		for _, arg := range prompt.Arguments {
			if want := arg.Name == "name"; arg.Required != want {
				t.Errorf("%s argument %q required = %t, want %t", prompt.Name, arg.Name, arg.Required, want)
			}
		}
	}
	for _, name := range []string{"default-greet", "zero-greet"} {
		if _, err := getPrompt(t, s, name, map[string]string{"greeting": "Hi"}); !strings.Contains(err, `requires argument "name"`) {
			t.Errorf("prompts/get %s without name error = %q, want missing argument", name, err)
		}
	}
	if got, err := getPrompt(t, s, "default-greet", map[string]string{"name": "Grace"}); err != "" || !cmp.Equal(got, []string{"Hello Grace"}) {
		t.Errorf("prompts/get default-greet = %q, %q, want default greeting", got, err)
	}
}

func TestMCPServerSkipsPromptsWithoutObjectInput(t *testing.T) {
	g := genkit.Init(context.Background())
	for name, schema := range map[string]map[string]any{
		"empty-object": {"type": "object"},
		"no-schema":    nil,
		"optional":     {"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}},
		"scalar":       {"type": "string"},
	} {
		genkit.DefinePrompt(g, name, ai.WithInputSchema(schema), ai.WithPrompt("Ready"))
	}
	s := NewMCPServer(g, MCPServerOptions{Name: "skip"})

	listed := len(listPrompts(t, s))
	names := s.ListRegisteredPrompts()
	if diff := cmp.Diff([]string{"empty-object", "no-schema", "optional"}, names); diff != "" {
		t.Fatalf("ListRegisteredPrompts mismatch (-want +got):\n%s", diff)
	}
	if listed != len(names) {
		t.Errorf("prompts/list returned %d prompts, want %d", listed, len(names))
	}
	for _, name := range names {
		if got, err := getPrompt(t, s, name, nil); err != "" || !cmp.Equal(got, []string{"Ready"}) {
			t.Errorf("prompts/get %s = %q, %q, want [Ready]", name, got, err)
		}
	}
}

func TestGenkitMessagesToMCP(t *testing.T) {
	got, err := genkitMessagesToMCP([]*ai.Message{
		ai.NewMessage(ai.RoleSystem, nil, ai.NewTextPart("Be brief."), ai.NewTextPart("Be kind.")),
		ai.NewMessage(ai.RoleSystem, nil, ai.NewMediaPart("", "data:image/png;base64,aGVsbG8="), ai.NewTextPart("Use the logo.")),
		ai.NewUserTextMessage("Hi"),
		ai.NewModelTextMessage("Hello"),
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []mcp.PromptMessage{
		{Role: mcp.RoleUser, Content: mcp.NewTextContent("System instructions:\nBe brief.")},
		{Role: mcp.RoleUser, Content: mcp.NewTextContent("Be kind.")},
		{Role: mcp.RoleUser, Content: mcp.NewTextContent("System instructions:")},
		{Role: mcp.RoleUser, Content: mcp.NewImageContent("aGVsbG8=", "image/png")},
		{Role: mcp.RoleUser, Content: mcp.NewTextContent("Use the logo.")},
		{Role: mcp.RoleUser, Content: mcp.NewTextContent("Hi")},
		{Role: mcp.RoleAssistant, Content: mcp.NewTextContent("Hello")},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("genkitMessagesToMCP mismatch (-want +got):\n%s", diff)
	}
	// MCP requires a messages array, so an empty render must not encode as null.
	if got, err := genkitMessagesToMCP(nil); err != nil || got == nil {
		t.Errorf("genkitMessagesToMCP(nil) = %#v, %v, want an empty slice", got, err)
	}
	if _, err := genkitMessagesToMCP([]*ai.Message{ai.NewMessage(ai.RoleTool, nil, ai.NewTextPart("result"))}); err == nil {
		t.Error("genkitMessagesToMCP accepted a tool message")
	}
}

func TestGenkitPartToMCP(t *testing.T) {
	for _, test := range []struct {
		name string
		part *ai.Part
		want mcp.Content // nil means an error is expected
	}{
		{"text", ai.NewTextPart("Hi"), mcp.NewTextContent("Hi")},
		{"data", ai.NewDataPart(`{"a":1}`), mcp.NewTextContent(`{"a":1}`)},
		{"image", ai.NewMediaPart("image/png", "data:image/png;base64,aGVsbG8="), mcp.NewImageContent("aGVsbG8=", "image/png")},
		{"audio", ai.NewMediaPart("", "data:audio/wav;base64,aGVsbG8="), mcp.NewAudioContent("aGVsbG8=", "audio/wav")},
		{"empty data URI", ai.NewMediaPart("image/png", "data:image/png;base64,"), nil},
		{"empty media", ai.NewMediaPart("audio/wav", ""), nil},
		{"signed remote URL", ai.NewMediaPart("image/png", "https://storage.example.com/a.png?X-Goog-Signature=secret"), nil},
		{"video", ai.NewMediaPart("video/mp4", "data:video/mp4;base64,aGVsbG8="), nil},
		{"percent-encoded data URI", ai.NewMediaPart("image/svg+xml", "data:image/svg+xml,%3Csvg%2F%3E"), nil},
		{"tool request", ai.NewToolRequestPart(&ai.ToolRequest{Name: "lookup"}), nil},
		{"tool response", ai.NewToolResponsePart(&ai.ToolResponse{Name: "lookup"}), nil},
		{"custom", ai.NewCustomPart(map[string]any{"a": 1}), nil},
		{"reasoning", ai.NewReasoningPart("thinking", nil), nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := genkitPartToMCP(test.part)
			if test.want == nil {
				if err == nil {
					t.Fatalf("genkitPartToMCP = %+v, want an error", got)
				}
				// Media URLs can be signed, so errors must not repeat them.
				if test.part.Text != "" && strings.Contains(err.Error(), test.part.Text) {
					t.Errorf("error %q repeats the part's media URL", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("genkitPartToMCP mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
