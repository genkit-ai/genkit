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

package genkit

import (
	"context"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
)

func TestStreamFlow(t *testing.T) {
	g := Init(context.Background())

	f := DefineStreamingFlow(g, "count", count)
	iter := f.Stream(context.Background(), 2)
	want := 0
	iter(func(val *core.StreamingFlowValue[int, int], err error) bool {
		if err != nil {
			t.Fatal(err)
		}
		var got int
		if val.Done {
			got = val.Output
		} else {
			got = val.Stream
		}
		if got != want {
			t.Errorf("got %d, want %d", got, want)
		}
		want++
		return true
	})
}

type csvTestFormatter struct{}

func (csvTestFormatter) Name() string { return "csv" }

func (csvTestFormatter) Handler(schema map[string]any) (ai.FormatHandler, error) {
	return csvTestHandler{}, nil
}

// jsonNameFormatter collides with the built-in "json" format.
type jsonNameFormatter struct{ csvTestFormatter }

func (jsonNameFormatter) Name() string { return "json" }

type csvTestHandler struct{}

func (csvTestHandler) ParseMessage(m *ai.Message) (*ai.Message, error) { return m, nil }

func (csvTestHandler) Instructions() string { return "Respond with CSV." }

func (csvTestHandler) Config() ai.ModelOutputConfig {
	return ai.ModelOutputConfig{Format: "csv", ContentType: "text/csv"}
}

func TestDefineFormats(t *testing.T) {
	ctx := context.Background()
	g := Init(ctx)

	echo := DefineModel(g, "test/echo", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("a,b,c")}, nil
	})

	DefineFormats(g, csvTestFormatter{})

	if !IsDefinedFormat(g, "csv") {
		t.Fatal("IsDefinedFormat() = false, want true")
	}

	res, err := Generate(ctx, g,
		ai.WithModel(echo),
		ai.WithPrompt("list some letters"),
		ai.WithOutputFormat("csv"),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if res.Request.Output.Format != "csv" {
		t.Errorf("output format = %q, want %q", res.Request.Output.Format, "csv")
	}
	if res.Request.Output.ContentType != "text/csv" {
		t.Errorf("output content type = %q, want %q", res.Request.Output.ContentType, "text/csv")
	}

	t.Run("deprecated DefineFormat registers under the given name", func(t *testing.T) {
		DefineFormat(g, "csv2", csvTestFormatter{})
		if !IsDefinedFormat(g, "csv2") {
			t.Error("IsDefinedFormat() = false, want true")
		}
	})

	t.Run("deprecated DefineFormat tolerates a prefixed name", func(t *testing.T) {
		DefineFormat(g, "/format/csv3", csvTestFormatter{})
		if !IsDefinedFormat(g, "csv3") {
			t.Error("IsDefinedFormat() = false, want true")
		}
		// IsDefinedFormat has to accept whatever DefineFormat accepted.
		if !IsDefinedFormat(g, "/format/csv3") {
			t.Error("IsDefinedFormat() = false for the prefixed name, want true")
		}
	})

	t.Run("DefineFormats panics on a built-in name", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("DefineFormats() should panic when the name collides with a built-in")
			}
		}()
		DefineFormats(g, jsonNameFormatter{})
	})
}

// count streams the numbers from 0 to n-1, then returns n.
func count(ctx context.Context, n int, cb func(context.Context, int) error) (int, error) {
	if cb != nil {
		for i := range n {
			if err := cb(ctx, i); err != nil {
				return 0, err
			}
		}
	}
	return n, nil
}

func TestDefineSchemaWithType(t *testing.T) {
	g := Init(context.Background())

	type UserInfo struct {
		Name string `json:"name"`
		Age  int    `json:"age,omitempty"`
	}

	DefineSchemasFor(g, UserInfo{})

	schema := g.reg.LookupSchema("UserInfo")
	if schema == nil {
		t.Fatal("Schema UserInfo not found")
	}

	if schema["type"] != "object" {
		t.Errorf("Expected type object, got %v", schema["type"])
	}

	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("Properties not found or invalid type")
	}

	if _, ok := props["name"]; !ok {
		t.Error("Property 'name' not found")
	}
	if _, ok := props["age"]; !ok {
		t.Error("Property 'age' not found")
	}

	required, ok := schema["required"].([]any)
	if !ok {
		t.Fatal("Required fields not found or invalid type")
	}
	// jsonschema reflection makes fields required by default unless omitempty
	foundName := false
	for _, r := range required {
		if r == "name" {
			foundName = true
			break
		}
	}
	if !foundName {
		t.Error("Expected 'name' to be required")
	}
}

func TestDefineSchemaWithType_Error(t *testing.T) {
	g := Init(context.Background())

	// We expect a panic because DefineSchemaWithType panics on error
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	type Invalid struct {
		Foo func() `json:"foo"`
	}

	DefineSchemasFor(g, Invalid{})
}

func TestDefineSchemasFor(t *testing.T) {
	t.Run("registers multiple schemas at once", func(t *testing.T) {
		g := Init(context.Background())

		type User struct {
			Name string `json:"name"`
		}
		type Order struct {
			ID string `json:"id"`
		}

		DefineSchemasFor(g, User{}, &Order{})

		for _, name := range []string{"User", "Order"} {
			if g.reg.LookupSchema(name) == nil {
				t.Errorf("Schema %s not found", name)
			}
		}
	})

	t.Run("panics on map value", func(t *testing.T) {
		g := Init(context.Background())

		defer func() {
			if recover() == nil {
				t.Error("expected panic for map value")
			}
		}()

		DefineSchemasFor(g, map[string]any{"type": "object"})
	})

	t.Run("panics on unnamed type", func(t *testing.T) {
		g := Init(context.Background())

		defer func() {
			if recover() == nil {
				t.Error("expected panic for unnamed type")
			}
		}()

		DefineSchemasFor(g, struct{ Name string }{})
	})

	t.Run("panics on nil value", func(t *testing.T) {
		g := Init(context.Background())

		defer func() {
			if recover() == nil {
				t.Error("expected panic for nil value")
			}
		}()

		DefineSchemasFor(g, nil)
	})
}

func TestDefineSchemaFor(t *testing.T) {
	g := Init(context.Background())

	type Legacy struct {
		Name string `json:"name"`
	}
	type LegacyPtr struct {
		Name string `json:"name"`
	}

	DefineSchemaFor[Legacy](g)
	DefineSchemaFor[*LegacyPtr](g)

	for _, name := range []string{"Legacy", "LegacyPtr"} {
		if g.reg.LookupSchema(name) == nil {
			t.Errorf("Schema %s not found", name)
		}
	}

	t.Run("guard panic names DefineSchemaFor", func(t *testing.T) {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic for map type")
			}
			if msg, ok := r.(string); !ok || !strings.HasPrefix(msg, "genkit.DefineSchemaFor:") {
				t.Errorf("panic = %v, want it to name genkit.DefineSchemaFor", r)
			}
		}()

		DefineSchemaFor[map[string]any](g)
	})
}

func TestWithPromptFS(t *testing.T) {
	tests := []struct {
		name       string
		fsys       fstest.MapFS
		promptDir  string
		promptName string
	}{
		{
			name: "with custom prompt directory",
			fsys: fstest.MapFS{
				"custom-prompts/test.prompt": &fstest.MapFile{
					Data: []byte(`---
model: googleai/gemini-2.5-flash
input:
  schema:
    text: string
---
{{text}}`),
				},
			},
			promptDir:  "custom-prompts",
			promptName: "test",
		},
		{
			name: "with default prompts directory",
			fsys: fstest.MapFS{
				"prompts/test.prompt": &fstest.MapFile{
					Data: []byte(`---
model: googleai/gemini-2.5-flash
input:
  schema:
    text: string
---
{{text}}`),
				},
			},
			promptDir:  "", // empty means use default
			promptName: "test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			var opts []GenkitOption
			opts = append(opts, WithPromptFS(tt.fsys))
			if tt.promptDir != "" {
				opts = append(opts, WithPromptDir(tt.promptDir))
			}

			g := Init(ctx, opts...)

			prompt := LookupPrompt(g, tt.promptName)
			if prompt == nil {
				t.Fatalf("Expected prompt %q to be loaded", tt.promptName)
			}
		})
	}
}

// pluginWithTool returns one of each primitive that a plugin is expected to be
// able to ship, matching the plugin-authoring example in the core package
// documentation.
type pluginWithTool struct{}

func (p *pluginWithTool) Name() string { return "toolplugin" }

func (p *pluginWithTool) Init(ctx context.Context) []api.Action {
	model := ai.NewModelAction("toolplugin/model", nil,
		func(ctx context.Context, req *ai.ModelRequest, _ any, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Message: ai.NewModelTextMessage("ok")}, nil
		})
	tool := ai.NewTool("toolplugin/tool", "a tool shipped by a plugin",
		func(ctx *ai.ToolContext, in string) (string, error) {
			return in + "!", nil
		})
	return []api.Action{model, tool}
}

// A plugin ships tools the same way it ships models: by returning them in its
// action slice. This is the shape the core package documentation prescribes,
// and it only compiles because *ai.ToolAction satisfies api.Action.
func TestPluginCanReturnTools(t *testing.T) {
	g := Init(context.Background(), WithPlugins(&pluginWithTool{}))

	if m := LookupModel(g, "toolplugin/model"); m == nil {
		t.Error("LookupModel() = nil, want the model the plugin returned")
	}

	tool := LookupTool(g, "toolplugin/tool")
	if tool == nil {
		t.Fatal("LookupTool() = nil, want the tool the plugin returned")
	}
	out, err := tool.RunRaw(context.Background(), "hi")
	if err != nil {
		t.Fatalf("RunRaw() error: %v", err)
	}
	if out != "hi!" {
		t.Errorf("RunRaw() = %v, want %q", out, "hi!")
	}
}
