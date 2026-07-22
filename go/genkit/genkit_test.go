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

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
)

// panickingPlugin is a plugin whose Init panics, used to verify that Init
// converts escaped panics into returned errors rather than crashing.
type panickingPlugin struct{}

func (panickingPlugin) Name() string { return "panicking" }

func (panickingPlugin) Init(ctx context.Context) []api.Action {
	panic("boom during plugin init")
}

func TestStreamFlow(t *testing.T) {
	g := MustInit(context.Background())

	f := g.DefineStreamingFlow("count", count)
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
	g := MustInit(context.Background())

	type UserInfo struct {
		Name string `json:"name"`
		Age  int    `json:"age,omitempty"`
	}

	g.DefineSchemaFor[UserInfo]()

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
	g := MustInit(context.Background())

	// We expect a panic because DefineSchemaWithType panics on error
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	type Invalid struct {
		Foo func() `json:"foo"`
	}

	g.DefineSchemaFor[Invalid]()
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

			g := MustInit(ctx, opts...)

			prompt := g.LookupPrompt(tt.promptName)
			if prompt == nil {
				t.Fatalf("Expected prompt %q to be loaded", tt.promptName)
			}
		})
	}
}

func TestInitReturnsErrorForMissingPromptDir(t *testing.T) {
	g, err := Init(context.Background(), WithPromptDir("./does-not-exist-prompts"))
	if err == nil {
		t.Fatal("expected an error for a missing explicit prompt directory, got nil")
	}
	if g != nil {
		t.Errorf("expected nil Genkit on error, got %v", g)
	}
}

func TestMustInitPanicsOnError(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected MustInit to panic on initialization error")
		}
	}()
	MustInit(context.Background(), WithPromptDir("./does-not-exist-prompts"))
}

func TestInitConvertsPluginPanicToError(t *testing.T) {
	g, err := Init(context.Background(), WithPlugins(panickingPlugin{}))
	if err == nil {
		t.Fatal("expected an error when a plugin's Init panics, got nil")
	}
	if g != nil {
		t.Errorf("expected nil Genkit on error, got %v", g)
	}
	if !strings.Contains(err.Error(), "boom during plugin init") {
		t.Errorf("expected error to wrap the panic value, got: %v", err)
	}
}
