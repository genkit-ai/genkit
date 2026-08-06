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

package ai

import (
	"context"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/core/api"
)

func TestNewToolWithSchema(t *testing.T) {
	inputSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
	}
	outputSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"weather": map[string]any{"type": "string"},
		},
	}

	created := NewToolWithSchema(
		"weather",
		"Gets the weather",
		ToolSchema{Input: inputSchema, Output: outputSchema},
		func(ctx *ToolContext, input any) (any, error) {
			return map[string]any{"weather": "sunny"}, nil
		},
	)

	if got := created.action.Desc().Type; got != api.ActionTypeToolV2 {
		t.Fatalf("action type = %q, want %q", got, api.ActionTypeToolV2)
	}
	if got := created.Definition().OutputSchema; !reflect.DeepEqual(got, outputSchema) {
		t.Fatalf("definition output schema = %#v, want %#v", got, outputSchema)
	}
	if got := created.action.Desc().OutputSchema; reflect.DeepEqual(got, outputSchema) {
		t.Fatal("action output schema must describe the multipart response, not the exposed tool output")
	}

	got, err := created.RunRaw(context.Background(), map[string]any{"city": "London"})
	if err != nil {
		t.Fatalf("RunRaw() error = %v", err)
	}
	want := map[string]any{"weather": "sunny"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RunRaw() = %#v, want %#v", got, want)
	}

	invalid := NewToolWithSchema(
		"invalid-weather",
		"Returns invalid weather",
		ToolSchema{Input: inputSchema, Output: outputSchema},
		func(ctx *ToolContext, input any) (any, error) {
			return map[string]any{"weather": 25}, nil
		},
	)
	if _, err := invalid.RunRaw(context.Background(), map[string]any{"city": "London"}); err == nil {
		t.Fatal("RunRaw() succeeded with output that does not match the custom schema")
	}
}
