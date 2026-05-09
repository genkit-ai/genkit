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

package core

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/registry"
)

func inc(_ context.Context, x int, _ noStream) (int, error) {
	return x + 1, nil
}

func TestActionRun(t *testing.T) {
	r := registry.New()
	a := defineAction(r, "test/inc", api.ActionTypeCustom, nil, nil, inc)
	got, err := a.Run(context.Background(), 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := 4; got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestActionRunJSON(t *testing.T) {
	r := registry.New()
	a := defineAction(r, "test/inc", api.ActionTypeCustom, nil, nil, inc)
	input := []byte("3")
	want := []byte("4")
	got, err := a.RunJSON(context.Background(), input, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
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

func TestActionStreaming(t *testing.T) {
	ctx := context.Background()
	r := registry.New()
	a := defineAction(r, "test/count", api.ActionTypeCustom, nil, nil, count)
	const n = 3

	// Non-streaming.
	got, err := a.Run(ctx, n, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != n {
		t.Fatalf("got %d, want %d", got, n)
	}

	// Streaming.
	var gotStreamed []int
	got, err = a.Run(ctx, n, func(_ context.Context, i int) error {
		gotStreamed = append(gotStreamed, i)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	wantStreamed := []int{0, 1, 2}
	if !slices.Equal(gotStreamed, wantStreamed) {
		t.Errorf("got %v, want %v", gotStreamed, wantStreamed)
	}
	if got != n {
		t.Errorf("got %d, want %d", got, n)
	}
}

func TestActionTracing(t *testing.T) {
	r := registry.New()
	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)
	name := api.NewName("test", "TestTracing-inc")
	a := defineAction(r, name, api.ActionTypeCustom, nil, nil, inc)
	if _, err := a.Run(context.Background(), 3, nil); err != nil {
		t.Fatal(err)
	}
	// The same trace store is used for all tests, so there might be several traces.
	// Look for this one, which has a unique name.
	for _, td := range tc.Traces {
		if td.DisplayName == name {
			// Spot check: expect a single span.
			if g, w := len(td.Spans), 1; g != w {
				t.Errorf("got %d spans, want %d", g, w)
			}
			return
		}
	}
	t.Fatalf("did not find trace named %q", name)
}

func TestNewAction(t *testing.T) {
	t.Run("creates unregistered action", func(t *testing.T) {
		fn := func(ctx context.Context, input string) (string, error) {
			return "Hello, " + input, nil
		}
		a := NewAction("greet", api.ActionTypeCustom, nil, nil, fn)

		if a == nil {
			t.Fatal("NewAction returned nil")
		}
		if a.Name() != "greet" {
			t.Errorf("Name() = %q, want %q", a.Name(), "greet")
		}
	})

	t.Run("action can be executed", func(t *testing.T) {
		fn := func(ctx context.Context, input int) (int, error) {
			return input * 2, nil
		}
		a := NewAction("double", api.ActionTypeCustom, nil, nil, fn)

		got, err := a.Run(context.Background(), 5, nil)
		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
		if got != 10 {
			t.Errorf("got %d, want 10", got)
		}
	})

	t.Run("action with custom input schema", func(t *testing.T) {
		customSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		fn := func(ctx context.Context, input any) (string, error) {
			return "ok", nil
		}
		a := NewAction("withSchema", api.ActionTypeCustom, nil, customSchema, fn)

		desc := a.Desc()
		if desc.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
	})

	t.Run("action with metadata", func(t *testing.T) {
		meta := map[string]any{
			"description": "A test action",
			"version":     "1.0",
		}
		fn := func(ctx context.Context, input struct{}) (bool, error) {
			return true, nil
		}
		a := NewAction("withMeta", api.ActionTypeCustom, meta, nil, fn)

		desc := a.Desc()
		if desc.Description != "A test action" {
			t.Errorf("Description = %q, want %q", desc.Description, "A test action")
		}
	})
}

func TestNewStreamingAction(t *testing.T) {
	t.Run("creates streaming action", func(t *testing.T) {
		fn := func(ctx context.Context, n int, cb func(context.Context, int) error) (int, error) {
			if cb != nil {
				for i := 0; i < n; i++ {
					if err := cb(ctx, i); err != nil {
						return 0, err
					}
				}
			}
			return n, nil
		}
		a := NewStreamingAction("counter", api.ActionTypeCustom, nil, nil, fn)

		if a == nil {
			t.Fatal("NewStreamingAction returned nil")
		}
		if a.Name() != "counter" {
			t.Errorf("Name() = %q, want %q", a.Name(), "counter")
		}
	})

	t.Run("streaming action streams correctly", func(t *testing.T) {
		fn := func(ctx context.Context, n int, cb func(context.Context, string) error) (int, error) {
			if cb != nil {
				for i := 0; i < n; i++ {
					if err := cb(ctx, "chunk"); err != nil {
						return 0, err
					}
				}
			}
			return n, nil
		}
		a := NewStreamingAction("streamer", api.ActionTypeCustom, nil, nil, fn)

		var chunks []string
		got, err := a.Run(context.Background(), 3, func(ctx context.Context, chunk string) error {
			chunks = append(chunks, chunk)
			return nil
		})

		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
		if got != 3 {
			t.Errorf("got %d, want 3", got)
		}
		if len(chunks) != 3 {
			t.Errorf("len(chunks) = %d, want 3", len(chunks))
		}
	})
}

func TestActionDesc(t *testing.T) {
	t.Run("returns action descriptor", func(t *testing.T) {
		meta := map[string]any{
			"description": "Test description",
			"custom":      "value",
		}
		fn := func(ctx context.Context, input struct {
			Name string `json:"name"`
		}) (struct {
			Greeting string `json:"greeting"`
		}, error) {
			return struct {
				Greeting string `json:"greeting"`
			}{Greeting: "Hello"}, nil
		}

		r := registry.New()
		a := DefineAction(r, "test/describe", api.ActionTypeCustom, meta, nil, fn)

		desc := a.Desc()

		if desc.Name != "test/describe" {
			t.Errorf("Name = %q, want %q", desc.Name, "test/describe")
		}
		if desc.Description != "Test description" {
			t.Errorf("Description = %q, want %q", desc.Description, "Test description")
		}
		if desc.Type != api.ActionTypeCustom {
			t.Errorf("Type = %v, want %v", desc.Type, api.ActionTypeCustom)
		}
		if desc.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
		if desc.OutputSchema == nil {
			t.Error("OutputSchema is nil")
		}
	})

	t.Run("resolves genkit schema references", func(t *testing.T) {
		r := registry.New()

		inputSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		}
		outputSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
			},
		}
		DefineSchema(r, "AgentRequest", inputSchema)
		DefineSchema(r, "AgentResponse", outputSchema)

		fn := func(ctx context.Context, input any) (any, error) { return nil, nil }
		a := NewAction("test/refs", api.ActionTypeCustom, nil, SchemaRef("AgentRequest"), fn)
		// Override the inferred output schema with a $ref so we can test resolution
		// of OutputSchema as well.
		a.desc.OutputSchema = SchemaRef("AgentResponse")
		a.Register(r)

		desc := a.Desc()

		if _, ok := desc.InputSchema["$ref"]; ok {
			t.Errorf("InputSchema still contains $ref: %v", desc.InputSchema)
		}
		if got, want := desc.InputSchema["type"], "object"; got != want {
			t.Errorf("InputSchema type = %v, want %v", got, want)
		}
		if _, ok := desc.OutputSchema["$ref"]; ok {
			t.Errorf("OutputSchema still contains $ref: %v", desc.OutputSchema)
		}
		if got, want := desc.OutputSchema["type"], "object"; got != want {
			t.Errorf("OutputSchema type = %v, want %v", got, want)
		}
	})

	t.Run("preserves $ref when schema is unregistered", func(t *testing.T) {
		r := registry.New()

		fn := func(ctx context.Context, input any) (any, error) { return nil, nil }
		a := DefineAction(r, "test/unresolved", api.ActionTypeCustom, nil, SchemaRef("Missing"), fn)

		desc := a.Desc()

		ref, ok := desc.InputSchema["$ref"].(string)
		if !ok {
			t.Fatalf("expected InputSchema to retain $ref, got: %v", desc.InputSchema)
		}
		if want := "genkit:Missing"; ref != want {
			t.Errorf("InputSchema $ref = %q, want %q", ref, want)
		}
	})
}

func TestActionRegister(t *testing.T) {
	t.Run("registers action with registry", func(t *testing.T) {
		r := registry.New()
		fn := func(ctx context.Context, input string) (string, error) {
			return input, nil
		}
		a := NewAction("test/register", api.ActionTypeCustom, nil, nil, fn)

		a.Register(r)

		key := api.KeyFromName(api.ActionTypeCustom, "test/register")
		found := r.LookupAction(key)
		if found == nil {
			t.Error("registered action not found in registry")
		}
	})
}

func TestResolveActionFor(t *testing.T) {
	t.Run("finds registered action", func(t *testing.T) {
		r := registry.New()
		fn := func(ctx context.Context, input int) (int, error) {
			return input + 1, nil
		}
		DefineAction(r, "test/resolvable", api.ActionTypeCustom, nil, nil, fn)

		found := ResolveActionFor[int, int, struct{}](r, api.ActionTypeCustom, "test/resolvable")

		if found == nil {
			t.Fatal("ResolveActionFor returned nil")
		}
		if found.Name() != "test/resolvable" {
			t.Errorf("Name() = %q, want %q", found.Name(), "test/resolvable")
		}
	})

	t.Run("returns nil for non-existent action", func(t *testing.T) {
		r := registry.New()

		found := ResolveActionFor[int, int, struct{}](r, api.ActionTypeCustom, "test/nonexistent")

		if found != nil {
			t.Errorf("ResolveActionFor returned %v, want nil", found)
		}
	})
}

func TestLookupActionFor(t *testing.T) {
	t.Run("finds registered action", func(t *testing.T) {
		r := registry.New()
		fn := func(ctx context.Context, input string) (string, error) {
			return "found: " + input, nil
		}
		DefineAction(r, "test/lookupable", api.ActionTypeCustom, nil, nil, fn)

		found := LookupActionFor[string, string, struct{}](r, api.ActionTypeCustom, "test/lookupable")

		if found == nil {
			t.Fatal("LookupActionFor returned nil")
		}
	})

	t.Run("returns nil for non-existent action", func(t *testing.T) {
		r := registry.New()

		found := LookupActionFor[string, string, struct{}](r, api.ActionTypeCustom, "test/missing")

		if found != nil {
			t.Errorf("LookupActionFor returned %v, want nil", found)
		}
	})
}

func TestRunJSONWithTelemetry(t *testing.T) {
	t.Run("returns telemetry info with result", func(t *testing.T) {
		r := registry.New()
		fn := func(ctx context.Context, input int) (int, error) {
			return input * 2, nil
		}
		a := DefineAction(r, "test/telemetry", api.ActionTypeCustom, nil, nil, fn)

		result, err := a.RunJSONWithTelemetry(context.Background(), []byte("5"), nil)

		if err != nil {
			t.Fatalf("RunJSONWithTelemetry error: %v", err)
		}
		if result == nil {
			t.Fatal("result is nil")
		}
		if string(result.Result) != "10" {
			t.Errorf("Result = %s, want %q", result.Result, "10")
		}
		// TraceId and SpanId should be set
		if result.TraceId == "" {
			t.Error("TraceId is empty")
		}
		if result.SpanId == "" {
			t.Error("SpanId is empty")
		}
	})

	t.Run("handles streaming callback", func(t *testing.T) {
		r := registry.New()
		fn := func(ctx context.Context, n int, cb func(context.Context, int) error) (int, error) {
			if cb != nil {
				for i := 0; i < n; i++ {
					if err := cb(ctx, i); err != nil {
						return 0, err
					}
				}
			}
			return n, nil
		}
		a := DefineStreamingAction(r, "test/streamTelemetry", api.ActionTypeCustom, nil, nil, fn)

		var chunks []string
		cb := func(ctx context.Context, chunk json.RawMessage) error {
			chunks = append(chunks, string(chunk))
			return nil
		}

		result, err := a.RunJSONWithTelemetry(context.Background(), []byte("3"), cb)

		if err != nil {
			t.Fatalf("RunJSONWithTelemetry error: %v", err)
		}
		if result == nil {
			t.Fatal("result is nil")
		}
		if len(chunks) != 3 {
			t.Errorf("len(chunks) = %d, want 3", len(chunks))
		}
	})

	t.Run("returns error for invalid JSON input", func(t *testing.T) {
		r := registry.New()
		fn := func(ctx context.Context, input int) (int, error) {
			return input, nil
		}
		a := DefineAction(r, "test/invalidInput", api.ActionTypeCustom, nil, nil, fn)

		_, err := a.RunJSONWithTelemetry(context.Background(), []byte("not valid json"), nil)

		if err == nil {
			t.Error("expected error for invalid JSON, got nil")
		}
	})
}
