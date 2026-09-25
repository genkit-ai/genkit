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
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/go-cmp/cmp"
)

func TestToolName(t *testing.T) {
	t.Run("Name returns string value", func(t *testing.T) {
		tn := ToolName("myTool")
		got := tn.Name()
		want := "myTool"
		if got != want {
			t.Errorf("Name() = %q, want %q", got, want)
		}
	})

	t.Run("empty tool name", func(t *testing.T) {
		tn := ToolName("")
		got := tn.Name()
		if got != "" {
			t.Errorf("Name() = %q, want empty string", got)
		}
	})
}

func TestIsToolInterruptError(t *testing.T) {
	t.Run("returns true for an interrupt error", func(t *testing.T) {
		meta := map[string]any{"reason": "user cancelled"}
		err := &base.ToolInterruptError{Data: meta}

		isInterrupt, gotMeta := IsToolInterruptError(err)

		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		if diff := cmp.Diff(meta, gotMeta); diff != "" {
			t.Errorf("metadata mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns true for a wrapped interrupt error", func(t *testing.T) {
		meta := map[string]any{"step": 3}
		innerErr := &base.ToolInterruptError{Data: meta}
		wrappedErr := errors.New("context: " + innerErr.Error())
		// Use proper wrapping
		wrappedErr = &wrappedInterruptError{cause: innerErr}

		isInterrupt, gotMeta := IsToolInterruptError(wrappedErr)

		if !isInterrupt {
			t.Error("IsToolInterruptError(wrapped) = false, want true")
		}
		if gotMeta["step"] != 3 {
			t.Errorf("metadata[step] = %v, want 3", gotMeta["step"])
		}
	})

	t.Run("returns false for regular error", func(t *testing.T) {
		err := errors.New("some error")

		isInterrupt, meta := IsToolInterruptError(err)

		if isInterrupt {
			t.Error("IsToolInterruptError(regular error) = true, want false")
		}
		if meta != nil {
			t.Errorf("metadata = %v, want nil", meta)
		}
	})

	t.Run("returns false for nil error", func(t *testing.T) {
		isInterrupt, meta := IsToolInterruptError(nil)

		if isInterrupt {
			t.Error("IsToolInterruptError(nil) = true, want false")
		}
		if meta != nil {
			t.Errorf("metadata = %v, want nil", meta)
		}
	})
}

// wrappedInterruptError is a helper for testing error unwrapping.
type wrappedInterruptError struct {
	cause error
}

func (e *wrappedInterruptError) Error() string {
	return "wrapped: " + e.cause.Error()
}

func (e *wrappedInterruptError) Unwrap() error {
	return e.cause
}

func TestDefineTool(t *testing.T) {
	t.Run("creates and registers tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/addNumbers", "Adds two numbers", func(ctx *ToolContext, input struct {
			A int `json:"a"`
			B int `json:"b"`
		}) (int, error) {
			return input.A + input.B, nil
		})

		if tl == nil {
			t.Fatal("DefineTool returned nil")
		}
		if tl.Name() != "provider/addNumbers" {
			t.Errorf("Name() = %q, want %q", tl.Name(), "provider/addNumbers")
		}

		def := tl.Definition()
		if def.Description != "Adds two numbers" {
			t.Errorf("Description = %q, want %q", def.Description, "Adds two numbers")
		}
	})

	t.Run("tool can be looked up after registration", func(t *testing.T) {
		r := newTestRegistry(t)
		defineTool(r, "provider/multiply", "Multiplies", func(ctx *ToolContext, input struct {
			X int `json:"x"`
			Y int `json:"y"`
		}) (int, error) {
			return input.X * input.Y, nil
		})

		found := LookupTool(r, "provider/multiply")
		if found == nil {
			t.Error("LookupTool returned nil for registered tool")
		}
	})

	t.Run("tool executes correctly", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/concat", "Concatenates strings", func(ctx *ToolContext, input struct {
			A string `json:"a"`
			B string `json:"b"`
		}) (string, error) {
			return input.A + input.B, nil
		})

		output, err := tl.RunRaw(context.Background(), map[string]any{
			"a": "hello",
			"b": "world",
		})

		if err != nil {
			t.Fatalf("RunRaw error: %v", err)
		}
		if output != "helloworld" {
			t.Errorf("output = %v, want %q", output, "helloworld")
		}
	})
}

func TestDefineToolWithInputSchema(t *testing.T) {
	t.Run("creates tool with custom input schema", func(t *testing.T) {
		r := newTestRegistry(t)
		customSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
			"required": []any{"query"},
		}

		tl := defineToolWithInputSchema(r, "provider/search", "Searches", customSchema,
			func(ctx *ToolContext, input any) (string, error) {
				m := input.(map[string]any)
				return "results for: " + m["query"].(string), nil
			})

		if tl == nil {
			t.Fatal("DefineToolWithInputSchema returned nil")
		}

		def := tl.Definition()
		if def.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
	})
}

func TestNewTool(t *testing.T) {
	t.Run("creates unregistered tool", func(t *testing.T) {
		tl := NewTool("dynamicTool", "A dynamic tool", func(ctx *ToolContext, input struct {
			Value int `json:"value"`
		}) (int, error) {
			return input.Value * 2, nil
		})

		if tl == nil {
			t.Fatal("NewTool returned nil")
		}
		if tl.Name() != "dynamicTool" {
			t.Errorf("Name() = %q, want %q", tl.Name(), "dynamicTool")
		}
	})

	t.Run("unregistered tool can be executed", func(t *testing.T) {
		tl := NewTool("double", "Doubles a number", func(ctx *ToolContext, input struct {
			N int `json:"n"`
		}) (int, error) {
			return input.N * 2, nil
		})

		output, err := tl.RunRaw(context.Background(), map[string]any{"n": 5})
		if err != nil {
			t.Fatalf("RunRaw error: %v", err)
		}
		// JSON unmarshalling returns float64 for numbers
		if output != float64(10) {
			t.Errorf("output = %v (%T), want 10", output, output)
		}
	})

	t.Run("tool can be registered later", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := NewTool("provider/laterTool", "Registered later", func(ctx *ToolContext, input struct{}) (string, error) {
			return "done", nil
		})

		tl.Register(r)

		found := LookupTool(r, "provider/laterTool")
		if found == nil {
			t.Error("LookupTool returned nil after registration")
		}
	})
}

func TestNewToolWithInputSchema(t *testing.T) {
	t.Run("creates tool with custom schema", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"data": map[string]any{"type": "array"},
			},
		}

		tl := NewToolWithInputSchema("process", "Processes data", schema,
			func(ctx *ToolContext, input any) (bool, error) {
				return true, nil
			})

		if tl == nil {
			t.Fatal("NewToolWithInputSchema returned nil")
		}

		def := tl.Definition()
		if def.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
	})
}

func TestDefineMultipartTool(t *testing.T) {
	t.Run("creates multipart tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineMultipartTool(r, "provider/imageGen", "Generates images",
			func(ctx *ToolContext, input struct {
				Prompt string `json:"prompt"`
			}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output: "generated",
					Content: []*Part{
						NewMediaPart("image/png", "data:image/png;base64,abc"),
					},
				}, nil
			})

		if tl == nil {
			t.Fatal("DefineMultipartTool returned nil")
		}

		// Check that it's a multipart tool via metadata
		def := tl.Definition()
		if def.Metadata == nil {
			t.Fatal("Metadata is nil")
		}
		if def.Metadata["multipart"] != true {
			t.Error("multipart metadata = false, want true")
		}
	})

	t.Run("multipart tool returns parts", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineMultipartTool(r, "provider/multiOut", "Returns multiple parts",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output: map[string]any{"status": "ok"},
					Content: []*Part{
						NewTextPart("additional text"),
						NewMediaPart("image/jpeg", "data:image/jpeg;base64,xyz"),
					},
				}, nil
			})

		resp, err := tl.RunRawMultipart(context.Background(), map[string]any{})
		if err != nil {
			t.Fatalf("RunRawMultipart error: %v", err)
		}

		if len(resp.Content) != 2 {
			t.Errorf("len(Content) = %d, want 2", len(resp.Content))
		}
	})
}

func TestNewMultipartTool(t *testing.T) {
	t.Run("creates unregistered multipart tool", func(t *testing.T) {
		tl := NewMultipartTool("dynamicMulti", "Dynamic multipart",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{Output: "test"}, nil
			})

		if tl == nil {
			t.Fatal("NewMultipartTool returned nil")
		}
		// Check via definition metadata
		def := tl.Definition()
		if def.Metadata["multipart"] != true {
			t.Error("multipart metadata = false, want true")
		}
	})

	t.Run("can be registered later", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := NewMultipartTool("provider/laterMulti", "Later registration",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{Output: "ok"}, nil
			})

		tl.Register(r)

		found := LookupTool(r, "provider/laterMulti")
		if found == nil {
			t.Error("LookupTool returned nil after registration")
		}
	})
}

func TestToolDefinition(t *testing.T) {
	t.Run("includes all fields", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/complete", "A complete tool", func(ctx *ToolContext, input struct {
			Query string `json:"query"`
		}) (struct {
			Result string `json:"result"`
		}, error) {
			return struct {
				Result string `json:"result"`
			}{Result: input.Query}, nil
		})

		def := tl.Definition()

		if def.Name != "provider/complete" {
			t.Errorf("Name = %q, want %q", def.Name, "provider/complete")
		}
		if def.Description != "A complete tool" {
			t.Errorf("Description = %q, want %q", def.Description, "A complete tool")
		}
		if def.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
		if def.OutputSchema == nil {
			t.Error("OutputSchema is nil")
		}
	})
}

// TestToolDefinition_OutputSchema pins what a tool advertises as its output:
// the schema of its output type, an explicit override when given, and nothing
// at all when the output type carries no schema. The action's own output schema
// is the multipart envelope every tool function is wrapped in, and that must
// never reach a model.
func TestToolDefinition_OutputSchema(t *testing.T) {
	type weather struct {
		Temp int    `json:"temp"`
		Sky  string `json:"sky"`
	}

	t.Run("typed output advertises its schema", func(t *testing.T) {
		tl := NewTool("typed", "d", func(ctx *ToolContext, _ struct{}) (weather, error) {
			return weather{}, nil
		})
		props, _ := tl.Definition().OutputSchema["properties"].(map[string]any)
		if _, ok := props["temp"]; !ok {
			t.Errorf("output schema = %#v, want the weather fields", tl.Definition().OutputSchema)
		}
	})

	t.Run("any output advertises no schema", func(t *testing.T) {
		tl := NewTool("anyOut", "d", func(ctx *ToolContext, _ struct{}) (any, error) {
			return "anything at all", nil
		})
		if got := tl.Definition().OutputSchema; got != nil {
			t.Errorf("output schema = %#v, want none: an unconstrained output is described by no schema", got)
		}
	})

	t.Run("multipart tool advertises no schema", func(t *testing.T) {
		tl := NewMultipartTool("multi", "d", func(ctx *ToolContext, _ struct{}) (*MultipartToolResponse, error) {
			return &MultipartToolResponse{Output: "ok"}, nil
		})
		got := tl.Definition().OutputSchema
		if props, ok := got["properties"].(map[string]any); ok {
			if _, leaked := props["content"]; leaked {
				t.Errorf("output schema leaked the multipart envelope: %#v", got)
			}
		}
		if got != nil {
			t.Errorf("output schema = %#v, want none", got)
		}
	})

	t.Run("explicit output schema wins", func(t *testing.T) {
		custom := map[string]any{
			"type":       "object",
			"properties": map[string]any{"custom": map[string]any{"type": "string"}},
		}
		tl := NewTool("override", "d",
			func(ctx *ToolContext, _ struct{}) (any, error) { return nil, nil },
			WithOutputSchema(custom))
		if diff := cmp.Diff(custom, tl.Definition().OutputSchema); diff != "" {
			t.Errorf("output schema mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestLookupTool(t *testing.T) {
	t.Run("returns nil for empty name", func(t *testing.T) {
		r := newTestRegistry(t)
		got := LookupTool(r, "")
		if got != nil {
			t.Errorf("LookupTool(\"\") = %v, want nil", got)
		}
	})

	t.Run("returns nil for non-existent tool", func(t *testing.T) {
		r := newTestRegistry(t)
		got := LookupTool(r, "nonexistent/tool")
		if got != nil {
			t.Errorf("LookupTool(nonexistent) = %v, want nil", got)
		}
	})

	t.Run("finds registered tool", func(t *testing.T) {
		r := newTestRegistry(t)
		defineTool(r, "test/findMe", "Find me", func(ctx *ToolContext, input struct{}) (bool, error) {
			return true, nil
		})

		got := LookupTool(r, "test/findMe")
		if got == nil {
			t.Error("LookupTool returned nil for registered tool")
		}
	})
}

// TestWithStrictSchema verifies the strict-schema flag round-trips through
// Definition().Metadata["strict"] and LookupTool, for both registered and
// dynamic tools.
func TestWithStrictSchema(t *testing.T) {
	type runOnTool func(*testing.T, Tool)

	t.Run("absent by default", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "strict/default", "no strict opt", func(ctx *ToolContext, input struct{}) (string, error) {
			return "", nil
		})
		def := tl.Definition()
		if _, ok := def.Metadata["strict"]; ok {
			t.Errorf("expected strict metadata to be absent by default, got %v", def.Metadata["strict"])
		}
	})

	check := func(want bool) runOnTool {
		return func(t *testing.T, tl Tool) {
			t.Helper()
			def := tl.Definition()
			got, ok := def.Metadata["strict"]
			if !ok {
				t.Fatalf("expected strict metadata to be present, got nothing")
			}
			if got != want {
				t.Errorf("strict metadata = %v, want %v", got, want)
			}
		}
	}

	t.Run("DefineTool with WithStrictSchema(true) is surfaced on Definition", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "strict/registered-true", "registered strict",
			func(ctx *ToolContext, input struct{}) (string, error) { return "", nil },
			WithStrictSchema(true),
		)
		check(true)(t, tl)
	})

	t.Run("DefineTool with WithStrictSchema(false) is surfaced on Definition", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "strict/registered-false", "registered loose",
			func(ctx *ToolContext, input struct{}) (string, error) { return "", nil },
			WithStrictSchema(false),
		)
		check(false)(t, tl)
	})

	t.Run("LookupTool round-trips the strict flag for registered tools", func(t *testing.T) {
		r := newTestRegistry(t)
		defineTool(r, "strict/lookup-true", "registered strict",
			func(ctx *ToolContext, input struct{}) (string, error) { return "", nil },
			WithStrictSchema(true),
		)
		found := LookupTool(r, "strict/lookup-true")
		if found == nil {
			t.Fatal("LookupTool returned nil")
		}
		check(true)(t, found)
	})

	t.Run("LookupTool round-trips the strict flag for dynamic tools after Register", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := NewTool("strict/dynamic-false", "dynamic loose",
			func(ctx *ToolContext, input struct{}) (string, error) { return "", nil },
			WithStrictSchema(false),
		)
		check(false)(t, tl)

		tl.Register(r)
		found := LookupTool(r, "strict/dynamic-false")
		if found == nil {
			t.Fatal("LookupTool returned nil")
		}
		check(false)(t, found)
	})

	t.Run("DefineMultipartTool plumbs strict the same way", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineMultipartTool(r, "strict/multipart", "multipart strict",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{}, nil
			},
			WithStrictSchema(true),
		)
		check(true)(t, tl)
	})

	t.Run("setting strict twice takes the last value", func(t *testing.T) {
		// WithStrictSchema fills a single slot, so repeating it overwrites
		// rather than failing.
		r := newTestRegistry(t)
		tl := defineTool(r, "strict/double-set", "double set",
			func(ctx *ToolContext, input struct{}) (string, error) { return "", nil },
			WithStrictSchema(true),
			WithStrictSchema(false),
		)
		check(false)(t, tl)
	})
}

func TestToolIsMultipart(t *testing.T) {
	t.Run("regular tool is not multipart", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/regular", "Regular tool", func(ctx *ToolContext, input struct{}) (string, error) {
			return "ok", nil
		})

		def := tl.Definition()
		if def.Metadata["multipart"] == true {
			t.Error("multipart metadata = true for regular tool, want false")
		}
	})

	t.Run("multipart tool is multipart", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineMultipartTool(r, "provider/multi", "Multi tool",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{}, nil
			})

		def := tl.Definition()
		if def.Metadata["multipart"] != true {
			t.Error("multipart metadata = false for multipart tool, want true")
		}
	})
}

func TestToolRunRaw(t *testing.T) {
	t.Run("returns output from regular tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/sum", "Sums numbers", func(ctx *ToolContext, input struct {
			Nums []int `json:"nums"`
		}) (int, error) {
			sum := 0
			for _, n := range input.Nums {
				sum += n
			}
			return sum, nil
		})

		output, err := tl.RunRaw(context.Background(), map[string]any{
			"nums": []any{1, 2, 3, 4, 5},
		})

		if err != nil {
			t.Fatalf("RunRaw error: %v", err)
		}
		// JSON unmarshalling returns float64 for numbers
		if output != float64(15) {
			t.Errorf("output = %v (%T), want 15", output, output)
		}
	})

	t.Run("returns error from tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/fail", "Always fails", func(ctx *ToolContext, input struct{}) (string, error) {
			return "", errors.New("intentional failure")
		})

		_, err := tl.RunRaw(context.Background(), map[string]any{})
		if err == nil {
			t.Error("expected error, got nil")
		}
	})
}

func TestToolRunRawMultipart(t *testing.T) {
	t.Run("returns full response from multipart tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineMultipartTool(r, "provider/fullResp", "Full response",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output: "main output",
					Content: []*Part{
						NewTextPart("extra"),
					},
				}, nil
			})

		resp, err := tl.RunRawMultipart(context.Background(), map[string]any{})
		if err != nil {
			t.Fatalf("RunRawMultipart error: %v", err)
		}

		if resp.Output != "main output" {
			t.Errorf("Output = %v, want %q", resp.Output, "main output")
		}
		if len(resp.Content) != 1 {
			t.Errorf("len(Content) = %d, want 1", len(resp.Content))
		}
	})
}

func TestToolRespond(t *testing.T) {
	r := newTestRegistry(t)
	tl := defineTool(r, "provider/responder", "Test responder", func(ctx *ToolContext, input struct{}) (string, error) {
		return "ok", nil
	})

	t.Run("creates response for tool request", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name:  "provider/responder",
			Ref:   "ref-123",
			Input: map[string]any{"x": 1},
		})
		reqPart.Interrupt = &ToolInterrupt{}

		resp := tl.Respond(reqPart, "output data", nil)

		if resp == nil {
			t.Fatal("Respond returned nil")
		}
		if !resp.IsToolResponse() {
			t.Error("response is not a tool response")
		}
		if resp.ToolResponse.Name != "provider/responder" {
			t.Errorf("Name = %q, want %q", resp.ToolResponse.Name, "provider/responder")
		}
		if resp.ToolResponse.Ref != "ref-123" {
			t.Errorf("Ref = %q, want %q", resp.ToolResponse.Ref, "ref-123")
		}
	})

	t.Run("returns nil for non-tool-request part", func(t *testing.T) {
		textPart := NewTextPart("not a tool request")

		resp := tl.Respond(textPart, "output", nil)

		if resp != nil {
			t.Errorf("Respond(textPart) = %v, want nil", resp)
		}
	})

	t.Run("returns nil for nil part", func(t *testing.T) {
		resp := tl.Respond(nil, "output", nil)

		if resp != nil {
			t.Errorf("Respond(nil) = %v, want nil", resp)
		}
	})

	t.Run("includes response options metadata", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name: "provider/responder",
		})
		reqPart.Interrupt = &ToolInterrupt{}

		opts := &RespondOptions{
			Metadata: map[string]any{"custom": "value"},
		}
		resp := tl.Respond(reqPart, "output", opts)

		if resp.Metadata == nil {
			t.Fatal("Metadata is nil")
		}
		if resp.Metadata["interruptResponse"] == nil {
			t.Error("interruptResponse not set in metadata")
		}
	})
}

func TestToolRestart(t *testing.T) {
	r := newTestRegistry(t)
	tl := defineTool(r, "provider/restarter", "Test restarter", func(ctx *ToolContext, input struct {
		Value int `json:"value"`
	}) (int, error) {
		return input.Value, nil
	})

	t.Run("creates restart for tool request", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name:  "provider/restarter",
			Ref:   "ref-456",
			Input: map[string]any{"value": 10},
		})
		reqPart.Interrupt = &ToolInterrupt{}

		restart := tl.Restart(reqPart, nil)

		if restart == nil {
			t.Fatal("Restart returned nil")
		}
		if !restart.IsToolRequest() {
			t.Error("restart is not a tool request")
		}
		if restart.ToolRequest.Name != "provider/restarter" {
			t.Errorf("Name = %q, want %q", restart.ToolRequest.Name, "provider/restarter")
		}
		if restart.Restart == nil || restart.Restart.Resume != nil {
			t.Errorf("Restart = %+v, want a bare restart", restart.Restart)
		}
		if restart.Interrupt != nil {
			t.Error("the restart part must not carry interrupt state")
		}
		// The typed state folds back into the JS-compatible wire keys.
		wire := wireMetadataOf(t, restart)
		if wire["resumed"] != true {
			t.Errorf("wire resumed = %v, want true", wire["resumed"])
		}
		if _, ok := wire["interrupt"]; ok {
			t.Error("interrupt should not be on the wire for a restart part")
		}
	})

	t.Run("returns nil for non-tool-request part", func(t *testing.T) {
		textPart := NewTextPart("text")

		restart := tl.Restart(textPart, nil)

		if restart != nil {
			t.Errorf("Restart(textPart) = %v, want nil", restart)
		}
	})

	t.Run("returns nil for nil part", func(t *testing.T) {
		restart := tl.Restart(nil, nil)

		if restart != nil {
			t.Errorf("Restart(nil) = %v, want nil", restart)
		}
	})

	t.Run("replaces input when specified", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name:  "provider/restarter",
			Input: map[string]any{"value": 10},
		})
		reqPart.Interrupt = &ToolInterrupt{}

		newInputVal := struct {
			Value int `json:"value"`
		}{Value: 20}
		opts := &RestartOptions{
			ReplaceInput: newInputVal,
		}
		restart := tl.Restart(reqPart, opts)

		newInput := restart.ToolRequest.Input.(struct {
			Value int `json:"value"`
		})
		if newInput.Value != 20 {
			t.Errorf("new input value = %v, want 20", newInput.Value)
		}
		if restart.Restart == nil || restart.Restart.OriginalInput == nil {
			t.Error("the original input was not preserved on the restart")
		}
		if wireMetadataOf(t, restart)["replacedInput"] == nil {
			t.Error("replacedInput not set on the wire")
		}
	})

	t.Run("sets resumed metadata when specified", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name: "provider/restarter",
		})
		reqPart.Interrupt = &ToolInterrupt{}

		opts := &RestartOptions{
			ResumedMetadata: map[string]any{"reason": "user confirmed"},
		}
		restart := tl.Restart(reqPart, opts)

		resumed := restart.Restart.Resume.(map[string]any)
		if resumed["reason"] != "user confirmed" {
			t.Errorf("resumed.reason = %v, want %q", resumed["reason"], "user confirmed")
		}
	})
}

func TestToolInterrupt(t *testing.T) {
	t.Run("tool can interrupt execution", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/interrupter", "Can interrupt",
			func(ctx *ToolContext, input struct {
				ShouldInterrupt bool `json:"shouldInterrupt"`
			}) (string, error) {
				if input.ShouldInterrupt {
					return "", ctx.Interrupt(&InterruptOptions{
						Metadata: map[string]any{"step": "confirmation"},
					})
				}
				return "completed", nil
			})

		_, err := tl.RunRaw(context.Background(), map[string]any{
			"shouldInterrupt": true,
		})

		if err == nil {
			t.Fatal("expected interrupt error, got nil")
		}

		isInterrupt, meta := IsToolInterruptError(err)
		if !isInterrupt {
			t.Errorf("IsToolInterruptError() = false, want true")
		}
		if meta["step"] != "confirmation" {
			t.Errorf("metadata[step] = %v, want %q", meta["step"], "confirmation")
		}
	})

	t.Run("tool completes without interrupt", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/noInterrupt", "No interrupt",
			func(ctx *ToolContext, input struct {
				ShouldInterrupt bool `json:"shouldInterrupt"`
			}) (string, error) {
				if input.ShouldInterrupt {
					return "", ctx.Interrupt(&InterruptOptions{})
				}
				return "completed", nil
			})

		output, err := tl.RunRaw(context.Background(), map[string]any{
			"shouldInterrupt": false,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if output != "completed" {
			t.Errorf("output = %v, want %q", output, "completed")
		}
	})
}

func TestToolWithInputSchemaOption(t *testing.T) {
	t.Run("DefineTool with WithInputSchema", func(t *testing.T) {
		r := newTestRegistry(t)
		customSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"customField": map[string]any{"type": "string"},
			},
		}

		tl := defineTool(r, "provider/customInput", "Custom input schema",
			func(ctx *ToolContext, input any) (string, error) {
				m := input.(map[string]any)
				return m["customField"].(string), nil
			},
			WithInputSchema(customSchema))

		def := tl.Definition()
		if def.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
	})

	t.Run("NewTool with WithInputSchema", func(t *testing.T) {
		customSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"field": map[string]any{"type": "number"},
			},
		}

		tl := NewTool("customNew", "Custom new tool",
			func(ctx *ToolContext, input any) (bool, error) {
				return true, nil
			},
			WithInputSchema(customSchema))

		def := tl.Definition()
		if def.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
	})
}

func TestToolWithOutputSchemaOptions(t *testing.T) {
	customSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
	}

	t.Run("registered tool with WithOutputSchema", func(t *testing.T) {
		r := newTestRegistry(t)

		tl := defineTool(r, "provider/customOutput", "Custom output schema",
			func(ctx *ToolContext, input struct{}) (any, error) { return nil, nil },
			WithOutputSchema(customSchema))

		def := tl.Definition()
		props, ok := def.OutputSchema["properties"].(map[string]any)
		if !ok || props["answer"] == nil {
			t.Errorf("OutputSchema = %v, want the custom schema", def.OutputSchema)
		}
	})

	t.Run("NewTool with WithOutputSchema", func(t *testing.T) {
		tl := NewTool("customOutputNew", "Custom output schema",
			func(ctx *ToolContext, input struct{}) (any, error) { return nil, nil },
			WithOutputSchema(customSchema))

		def := tl.Definition()
		props, ok := def.OutputSchema["properties"].(map[string]any)
		if !ok || props["answer"] == nil {
			t.Errorf("OutputSchema = %v, want the custom schema", def.OutputSchema)
		}
	})

	t.Run("WithOutputSchemaName resolves through the registry", func(t *testing.T) {
		r := newTestRegistry(t)
		r.RegisterSchema("Answer", customSchema)

		tl := defineTool(r, "provider/namedOutput", "Named output schema",
			func(ctx *ToolContext, input struct{}) (any, error) { return nil, nil },
			WithOutputSchemaName("Answer"))

		def := tl.Definition()
		props, ok := def.OutputSchema["properties"].(map[string]any)
		if !ok || props["answer"] == nil {
			t.Errorf("OutputSchema = %v, want the registered Answer schema", def.OutputSchema)
		}
	})

	t.Run("panics when Out is not any", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("expected panic for concrete Out with a custom output schema")
			}
		}()

		NewTool("badOut", "Concrete out",
			func(ctx *ToolContext, input struct{}) (string, error) { return "", nil },
			WithOutputSchema(customSchema))
	})

	// The multipart constructor honors the same input schema option, so it
	// needs the same guard: a concrete In would advertise the custom schema
	// and then decode into a zero value, with no error anywhere.
	t.Run("NewMultipartTool panics when In is not any", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("expected panic for concrete In with a custom input schema")
			}
		}()

		NewMultipartTool("badMultipartIn", "Concrete in",
			func(ctx *ToolContext, input struct{ City string }) (*MultipartToolResponse, error) { return nil, nil },
			WithInputSchema(customSchema))
	})

	t.Run("last output schema wins", func(t *testing.T) {
		tl := NewTool("doubleOut", "Two output schemas",
			func(ctx *ToolContext, input struct{}) (any, error) { return nil, nil },
			WithOutputSchemaName("Answer"), WithOutputSchema(customSchema))

		def := tl.Definition()
		props, ok := def.OutputSchema["properties"].(map[string]any)
		if !ok || props["answer"] == nil {
			t.Errorf("OutputSchema = %v, want the last schema set", def.OutputSchema)
		}
	})

	t.Run("multipart tools advertise the custom schema over the envelope", func(t *testing.T) {
		tl := NewMultipartTool("multipartOut", "Multipart with output schema",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) { return nil, nil },
			WithOutputSchema(customSchema))

		def := tl.Definition()
		props, ok := def.OutputSchema["properties"].(map[string]any)
		if !ok || props["answer"] == nil {
			t.Errorf("OutputSchema = %v, want the custom schema, not the envelope", def.OutputSchema)
		}
	})
}

func TestResolveUniqueTools(t *testing.T) {
	t.Run("resolves tools from registry", func(t *testing.T) {
		r := newTestRegistry(t)
		defineTool(r, "provider/tool1", "Tool 1", func(ctx *ToolContext, input struct{}) (bool, error) {
			return true, nil
		})
		defineTool(r, "provider/tool2", "Tool 2", func(ctx *ToolContext, input struct{}) (bool, error) {
			return true, nil
		})

		toolRefs := []ToolRef{
			ToolName("provider/tool1"),
			ToolName("provider/tool2"),
		}

		names, newTools, err := resolveUniqueTools(r, toolRefs)

		if err != nil {
			t.Fatalf("resolveUniqueTools error: %v", err)
		}
		if len(names) != 2 {
			t.Errorf("len(names) = %d, want 2", len(names))
		}
		if len(newTools) != 0 {
			t.Errorf("len(newTools) = %d, want 0 (tools already registered)", len(newTools))
		}
	})

	t.Run("returns error for duplicate tools", func(t *testing.T) {
		r := newTestRegistry(t)
		toolRefs := []ToolRef{
			ToolName("provider/dup"),
			ToolName("provider/dup"),
		}

		_, _, err := resolveUniqueTools(r, toolRefs)

		if err == nil {
			t.Error("expected error for duplicate tools, got nil")
		}
	})

	t.Run("identifies new tools to register", func(t *testing.T) {
		r := newTestRegistry(t)
		newTl := NewTool("provider/brandNew", "Brand new", func(ctx *ToolContext, input struct{}) (string, error) {
			return "new", nil
		})

		toolRefs := []ToolRef{newTl}

		names, newTools, err := resolveUniqueTools(r, toolRefs)

		if err != nil {
			t.Fatalf("resolveUniqueTools error: %v", err)
		}
		if len(names) != 1 {
			t.Errorf("len(names) = %d, want 1", len(names))
		}
		if len(newTools) != 1 {
			t.Errorf("len(newTools) = %d, want 1", len(newTools))
		}
	})
}

func TestIsMultipart(t *testing.T) {
	t.Run("returns false for standard tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineTool(r, "provider/standard", "Standard tool",
			func(ctx *ToolContext, input struct{}) (string, error) {
				return "result", nil
			})

		if tl.IsMultipart() {
			t.Error("IsMultipart() = true for standard tool, want false")
		}
	})

	t.Run("returns false for NewTool", func(t *testing.T) {
		tl := NewTool("standard", "Standard",
			func(ctx *ToolContext, input struct{}) (string, error) {
				return "result", nil
			})

		if tl.IsMultipart() {
			t.Error("IsMultipart() = true for NewTool, want false")
		}
	})

	t.Run("returns true for multipart tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := defineMultipartTool(r, "provider/multipart", "Multipart tool",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Content: []*Part{NewTextPart("hello"), NewTextPart("world")},
				}, nil
			})

		if !tl.IsMultipart() {
			t.Error("IsMultipart() = false for multipart tool, want true")
		}
	})

	t.Run("returns true for NewMultipartTool", func(t *testing.T) {
		tl := NewMultipartTool("multipart", "Multipart",
			func(ctx *ToolContext, input struct{}) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Content: []*Part{NewTextPart("content")},
				}, nil
			})

		if !tl.IsMultipart() {
			t.Error("IsMultipart() = false for NewMultipartTool, want true")
		}
	})
}

func TestToolContextIsResumed(t *testing.T) {
	t.Run("returns false when Resumed is nil", func(t *testing.T) {
		tc := &ToolContext{
			Context: context.Background(),
			Resumed: nil,
		}

		if tc.IsResumed() {
			t.Error("IsResumed() = true, want false")
		}
	})

	t.Run("returns true when Resumed is set", func(t *testing.T) {
		tc := &ToolContext{
			Context: context.Background(),
			Resumed: map[string]any{"step": "confirm"},
		}

		if !tc.IsResumed() {
			t.Error("IsResumed() = false, want true")
		}
	})

	t.Run("returns true even for empty map", func(t *testing.T) {
		tc := &ToolContext{
			Context: context.Background(),
			Resumed: map[string]any{},
		}

		if !tc.IsResumed() {
			t.Error("IsResumed() = false for empty map, want true")
		}
	})
}

func TestResumedValue(t *testing.T) {
	// ctxWithResumed builds a ToolContext whose embedded context carries the
	// resumed metadata, mirroring how wrapToolFunc constructs one at runtime.
	ctxWithResumed := func(m map[string]any) *ToolContext {
		ctx := context.Background()
		if m != nil {
			ctx = base.ToolResumeKey.NewContext(ctx, m)
		}
		return &ToolContext{Context: ctx, Resumed: m}
	}

	t.Run("returns value when key exists and type matches", func(t *testing.T) {
		tc := ctxWithResumed(map[string]any{
			"step":  "confirmation",
			"count": 42,
		})

		step, ok := ResumedValue[string](tc, "step")
		if !ok {
			t.Error("ResumedValue[string] ok = false, want true")
		}
		if step != "confirmation" {
			t.Errorf("step = %q, want %q", step, "confirmation")
		}

		count, ok := ResumedValue[int](tc, "count")
		if !ok {
			t.Error("ResumedValue[int] ok = false, want true")
		}
		if count != 42 {
			t.Errorf("count = %d, want %d", count, 42)
		}
	})

	t.Run("returns false when key does not exist", func(t *testing.T) {
		tc := ctxWithResumed(map[string]any{"other": "value"})

		val, ok := ResumedValue[string](tc, "missing")
		if ok {
			t.Error("ResumedValue ok = true for missing key, want false")
		}
		if val != "" {
			t.Errorf("val = %q, want zero value", val)
		}
	})

	t.Run("returns false when type does not match", func(t *testing.T) {
		tc := ctxWithResumed(map[string]any{"count": "not a number"})

		val, ok := ResumedValue[int](tc, "count")
		if ok {
			t.Error("ResumedValue ok = true for wrong type, want false")
		}
		if val != 0 {
			t.Errorf("val = %d, want zero value", val)
		}
	})

	t.Run("returns false when Resumed is nil", func(t *testing.T) {
		tc := ctxWithResumed(nil)

		val, ok := ResumedValue[string](tc, "anything")
		if ok {
			t.Error("ResumedValue ok = true for nil Resumed, want false")
		}
		if val != "" {
			t.Errorf("val = %q, want zero value", val)
		}
	})

	t.Run("works with complex types", func(t *testing.T) {
		tc := ctxWithResumed(map[string]any{
			"options": []string{"a", "b", "c"},
			"nested":  map[string]any{"key": "value"},
		})

		options, ok := ResumedValue[[]string](tc, "options")
		if !ok {
			t.Error("ResumedValue[[]string] ok = false, want true")
		}
		if len(options) != 3 {
			t.Errorf("len(options) = %d, want 3", len(options))
		}

		nested, ok := ResumedValue[map[string]any](tc, "nested")
		if !ok {
			t.Error("ResumedValue[map[string]any] ok = false, want true")
		}
		if nested["key"] != "value" {
			t.Errorf("nested[key] = %v, want %q", nested["key"], "value")
		}
	})

	t.Run("works with a plain context.Context (middleware use)", func(t *testing.T) {
		ctx := base.ToolResumeKey.NewContext(context.Background(), map[string]any{
			"toolApproved": true,
		})

		approved, ok := ResumedValue[bool](ctx, "toolApproved")
		if !ok || !approved {
			t.Errorf("ResumedValue[bool] = (%v, %v), want (true, true)", approved, ok)
		}
	})
}

func TestOriginalInputAs(t *testing.T) {
	type MyInput struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}

	t.Run("returns typed input when type matches", func(t *testing.T) {
		original := MyInput{Query: "test", Limit: 10}
		tc := &ToolContext{
			Context:       context.Background(),
			OriginalInput: original,
		}

		input, ok := OriginalInputAs[MyInput](tc)
		if !ok {
			t.Error("OriginalInputAs ok = false, want true")
		}
		if input.Query != "test" {
			t.Errorf("input.Query = %q, want %q", input.Query, "test")
		}
		if input.Limit != 10 {
			t.Errorf("input.Limit = %d, want %d", input.Limit, 10)
		}
	})

	t.Run("returns false when OriginalInput is nil", func(t *testing.T) {
		tc := &ToolContext{
			Context:       context.Background(),
			OriginalInput: nil,
		}

		input, ok := OriginalInputAs[MyInput](tc)
		if ok {
			t.Error("OriginalInputAs ok = true for nil, want false")
		}
		if input.Query != "" || input.Limit != 0 {
			t.Errorf("input = %+v, want zero value", input)
		}
	})

	t.Run("returns false when type does not match", func(t *testing.T) {
		tc := &ToolContext{
			Context:       context.Background(),
			OriginalInput: "wrong type",
		}

		input, ok := OriginalInputAs[MyInput](tc)
		if ok {
			t.Error("OriginalInputAs ok = true for wrong type, want false")
		}
		if input.Query != "" || input.Limit != 0 {
			t.Errorf("input = %+v, want zero value", input)
		}
	})

	t.Run("works with map type", func(t *testing.T) {
		original := map[string]any{"query": "test", "limit": 10}
		tc := &ToolContext{
			Context:       context.Background(),
			OriginalInput: original,
		}

		input, ok := OriginalInputAs[map[string]any](tc)
		if !ok {
			t.Error("OriginalInputAs ok = false, want true")
		}
		if input["query"] != "test" {
			t.Errorf("input[query] = %v, want %q", input["query"], "test")
		}
	})

	t.Run("works with pointer types", func(t *testing.T) {
		original := &MyInput{Query: "pointer", Limit: 5}
		tc := &ToolContext{
			Context:       context.Background(),
			OriginalInput: original,
		}

		input, ok := OriginalInputAs[*MyInput](tc)
		if !ok {
			t.Error("OriginalInputAs ok = false, want true")
		}
		if input.Query != "pointer" {
			t.Errorf("input.Query = %q, want %q", input.Query, "pointer")
		}
	})
}

func TestToolContextInterruptMethod(t *testing.T) {
	t.Run("interrupt with nil options", func(t *testing.T) {
		tc := &ToolContext{
			Context: context.Background(),
		}

		err := tc.Interrupt(nil)
		if err == nil {
			t.Fatal("Interrupt(nil) = nil, want error")
		}

		isInterrupt, meta := IsToolInterruptError(err)
		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		if meta != nil {
			t.Errorf("metadata = %v, want nil", meta)
		}
	})

	t.Run("interrupt with empty options", func(t *testing.T) {
		tc := &ToolContext{
			Context: context.Background(),
		}

		err := tc.Interrupt(&InterruptOptions{})
		if err == nil {
			t.Fatal("Interrupt() = nil, want error")
		}

		isInterrupt, meta := IsToolInterruptError(err)
		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		if meta != nil {
			t.Errorf("metadata = %v, want nil", meta)
		}
	})

	t.Run("interrupt with metadata", func(t *testing.T) {
		tc := &ToolContext{
			Context: context.Background(),
		}

		err := tc.Interrupt(&InterruptOptions{
			Metadata: map[string]any{
				"step":    "confirm",
				"preview": "deleting files",
			},
		})
		if err == nil {
			t.Fatal("Interrupt() = nil, want error")
		}

		isInterrupt, meta := IsToolInterruptError(err)
		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		if meta["step"] != "confirm" {
			t.Errorf("meta[step] = %v, want %q", meta["step"], "confirm")
		}
		if meta["preview"] != "deleting files" {
			t.Errorf("meta[preview] = %v, want %q", meta["preview"], "deleting files")
		}
	})
}

func TestInterruptFor(t *testing.T) {
	type ConfirmMeta struct {
		Reason    string  `json:"reason"`
		Amount    float64 `json:"amount"`
		Recipient string  `json:"recipient"`
	}

	t.Run("creates interrupt with typed metadata", func(t *testing.T) {
		tc := &ToolContext{Context: context.Background()}

		err := InterruptWith(tc, ConfirmMeta{
			Reason:    "new recipient",
			Amount:    50.0,
			Recipient: "Alice",
		})

		if err == nil {
			t.Fatal("InterruptFor() = nil, want error")
		}

		isInterrupt, meta := IsToolInterruptError(err)
		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		if meta["reason"] != "new recipient" {
			t.Errorf("meta[reason] = %v, want %q", meta["reason"], "new recipient")
		}
		if meta["amount"] != 50.0 {
			t.Errorf("meta[amount] = %v, want %v", meta["amount"], 50.0)
		}
		if meta["recipient"] != "Alice" {
			t.Errorf("meta[recipient] = %v, want %q", meta["recipient"], "Alice")
		}
	})

	t.Run("handles nested structs", func(t *testing.T) {
		type Nested struct {
			Inner struct {
				Value string `json:"value"`
			} `json:"inner"`
		}

		tc := &ToolContext{Context: context.Background()}
		err := InterruptWith(tc, Nested{Inner: struct {
			Value string `json:"value"`
		}{Value: "test"}})

		isInterrupt, meta := IsToolInterruptError(err)
		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		inner, ok := meta["inner"].(map[string]any)
		if !ok {
			t.Fatal("meta[inner] is not a map")
		}
		if inner["value"] != "test" {
			t.Errorf("inner[value] = %v, want %q", inner["value"], "test")
		}
	})
}

func TestInterruptMetadata(t *testing.T) {
	type ConfirmMeta struct {
		Reason    string  `json:"reason"`
		Amount    float64 `json:"amount"`
		Recipient string  `json:"recipient"`
	}

	t.Run("extracts typed metadata from interrupt part", func(t *testing.T) {
		part := NewToolRequestPart(&ToolRequest{
			Name:  "testTool",
			Input: map[string]any{},
		})
		part.Interrupt = &ToolInterrupt{Data: map[string]any{
			"reason":    "large amount",
			"amount":    200.0,
			"recipient": "Bob",
		}}

		meta, ok := InterruptAs[ConfirmMeta](part)
		if !ok {
			t.Fatal("InterruptMetadata() ok = false, want true")
		}
		if meta.Reason != "large amount" {
			t.Errorf("meta.Reason = %q, want %q", meta.Reason, "large amount")
		}
		if meta.Amount != 200.0 {
			t.Errorf("meta.Amount = %v, want %v", meta.Amount, 200.0)
		}
		if meta.Recipient != "Bob" {
			t.Errorf("meta.Recipient = %q, want %q", meta.Recipient, "Bob")
		}
	})

	t.Run("returns false for non-interrupt part", func(t *testing.T) {
		part := NewTextPart("not an interrupt")

		_, ok := InterruptAs[ConfirmMeta](part)
		if ok {
			t.Error("InterruptMetadata() ok = true for text part, want false")
		}
	})

	t.Run("returns false for nil part", func(t *testing.T) {
		_, ok := InterruptAs[ConfirmMeta](nil)
		if ok {
			t.Error("InterruptMetadata() ok = true for nil, want false")
		}
	})

	t.Run("returns false when the interrupt carries no data", func(t *testing.T) {
		part := NewToolRequestPart(&ToolRequest{Name: "test"})
		part.Interrupt = &ToolInterrupt{} // bare interrupt, no data

		_, ok := InterruptAs[ConfirmMeta](part)
		if ok {
			t.Error("InterruptMetadata() ok = true for a bare interrupt, want false")
		}
	})
}

// wireMetadataOf marshals a part and returns the metadata map it produced, so
// tests can assert the wire contract (the JS-compatible metadata keys) that the
// typed Interrupt and Restart fields fold into.
func wireMetadataOf(t *testing.T, p *Part) map[string]any {
	t.Helper()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal part: %v", err)
	}
	var wire struct {
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("unmarshal part: %v", err)
	}
	return wire.Metadata
}

// TestPartClone_IsolatesInterruptPayloads pins that Clone copies the top-level
// container of every payload the interrupt and restart state carry, the way it
// does for Data, so mutating the clone's payload maps leaves the original
// untouched.
func TestPartClone_IsolatesInterruptPayloads(t *testing.T) {
	part := NewToolRequestPart(&ToolRequest{Name: "transfer", Input: map[string]any{"amount": 200}})
	part.Interrupt = &ToolInterrupt{Data: map[string]any{"reason": "confirm"}}
	part.Restart = &ToolRestart{
		Resume:        map[string]any{"approved": true},
		OriginalInput: []any{"original"},
	}

	cp := part.Clone()
	cp.Interrupt.Data.(map[string]any)["reason"] = "changed"
	cp.Restart.Resume.(map[string]any)["approved"] = false
	cp.Restart.OriginalInput.([]any)[0] = "changed"

	if got := part.Interrupt.Data.(map[string]any)["reason"]; got != "confirm" {
		t.Errorf("original Interrupt.Data[reason] = %v, want confirm", got)
	}
	if got := part.Restart.Resume.(map[string]any)["approved"]; got != true {
		t.Errorf("original Restart.Resume[approved] = %v, want true", got)
	}
	if got := part.Restart.OriginalInput.([]any)[0]; got != "original" {
		t.Errorf("original Restart.OriginalInput[0] = %v, want original", got)
	}

	// A struct payload is shared as is: there is no container to copy.
	type reason struct{ Why string }
	part.Interrupt.Data = reason{Why: "struct"}
	if got := part.Clone().Interrupt.Data; got != (reason{Why: "struct"}) {
		t.Errorf("cloned struct payload = %v, want the value copied through", got)
	}
}

// TestDeprecatedRestart_CarriesResumeAsGiven pins that the deprecated Restart
// builds a part from whatever resume data it is handed, as it did before the
// typed verbs existed: a non-object value rides to the loop, which reads it as
// a bare restart, rather than turning the whole call into a nil part.
func TestDeprecatedRestart_CarriesResumeAsGiven(t *testing.T) {
	tl := NewTool("t", "d", func(ctx *ToolContext, _ struct{}) (string, error) { return "", nil })
	part := NewToolRequestPart(&ToolRequest{Name: "t"})
	part.Interrupt = &ToolInterrupt{}

	restart := tl.Restart(part, &RestartOptions{ResumedMetadata: "approved"})
	if restart == nil {
		t.Fatal("Restart returned nil for a scalar resume value")
	}
	if !restart.IsRestart() || restart.Restart.Resume != "approved" {
		t.Errorf("restart state = %+v, want the resume value as given", restart.Restart)
	}
}

// TestInterruptVerbs_NilToolRequest pins that a tool request part with no
// request, which the kind check alone lets through, is declined by every
// verb that would otherwise dereference it: the typed claim reports false,
// the part verbs return an error, and the deprecated verbs return nil or an
// error as their contracts say.
func TestInterruptVerbs_NilToolRequest(t *testing.T) {
	tl := NewTool("t", "d", func(ctx *ToolContext, _ struct{}) (string, error) { return "", nil })

	rawKeyed := NewToolRequestPart(nil)
	rawKeyed.Metadata = map[string]any{"interrupt": true}
	typed := &Part{Kind: PartToolRequest, Interrupt: &ToolInterrupt{}}

	for name, part := range map[string]*Part{"raw key": rawKeyed, "typed field": typed} {
		t.Run(name, func(t *testing.T) {
			if part.IsInterrupt() || part.IsRestart() {
				t.Error("a tool request part with no ToolRequest must not read as an interrupt or a restart")
			}
			if err := part.Validate(); err == nil {
				t.Error("Validate() = nil, want the missing request reported")
			}
			if _, ok := tl.Interrupted(part); ok {
				t.Error("Interrupted claimed a part with no request")
			}
			if _, err := part.ToToolRestart(nil); err == nil {
				t.Error("ToToolRestart accepted a part with no request")
			}
			if _, err := part.ToToolRestartWithInput(struct{}{}, nil); err == nil {
				t.Error("ToToolRestartWithInput accepted a part with no request")
			}
			if _, err := part.ToToolResponse("out"); err == nil {
				t.Error("ToToolResponse accepted a part with no request")
			}
			if tl.Respond(part, "out", nil) != nil {
				t.Error("Respond built a part from a request part with no request")
			}
			if tl.Restart(part, nil) != nil {
				t.Error("Restart built a part from a request part with no request")
			}
			if _, err := tl.RespondWith(part, "out"); err == nil {
				t.Error("RespondWith accepted a part with no request")
			}
			if _, err := tl.RestartWith(part); err == nil {
				t.Error("RestartWith accepted a part with no request")
			}
		})
	}
}

// TestRestartWithInput_ReplacementIsExplicit pins that the verbs decide
// whether the input is replaced, not the value: the typed verb replaces with
// whatever it is given, a nil pointer included; the untyped part verb refuses
// a nil input outright; and the deprecated option-driven verbs treat a nil of
// any type as "not set", since replacing is optional there.
func TestRestartWithInput_ReplacementIsExplicit(t *testing.T) {
	type in struct {
		Amount int `json:"amount"`
	}
	type confirmation struct {
		Approved bool `json:"approved"`
	}
	original := map[string]any{"amount": float64(200)}
	interrupted := func() *Part {
		p := NewToolRequestPart(&ToolRequest{Name: "transfer", Input: original})
		p.Interrupt = &ToolInterrupt{}
		return p
	}

	t.Run("typed verb replaces with a nil pointer", func(t *testing.T) {
		tl := NewResumableTool("transfer", "d",
			func(ctx context.Context, _ *in, _ *confirmation) (string, error) { return "", nil })
		call, ok := tl.Interrupted(interrupted())
		if !ok {
			t.Fatal("Interrupted did not claim the part")
		}
		restart := call.RestartWithInput(nil, confirmation{Approved: true})
		if got, ok := restart.ToolRequest.Input.(*in); !ok || got != nil {
			t.Errorf("input = %#v, want the nil pointer the caller gave", restart.ToolRequest.Input)
		}
		if diff := cmp.Diff(original, restart.Restart.OriginalInput); diff != "" {
			t.Errorf("OriginalInput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("part verb refuses a nil input", func(t *testing.T) {
		for name, input := range map[string]any{"untyped nil": nil, "typed nil": (*in)(nil)} {
			if _, err := interrupted().ToToolRestartWithInput(input, nil); err == nil {
				t.Errorf("ToToolRestartWithInput(%s) = nil error, want a rejection", name)
			}
		}
	})

	t.Run("deprecated verbs treat a nil replacement as not set", func(t *testing.T) {
		tl := NewTool("transfer", "d", func(ctx *ToolContext, _ *in) (string, error) { return "", nil })
		restart := tl.Restart(interrupted(), &RestartOptions{ReplaceInput: (*in)(nil)})
		if diff := cmp.Diff(original, restart.ToolRequest.Input); diff != "" || restart.Restart.OriginalInput != nil {
			t.Errorf("Restart replaced the input with a typed nil: input diff %s, original %v", diff, restart.Restart.OriginalInput)
		}
		restart, err := tl.RestartWith(interrupted(), WithNewInput[*in](nil))
		if err != nil {
			t.Fatalf("RestartWith: %v", err)
		}
		if diff := cmp.Diff(original, restart.ToolRequest.Input); diff != "" || restart.Restart.OriginalInput != nil {
			t.Errorf("RestartWith replaced the input with a typed nil: input diff %s, original %v", diff, restart.Restart.OriginalInput)
		}
	})
}

// TestBuildRestartPart_DropsPendingBookkeeping pins that a restart part does
// not carry the loop's record of a completed sibling (pendingOutput and its
// companions) off the interrupted part it is built from: that record
// describes the request in history, and a restart must not replay it.
func TestBuildRestartPart_DropsPendingBookkeeping(t *testing.T) {
	part := NewToolRequestPart(&ToolRequest{Name: "transfer"})
	part.Interrupt = &ToolInterrupt{}
	part.Metadata = map[string]any{"pendingOutput": "stale", "pendingMetadata": map[string]any{}, "pendingContent": []any{}, "keep": "me"}

	restart, err := part.ToToolRestart(nil)
	if err != nil {
		t.Fatalf("ToToolRestart: %v", err)
	}
	if diff := cmp.Diff(map[string]any{"keep": "me"}, restart.Metadata); diff != "" {
		t.Errorf("restart metadata mismatch (-want +got):\n%s", diff)
	}
}

// TestPartToRestart_PreservesIdentity pins the shape of a restart part: the
// identity the generate loop matches on survives, unrelated metadata is carried
// over, the interrupt state is dropped, and the source part is left alone. It
// also pins the wire keys the typed state folds into, which the JS runtime
// reads.
func TestPartToRestart_PreservesIdentity(t *testing.T) {
	part := NewToolRequestPart(&ToolRequest{
		Name:  "transfer",
		Ref:   "call-1",
		Input: map[string]any{"amount": float64(200)},
	})
	part.Interrupt = &ToolInterrupt{Data: map[string]any{"reason": "large_amount"}}
	part.Metadata = map[string]any{"keep": "me"}

	type confirmation struct {
		Approved bool `json:"approved"`
	}
	got, err := part.ToToolRestart(confirmation{Approved: true})
	if err != nil {
		t.Fatalf("ToToolRestart: %v", err)
	}
	if !got.IsToolRequest() {
		t.Fatal("ToToolRestart must produce a tool request part")
	}
	if got.IsInterrupt() || got.Interrupt != nil {
		t.Error("the restart part must not carry interrupt state")
	}
	if got.ToolRequest.Name != "transfer" || got.ToolRequest.Ref != "call-1" {
		t.Errorf("identity = %q/%q, want transfer/call-1", got.ToolRequest.Name, got.ToolRequest.Ref)
	}
	if got.Restart == nil || got.Restart.Resume.(confirmation) != (confirmation{Approved: true}) {
		t.Errorf("Restart = %+v, want the resume payload", got.Restart)
	}
	if got.Metadata["keep"] != "me" {
		t.Errorf("unrelated metadata was dropped: %v", got.Metadata)
	}
	if part.Interrupt == nil || part.Interrupt.Resolved {
		t.Error("ToToolRestart must not mutate the source part")
	}

	// On the wire the typed state becomes the JS-compatible metadata keys.
	wire := wireMetadataOf(t, got)
	resumed, ok := wire["resumed"].(map[string]any)
	if !ok {
		t.Fatalf("wire resumed = %T, want map[string]any", wire["resumed"])
	}
	if resumed["approved"] != true {
		t.Errorf("wire resumed[approved] = %v, want true", resumed["approved"])
	}
	if _, ok := wire["interrupt"]; ok {
		t.Error("the restart part must not carry an interrupt key on the wire")
	}
	if wire["keep"] != "me" {
		t.Errorf("unrelated metadata missing from the wire: %v", wire)
	}

	// A bare restart marks the call as resumed without carrying data.
	bare, err := part.ToToolRestart(nil)
	if err != nil {
		t.Fatalf("bare ToToolRestart: %v", err)
	}
	if bare.Restart == nil || bare.Restart.Resume != nil {
		t.Errorf("bare restart = %+v, want no resume payload", bare.Restart)
	}
	if wireMetadataOf(t, bare)["resumed"] != true {
		t.Error("a bare restart must be marked resumed on the wire")
	}
}

// TestPartToRestartToResponse_RejectNonInterrupt keeps the part verbs from
// building parts out of anything that isn't an interrupted tool request.
func TestPartToRestartToResponse_RejectNonInterrupt(t *testing.T) {
	plain := NewToolRequestPart(&ToolRequest{Name: "x"}) // never interrupted
	resolved := NewToolRequestPart(&ToolRequest{Name: "x"})
	resolved.Interrupt = &ToolInterrupt{Resolved: true}

	for _, tc := range []struct {
		name string
		part *Part
	}{
		{"non-interrupt tool request", plain},
		{"already resolved interrupt", resolved},
		{"text part", NewTextPart("hi")},
	} {
		if _, err := tc.part.ToToolRestart(nil); err == nil {
			t.Errorf("ToToolRestart(%s) must error", tc.name)
		}
		if _, err := tc.part.ToToolRestartWithInput(map[string]any{}, nil); err == nil {
			t.Errorf("ToToolRestartWithInput(%s) must error", tc.name)
		}
		if _, err := tc.part.ToToolResponse("out"); err == nil {
			t.Errorf("ToToolResponse(%s) must error", tc.name)
		}
	}
}

// TestPartToRestart_NonObjectResume covers the documented constraint: resume
// data must serialize to a JSON object, and a scalar yields an actionable error
// rather than an opaque json failure.
func TestPartToRestart_NonObjectResume(t *testing.T) {
	part := NewToolRequestPart(&ToolRequest{Name: "x"})
	part.Interrupt = &ToolInterrupt{}

	_, err := part.ToToolRestart("just a string")
	if err == nil {
		t.Fatal("expected an error restarting with non-object resume data")
	}
	if !strings.Contains(err.Error(), "JSON object") {
		t.Errorf("error = %q, want it to mention the JSON object constraint", err)
	}
}

// TestPartToResponse_MarksInterruptResponse pins the marker the generate loop
// keys on to resolve an interrupt instead of re-executing the tool.
func TestPartToResponse_MarksInterruptResponse(t *testing.T) {
	part := NewToolRequestPart(&ToolRequest{Name: "transfer", Ref: "call-1"})
	part.Interrupt = &ToolInterrupt{}

	got, err := part.ToToolResponse(map[string]any{"status": "cancelled"})
	if err != nil {
		t.Fatalf("ToToolResponse: %v", err)
	}
	if !got.IsToolResponse() {
		t.Fatal("ToToolResponse must produce a tool response part")
	}
	if got.ToolResponse.Name != "transfer" || got.ToolResponse.Ref != "call-1" {
		t.Errorf("identity = %q/%q, want transfer/call-1", got.ToolResponse.Name, got.ToolResponse.Ref)
	}
	if got.Metadata["interruptResponse"] != true {
		t.Errorf("interruptResponse = %v, want true", got.Metadata["interruptResponse"])
	}
}

// TestPartToRestart_LiftsRawInterruptMetadata keeps the part verbs as lenient
// as the type-erased tool verbs they replace: a part hand-assembled with the
// JS "interrupt" metadata key restarts, the caller's map is untouched, and the
// key does not ride along onto the part built from it.
func TestPartToRestart_LiftsRawInterruptMetadata(t *testing.T) {
	raw := map[string]any{"interrupt": map[string]any{"reason": "confirm"}, "keep": "me"}
	part := NewToolRequestPart(&ToolRequest{Name: "transfer", Input: map[string]any{"amount": float64(200)}})
	part.Metadata = raw

	restart, err := part.ToToolRestart(map[string]any{"approved": true})
	if err != nil {
		t.Fatalf("ToToolRestart: %v", err)
	}
	if !restart.IsRestart() || restart.Interrupt != nil {
		t.Errorf("restart = %+v, want typed restart state and no interrupt state", restart)
	}
	if _, ok := restart.Metadata["interrupt"]; ok {
		t.Error("the raw interrupt key rode along onto the restart part")
	}
	if restart.Metadata["keep"] != "me" {
		t.Errorf("unrelated metadata was dropped: %v", restart.Metadata)
	}
	if _, ok := raw["interrupt"]; !ok || part.Interrupt != nil {
		t.Error("ToToolRestart must not mutate the source part")
	}
	wire := wireMetadataOf(t, restart)
	if _, ok := wire["interrupt"]; ok {
		t.Errorf("wire metadata = %v, want no interrupt key", wire)
	}
	if wire["resumed"] == nil {
		t.Errorf("wire metadata = %v, want resumed", wire)
	}

	response, err := part.ToToolResponse("out")
	if err != nil {
		t.Fatalf("ToToolResponse: %v", err)
	}
	if !response.IsToolResponse() {
		t.Error("ToToolResponse must produce a tool response part")
	}

	resolved := NewToolRequestPart(&ToolRequest{Name: "transfer"})
	resolved.Metadata = map[string]any{"resolvedInterrupt": true}
	if _, err := resolved.ToToolRestart(nil); err == nil {
		t.Error("a raw resolvedInterrupt part must not restart")
	}
}

// TestInterrupted_LiftsRawInterruptMetadata is the same leniency on the claim:
// a part hand-assembled with the JS "interrupt" metadata key is claimed on a
// copy, so the parts built from it carry typed state and the source part is
// untouched.
func TestInterrupted_LiftsRawInterruptMetadata(t *testing.T) {
	type approval struct {
		Approved bool `json:"approved"`
	}
	transfer := NewResumableTool("transfer", "d",
		func(ctx context.Context, _ struct{}, _ *approval) (string, error) { return "", nil })

	part := NewToolRequestPart(&ToolRequest{Name: "transfer"})
	part.Metadata = map[string]any{"interrupt": true}

	call, ok := transfer.Interrupted(part)
	if !ok {
		t.Fatal("Interrupted must lift raw interrupt metadata and claim the part")
	}
	restart := call.Restart(approval{Approved: true})
	if !restart.IsRestart() || restart.Metadata != nil {
		t.Errorf("restart = %+v (metadata %v), want typed restart state and no leftover metadata", restart, restart.Metadata)
	}
	if resp := call.Respond("declined"); !resp.IsToolResponse() {
		t.Errorf("Respond = %+v, want a tool response part", resp)
	}
	if part.Interrupt != nil || part.Metadata["interrupt"] != true {
		t.Error("the claim must not mutate the source part")
	}
}

// TestInterruptedCall_Verbs pins the parts each verb of a claimed call builds,
// on a ToolAction, whose resume type is a map.
func TestInterruptedCall_Verbs(t *testing.T) {
	type in struct {
		Value int `json:"value"`
	}
	tl := NewTool("restarter", "d", func(ctx *ToolContext, input in) (int, error) { return input.Value, nil })
	part := NewToolRequestPart(&ToolRequest{Name: "restarter", Ref: "r1", Input: map[string]any{"value": 10}})
	part.Interrupt = &ToolInterrupt{}
	call, ok := tl.Interrupted(part)
	if !ok {
		t.Fatal("Interrupted did not claim the tool's own interrupt")
	}
	if call.Input.Value != 10 {
		t.Errorf("call.Input = %+v, want the input decoded to the tool's In type", call.Input)
	}

	t.Run("nil map is a bare restart", func(t *testing.T) {
		restart := call.Restart(nil)
		if !restart.IsRestart() || restart.Restart.Resume != nil {
			t.Errorf("Restart = %+v, want a bare restart", restart.Restart)
		}
		if wireMetadataOf(t, restart)["resumed"] != true {
			t.Error("a bare restart must be marked resumed on the wire")
		}
		if restart.ToolRequest.Name != "restarter" || restart.ToolRequest.Ref != "r1" {
			t.Errorf("identity = %q/%q, want restarter/r1", restart.ToolRequest.Name, restart.ToolRequest.Ref)
		}
	})

	t.Run("map resume rides on the part", func(t *testing.T) {
		restart := call.Restart(map[string]any{"approved": true})
		resumed, _ := restart.Restart.Resume.(map[string]any)
		if resumed["approved"] != true {
			t.Errorf("Resume = %v, want the map", restart.Restart.Resume)
		}
	})

	t.Run("RestartWithInput preserves the original", func(t *testing.T) {
		restart := call.RestartWithInput(in{Value: 20}, nil)
		if got, _ := restart.ToolRequest.Input.(in); got.Value != 20 {
			t.Errorf("input = %+v, want the new input {20}", restart.ToolRequest.Input)
		}
		orig, _ := restart.Restart.OriginalInput.(map[string]any)
		if orig["value"] != 10 {
			t.Errorf("OriginalInput = %v, want the original {10}", restart.Restart.OriginalInput)
		}
		if wireMetadataOf(t, restart)["replacedInput"] == nil {
			t.Error("replacedInput not set on the wire")
		}
	})

	t.Run("Respond marks the interrupt response", func(t *testing.T) {
		resp := call.Respond(7)
		if !resp.IsToolResponse() || resp.ToolResponse.Output != 7 {
			t.Errorf("Respond = %+v, want a tool response carrying 7", resp)
		}
		if resp.ToolResponse.Name != "restarter" || resp.ToolResponse.Ref != "r1" {
			t.Errorf("identity = %q/%q, want restarter/r1", resp.ToolResponse.Name, resp.ToolResponse.Ref)
		}
		if resp.Metadata["interruptResponse"] != true {
			t.Errorf("interruptResponse = %v, want true", resp.Metadata["interruptResponse"])
		}
	})
}
