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
	"errors"
	"testing"

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

func TestToolInterruptError(t *testing.T) {
	t.Run("Error includes metadata when present", func(t *testing.T) {
		err := &toolInterruptError{Metadata: map[string]any{"key": "value"}}
		got := err.Error()
		want := "tool execution interrupted: \n\n{\n  \"key\": \"value\"\n}"
		if got != want {
			t.Errorf("Error() = %q, want %q", got, want)
		}
	})

	t.Run("Error returns simple message when no metadata", func(t *testing.T) {
		err := &toolInterruptError{}
		got := err.Error()
		want := "tool execution interrupted"
		if got != want {
			t.Errorf("Error() = %q, want %q", got, want)
		}
	})
}

func TestIsToolInterruptError(t *testing.T) {
	t.Run("returns true for toolInterruptError", func(t *testing.T) {
		meta := map[string]any{"reason": "user cancelled"}
		err := &toolInterruptError{Metadata: meta}

		isInterrupt, gotMeta := IsToolInterruptError(err)

		if !isInterrupt {
			t.Error("IsToolInterruptError() = false, want true")
		}
		if diff := cmp.Diff(meta, gotMeta); diff != "" {
			t.Errorf("metadata mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns true for wrapped toolInterruptError", func(t *testing.T) {
		meta := map[string]any{"step": 3}
		innerErr := &toolInterruptError{Metadata: meta}
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
		tl := DefineTool(r, "provider/addNumbers", "Adds two numbers", func(ctx *ToolContext, input struct {
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
		DefineTool(r, "provider/multiply", "Multiplies", func(ctx *ToolContext, input struct {
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
		tl := DefineTool(r, "provider/concat", "Concatenates strings", func(ctx *ToolContext, input struct {
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

		tl := DefineToolWithInputSchema(r, "provider/search", "Searches", customSchema,
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
		tl := DefineMultipartTool(r, "provider/imageGen", "Generates images",
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
		tl := DefineMultipartTool(r, "provider/multiOut", "Returns multiple parts",
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
		tl := DefineTool(r, "provider/complete", "A complete tool", func(ctx *ToolContext, input struct {
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
		DefineTool(r, "test/findMe", "Find me", func(ctx *ToolContext, input struct{}) (bool, error) {
			return true, nil
		})

		got := LookupTool(r, "test/findMe")
		if got == nil {
			t.Error("LookupTool returned nil for registered tool")
		}
	})
}

func TestToolIsMultipart(t *testing.T) {
	t.Run("regular tool is not multipart", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "provider/regular", "Regular tool", func(ctx *ToolContext, input struct{}) (string, error) {
			return "ok", nil
		})

		def := tl.Definition()
		if def.Metadata["multipart"] == true {
			t.Error("multipart metadata = true for regular tool, want false")
		}
	})

	t.Run("multipart tool is multipart", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineMultipartTool(r, "provider/multi", "Multi tool",
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
		tl := DefineTool(r, "provider/sum", "Sums numbers", func(ctx *ToolContext, input struct {
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
		tl := DefineTool(r, "provider/fail", "Always fails", func(ctx *ToolContext, input struct{}) (string, error) {
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
		tl := DefineMultipartTool(r, "provider/fullResp", "Full response",
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
	tl := DefineTool(r, "provider/responder", "Test responder", func(ctx *ToolContext, input struct{}) (string, error) {
		return "ok", nil
	})

	t.Run("creates response for tool request", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name:  "provider/responder",
			Ref:   "ref-123",
			Input: map[string]any{"x": 1},
		})
		reqPart.Metadata = map[string]any{"interrupt": true}

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
		reqPart.Metadata = map[string]any{"interrupt": true}

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
	tl := DefineTool(r, "provider/restarter", "Test restarter", func(ctx *ToolContext, input struct {
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
		reqPart.Metadata = map[string]any{"interrupt": true}

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
		if restart.Metadata["resumed"] != true {
			t.Errorf("resumed = %v, want true", restart.Metadata["resumed"])
		}
		if restart.Metadata["interrupt"] != nil {
			t.Error("interrupt should be removed from metadata")
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
		reqPart.Metadata = map[string]any{"interrupt": true}

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
		if restart.Metadata["replacedInput"] == nil {
			t.Error("replacedInput not set in metadata")
		}
	})

	t.Run("sets resumed metadata when specified", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name: "provider/restarter",
		})
		reqPart.Metadata = map[string]any{"interrupt": true}

		opts := &RestartOptions{
			ResumedMetadata: map[string]any{"reason": "user confirmed"},
		}
		restart := tl.Restart(reqPart, opts)

		resumed := restart.Metadata["resumed"].(map[string]any)
		if resumed["reason"] != "user confirmed" {
			t.Errorf("resumed.reason = %v, want %q", resumed["reason"], "user confirmed")
		}
	})
}

func TestToolInterrupt(t *testing.T) {
	t.Run("tool can interrupt execution", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "provider/interrupter", "Can interrupt",
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
		tl := DefineTool(r, "provider/noInterrupt", "No interrupt",
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

		tl := DefineTool(r, "provider/customInput", "Custom input schema",
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

func TestResolveUniqueTools(t *testing.T) {
	t.Run("resolves tools from registry", func(t *testing.T) {
		r := newTestRegistry(t)
		DefineTool(r, "provider/tool1", "Tool 1", func(ctx *ToolContext, input struct{}) (bool, error) {
			return true, nil
		})
		DefineTool(r, "provider/tool2", "Tool 2", func(ctx *ToolContext, input struct{}) (bool, error) {
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
		tl := DefineTool(r, "provider/standard", "Standard tool",
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
		tl := DefineMultipartTool(r, "provider/multipart", "Multipart tool",
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
			ctx = resumedCtxKey.NewContext(ctx, m)
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
		ctx := resumedCtxKey.NewContext(context.Background(), map[string]any{
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
		part.Metadata = map[string]any{
			"interrupt": map[string]any{
				"reason":    "large amount",
				"amount":    200.0,
				"recipient": "Bob",
			},
		}

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

	t.Run("returns false when interrupt metadata is not a map", func(t *testing.T) {
		part := NewToolRequestPart(&ToolRequest{Name: "test"})
		part.Metadata = map[string]any{
			"interrupt": true, // bool instead of map
		}

		_, ok := InterruptAs[ConfirmMeta](part)
		if ok {
			t.Error("InterruptMetadata() ok = true for bool interrupt, want false")
		}
	})
}
