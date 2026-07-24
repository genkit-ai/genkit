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
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

func TestToolName(t *testing.T) {
	tn := ToolName("myTool")
	if tn.Name() != "myTool" {
		t.Errorf("Name() = %q, want %q", tn.Name(), "myTool")
	}
}

// defineToolThenFinishModel defines "test/model": on the first turn it returns
// reqs (typically tool requests), and once a tool response is in history it
// returns the final text "done". This drives a single tool round per Generate.
func defineToolThenFinishModel(r api.Registry, reqs ...*Part) {
	DefineModel(r, "test/model",
		&ModelOptions{Supports: &ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ModelRequest, _ any, cb ModelStreamCallback) (*ModelResponse, error) {
			for _, m := range req.Messages {
				if m.Role == RoleTool {
					return &ModelResponse{
						Request:      req,
						Message:      NewModelTextMessage("done"),
						FinishReason: FinishReasonStop,
					}, nil
				}
			}
			return &ModelResponse{
				Request:      req,
				Message:      &Message{Role: RoleModel, Content: reqs},
				FinishReason: FinishReasonStop,
			}, nil
		})
}

type weatherIn struct {
	City string `json:"city"`
}

func TestDefineTool(t *testing.T) {
	t.Run("creates and registers tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "provider/addNumbers", "Adds two numbers", func(ctx context.Context, input struct {
			A int `json:"a"`
			B int `json:"b"`
		}) (int, error) {
			return input.A + input.B, nil
		})

		if tl.Name() != "provider/addNumbers" {
			t.Errorf("Name() = %q, want %q", tl.Name(), "provider/addNumbers")
		}
		def := tl.Definition()
		if def.Description != "Adds two numbers" {
			t.Errorf("Description = %q, want %q", def.Description, "Adds two numbers")
		}
		if LookupTool(r, "provider/addNumbers") == nil {
			t.Error("LookupTool returned nil for registered tool")
		}
	})

	t.Run("tool executes correctly", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "provider/concat", "Concatenates strings", func(ctx context.Context, input struct {
			A string `json:"a"`
			B string `json:"b"`
		}) (string, error) {
			return input.A + input.B, nil
		})

		resp, err := tl.RunRaw(context.Background(), map[string]any{"a": "hello", "b": "world"})
		if err != nil {
			t.Fatalf("RunRaw error: %v", err)
		}
		if resp.Output != "helloworld" {
			t.Errorf("output = %v, want %q", resp.Output, "helloworld")
		}
	})

	// The headline ergonomics: a plain-context tool function driven end to end
	// through Generate.
	t.Run("end to end through Generate", func(t *testing.T) {
		r := newTestRegistry(t)
		defineToolThenFinishModel(r, NewToolRequestPart(&ToolRequest{
			Name: "getWeather", Input: map[string]any{"city": "Paris"},
		}))

		var gotCity string
		weather := DefineTool(r, "getWeather", "fetches the weather",
			func(ctx context.Context, in weatherIn) (string, error) {
				gotCity = in.City
				return "Sunny", nil
			})

		resp, err := Generate(context.Background(), r,
			WithModelName("test/model"),
			WithPrompt("weather in Paris?"),
			WithTools(weather))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if gotCity != "Paris" {
			t.Errorf("tool saw city %q, want %q", gotCity, "Paris")
		}
		if got := resp.Text(); got != "done" {
			t.Errorf("final text = %q, want %q", got, "done")
		}
	})
}

func TestNewTool(t *testing.T) {
	t.Run("unregistered tool can be executed", func(t *testing.T) {
		tl := NewTool("double", "Doubles a number", func(ctx context.Context, input struct {
			N int `json:"n"`
		}) (int, error) {
			return input.N * 2, nil
		})

		resp, err := tl.RunRaw(context.Background(), map[string]any{"n": 5})
		if err != nil {
			t.Fatalf("RunRaw error: %v", err)
		}
		// JSON unmarshalling returns float64 for numbers
		if resp.Output != float64(10) {
			t.Errorf("output = %v (%T), want 10", resp.Output, resp.Output)
		}
	})

	t.Run("tool can be registered later", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := NewTool("provider/laterTool", "Registered later", func(ctx context.Context, input struct{}) (string, error) {
			return "done", nil
		})

		tl.Register(r)

		if LookupTool(r, "provider/laterTool") == nil {
			t.Error("LookupTool returned nil after registration")
		}
	})
}

type reportItem struct {
	Name string `json:"name"`
}

type reportOut struct {
	Title string       `json:"title"`
	Items []reportItem `json:"items"`
}

func TestToolDefinition(t *testing.T) {
	t.Run("includes all fields", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "provider/complete", "A complete tool", func(ctx context.Context, input struct {
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

	// Guards against the multipart envelope leaking into the tool definition.
	// The underlying action's function returns *MultipartToolResponse, so
	// without the recorded output schema the model and Dev UI would see the
	// envelope ({content, output, metadata}) instead of the actual Out type.
	t.Run("output schema is the Out type, not the envelope", func(t *testing.T) {
		want := core.InferSchemaMap(reportOut{})

		simple := NewTool("simple", "d",
			func(ctx context.Context, _ weatherIn) (reportOut, error) { return reportOut{}, nil })
		interruptible := NewInterruptibleTool("interruptible", "d",
			func(ctx context.Context, _ weatherIn, _ *confirmation) (reportOut, error) { return reportOut{}, nil })

		for _, tc := range []struct {
			name string
			got  map[string]any
		}{
			{"NewTool", simple.Definition().OutputSchema},
			{"NewInterruptibleTool", interruptible.Definition().OutputSchema},
		} {
			if !reflect.DeepEqual(tc.got, want) {
				t.Errorf("%s output schema = %#v\nwant %#v", tc.name, tc.got, want)
			}
			props, _ := tc.got["properties"].(map[string]any)
			if _, ok := props["title"]; !ok {
				t.Errorf("%s output schema missing the real %q field: %#v", tc.name, "title", tc.got)
			}
			if _, ok := props["content"]; ok {
				t.Errorf("%s output schema leaked the multipart envelope (has %q): %#v", tc.name, "content", tc.got)
			}
		}
	})

	// The real output schema must survive registration + lookup, since the
	// type parameters are erased on the LookupTool path.
	t.Run("output schema survives LookupTool", func(t *testing.T) {
		r := newTestRegistry(t)
		DefineTool(r, "provider/report", "d",
			func(ctx context.Context, _ weatherIn) (reportOut, error) { return reportOut{}, nil })

		found := LookupTool(r, "provider/report")
		if found == nil {
			t.Fatal("LookupTool returned nil")
		}
		want := core.InferSchemaMap(reportOut{})
		if !reflect.DeepEqual(found.Definition().OutputSchema, want) {
			t.Errorf("looked-up output schema = %#v\nwant %#v", found.Definition().OutputSchema, want)
		}
	})
}

func TestLookupTool(t *testing.T) {
	t.Run("returns nil for empty name", func(t *testing.T) {
		r := newTestRegistry(t)
		if got := LookupTool(r, ""); got != nil {
			t.Errorf("LookupTool(\"\") = %v, want nil", got)
		}
	})

	t.Run("returns nil for non-existent tool", func(t *testing.T) {
		r := newTestRegistry(t)
		if got := LookupTool(r, "nonexistent/tool"); got != nil {
			t.Errorf("LookupTool(nonexistent) = %v, want nil", got)
		}
	})
}

// TestWithStrictSchema verifies the strict-schema flag round-trips through
// Definition().Metadata["strict"] and LookupTool, for both registered and
// dynamic tools.
func TestWithStrictSchema(t *testing.T) {
	t.Run("absent by default", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "strict/default", "no strict opt", func(ctx context.Context, input struct{}) (string, error) {
			return "", nil
		})
		def := tl.Definition()
		if _, ok := def.Metadata["strict"]; ok {
			t.Errorf("expected strict metadata to be absent by default, got %v", def.Metadata["strict"])
		}
	})

	check := func(want bool) func(*testing.T, AnyTool) {
		return func(t *testing.T, tl AnyTool) {
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

	t.Run("DefineTool surfaces the flag on Definition", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "strict/registered-true", "registered strict",
			func(ctx context.Context, input struct{}) (string, error) { return "", nil },
			WithStrictSchema(true),
		)
		check(true)(t, tl)

		found := LookupTool(r, "strict/registered-true")
		if found == nil {
			t.Fatal("LookupTool returned nil")
		}
		check(true)(t, found)
	})

	t.Run("NewTool round-trips the flag through Register and LookupTool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := NewTool("strict/dynamic-false", "dynamic loose",
			func(ctx context.Context, input struct{}) (string, error) { return "", nil },
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

	t.Run("setting strict twice panics", func(t *testing.T) {
		assertPanic(t, func() {
			r := newTestRegistry(t)
			DefineTool(r, "strict/double-set", "double set",
				func(ctx context.Context, input struct{}) (string, error) { return "", nil },
				WithStrictSchema(true),
				WithStrictSchema(false),
			)
		}, "strict schema")
	})
}

func TestToolRunRaw(t *testing.T) {
	t.Run("returns error from tool", func(t *testing.T) {
		r := newTestRegistry(t)
		tl := DefineTool(r, "provider/fail", "Always fails", func(ctx context.Context, input struct{}) (string, error) {
			return "", errors.New("intentional failure")
		})

		if _, err := tl.RunRaw(context.Background(), map[string]any{}); err == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("nil tool returns clear error", func(t *testing.T) {
		var tl *Tool[struct{}, string]
		_, err := tl.RunRaw(context.Background(), map[string]any{})
		assertError(t, err, "nil tool")
	})
}

// TestAttachParts verifies tool.AttachParts (via the part sink installed by
// the tool wrapper) folds extra content into the tool's multipart response
// without changing the function signature.
func TestAttachParts(t *testing.T) {
	r := newTestRegistry(t)
	shot := DefineTool(r, "screenshot", "takes a screenshot",
		func(ctx context.Context, _ struct{}) (string, error) {
			if sink := base.ToolPartSinkKey.FromContext(ctx); sink != nil {
				sink(NewMediaPart("image/png", "pngbytes"))
			}
			return "captured", nil
		})

	resp, err := shot.RunRaw(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRaw: %v", err)
	}
	if resp.Output != "captured" {
		t.Errorf("output = %v, want %q", resp.Output, "captured")
	}
	if len(resp.Content) != 1 || !resp.Content[0].IsMedia() {
		t.Fatalf("expected one attached media part, got %+v", resp.Content)
	}
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
			func(ctx context.Context, input any) (string, error) {
				m := input.(map[string]any)
				return m["customField"].(string), nil
			},
			WithInputSchema(customSchema))

		def := tl.Definition()
		if def.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
	})

	t.Run("WithInputSchema requires In to be any", func(t *testing.T) {
		assertPanic(t, func() {
			NewTool("typedIn", "d",
				func(ctx context.Context, input weatherIn) (string, error) { return "", nil },
				WithInputSchema(map[string]any{"type": "object"}))
		}, "requires In to be of type 'any'")
	})
}

func TestResolveUniqueTools(t *testing.T) {
	t.Run("resolves tools from registry", func(t *testing.T) {
		r := newTestRegistry(t)
		DefineTool(r, "provider/tool1", "Tool 1", func(ctx context.Context, input struct{}) (bool, error) {
			return true, nil
		})
		DefineTool(r, "provider/tool2", "Tool 2", func(ctx context.Context, input struct{}) (bool, error) {
			return true, nil
		})

		names, newTools, err := resolveUniqueTools(r, []ToolArg{
			ToolName("provider/tool1"),
			ToolName("provider/tool2"),
		})
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
		_, _, err := resolveUniqueTools(r, []ToolArg{
			ToolName("provider/dup"),
			ToolName("provider/dup"),
		})
		if err == nil {
			t.Error("expected error for duplicate tools, got nil")
		}
	})

	t.Run("identifies new tools to register", func(t *testing.T) {
		r := newTestRegistry(t)
		newTl := NewTool("provider/brandNew", "Brand new", func(ctx context.Context, input struct{}) (string, error) {
			return "new", nil
		})

		names, newTools, err := resolveUniqueTools(r, []ToolArg{newTl})
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

type confirmation struct {
	Approved bool `json:"approved"`
}

// TestInterrupt_NonObjectData_ReturnsClearError covers the documented
// constraint: interrupt data must serialize to a JSON object, and a scalar
// yields an actionable error where the tool runs.
func TestInterrupt_NonObjectData_ReturnsClearError(t *testing.T) {
	r := newTestRegistry(t)
	tl := DefineInterruptibleTool(r, "bad", "interrupts with a scalar",
		func(ctx context.Context, _ struct{}, _ *struct{}) (string, error) {
			return "", &InterruptError{Data: "not an object"}
		})

	_, err := tl.RunRaw(context.Background(), struct{}{})
	if err == nil {
		t.Fatal("expected an error interrupting with non-object data")
	}
	if !strings.Contains(err.Error(), "JSON object") {
		t.Errorf("error = %q, want it to mention the JSON object constraint", err)
	}
}

// TestSendPartial_StreamsPartialToolResponse asserts a tool's partial sends
// arrive on the stream as partial tool responses, distinguishable via
// IsPartial / ToolResponses.
func TestSendPartial_StreamsPartialToolResponse(t *testing.T) {
	r := newTestRegistry(t)
	defineToolThenFinishModel(r, NewToolRequestPart(&ToolRequest{Name: "progressTool", Input: map[string]any{}}))

	DefineTool(r, "progressTool", "streams progress",
		func(ctx context.Context, _ struct{}) (string, error) {
			if send := base.ToolPartialSenderKey.FromContext(ctx); send != nil {
				send(ctx, map[string]any{"progress": 50})
			}
			return "complete", nil
		})

	var partials []*Part
	for val, err := range GenerateStream(context.Background(), r,
		WithModelName("test/model"),
		WithPrompt("go"),
		WithTools(ToolName("progressTool"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
		if val.Done {
			continue
		}
		for _, p := range val.Chunk.ToolResponses() {
			if p.IsPartial() {
				partials = append(partials, p)
			}
		}
	}

	if len(partials) == 0 {
		t.Fatal("expected at least one partial tool response on the stream")
	}
	if partials[0].ToolResponse.Name != "progressTool" {
		t.Errorf("partial tool name = %q, want %q", partials[0].ToolResponse.Name, "progressTool")
	}
}

// TestConcurrentStreamingTools_NoDataRace is the regression for the streaming
// race: when a model emits multiple tool calls in one turn and more than one
// streams partial responses, the per-tool senders run on concurrent
// goroutines. They must be serialized so they don't race on the shared stream
// callback. Run under `go test -race` to detect a regression.
func TestConcurrentStreamingTools_NoDataRace(t *testing.T) {
	r := newTestRegistry(t)
	defineToolThenFinishModel(r,
		NewToolRequestPart(&ToolRequest{Name: "toolA", Input: map[string]any{}}),
		NewToolRequestPart(&ToolRequest{Name: "toolB", Input: map[string]any{}}))

	// A rendezvous so both tools enter their send loops at the same time,
	// maximizing the chance of overlapping callback invocations.
	var ready sync.WaitGroup
	ready.Add(2)
	start := make(chan struct{})
	go func() { ready.Wait(); close(start) }()

	streamer := func(ctx context.Context, _ struct{}) (string, error) {
		ready.Done()
		<-start
		send := base.ToolPartialSenderKey.FromContext(ctx)
		for i := 0; i < 200; i++ {
			if send != nil {
				send(ctx, map[string]any{"n": i})
			}
		}
		return "ok", nil
	}
	DefineTool(r, "toolA", "streams", streamer)
	DefineTool(r, "toolB", "streams", streamer)

	for _, err := range GenerateStream(context.Background(), r,
		WithModelName("test/model"),
		WithPrompt("go"),
		WithTools(ToolName("toolA"), ToolName("toolB"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
	}
}

// TestConcurrentAttachParts_NoDataRace is the regression for concurrent part
// attachment: a tool may fan work out to goroutines that each attach parts
// (mirroring tool.SendPartial, which is safe in the same pattern), so the
// per-invocation sink must be synchronized. Run under `go test -race` to
// detect a regression.
func TestConcurrentAttachParts_NoDataRace(t *testing.T) {
	const goroutines, perGoroutine = 4, 100

	tl := NewTool("fanout", "attaches parts from goroutines",
		func(ctx context.Context, _ struct{}) (string, error) {
			sink := base.ToolPartSinkKey.FromContext(ctx)
			var wg sync.WaitGroup
			for range goroutines {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < perGoroutine; i++ {
						sink(NewTextPart("part"))
					}
				}()
			}
			wg.Wait()
			return "ok", nil
		})

	resp, err := tl.RunRaw(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRaw: %v", err)
	}
	if got, want := len(resp.Content), goroutines*perGoroutine; got != want {
		t.Errorf("attached %d parts, want %d", got, want)
	}
}
