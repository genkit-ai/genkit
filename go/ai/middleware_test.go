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
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
)

// --- counter: a config whose BuildMiddleware tracks hook invocations ---

type counterConfig struct {
	Label string `json:"label,omitempty"`

	// Plugin-level state lives on unexported fields and is preserved by
	// the descriptor's value-copy across JSON-dispatch calls.
	sharedGenerateCalls *int32
	sharedModelCalls    *int32
	sharedToolCalls     *int32
}

func (counterConfig) Name() string { return "test/counter" }

func (c counterConfig) New(ctx context.Context) (*Hooks, error) {
	return &Hooks{
		WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
			if c.sharedGenerateCalls != nil {
				atomic.AddInt32(c.sharedGenerateCalls, 1)
			}
			return next(ctx, p)
		},
		WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
			if c.sharedModelCalls != nil {
				atomic.AddInt32(c.sharedModelCalls, 1)
			}
			return next(ctx, p)
		},
		WrapTool: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
			if c.sharedToolCalls != nil {
				atomic.AddInt32(c.sharedToolCalls, 1)
			}
			return next(ctx, p)
		},
	}, nil
}

// --- core descriptor tests ---

func TestNewMiddleware_NameFromPrototype(t *testing.T) {
	desc := NewMiddleware("tracks calls", counterConfig{})
	if desc.Name != "test/counter" {
		t.Errorf("got name %q, want %q", desc.Name, "test/counter")
	}
	if desc.Description != "tracks calls" {
		t.Errorf("got description %q, want %q", desc.Description, "tracks calls")
	}
}

func TestBuildFromJSON(t *testing.T) {
	desc := NewMiddleware("desc", counterConfig{})
	mw, err := desc.buildFromJSON(testCtx, []byte(`{"label": "custom"}`))
	if err != nil {
		t.Fatalf("buildFromJSON failed: %v", err)
	}
	if mw == nil || mw.WrapModel == nil {
		t.Fatal("expected middleware with WrapModel hook")
	}
}

func TestBuildFromJSON_InvalidJSON(t *testing.T) {
	desc := NewMiddleware("desc", counterConfig{})
	_, err := desc.buildFromJSON(testCtx, []byte(`not-json`))
	if err == nil {
		t.Fatal("expected error from invalid JSON")
	}
}

// --- plugin-level state: prototype unexported fields preserved across calls ---

func TestPluginStateCarriedThroughPrototype(t *testing.T) {
	// Simulate the plugin's Middlewares() building a prototype that holds
	// an "expensive client" (here, a shared counter). JSON dispatch must
	// preserve this plugin-level state across invocations.
	var shared int32
	desc := NewMiddleware("desc", counterConfig{sharedModelCalls: &shared})

	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})
	desc.Register(r)

	// Simulate Dev UI JSON dispatch: ref.Config is a map, not a typed config.
	refs := []*MiddlewareRef{{Name: "test/counter", Config: map[string]any{"label": "dev-ui"}}}
	for i := 0; i < 3; i++ {
		_, err := GenerateWithRequest(testCtx, r, &GenerateActionOptions{
			Model:    m.Name(),
			Messages: []*Message{NewUserTextMessage("go")},
			Use:      refs,
		}, nil, nil)
		assertNoError(t, err)
	}
	if got := atomic.LoadInt32(&shared); got != 3 {
		t.Errorf("shared counter = %d, want 3 (plugin state should persist across JSON dispatches)", got)
	}
}

// --- per-call isolation: JSON dispatch never writes through to the prototype ---

// isolationLog records the config each New() was built with. Dispatches in
// these tests are sequential, so no locking.
type isolationLog struct {
	seen []isolationConfig
}

func (l *isolationLog) at(t *testing.T, i int) isolationConfig {
	t.Helper()
	if i >= len(l.seen) {
		t.Fatalf("wanted config %d, only %d recorded", i, len(l.seen))
	}
	return l.seen[i]
}

// isolationConfig mirrors the built-in middleware shape: exported fields are
// per-call user config, unexported fields are plugin state.
type isolationConfig struct {
	Label    string            `json:"label,omitempty"`
	Statuses []string          `json:"statuses,omitempty"`
	Tags     map[string]string `json:"tags,omitempty"`

	log *isolationLog
}

func (isolationConfig) Name() string { return "test/isolation" }

func (c isolationConfig) New(ctx context.Context) (*Hooks, error) {
	if c.log != nil {
		c.log.seen = append(c.log.seen, c)
	}
	return &Hooks{}, nil
}

// Registering by value is the documented shape, but [NewMiddleware] accepts
// any [Middleware], so a pointer prototype has to isolate just the same.
func forEachPrototypeShape(t *testing.T, run func(t *testing.T, desc *MiddlewareDesc, log *isolationLog, snapshot func() isolationConfig)) {
	t.Helper()
	for _, tc := range []struct {
		name  string
		build func(isolationConfig) (*MiddlewareDesc, func() isolationConfig)
	}{
		{"value prototype", func(p isolationConfig) (*MiddlewareDesc, func() isolationConfig) {
			return NewMiddleware("desc", p), func() isolationConfig { return p }
		}},
		{"pointer prototype", func(p isolationConfig) (*MiddlewareDesc, func() isolationConfig) {
			ptr := &p
			return NewMiddleware("desc", ptr), func() isolationConfig { return *ptr }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log := &isolationLog{}
			desc, snapshot := tc.build(isolationConfig{log: log})
			run(t, desc, log, snapshot)
		})
	}
}

func TestBuildFromJSON_ConfigDoesNotLeakBetweenCalls(t *testing.T) {
	forEachPrototypeShape(t, func(t *testing.T, desc *MiddlewareDesc, log *isolationLog, _ func() isolationConfig) {
		buildOrFail(t, desc, `{"label":"first","statuses":["a","b"],"tags":{"k":"v"}}`)
		buildOrFail(t, desc, `{}`)

		second := log.at(t, 1)
		if second.Label != "" {
			t.Errorf("Label = %q, want empty (leaked from the previous call)", second.Label)
		}
		if second.Statuses != nil {
			t.Errorf("Statuses = %v, want nil (leaked from the previous call)", second.Statuses)
		}
		if second.Tags != nil {
			t.Errorf("Tags = %v, want nil (leaked from the previous call)", second.Tags)
		}
	})
}

func TestBuildFromJSON_PrototypeNotMutated(t *testing.T) {
	forEachPrototypeShape(t, func(t *testing.T, desc *MiddlewareDesc, _ *isolationLog, snapshot func() isolationConfig) {
		buildOrFail(t, desc, `{"label":"first","statuses":["a","b"],"tags":{"k":"v"}}`)

		got := snapshot()
		if got.Label != "" || got.Statuses != nil || got.Tags != nil {
			t.Errorf("prototype mutated by dispatch: %+v", got)
		}
	})
}

// ptrRecvConfig takes pointer receivers, so Name() survives being read off a
// nil prototype during registration.
type ptrRecvConfig struct {
	Label string `json:"label,omitempty"`
}

func (*ptrRecvConfig) Name() string { return "test/ptr-recv" }

func (c *ptrRecvConfig) New(context.Context) (*Hooks, error) {
	return &Hooks{
		WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
			_ = c.Label // A nil receiver surfaces here, not in New.
			return next(ctx, p)
		},
	}, nil
}

// A nil pointer prototype is a programming error, but it must not panic the
// process. A call that sends no config has nothing to unmarshal, so nothing
// allocates through the pointer and New would otherwise get a nil receiver.
func TestBuildFromJSON_NilPrototype(t *testing.T) {
	desc := NewMiddleware("desc", (*ptrRecvConfig)(nil))
	h, err := desc.buildFromJSON(testCtx, nil)
	if err != nil {
		t.Fatalf("buildFromJSON failed: %v", err)
	}
	next := func(context.Context, *ModelParams) (*ModelResponse, error) { return &ModelResponse{}, nil }
	if _, err := h.WrapModel(testCtx, &ModelParams{}, next); err != nil {
		t.Fatalf("WrapModel failed: %v", err)
	}
}

func buildOrFail(t *testing.T, desc *MiddlewareDesc, configJSON string) {
	t.Helper()
	if _, err := desc.buildFromJSON(testCtx, []byte(configJSON)); err != nil {
		t.Fatalf("buildFromJSON(%s) failed: %v", configJSON, err)
	}
}

// --- call-level state: each Generate gets fresh BuildMiddleware scope ---

type perCallConfig struct {
	checker func(n int32)
}

func (perCallConfig) Name() string { return "test/per-call" }

func (c perCallConfig) New(ctx context.Context) (*Hooks, error) {
	var counter int32
	return &Hooks{
		WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
			n := atomic.AddInt32(&counter, 1)
			if c.checker != nil {
				c.checker(n)
			}
			return next(ctx, p)
		},
	}, nil
}

func TestCallLevelStateIsolation(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	cfg := perCallConfig{checker: func(n int32) {
		if n != 1 {
			t.Errorf("call-level counter leaked: got %d, want 1", n)
		}
	}}
	for i := 0; i < 3; i++ {
		_, err := Generate(testCtx, r, WithModel(m), WithPrompt("go"), WithUse(cfg))
		assertNoError(t, err)
	}
}

// --- pure Go usage: no registration required for local calls ---

func TestWithUseNoRegistrationNeeded(t *testing.T) {
	// The whole point: user creates a Genkit with no middleware plugins and
	// still calls WithUse(middleware.Retry{...}). The config's BuildMiddleware
	// method runs directly; the registry is never consulted.
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	var called int32
	cfg := counterConfig{sharedModelCalls: &called}
	// Note: no Register call anywhere.

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hi"), WithUse(cfg))
	assertNoError(t, err)
	if atomic.LoadInt32(&called) != 1 {
		t.Errorf("expected 1 model-hook call, got %d", called)
	}
}

// --- hook invocation: model, tool, generate ---

func TestMiddlewareModelHook(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	var called int32
	tracker := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
				atomic.AddInt32(&called, 1)
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hello"), WithUse(tracker))
	assertNoError(t, err)
	if atomic.LoadInt32(&called) == 0 {
		t.Error("expected model hook to be called")
	}
}

func TestMiddlewareToolHook(t *testing.T) {
	r := newTestRegistry(t)
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/toolModel",
		handler: toolCallingModelHandler("myTool", map[string]any{"value": "test"}, "done"),
	})
	defineFakeTool(t, r, "myTool", "A test tool")

	var called int32
	tracker := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapTool: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				atomic.AddInt32(&called, 1)
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/toolModel"),
		WithPrompt("use the tool"),
		WithTools(ToolName("myTool")),
		WithUse(tracker),
	)
	assertNoError(t, err)
	if atomic.LoadInt32(&called) == 0 {
		t.Error("expected tool hook to be called at least once")
	}
}

func TestMiddlewareGenerateHook(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	var called int32
	tracker := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
				atomic.AddInt32(&called, 1)
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hello"), WithUse(tracker))
	assertNoError(t, err)
	if atomic.LoadInt32(&called) == 0 {
		t.Error("expected generate hook to be called")
	}
}

// --- ordering: first middleware wraps outermost ---

func TestMiddlewareOrdering(t *testing.T) {
	var mu sync.Mutex
	var order []string
	appendOrder := func(s string) {
		mu.Lock()
		defer mu.Unlock()
		order = append(order, s)
	}
	tracker := func(label string) Middleware {
		return MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
					appendOrder(label + "-before")
					resp, err := next(ctx, p)
					appendOrder(label + "-after")
					return resp, err
				},
			}, nil
		})
	}

	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	_, err := Generate(testCtx, r,
		WithModel(m),
		WithPrompt("hello"),
		WithUse(tracker("A"), tracker("B")),
	)
	assertNoError(t, err)

	want := []string{"A-before", "B-before", "B-after", "A-after"}
	if len(order) != len(want) {
		t.Fatalf("got %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, order[i], want[i])
		}
	}
}

// --- MiddlewareFunc adapter basics ---

func TestMiddlewareFunc(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	var called bool
	mw := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
				called = true
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hello"), WithUse(mw))
	assertNoError(t, err)
	if !called {
		t.Error("inline middleware hook not called")
	}
}

func TestMiddlewareFuncCoexist(t *testing.T) {
	// Two MiddlewareFunc adapter instances should be able to coexist in a
	// single WithUse call.
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	var a, b int32
	useA := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
			atomic.AddInt32(&a, 1)
			return next(ctx, p)
		}}, nil
	})
	useB := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
			atomic.AddInt32(&b, 1)
			return next(ctx, p)
		}}, nil
	})

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hi"), WithUse(useA, useB))
	assertNoError(t, err)
	if atomic.LoadInt32(&a) != 1 || atomic.LoadInt32(&b) != 1 {
		t.Errorf("expected both hooks called once, got a=%d b=%d", a, b)
	}
}

// --- optional hooks: nil hook fields must pass through ---

func TestNilHookFieldsPassThrough(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	passthrough := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{}, nil
	})

	resp, err := Generate(testCtx, r, WithModel(m), WithPrompt("hi"), WithUse(passthrough))
	assertNoError(t, err)
	if resp == nil {
		t.Fatal("expected response")
	}
}

// --- streaming: middleware-emitted chunks accumulate with model chunks ---

func TestMiddlewareStreamsAccumulateWithModel(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{
		name:    "test/streamModel",
		handler: streamingModelHandler([]string{"model chunk"}, "done"),
	})

	midChunk := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
				if p.Callback != nil {
					if err := p.Callback(ctx, &ModelResponseChunk{
						Content: []*Part{NewTextPart("middleware chunk ")},
					}); err != nil {
						return nil, err
					}
				}
				return next(ctx, p)
			},
		}, nil
	})

	var chunks []*ModelResponseChunk
	_, err := Generate(testCtx, r,
		WithModel(m),
		WithPrompt("go"),
		WithUse(midChunk),
		WithStreaming(func(_ context.Context, c *ModelResponseChunk) error {
			chunks = append(chunks, c)
			return nil
		}),
	)
	assertNoError(t, err)

	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}
	if chunks[0].Role != RoleModel {
		t.Errorf("chunks[0].Role = %q, want %q", chunks[0].Role, RoleModel)
	}
	if chunks[0].Index != chunks[1].Index {
		t.Errorf("chunks[0].Index=%d chunks[1].Index=%d; want equal", chunks[0].Index, chunks[1].Index)
	}

	var midText string
	if err := chunks[0].Output(&midText); err != nil {
		t.Fatalf("middleware chunk Output error: %v", err)
	}
	if midText != "middleware chunk " {
		t.Errorf("middleware chunk Output = %q, want %q", midText, "middleware chunk ")
	}

	var modelText string
	if err := chunks[1].Output(&modelText); err != nil {
		t.Fatalf("model chunk Output error: %v", err)
	}
	if modelText != "middleware chunk model chunk" {
		t.Errorf("model chunk Output = %q, want %q", modelText, "middleware chunk model chunk")
	}
}

// --- tool contribution: Tools on *Middleware ---

func TestMiddlewareContributesTool(t *testing.T) {
	r := newTestRegistry(t)
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/toolModel",
		handler: toolCallingModelHandler("mw/tool", map[string]any{"value": "x"}, "done"),
	})

	injectTool := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			Tools: []Tool{NewTool("mw/tool", "injected",
				func(tc *ToolContext, in struct {
					Value string `json:"value"`
				}) (string, error) {
					return "ok", nil
				})},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/toolModel"),
		WithPrompt("use it"),
		WithUse(injectTool),
	)
	assertNoError(t, err)
}

// Middleware-contributed tools are registered before their definitions are
// captured, so schema names (WithInputSchemaName/WithOutputSchemaName) reach
// the model resolved rather than as raw {"$ref": "genkit:..."} maps.
func TestMiddlewareToolSchemaNamesResolve(t *testing.T) {
	r := newTestRegistry(t)
	r.RegisterSchema("Answer", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
	})

	var captured *ModelRequest
	defineFakeModel(t, r, fakeModelConfig{
		name: "test/refModel",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			return &ModelResponse{Request: req, Message: NewModelTextMessage("done")}, nil
		},
	})

	inject := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			Tools: []Tool{NewTool("mw/schemaTool", "named schemas",
				func(tc *ToolContext, in any) (any, error) { return "ok", nil },
				WithInputSchemaName("Answer"), WithOutputSchemaName("Answer"))},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/refModel"),
		WithPrompt("hi"),
		WithUse(inject),
	)
	assertNoError(t, err)

	if captured == nil {
		t.Fatal("model was never called")
	}
	var def *ToolDefinition
	for _, td := range captured.Tools {
		if td.Name == "mw/schemaTool" {
			def = td
		}
	}
	if def == nil {
		t.Fatalf("middleware tool not sent to the model: %v", captured.Tools)
	}
	for slot, schema := range map[string]map[string]any{
		"InputSchema":  def.InputSchema,
		"OutputSchema": def.OutputSchema,
	} {
		if _, ok := schema["$ref"]; ok {
			t.Errorf("%s = %v, want the resolved Answer schema, not a $ref", slot, schema)
		}
		if props, ok := schema["properties"].(map[string]any); !ok || props["answer"] == nil {
			t.Errorf("%s = %v, want the registered Answer schema", slot, schema)
		}
	}
}

// --- duplicate tool collision: two middleware with same tool name ---

func TestDuplicateMiddlewareToolRejected(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	makeInjector := func() Middleware {
		return MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				Tools: []Tool{NewTool("dup/tool", "d",
					func(tc *ToolContext, in struct{}) (string, error) { return "x", nil })},
			}, nil
		})
	}

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hi"),
		WithUse(makeInjector(), makeInjector()))
	if err == nil {
		t.Fatal("expected duplicate tool error, got nil")
	}
}

// --- error propagation from BuildMiddleware ---

func TestBuildMiddlewareErrorPropagates(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{})

	bad := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return nil, errors.New("boom")
	})

	_, err := Generate(testCtx, r, WithModel(m), WithPrompt("hi"), WithUse(bad))
	if err == nil {
		t.Fatal("expected BuildMiddleware error, got nil")
	}
}

// --- tool interrupt from WrapTool ---

func TestWrapToolInterrupts(t *testing.T) {
	r := newTestRegistry(t)
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/toolModel",
		handler: toolCallingModelHandler("myTool", map[string]any{"value": "x"}, "done"),
	})
	defineFakeTool(t, r, "myTool", "A test tool")

	interrupter := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapTool: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				return nil, NewToolInterruptError(map[string]any{"reason": "blocked"})
			},
		}, nil
	})

	resp, err := Generate(testCtx, r,
		WithModelName("test/toolModel"),
		WithPrompt("use it"),
		WithTools(ToolName("myTool")),
		WithUse(interrupter),
	)
	assertNoError(t, err)
	if resp.FinishReason != "interrupted" {
		t.Errorf("expected FinishReason=interrupted, got %q", resp.FinishReason)
	}
	if len(resp.Interrupts()) == 0 {
		t.Error("expected at least one interrupt part in response")
	}
}

// --- WrapTool short-circuits are attributed to the tool in traces ---

// defineCountingTool defines a tool that records how many times its function
// body ran, so a test can tell a short-circuited call from a real one.
func defineCountingTool(t *testing.T, r api.Registry, name string, calls *int32) Tool {
	t.Helper()
	return defineTool(r, name, "A test tool",
		func(ctx *ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			atomic.AddInt32(calls, 1)
			return "tool result: " + input.Value, nil
		})
}

// TestWrapToolShortCircuitEmitsToolSpan verifies that a WrapTool hook which
// resolves a call without invoking next (cached response, interrupt) is still
// attributed to the tool in traces, with the span core/action.go would have
// emitted had the tool run. Middleware therefore does not hand-roll its own.
func TestWrapToolShortCircuitEmitsToolSpan(t *testing.T) {
	tests := []struct {
		name      string
		hook      func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error)
		wantState string
		// wantGenerateErr is true for the case that fails the whole call
		// rather than resolving the turn; the span is recorded either way.
		wantGenerateErr bool
	}{
		{
			name: "cached response",
			hook: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{Output: "from cache"}, nil
			},
			wantState: "success",
		},
		{
			name: "interrupt",
			hook: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				return nil, NewToolInterruptError(map[string]any{"reason": "blocked"})
			},
			wantState: "error",
		},
		{
			name: "injected error",
			hook: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				return nil, errors.New("denied")
			},
			wantState:       "error",
			wantGenerateErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestRegistry(t)
			defineFakeModel(t, r, fakeModelConfig{
				name:    "test/toolModel",
				handler: toolCallingModelHandler("myTool", map[string]any{"value": "x"}, "done"),
			})
			var toolCalls int32
			tool := defineCountingTool(t, r, "myTool", &toolCalls)
			spans := collectSpans(t)

			shortCircuit := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
				return &Hooks{WrapTool: tt.hook}, nil
			})

			_, err := Generate(testCtx, r,
				WithModelName("test/toolModel"),
				WithPrompt("use it"),
				WithTools(tool),
				WithUse(shortCircuit),
			)
			if tt.wantGenerateErr {
				if err == nil {
					t.Fatal("expected the injected error to fail the call, got nil")
				}
			} else {
				assertNoError(t, err)
			}

			if got := atomic.LoadInt32(&toolCalls); got != 0 {
				t.Errorf("tool ran %d times, want 0 (hook short-circuited)", got)
			}

			toolSpans := spans.allByName(tool.Name())
			if len(toolSpans) != 1 {
				t.Fatalf("got %d spans named %q, want exactly 1", len(toolSpans), tool.Name())
			}
			span := toolSpans[0]
			assertSpanAttr(t, span, "genkit:type", "action")
			assertSpanAttr(t, span, "genkit:metadata:subtype", string(api.ActionTypeToolV2))
			assertSpanAttr(t, span, "genkit:state", tt.wantState)
			assertSpanAttr(t, span, "genkit:input", `{"value":"x"}`)
		})
	}
}

// TestWrapToolPassThroughEmitsSingleToolSpan verifies the engine does not
// double-count a hook that runs the tool: the action's own span is the only
// one recorded, however the hook chose to hand the call down. The rebuilt-params
// case is why the ran flag rides the context and not ToolParams, which a hook
// is free to replace.
func TestWrapToolPassThroughEmitsSingleToolSpan(t *testing.T) {
	rewritten := map[string]any{"value": "rewritten"}

	tests := []struct {
		name string
		hook func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error)
	}{
		{
			name: "mutates the params it was handed",
			hook: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				p.Request = &ToolRequest{Name: p.Request.Name, Ref: p.Request.Ref, Input: rewritten}
				return next(ctx, p)
			},
		},
		{
			name: "hands next a params struct of its own",
			hook: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				return next(ctx, &ToolParams{
					Request: &ToolRequest{Name: p.Request.Name, Ref: p.Request.Ref, Input: rewritten},
					Tool:    p.Tool,
				})
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestRegistry(t)
			defineFakeModel(t, r, fakeModelConfig{
				name:    "test/toolModel",
				handler: toolCallingModelHandler("myTool", map[string]any{"value": "x"}, "done"),
			})
			var toolCalls int32
			tool := defineCountingTool(t, r, "myTool", &toolCalls)
			spans := collectSpans(t)

			rewriter := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
				return &Hooks{WrapTool: tt.hook}, nil
			})

			_, err := Generate(testCtx, r,
				WithModelName("test/toolModel"),
				WithPrompt("use it"),
				WithTools(tool),
				WithUse(rewriter),
			)
			assertNoError(t, err)

			if got := atomic.LoadInt32(&toolCalls); got != 1 {
				t.Errorf("tool ran %d times, want 1", got)
			}
			toolSpans := spans.allByName(tool.Name())
			if len(toolSpans) != 1 {
				t.Fatalf("got %d spans named %q, want exactly 1", len(toolSpans), tool.Name())
			}
			assertSpanAttr(t, toolSpans[0], "genkit:input", `{"value":"rewritten"}`)
			// The short-circuit span is meant to be indistinguishable from
			// this one, so pin them to the same shape from both sides.
			assertSpanAttr(t, toolSpans[0], "genkit:type", "action")
			assertSpanAttr(t, toolSpans[0], "genkit:metadata:subtype", string(api.ActionTypeToolV2))
		})
	}
}

// --- WrapTool validation errors returned to model ---

func TestWrapToolValidationErrorReturnedToModel(t *testing.T) {
	r := newTestRegistry(t)

	var callCount int32

	modelHandler := func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		c := atomic.AddInt32(&callCount, 1)

		if c <= 2 {
			var toolInput map[string]any
			if c == 1 {
				toolInput = map[string]any{"value": "not_a_number"}
			} else {
				toolInput = map[string]any{"value": 42}
			}
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role: RoleModel,
					Content: []*Part{NewToolRequestPart(&ToolRequest{
						Name:  "validateMe",
						Input: toolInput,
					})},
				},
			}, nil
		}

		return &ModelResponse{
			Request:      req,
			Message:      NewModelTextMessage("done"),
			FinishReason: FinishReasonStop,
		}, nil
	}

	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/model",
		handler: modelHandler,
	})

	defineTool(r, "validateMe", "A tool that requires a numeric value",
		func(ctx *ToolContext, input any) (string, error) {
			m := input.(map[string]any)
			return fmt.Sprintf("success: %v", m["value"]), nil
		},
		WithInputSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"value": map[string]any{
					"type": "number",
				},
			},
			"required": []string{"value"},
		}),
	)

	errorHandler := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapTool: func(ctx context.Context, params *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				resp, err := next(ctx, params)
				if errors.Is(err, status.ErrInvalidInput) {
					return &MultipartToolResponse{
						Content: []*Part{NewTextPart(fmt.Sprintf("Validation error: %v", err))},
						Output:  "tool call failed; see content for details",
					}, nil
				}
				return resp, err
			},
		}, nil
	})

	resp, err := Generate(testCtx, r,
		WithModelName("test/model"),
		WithPrompt("use the tool"),
		WithTools(ToolName("validateMe")),
		WithUse(errorHandler),
	)
	assertNoError(t, err)
	if resp.FinishReason != FinishReasonStop {
		t.Errorf("expected FinishReason=stop, got %q", resp.FinishReason)
	}
	if c := atomic.LoadInt32(&callCount); c < 3 {
		t.Errorf("expected >=3 model calls (invalid → retry → done), got %d", c)
	}
}

// --- WrapGenerate fires per tool-loop iteration ---

func TestGenerateHookFiresEachIteration(t *testing.T) {
	r := newTestRegistry(t)
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/toolLoop",
		handler: toolCallingModelHandler("myTool", map[string]any{"value": "x"}, "done"),
	})
	defineFakeTool(t, r, "myTool", "A test tool")

	var iters int32
	tracker := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
				atomic.AddInt32(&iters, 1)
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/toolLoop"),
		WithPrompt("use it"),
		WithTools(ToolName("myTool")),
		WithUse(tracker),
	)
	assertNoError(t, err)
	if got := atomic.LoadInt32(&iters); got < 2 {
		t.Errorf("expected WrapGenerate to fire >=2 times, got %d", got)
	}
}

// --- WrapTool metadata survives round trip ---

func TestWrapToolPreservesMetadata(t *testing.T) {
	r := newTestRegistry(t)
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/toolModel",
		handler: toolCallingModelHandler("myTool", map[string]any{"value": "x"}, "done"),
	})
	defineFakeTool(t, r, "myTool", "A test tool")

	var sawMetadata map[string]any
	reader := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapTool: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				resp, err := next(ctx, p)
				if err != nil {
					return nil, err
				}
				if resp.Metadata == nil {
					resp.Metadata = map[string]any{}
				}
				resp.Metadata["traced"] = true
				sawMetadata = resp.Metadata
				return resp, nil
			},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/toolModel"),
		WithPrompt("use it"),
		WithTools(ToolName("myTool")),
		WithUse(reader),
	)
	assertNoError(t, err)
	if sawMetadata == nil || sawMetadata["traced"] != true {
		t.Errorf("metadata not threaded, got %v", sawMetadata)
	}
}

// --- hook ordering on tool restart: outer generate > restarted tool > inner generate > model ---

func TestMiddlewareHookOrderOnToolRestart(t *testing.T) {
	r := newTestRegistry(t)

	type restartInput struct {
		Interrupt bool `json:"interrupt"`
	}

	tool := defineTool(r, "restartable", "interrupts, then runs on resume",
		func(ctx *ToolContext, in restartInput) (string, error) {
			if in.Interrupt {
				return "", ctx.Interrupt(&InterruptOptions{})
			}
			return "ok", nil
		},
	)

	// Requests the tool on the first turn, returns a final text response once a
	// tool response is present in history.
	model := defineModel(r, "test/restartModel", &ModelOptions{
		Supports: &ModelSupports{Multiturn: true, Tools: true},
	}, func(ctx context.Context, req *ModelRequest, _ ModelStreamCallback) (*ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, p := range msg.Content {
				if p.IsToolResponse() {
					return &ModelResponse{
						Request: req,
						Message: NewModelTextMessage("done"),
					}, nil
				}
			}
		}
		return &ModelResponse{
			Request: req,
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{NewToolRequestPart(&ToolRequest{
					Name:  "restartable",
					Ref:   "t1",
					Input: map[string]any{"interrupt": true},
				})},
			},
		}, nil
	})

	first, err := Generate(testCtx, r, WithModel(model), WithPrompt("go"), WithTools(tool))
	assertNoError(t, err)
	if first.FinishReason != "interrupted" {
		t.Fatalf("expected FinishReason=interrupted, got %q", first.FinishReason)
	}
	interruptedPart := first.Message.Content[0]

	var mu sync.Mutex
	var order []string
	record := func(s string) {
		mu.Lock()
		defer mu.Unlock()
		order = append(order, s)
	}

	tracker := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
				record("generate")
				return next(ctx, p)
			},
			WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
				record("model")
				return next(ctx, p)
			},
			WrapTool: func(ctx context.Context, p *ToolParams, next ToolNext) (*MultipartToolResponse, error) {
				record("tool")
				return next(ctx, p)
			},
		}, nil
	})

	restartPart, err := tool.RestartWith(interruptedPart, WithNewInput[restartInput](restartInput{Interrupt: false}))
	assertNoError(t, err)

	resumed, err := Generate(testCtx, r,
		WithModel(model),
		WithMessages(first.History()...),
		WithTools(tool),
		WithToolRestarts(restartPart),
		WithUse(tracker),
	)
	assertNoError(t, err)
	if resumed.FinishReason == "interrupted" {
		t.Fatalf("expected completion after restart, got interrupted")
	}

	// Resume handling lives inside the outer generate span, so the restarted
	// tool fires before the recursive follow-up iteration's generate+model.
	want := []string{"generate", "tool", "generate", "model"}
	if len(order) != len(want) {
		t.Fatalf("hook order: got %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, order[i], want[i])
		}
	}
}

var testCtx = context.Background()

// --- middlewareRefArg: lazy reference used by the dotprompt loader ---

func TestConfigsToRefs_StripsLazyAdapter(t *testing.T) {
	refs, err := configsToRefs([]Middleware{
		middlewareRefArg{name: "test/foo"},
		middlewareRefArg{name: "test/bar", config: map[string]any{"k": "v"}},
		counterConfig{},
	})
	assertNoError(t, err)
	if len(refs) != 3 {
		t.Fatalf("got %d refs, want 3", len(refs))
	}

	if refs[0].Name != "test/foo" || refs[0].Config != nil {
		t.Errorf("name-only adapter: got {Name:%q Config:%v}, want {test/foo nil}", refs[0].Name, refs[0].Config)
	}

	cfg, ok := refs[1].Config.(map[string]any)
	if refs[1].Name != "test/bar" || !ok || cfg["k"] != "v" {
		t.Errorf("adapter with config: got {Name:%q Config:%v}, want {test/bar map[k:v]}", refs[1].Name, refs[1].Config)
	}

	if _, ok := refs[2].Config.(Middleware); !ok {
		t.Errorf("regular middleware ref: Config should retain Middleware value for fast path, got %T", refs[2].Config)
	}
}

func TestMiddlewareRefArg_NewErrors(t *testing.T) {
	// Defensive: configsToRefs strips the adapter, so resolveRefs should
	// never call New on it. If routing ever regresses, fail loudly instead
	// of silently producing nil hooks.
	if _, err := (middlewareRefArg{name: "x"}).New(testCtx); err == nil {
		t.Fatal("expected middlewareRefArg.New to return an error")
	}
}

// --- hook logging: runs are attributed and short-circuits flagged ---

// hookLogRecorder is a slog.Handler that captures records at or above min.
type hookLogRecorder struct {
	min     slog.Level
	mu      sync.Mutex
	records []slog.Record
}

func (h *hookLogRecorder) Enabled(_ context.Context, l slog.Level) bool { return l >= h.min }

func (h *hookLogRecorder) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, r.Clone())
	return nil
}

func (h *hookLogRecorder) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *hookLogRecorder) WithGroup(string) slog.Handler      { return h }

func TestMiddlewareHookLogging(t *testing.T) {
	base := func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{}, nil
	}
	passThrough := namedHooks{name: "test/outer", hooks: &Hooks{
		WrapModel: func(ctx context.Context, params *ModelParams, next ModelNext) (*ModelResponse, error) {
			return next(ctx, params)
		},
	}}
	shortCircuit := namedHooks{name: "test/inner", hooks: &Hooks{
		WrapModel: func(ctx context.Context, params *ModelParams, next ModelNext) (*ModelResponse, error) {
			return &ModelResponse{}, nil // resolves the call without invoking next
		},
	}}
	chain := buildModelChain([]namedHooks{passThrough, shortCircuit}, base)

	rec := &hookLogRecorder{min: slog.LevelDebug}
	ctx := logger.WithContext(context.Background(), slog.New(rec))
	if _, err := chain(ctx, &ModelRequest{}, nil); err != nil {
		t.Fatal(err)
	}

	type entry struct {
		msg, mw        string
		shortCircuited bool
	}
	var got []entry
	for _, r := range rec.records {
		e := entry{msg: r.Message}
		r.Attrs(func(a slog.Attr) bool {
			switch a.Key {
			case "middleware":
				e.mw = a.Value.String()
			case "shortCircuited":
				e.shortCircuited = a.Value.Bool()
			}
			return true
		})
		got = append(got, e)
	}
	want := []entry{
		{msg: "middleware hook started", mw: "test/outer"},
		{msg: "middleware hook started", mw: "test/inner"},
		{msg: "middleware hook finished", mw: "test/inner", shortCircuited: true},
		{msg: "middleware hook finished", mw: "test/outer"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("hook log entries = %v, want %v", got, want)
	}
}

func TestMiddlewareHookLoggingDisabled(t *testing.T) {
	ran := false
	base := func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		ran = true
		return &ModelResponse{}, nil
	}
	chain := buildModelChain([]namedHooks{{name: "test/mw", hooks: &Hooks{
		WrapModel: func(ctx context.Context, params *ModelParams, next ModelNext) (*ModelResponse, error) {
			return next(ctx, params)
		},
	}}}, base)

	rec := &hookLogRecorder{min: slog.LevelInfo}
	ctx := logger.WithContext(context.Background(), slog.New(rec))
	if _, err := chain(ctx, &ModelRequest{}, nil); err != nil {
		t.Fatal(err)
	}
	if !ran {
		t.Fatal("base model fn did not run")
	}
	if len(rec.records) != 0 {
		t.Errorf("got %d log records with debug disabled, want 0", len(rec.records))
	}
}

// TestWrapGenerateOptionsAreIsolatedPerIteration checks that a WrapGenerate
// hook writing to its Options disturbs neither later turns nor their spans.
func TestWrapGenerateOptionsAreIsolatedPerIteration(t *testing.T) {
	r := newTestRegistry(t)
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/toolModel",
		handler: toolCallingModelHandler("myTool", map[string]any{"value": "x"}, "done"),
	})
	var toolCalls int32
	tool := defineCountingTool(t, r, "myTool", &toolCalls)
	spans := collectSpans(t)

	var seen []int
	clobber := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
				seen = append(seen, len(p.Options.Messages))
				p.Options.Messages = nil
				p.Options.StepName = "clobbered"
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/toolModel"),
		WithPrompt("use it"),
		WithTools(tool),
		WithUse(clobber),
	)
	assertNoError(t, err)

	// One tool call and a final answer; each turn sees the call's own message.
	want := []int{1, 1}
	if !slices.Equal(seen, want) {
		t.Errorf("hook saw %v messages per iteration, want %v", seen, want)
	}
	if got := len(spans.allByName("clobbered")); got != 0 {
		t.Errorf("got %d spans named %q, want 0: a hook's step name must not rename the loop's spans", got, "clobbered")
	}
	if got := len(spans.allByName("generate")); got != len(want) {
		t.Errorf("got %d generate spans, want %d (one per iteration)", got, len(want))
	}
}
