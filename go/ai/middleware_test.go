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
	"sync"
	"sync/atomic"
	"testing"
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

	tool := DefineTool(r, "restartable", "interrupts, then runs on resume",
		func(ctx *ToolContext, in restartInput) (string, error) {
			if in.Interrupt {
				return "", ctx.Interrupt(&InterruptOptions{})
			}
			return "ok", nil
		},
	)

	// Requests the tool on the first turn, returns a final text response once a
	// tool response is present in history.
	model := DefineModel(r, "test/restartModel", &ModelOptions{
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
