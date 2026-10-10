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
//
// SPDX-License-Identifier: Apache-2.0

package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
)

func newTestGenkit(t *testing.T) *genkit.Genkit {
	t.Helper()
	return genkit.Init(context.Background())
}

func defineTestModel(t *testing.T, g *genkit.Genkit, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return genkit.DefineModel(g, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, fn)
}

func TestFallbackNotTriggeredOnSuccess(t *testing.T) {
	g := newTestGenkit(t)
	primaryCalls := 0
	secondaryCalls := 0

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		primaryCalls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("primary")}, nil
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		secondaryCalls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "primary" {
		t.Errorf("got %q, want %q", resp.Text(), "primary")
	}
	if primaryCalls != 1 {
		t.Errorf("primary called %d times, want 1", primaryCalls)
	}
	if secondaryCalls != 0 {
		t.Errorf("secondary called %d times, want 0", secondaryCalls)
	}
}

func TestFallbackTriggeredOnRetryableError(t *testing.T) {
	g := newTestGenkit(t)

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "secondary ok" {
		t.Errorf("got %q, want %q", resp.Text(), "secondary ok")
	}
}

func TestFallbackTriesMultipleModels(t *testing.T) {
	g := newTestGenkit(t)
	var callOrder []string

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		callOrder = append(callOrder, "primary")
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		callOrder = append(callOrder, "secondary")
		return nil, core.NewError(core.RESOURCE_EXHAUSTED, "secondary exhausted")
	})
	tertiary := defineTestModel(t, g, "test/tertiary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		callOrder = append(callOrder, "tertiary")
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("tertiary ok")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil), ai.NewModelRef(tertiary.Name(), nil)}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "tertiary ok" {
		t.Errorf("got %q, want %q", resp.Text(), "tertiary ok")
	}
	want := []string{"primary", "secondary", "tertiary"}
	if len(callOrder) != len(want) {
		t.Fatalf("got call order %v, want %v", callOrder, want)
	}
	for i := range want {
		if callOrder[i] != want[i] {
			t.Errorf("callOrder[%d] = %q, want %q", i, callOrder[i], want[i])
		}
	}
}

func TestFallbackAllModelsFail(t *testing.T) {
	g := newTestGenkit(t)

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "secondary down")
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "secondary down") {
		t.Errorf("error %q does not contain %q", err.Error(), "secondary down")
	}
}

func TestFallbackDoesNotTriggerOnNonRetryableError(t *testing.T) {
	g := newTestGenkit(t)
	secondaryCalls := 0

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.INVALID_ARGUMENT, "bad input")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		secondaryCalls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "bad input") {
		t.Errorf("error %q does not contain %q", err.Error(), "bad input")
	}
	if secondaryCalls != 0 {
		t.Errorf("secondary called %d times, want 0 (non-retryable error)", secondaryCalls)
	}
}

// An unclassified error propagates immediately, per the v1 contract: failing
// over to a different billed model requires an explicit classification, or a
// deterministic bug in a model plugin would silently reroute every request to
// the fallback. (Retry treats the same error the opposite way, retrying it
// unconditionally; that asymmetry is also v1's.)
func TestFallbackPropagatesUnclassifiedError(t *testing.T) {
	g := newTestGenkit(t)
	secondaryCalls := 0

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, fmt.Errorf("plain error")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		secondaryCalls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err == nil {
		t.Fatal("expected the unclassified error to propagate, got nil")
	}
	if !strings.Contains(err.Error(), "plain error") {
		t.Errorf("error %q does not contain %q", err.Error(), "plain error")
	}
	if secondaryCalls != 0 {
		t.Errorf("secondary called %d times, want 0 (unclassified errors never trigger fallback)", secondaryCalls)
	}
}

// A cancelled context reports CANCELLED, which is not in the default set, so it
// must not burn through the fallback chain.
func TestFallbackDoesNotTriggerOnCancelledContext(t *testing.T) {
	g := newTestGenkit(t)
	secondaryCalls := 0

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, context.Canceled
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		secondaryCalls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	if _, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb)); err == nil {
		t.Fatal("expected error, got nil")
	}
	if secondaryCalls != 0 {
		t.Errorf("secondary called %d times, want 0 (cancelled)", secondaryCalls)
	}
}

func TestFallbackStopsOnNonRetryableFallbackError(t *testing.T) {
	g := newTestGenkit(t)
	tertiaryCalls := 0

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.INVALID_ARGUMENT, "bad request from secondary")
	})
	tertiary := defineTestModel(t, g, "test/tertiary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		tertiaryCalls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("tertiary")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil), ai.NewModelRef(tertiary.Name(), nil)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "bad request from secondary") {
		t.Errorf("error %q does not contain %q", err.Error(), "bad request from secondary")
	}
	if tertiaryCalls != 0 {
		t.Errorf("tertiary called %d times, want 0", tertiaryCalls)
	}
}

func TestFallbackCustomStatuses(t *testing.T) {
	g := newTestGenkit(t)

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.PERMISSION_DENIED, "forbidden")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	fb := &Fallback{
		Models:   []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)},
		Statuses: []core.StatusName{core.PERMISSION_DENIED},
	}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "secondary ok" {
		t.Errorf("got %q, want %q", resp.Text(), "secondary ok")
	}
}

func TestFallbackUsesRefConfig(t *testing.T) {
	g := newTestGenkit(t)
	var secondaryConfig any

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		secondaryConfig = req.Config
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	refConfig := map[string]any{"temperature": 0.5, "source": "ref"}
	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), refConfig)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithConfig(map[string]any{"temperature": 0.9, "source": "req"}), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	got, ok := secondaryConfig.(map[string]any)
	if !ok {
		t.Fatalf("secondary config type = %T, want map[string]any", secondaryConfig)
	}
	if got["source"] != "ref" {
		t.Errorf("secondary config source = %v, want %q (fallback must use ref config)", got["source"], "ref")
	}
}

func TestFallbackPassesNilConfigWhenRefHasNone(t *testing.T) {
	g := newTestGenkit(t)
	var secondaryConfig any

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		secondaryConfig = req.Config
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithConfig(map[string]any{"source": "req"}), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if secondaryConfig != nil {
		t.Errorf("secondary config = %v, want nil (ref has no config, request config must not leak through)", secondaryConfig)
	}
}

func TestFallbackDoesNotMutateOriginalRequest(t *testing.T) {
	g := newTestGenkit(t)
	var primaryConfig any

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		primaryConfig = req.Config
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	refConfig := map[string]any{"source": "ref"}
	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), refConfig)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithConfig(map[string]any{"source": "req"}), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	got, ok := primaryConfig.(map[string]any)
	if !ok {
		t.Fatalf("primary config type = %T, want map[string]any", primaryConfig)
	}
	if got["source"] != "req" {
		t.Errorf("primary config source = %v, want %q (fallback must not mutate request seen by primary)", got["source"], "req")
	}
}

func TestFallbackModelNotFound(t *testing.T) {
	g := newTestGenkit(t)

	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef("test/nonexistent", nil)}}

	_, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error %q does not contain %q", err.Error(), "not found")
	}
}

// TestFallbackThroughGenerateAction runs Fallback the way the Dev UI does:
// through the registered /util/generate action, with the middleware referenced
// by name. Guards against the action running without a Genkit on the context,
// which made the fallback lookup dereference a nil *genkit.Genkit.
func TestFallbackThroughGenerateAction(t *testing.T) {
	g := genkit.Init(context.Background(), genkit.WithPlugins(&Middleware{}))

	defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	action := genkit.LookupAction(g, "/util/generate")
	if action == nil {
		t.Fatal("generate action not registered")
	}
	input := `{
		"model": "test/primary",
		"messages": [{"role": "user", "content": [{"text": "hello"}]}],
		"use": [{"name": "genkit-middleware/fallback", "config": {"models": [{"name": "test/secondary"}]}}]
	}`
	out, err := action.RunJSON(context.Background(), json.RawMessage(input), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp ai.ModelResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "secondary ok" {
		t.Errorf("got %q, want %q", got, "secondary ok")
	}
}

// TestFallbackOnBareRegistry runs Fallback through ai.Generate on a bare
// registry rather than genkit.Generate. The loop puts a Genkit backing the
// registry on the context itself, so the fallback model still resolves.
func TestFallbackOnBareRegistry(t *testing.T) {
	r := newTestRegistry(t)
	primary := defineModel(t, r, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	defineModel(t, r, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef("test/secondary", nil)}}

	resp, err := ai.Generate(ctx, r, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "secondary ok" {
		t.Errorf("got %q, want %q", got, "secondary ok")
	}
}

// The reroute warning names the model that failed and the one tried next as
// separate attributes; it once logged only the next model under "model".
func TestFallbackLogsFailedModel(t *testing.T) {
	g := newTestGenkit(t)
	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "primary down")
	})
	secondary := defineTestModel(t, g, "test/secondary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("secondary ok")}, nil
	})

	var buf bytes.Buffer
	logCtx := logger.WithContext(ctx, slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}
	if _, err := genkit.Generate(logCtx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithUse(fb)); err != nil {
		t.Fatal(err)
	}

	var rec map[string]any
	for line := range strings.Lines(buf.String()) {
		var r map[string]any
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatal(err)
		}
		if r["msg"] == "model call failed, falling back" {
			rec = r
		}
	}
	if rec == nil {
		t.Fatalf("no fallback warning logged; got:\n%s", buf.String())
	}
	if rec["model"] != "test/primary" || rec["fallbackModel"] != "test/secondary" {
		t.Errorf("got model=%v fallbackModel=%v, want model=test/primary fallbackModel=test/secondary", rec["model"], rec["fallbackModel"])
	}
}

// A model that fails with a sticky status is skipped for the rest of the
// generate call, so a three-turn tool loop reaches the dead primary once. The
// next generate call starts over, so the state is per call and never global.
func TestFallbackSkipsDeadModelForRestOfCall(t *testing.T) {
	g := newTestGenkit(t)
	primaryCalls, secondaryCalls := 0, 0
	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		primaryCalls++
		return nil, core.NewError(core.NOT_FOUND, "model not found")
	})
	secondary := defineTestModel(t, g, "test/secondary", echoLoopModel(&secondaryCalls, 3))
	tool := genkit.DefineTool(g, "echo", "Echoes.", func(ctx *ai.ToolContext, in struct {
		V string `json:"v"`
	}) (string, error) {
		return in.V, nil
	})
	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	for run := 1; run <= 2; run++ {
		secondaryCalls = 0
		resp, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithTools(tool), ai.WithUse(fb))
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() != "done" {
			t.Errorf("run %d: got %q, want %q", run, resp.Text(), "done")
		}
		if secondaryCalls != 3 {
			t.Errorf("run %d: secondary called %d times, want 3", run, secondaryCalls)
		}
		if primaryCalls != run {
			t.Errorf("after run %d: primary called %d times, want %d", run, primaryCalls, run)
		}
	}
}

// A transient failure does not skip the model: the next turn tries the
// primary again.
func TestFallbackRetriesTransientFailureNextTurn(t *testing.T) {
	g := newTestGenkit(t)
	primaryCalls, secondaryCalls := 0, 0
	primary := defineTestModel(t, g, "test/primary", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		primaryCalls++
		if primaryCalls == 1 {
			return nil, core.NewError(core.UNAVAILABLE, "primary down")
		}
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("primary ok")}, nil
	})
	secondary := defineTestModel(t, g, "test/secondary", echoLoopModel(&secondaryCalls, 2))
	tool := genkit.DefineTool(g, "echo", "Echoes.", func(ctx *ai.ToolContext, in struct {
		V string `json:"v"`
	}) (string, error) {
		return in.V, nil
	})
	fb := &Fallback{Models: []ai.ModelRef{ai.NewModelRef(secondary.Name(), nil)}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(primary), ai.WithPrompt("hello"), ai.WithTools(tool), ai.WithUse(fb))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "primary ok" {
		t.Errorf("got %q, want %q", resp.Text(), "primary ok")
	}
	if primaryCalls != 2 || secondaryCalls != 1 {
		t.Errorf("primary called %d times and secondary %d, want 2 and 1", primaryCalls, secondaryCalls)
	}
}

// echoLoopModel returns a model that requests the echo tool on each of its
// first turns-1 calls and answers "done" after that, counting calls in calls.
func echoLoopModel(calls *int, turns int) ai.ModelFunc {
	return func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		*calls++
		if *calls >= turns {
			return &ai.ModelResponse{Message: ai.NewModelTextMessage("done")}, nil
		}
		return &ai.ModelResponse{Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: "echo", Input: map[string]any{"v": "x"}})},
		}}, nil
	}
}
