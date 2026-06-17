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
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

func newTestGenkit(t *testing.T) *genkit.Genkit {
	t.Helper()
	return genkit.Init(context.Background())
}

func defineTestModel(t *testing.T, g *genkit.Genkit, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return genkit.DefineModel(g, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
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

func TestFallbackDoesNotTriggerOnNonGenkitError(t *testing.T) {
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
		t.Fatal("expected error, got nil")
	}
	if secondaryCalls != 0 {
		t.Errorf("secondary called %d times, want 0 (non-GenkitError)", secondaryCalls)
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
