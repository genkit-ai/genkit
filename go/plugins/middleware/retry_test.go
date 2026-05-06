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
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/registry"
)

func newTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	r := registry.New()
	ai.ConfigureFormats(r)
	return r
}

func defineModel(t *testing.T, r *registry.Registry, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return ai.DefineModel(r, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	}, fn)
}

func init() {
	// Disable real sleeping in tests.
	sleepFunc = func(context.Context, time.Duration) error { return nil }
}

func TestRetrySucceedsOnFirstAttempt(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/ok", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("ok")}, nil
	})

	retry := &Retry{}
	ai.DefineMiddleware(r, "retry", retry)

	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "ok" {
		t.Errorf("got %q, want %q", resp.Text(), "ok")
	}
	if calls != 1 {
		t.Errorf("got %d calls, want 1", calls)
	}
}

func TestRetryRecoversAfterTransientError(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/transient", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		if calls <= 2 {
			return nil, core.NewError(core.UNAVAILABLE, "service down")
		}
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("recovered")}, nil
	})

	retry := &Retry{}
	ai.DefineMiddleware(r, "retry", retry)

	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "recovered" {
		t.Errorf("got %q, want %q", resp.Text(), "recovered")
	}
	if calls != 3 {
		t.Errorf("got %d calls, want 3", calls)
	}
}

func TestRetryExhaustsMaxRetries(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/alwaysfail", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		return nil, core.NewError(core.UNAVAILABLE, "always failing")
	})

	retry := &Retry{MaxRetries: 2}
	ai.DefineMiddleware(r, "retry", retry)

	_, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// 1 initial + 2 retries = 3 calls
	if calls != 3 {
		t.Errorf("got %d calls, want 3", calls)
	}
}

func TestRetryDoesNotRetryNonMatchingGenkitError(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/badarg", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		return nil, core.NewError(core.INVALID_ARGUMENT, "bad input")
	})

	retry := &Retry{}
	ai.DefineMiddleware(r, "retry", retry)

	_, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if calls != 1 {
		t.Errorf("got %d calls, want 1 (no retries for INVALID_ARGUMENT)", calls)
	}
}

func TestRetryRetriesNonGenkitErrors(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/plainerr", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return nil, fmt.Errorf("network timeout")
		}
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("ok")}, nil
	})

	retry := &Retry{}
	ai.DefineMiddleware(r, "retry", retry)

	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "ok" {
		t.Errorf("got %q, want %q", resp.Text(), "ok")
	}
	if calls != 2 {
		t.Errorf("got %d calls, want 2", calls)
	}
}

func TestRetryCustomStatuses(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/forbidden", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		return nil, core.NewError(core.PERMISSION_DENIED, "forbidden")
	})

	retry := &Retry{
		Statuses:   []core.StatusName{core.PERMISSION_DENIED},
		MaxRetries: 1,
	}
	ai.DefineMiddleware(r, "retry", retry)

	_, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// 1 initial + 1 retry = 2 calls
	if calls != 2 {
		t.Errorf("got %d calls, want 2", calls)
	}
}

func TestRetryBackoffDelays(t *testing.T) {
	r := newTestRegistry(t)
	m := defineModel(t, r, "test/delays", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "down")
	})

	var delays []time.Duration
	origSleep := sleepFunc
	sleepFunc = func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }
	defer func() { sleepFunc = origSleep }()

	retry := &Retry{
		MaxRetries:     3,
		InitialDelayMs: 100,
		BackoffFactor:  2,
		NoJitter:       true,
	}
	ai.DefineMiddleware(r, "retry", retry)

	_, _ = ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))

	if len(delays) != 3 {
		t.Fatalf("got %d delays, want 3", len(delays))
	}
	// With no jitter and factor 2: 100ms, 200ms, 400ms
	want := []time.Duration{100 * time.Millisecond, 200 * time.Millisecond, 400 * time.Millisecond}
	for i, got := range delays {
		if got != want[i] {
			t.Errorf("delay[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestRetryMaxDelayClamp(t *testing.T) {
	r := newTestRegistry(t)
	m := defineModel(t, r, "test/clamp", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, core.NewError(core.UNAVAILABLE, "down")
	})

	var delays []time.Duration
	origSleep := sleepFunc
	sleepFunc = func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }
	defer func() { sleepFunc = origSleep }()

	retry := &Retry{
		MaxRetries:     3,
		InitialDelayMs: 500,
		MaxDelayMs:     600,
		BackoffFactor:  2,
		NoJitter:       true,
	}
	ai.DefineMiddleware(r, "retry", retry)

	_, _ = ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))

	if len(delays) != 3 {
		t.Fatalf("got %d delays, want 3", len(delays))
	}
	// 500ms, then 1000ms clamped to 600ms, then still 600ms (clamped again)
	want := []time.Duration{500 * time.Millisecond, 600 * time.Millisecond, 600 * time.Millisecond}
	for i, got := range delays {
		if got != want[i] {
			t.Errorf("delay[%d] = %v, want %v", i, got, want[i])
		}
	}
}

// If the caller cancels while we are sleeping between attempts, the retry
// loop should bail out immediately rather than running more attempts.
func TestRetryStopsWhenContextCanceledDuringBackoff(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/ctxcancel", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		return nil, core.NewError(core.UNAVAILABLE, "down")
	})

	reqCtx, cancel := context.WithCancel(context.Background())
	origSleep := sleepFunc
	sleepFunc = func(c context.Context, _ time.Duration) error {
		cancel() // simulate caller disconnecting mid-backoff
		return c.Err()
	}
	defer func() { sleepFunc = origSleep }()

	retry := &Retry{MaxRetries: 5}
	ai.DefineMiddleware(r, "retry", retry)

	_, err := ai.Generate(reqCtx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(retry))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Only the initial attempt should have run; the backoff was aborted.
	if calls != 1 {
		t.Errorf("got %d calls, want 1 (aborted after first attempt)", calls)
	}
}

var ctx = context.Background()
