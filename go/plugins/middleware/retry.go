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

// Package middleware provides reusable middleware for Genkit model generation,
// including retry with exponential backoff and model fallback.
package middleware

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"slices"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

// defaultRetryStatuses are the status codes that trigger a retry by default.
var defaultRetryStatuses = []core.StatusName{
	core.UNAVAILABLE,
	core.DEADLINE_EXCEEDED,
	core.RESOURCE_EXHAUSTED,
	core.ABORTED,
	core.INTERNAL,
}

// sleepFunc is the function used for delays. It blocks for d or until ctx is
// canceled, returning ctx.Err() in the latter case. Overridable for testing.
var sleepFunc = func(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Retry is a middleware that retries failed model calls with exponential backoff.
//
// It only hooks the Model stage — individual model API calls are retried,
// not the entire generate loop.
//
// By default, retries occur for non-[core.GenkitError] errors (e.g. network failures)
// and for [core.GenkitError] errors whose status is one of UNAVAILABLE, DEADLINE_EXCEEDED,
// RESOURCE_EXHAUSTED, ABORTED, or INTERNAL_SERVER_ERROR.
//
// Usage:
//
//	resp, err := ai.Generate(ctx, r,
//	    ai.WithModel(m),
//	    ai.WithPrompt("hello"),
//	    ai.WithUse(&middleware.Retry{MaxRetries: 3}),
//	)
type Retry struct {
	// MaxRetries is the maximum number of retry attempts. Defaults to 3.
	MaxRetries int `json:"maxRetries,omitempty"`
	// Statuses is the set of status codes that trigger a retry for [core.GenkitError] errors.
	// Non-GenkitError errors are always retried regardless of this setting.
	// Defaults to [defaultRetryStatuses].
	Statuses []core.StatusName `json:"statuses,omitempty"`
	// InitialDelayMs is the delay before the first retry, in milliseconds. Defaults to 1000.
	InitialDelayMs int `json:"initialDelayMs,omitempty"`
	// MaxDelayMs is the upper bound on retry delay, in milliseconds. Defaults to 60000.
	MaxDelayMs int `json:"maxDelayMs,omitempty"`
	// BackoffFactor is the multiplier applied to the delay after each retry. Defaults to 2.
	BackoffFactor float64 `json:"backoffFactor,omitempty"`
	// NoJitter disables random jitter on the delay. Jitter helps prevent
	// thundering-herd problems when many clients retry simultaneously.
	NoJitter bool `json:"noJitter,omitempty"`
}

func (r *Retry) Name() string { return provider + "/retry" }

func (r *Retry) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{
		WrapModel: r.wrapModel,
	}, nil
}

func (r *Retry) maxRetries() int {
	if r.MaxRetries > 0 {
		return r.MaxRetries
	}
	return 3
}

func (r *Retry) statuses() []core.StatusName {
	if len(r.Statuses) > 0 {
		return r.Statuses
	}
	return defaultRetryStatuses
}

func (r *Retry) initialDelay() time.Duration {
	if r.InitialDelayMs > 0 {
		return time.Duration(r.InitialDelayMs) * time.Millisecond
	}
	return time.Second
}

func (r *Retry) maxDelay() time.Duration {
	if r.MaxDelayMs > 0 {
		return time.Duration(r.MaxDelayMs) * time.Millisecond
	}
	return 60 * time.Second
}

func (r *Retry) backoffFactor() float64 {
	if r.BackoffFactor > 0 {
		return r.BackoffFactor
	}
	return 2
}

func (r *Retry) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	maxRetries := r.maxRetries()
	statuses := r.statuses()
	currentDelay := r.initialDelay()

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err := next(ctx, params)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		if attempt == maxRetries {
			break
		}

		if !isRetryable(err, statuses) {
			return nil, err
		}

		delay := currentDelay
		if !r.NoJitter {
			jitter := time.Duration(float64(time.Second) * math.Pow(2, float64(attempt)) * rand.Float64())
			delay += jitter
		}

		// Bail out if the caller disconnected mid-backoff; no reason to wait
		// out the delay (or issue another retry) for a caller who has left.
		if err := sleepFunc(ctx, delay); err != nil {
			return nil, lastErr
		}

		currentDelay = min(time.Duration(float64(currentDelay)*r.backoffFactor()), r.maxDelay())
	}
	return nil, lastErr
}

// isRetryable reports whether err should trigger a retry.
// Non-GenkitError errors are always retried. GenkitErrors are retried
// only if their status is in the provided list.
func isRetryable(err error, statuses []core.StatusName) bool {
	var ge *core.GenkitError
	if !errors.As(err, &ge) {
		return true // unknown errors are retryable
	}
	return slices.Contains(statuses, ge.Status)
}
