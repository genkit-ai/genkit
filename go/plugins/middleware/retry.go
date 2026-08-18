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
	"math"
	"math/rand"
	"slices"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
)

// defaultRetryStatuses are the status codes that trigger a retry by default.
var defaultRetryStatuses = []status.Name{
	status.Unavailable,
	status.DeadlineExceeded,
	status.ResourceExhausted,
	status.Aborted,
	status.Internal,
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
// A classified error is retried when its status is in Statuses, which defaults
// to UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, ABORTED, and INTERNAL.
// An unclassified error (no [status.Error] or sentinel in its chain) is always
// retried, regardless of Statuses. A cancelled context reports CANCELLED and is
// not retried.
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
	MaxRetries int `json:"maxRetries,omitempty" jsonschema_description:"Maximum number of retry attempts. Defaults to 3."`
	// Statuses is the set of status codes that trigger a retry for classified
	// errors; unclassified errors are always retried regardless of this list.
	// Defaults to [defaultRetryStatuses].
	Statuses []status.Name `json:"statuses,omitempty" jsonschema_description:"Status codes that trigger a retry for classified errors. Unclassified errors are always retried regardless of this list. Defaults to UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, ABORTED and INTERNAL." jsonschema:"enum=OK,enum=CANCELLED,enum=UNKNOWN,enum=INVALID_ARGUMENT,enum=DEADLINE_EXCEEDED,enum=NOT_FOUND,enum=ALREADY_EXISTS,enum=PERMISSION_DENIED,enum=UNAUTHENTICATED,enum=RESOURCE_EXHAUSTED,enum=FAILED_PRECONDITION,enum=ABORTED,enum=OUT_OF_RANGE,enum=UNIMPLEMENTED,enum=INTERNAL,enum=UNAVAILABLE,enum=DATA_LOSS"`
	// InitialDelayMs is the delay before the first retry, in milliseconds. Defaults to 1000.
	InitialDelayMs int `json:"initialDelayMs,omitempty" jsonschema_description:"Delay before the first retry, in milliseconds. Defaults to 1000."`
	// MaxDelayMs is the upper bound on retry delay, in milliseconds. Defaults to 60000.
	MaxDelayMs int `json:"maxDelayMs,omitempty" jsonschema_description:"Upper bound on the retry delay, in milliseconds. Defaults to 60000."`
	// BackoffFactor is the multiplier applied to the delay after each retry. Defaults to 2.
	BackoffFactor float64 `json:"backoffFactor,omitempty" jsonschema_description:"Multiplier applied to the delay after each retry. Defaults to 2."`
	// NoJitter disables random jitter on the delay. Jitter helps prevent
	// thundering-herd problems when many clients retry simultaneously.
	NoJitter bool `json:"noJitter,omitempty" jsonschema_description:"Disables random jitter on the delay. Jitter helps prevent thundering-herd problems when many clients retry at the same time. Defaults to false."`
}

// Name implements [ai.Middleware].
func (r Retry) Name() string { return provider + "/retry" }

// New implements [ai.Middleware], hooking the model stage.
func (r Retry) New(ctx context.Context) (*ai.Hooks, error) {
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

func (r *Retry) statuses() []status.Name {
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

		logger.Debug(ctx, "model call failed, retrying",
			"attempt", attempt+1,
			"maxRetries", maxRetries,
			"delay", delay.Round(time.Millisecond),
			"error", err)

		// Bail out if the caller disconnected mid-backoff; no reason to wait
		// out the delay (or issue another retry) for a caller who has left.
		if err := sleepFunc(ctx, delay); err != nil {
			return nil, lastErr
		}

		currentDelay = min(time.Duration(float64(currentDelay)*r.backoffFactor()), r.maxDelay())
	}
	return nil, lastErr
}

// isRetryable reports whether err should trigger a retry: a classified error's
// status must be in statuses, and an unclassified error is always retryable,
// preserving the v1 contract that non-GenkitError errors are retried
// regardless of the Statuses setting.
func isRetryable(err error, statuses []status.Name) bool {
	if s, ok := status.Classified(err); ok {
		return slices.Contains(statuses, s)
	}
	return true
}
