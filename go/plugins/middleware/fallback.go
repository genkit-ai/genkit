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
	"slices"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
)

// defaultFallbackStatuses are the status codes that trigger a fallback by default.
var defaultFallbackStatuses = []status.Name{
	status.Unavailable,
	status.DeadlineExceeded,
	status.ResourceExhausted,
	status.Aborted,
	status.Internal,
	status.NotFound,
	status.Unimplemented,
}

// Fallback is a middleware that tries alternative models when the primary model
// fails with a retryable error status.
//
// It only hooks the Model stage -- when a model API call fails with a matching
// status, the request is forwarded to the next model in the list.
//
// Models are specified as [ai.ModelRef] values (created via [ai.NewModelRef])
// and resolved via the [genkit.Genkit] instance at call time.
//
// A model that fails with a status that does not clear up between calls
// (NOT_FOUND, PERMISSION_DENIED, UNAUTHENTICATED, INVALID_ARGUMENT or
// UNIMPLEMENTED) is skipped for the rest of the generate call, so each turn of
// a tool loop starts at the next model instead of repeating the failure. The
// next generate call tries every model again.
//
// Provider SDKs can retry on their own before an error reaches this
// middleware (anthropic-sdk-go retries twice by default). Those retries
// multiply with [Retry] and delay the fallback; disable one layer, for example
// with the plugin's SDK options, when combining them.
//
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(primary),
//	    ai.WithPrompt("hello"),
//	    ai.WithUse(&middleware.Fallback{Models: []ai.ModelRef{
//	        googlegenai.ModelRef("googleai/gemini-flash-latest", ...),
//	        googlegenai.ModelRef("vertexai/gemini-flash-latest", ...),
//	    }}),
//	)
type Fallback struct {
	// Models is the ordered list of fallback models to try.
	// These are tried in order after the primary model fails. Each ref's
	// Config is used verbatim for that model -- the original request's
	// Config is not inherited. Use [ai.NewModelRef] to attach config.
	Models []ai.ModelRef `json:"models,omitempty" jsonschema_description:"Ordered list of fallback models to try after the primary model fails. Each ref's config is used verbatim for that model, and the original request's config is not inherited."`
	// Statuses is the set of status codes that trigger a fallback for
	// classified errors; unclassified errors propagate immediately and never
	// trigger one. Defaults to [defaultFallbackStatuses].
	Statuses []status.Name `json:"statuses,omitempty" jsonschema_description:"Status codes that trigger a fallback for classified errors. Unclassified errors propagate immediately and never trigger one. Defaults to UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, ABORTED, INTERNAL, NOT_FOUND and UNIMPLEMENTED." jsonschema:"enum=OK,enum=CANCELLED,enum=UNKNOWN,enum=INVALID_ARGUMENT,enum=DEADLINE_EXCEEDED,enum=NOT_FOUND,enum=ALREADY_EXISTS,enum=PERMISSION_DENIED,enum=UNAUTHENTICATED,enum=RESOURCE_EXHAUSTED,enum=FAILED_PRECONDITION,enum=ABORTED,enum=OUT_OF_RANGE,enum=UNIMPLEMENTED,enum=INTERNAL,enum=UNAVAILABLE,enum=DATA_LOSS"`
}

// Name implements [ai.Middleware].
func (f Fallback) Name() string { return provider + "/fallback" }

// New implements [ai.Middleware], hooking the model stage. It also hooks the
// generate stage to learn the primary model's name, which the model stage
// does not carry.
func (f Fallback) New(ctx context.Context) (*ai.Hooks, error) {
	run := &fallbackRun{f: &f, failed: map[string]error{}}
	return &ai.Hooks{
		WrapGenerate: run.wrapGenerate,
		WrapModel:    run.wrapModel,
	}, nil
}

func (f *Fallback) statuses() []status.Name {
	if len(f.Statuses) > 0 {
		return f.Statuses
	}
	return defaultFallbackStatuses
}

// stickyFallbackStatuses are the statuses that describe the model, its
// credentials or the request rather than the provider's load, so a later turn
// of the same generate call would fail the same way.
var stickyFallbackStatuses = []status.Name{
	status.NotFound,
	status.PermissionDenied,
	status.Unauthenticated,
	status.InvalidArgument,
	status.Unimplemented,
}

// fallbackRun is the state of [Fallback] for one generate call. The tool loop
// calls the model one turn at a time, so it needs no synchronization.
type fallbackRun struct {
	f *Fallback
	// primary is the name of the model the generate call targets.
	primary string
	// failed maps the name of each model skipped for the rest of the call to
	// the error that put it there.
	failed map[string]error
}

func (r *fallbackRun) wrapGenerate(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
	r.primary = params.Options.Model
	return next(ctx, params)
}

func (r *fallbackRun) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	statuses := r.f.statuses()
	// failedModel names the last model called in this turn that failed, and
	// lastErr holds its error, or the first skipped model's error when no
	// call has failed yet. A skip leaves failedModel empty, so the reroute
	// warning fires only on the turn that recorded the failure.
	var failedModel string
	var lastErr error
	if err, ok := r.failed[r.primary]; ok {
		logger.Debug(ctx, "skipping model that failed earlier in this generate call", "model", r.primary, "error", err)
		lastErr = err
	} else {
		resp, err := next(ctx, params)
		if err == nil {
			return resp, nil
		}
		if !isFallbackRetryable(err, statuses) {
			return nil, err
		}
		r.recordFailure(r.primary, err)
		failedModel, lastErr = r.primary, err
	}

	g := genkit.FromContext(ctx)
	if g == nil && len(r.f.Models) > 0 {
		return nil, status.Errorf(status.ErrFailedPrecondition, "fallback: no Genkit instance on the context to resolve fallback models (primary model error: %w)", lastErr)
	}
	for _, ref := range r.f.Models {
		name := ref.Name()
		if err, ok := r.failed[name]; ok {
			logger.Debug(ctx, "skipping model that failed earlier in this generate call", "model", name, "error", err)
			if lastErr == nil {
				lastErr = err
			}
			continue
		}
		if failedModel != "" {
			// A fallback reroutes the request to a different (billed) model,
			// so it warrants more than debug visibility.
			logger.Warn(ctx, "model call failed, falling back", "model", failedModel, "fallbackModel", name, "error", lastErr)
		}
		m := genkit.LookupModel(g, name)
		if m == nil {
			return nil, status.Errorf(ai.ErrModelNotFound, "fallback: model %q not found", name)
		}
		req := *params.Request
		req.Config = ref.Config()
		resp, err := m.Generate(ctx, &req, params.Callback)
		if err == nil {
			return resp, nil
		}
		if !isFallbackRetryable(err, statuses) {
			return nil, err
		}
		r.recordFailure(name, err)
		failedModel, lastErr = name, err
	}
	return nil, lastErr
}

// recordFailure marks the model to be skipped for the rest of the generate
// call when err has a sticky status.
func (r *fallbackRun) recordFailure(name string, err error) {
	if s, ok := status.Classified(err); ok && slices.Contains(stickyFallbackStatuses, s) {
		r.failed[name] = err
	}
}

// isFallbackRetryable reports whether err should trigger trying the next model:
// a classified error's status must be in statuses, and an unclassified error
// propagates immediately, preserving the v1 contract. Failing over to a
// different billed model is a bigger action than retrying the same one, so it
// requires an explicit classification; without this, a deterministic bug in a
// model plugin would silently reroute every request to the fallback.
func isFallbackRetryable(err error, statuses []status.Name) bool {
	if s, ok := status.Classified(err); ok {
		return slices.Contains(statuses, s)
	}
	return false
}
