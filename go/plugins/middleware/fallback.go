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

	genkit "github.com/firebase/genkit/go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
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
// Models are specified as [ai.ActionRef] values (created via [ai.NewActionRef])
// and resolved via the [genkit.Genkit] instance at call time.
//
// Usage:
//
//	resp, err := g.Generate(ctx,
//	    ai.WithModel(primary),
//	    ai.WithPrompt("hello"),
//	    ai.WithUse(&middleware.Fallback{Models: []ai.ActionRef{
//	        googlegenai.ModelRef("googleai/gemini-2.5-flash", ...),
//	        googlegenai.ModelRef("vertexai/gemini-2.5-flash", ...),
//	    }}),
//	)
type Fallback struct {
	// Models is the ordered list of fallback models to try.
	// These are tried in order after the primary model fails. Each ref's
	// Config is used verbatim for that model -- the original request's
	// Config is not inherited. Use [ai.NewActionRef] to attach config.
	Models []ai.ActionRef `json:"models,omitempty"`
	// Statuses is the set of status codes that trigger a fallback. An error
	// carrying no status counts as INTERNAL for this check.
	// Defaults to [defaultFallbackStatuses].
	Statuses []status.Name `json:"statuses,omitempty"`
}

func (f *Fallback) Name() string { return provider + "/fallback" }

func (f *Fallback) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{
		WrapModel: f.wrapModel,
	}, nil
}

func (f *Fallback) statuses() []status.Name {
	if len(f.Statuses) > 0 {
		return f.Statuses
	}
	return defaultFallbackStatuses
}

func (f *Fallback) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	resp, err := next(ctx, params)
	if err == nil {
		return resp, nil
	}

	if !isFallbackRetryable(err, f.statuses()) {
		return nil, err
	}

	lastErr := err
	for _, ref := range f.Models {
		name := ref.Name()
		m := genkit.FromContext(ctx).LookupModel(name)
		if m == nil {
			return nil, status.Errorf(ai.ErrModelNotFound, "fallback: model %q not found", name)
		}
		req := *params.Request
		req.Config = ref.Config()
		resp, err := m.Generate(ctx, &req, params.Callback)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !isFallbackRetryable(err, f.statuses()) {
			return nil, err
		}
	}
	return nil, lastErr
}

// isFallbackRetryable reports whether err should trigger trying the next model:
// its status must be in the provided list. An error carrying no status counts
// as INTERNAL, which the default list includes, so an unclassified provider
// failure still falls through to the next model.
func isFallbackRetryable(err error, statuses []status.Name) bool {
	return slices.Contains(statuses, status.Of(err))
}
