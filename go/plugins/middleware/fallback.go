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
	"errors"
	"slices"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// defaultFallbackStatuses are the status codes that trigger a fallback by default.
var defaultFallbackStatuses = []core.StatusName{
	core.UNAVAILABLE,
	core.DEADLINE_EXCEEDED,
	core.RESOURCE_EXHAUSTED,
	core.ABORTED,
	core.INTERNAL,
	core.NOT_FOUND,
	core.UNIMPLEMENTED,
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
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(primary),
//	    ai.WithPrompt("hello"),
//	    ai.WithUse(&middleware.Fallback{Models: []ai.ModelRef{
//	        googlegenai.ModelRef("googleai/gemini-2.5-flash", ...),
//	        googlegenai.ModelRef("vertexai/gemini-2.5-flash", ...),
//	    }}),
//	)
type Fallback struct {
	// Models is the ordered list of fallback models to try.
	// These are tried in order after the primary model fails. Each ref's
	// Config is used verbatim for that model -- the original request's
	// Config is not inherited. Use [ai.NewModelRef] to attach config.
	Models []ai.ModelRef `json:"models,omitempty"`
	// Statuses is the set of status codes that trigger a fallback.
	// Only [core.GenkitError] errors with a matching status will trigger fallback;
	// non-GenkitError errors propagate immediately.
	// Defaults to [defaultFallbackStatuses].
	Statuses []core.StatusName `json:"statuses,omitempty"`
}

func (f *Fallback) Name() string { return provider + "/fallback" }

func (f *Fallback) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{
		WrapModel: f.wrapModel,
	}, nil
}

func (f *Fallback) statuses() []core.StatusName {
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
		m := genkit.LookupModel(genkit.FromContext(ctx), name)
		if m == nil {
			return nil, core.NewError(core.NOT_FOUND, "fallback: model %q not found", name)
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

// isFallbackRetryable reports whether err should trigger trying the next model.
// Only GenkitErrors with a matching status trigger fallback.
func isFallbackRetryable(err error, statuses []core.StatusName) bool {
	var ge *core.GenkitError
	if !errors.As(err, &ge) {
		return false
	}
	return slices.Contains(statuses, ge.Status)
}
