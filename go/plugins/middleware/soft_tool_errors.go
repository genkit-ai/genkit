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
	"github.com/firebase/genkit/go/internal/base"
)

// SoftToolErrors is a middleware that returns tool errors to the model
// instead of failing the generation. The model receives the error as the
// tool's response, {"error": "<message>"}, and can correct its input and try
// again.
//
// It covers the errors a tool returns, input validation included, and calls
// to tool names that do not exist. These still stop the generation: errors
// that a WrapTool hook raises in place of the tool's own, interrupts, and
// cancellation. Other WrapTool hooks, such as a retry, see the tool's error
// before it becomes a response, wherever SoftToolErrors sits in the chain.
//
// A tool author can return a specific error to the model without this
// middleware, with tool.Fail from the ai/exp/tool package.
//
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithPrompt("what is the weather in Paris?"),
//	    ai.WithTools(weatherTool),
//	    ai.WithUse(&middleware.SoftToolErrors{}),
//	)
type SoftToolErrors struct {
	// Tools limits the middleware to these tool names. Empty covers every
	// tool, and every tool name the model asks for that does not exist.
	Tools []string `json:"tools,omitempty" jsonschema_description:"Tool names whose errors are returned to the model. Empty covers every tool, and calls to tools that do not exist."`
}

// Name implements [ai.Middleware].
func (t SoftToolErrors) Name() string { return provider + "/softToolErrors" }

// New implements [ai.Middleware], hooking each tool-loop iteration.
func (t SoftToolErrors) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{WrapGenerate: returnToolErrors(t.covers)}, nil
}

func (t SoftToolErrors) covers(name string) bool {
	return len(t.Tools) == 0 || slices.Contains(t.Tools, name)
}

// returnToolErrors returns a WrapGenerate hook that has the tool loop return
// the errors of the tools covers accepts to the model, with the rules
// [SoftToolErrors] describes.
func returnToolErrors(covers func(toolName string) bool) func(context.Context, *ai.GenerateParams, ai.GenerateNext) (*ai.ModelResponse, error) {
	return func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		// Every turn after the first runs inside the first one's context,
		// so later turns find the policy there.
		if params.Iteration > 0 {
			return next(ctx, params)
		}
		policy := &base.SoftToolErrors{Covers: covers, Outer: base.SoftToolErrorsKey.FromContext(ctx)}
		return next(base.SoftToolErrorsKey.NewContext(ctx, policy), params)
	}
}
