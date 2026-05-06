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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

const provider = "genkit-middleware"

// Middleware provides the built-in middleware (Retry, Fallback, ToolApproval)
// as a Genkit plugin. Register it with [genkit.WithPlugins] during [genkit.Init].
type Middleware struct{}

func (p *Middleware) Name() string { return provider }

func (p *Middleware) Init(ctx context.Context) []api.Action { return nil }

func (p *Middleware) Middlewares(ctx context.Context) ([]*ai.MiddlewareDesc, error) {
	return []*ai.MiddlewareDesc{
		ai.NewMiddleware("Retry failed model calls with exponential backoff", &Retry{}),
		ai.NewMiddleware("Try alternative models when the primary model fails", &Fallback{}),
		ai.NewMiddleware("Require explicit approval before executing tools", &ToolApproval{}),
		ai.NewMiddleware("Expose a local library of skills as loadable system instructions", &Skills{}),
		ai.NewMiddleware("Grant the model file access scoped to a directory", &Filesystem{}),
	}, nil
}
