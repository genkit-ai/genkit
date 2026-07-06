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

// Package exp provides experimental middleware for the agent APIs in
// [github.com/firebase/genkit/go/ai/exp]: [Agents] for sub-agent delegation and
// [Artifacts] for session artifact access. These middlewares are experimental
// and may change in any minor release, tracking the agent APIs they build on.
package exp

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

// provider names the experimental middleware plugin and prefixes the
// registered middleware names (e.g. genkit-middleware-exp/agents).
const provider = "genkit-middleware-exp"

// Middleware provides the experimental agent middleware ([Agents], [Artifacts])
// as a Genkit plugin. Register it with [genkit.WithPlugins] during
// [genkit.Init] to make them resolvable by name (e.g. for the Dev UI). Using
// them directly via [ai.WithUse] does not require the plugin.
type Middleware struct{}

func (p *Middleware) Name() string { return provider }

func (p *Middleware) Init(ctx context.Context) []api.Action { return nil }

func (p *Middleware) Middlewares(ctx context.Context) ([]*ai.MiddlewareDesc, error) {
	return []*ai.MiddlewareDesc{
		ai.NewMiddleware("Delegate tasks to registered sub-agents via per-agent tools", &Agents{}),
		ai.NewMiddleware("Provide read/write tools for session artifacts", &Artifacts{}),
	}, nil
}
