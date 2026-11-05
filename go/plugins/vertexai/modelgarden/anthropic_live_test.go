// Copyright 2025 Google LLC
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

package modelgarden_test

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal/anthropic/anthropiclive"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

// vertexEnv gates a Model Garden live test on the project and location the
// plugin reads. Authentication is ambient (Application Default Credentials).
func vertexEnv(t *testing.T) {
	t.Helper()
	livetest.Env(t, "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT")
	livetest.Env(t, "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION")
}

func TestAnthropicLive(t *testing.T) {
	vertexEnv(t)
	g := livetest.Init(t, &modelgarden.Anthropic{})

	const model = "vertexai/claude-haiku-4-5"
	anthropiclive.Run(t, g, livetest.Suite{
		Model: ai.NewModelRef(model, nil),
		ReasoningModel: ai.NewModelRef(model, &anthropic.MessageNewParams{
			MaxTokens: 4096,
			Thinking: anthropic.ThinkingConfigParamUnion{
				OfEnabled: &anthropic.ThinkingConfigEnabledParam{BudgetTokens: 2048},
			},
		}),
		ReasoningContent:  true,
		LimitConfig:       &anthropic.MessageNewParams{MaxTokens: 16},
		ToolResponseMedia: true,
	})
}
