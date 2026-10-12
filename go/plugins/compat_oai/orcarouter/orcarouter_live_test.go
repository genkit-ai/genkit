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

package orcarouter_test

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/orcarouter"
)

// The models the live checks spend on. They are ordinary catalog entries
// rather than anything the plugin knows about, so swap in whatever the key
// has credit for; the plugin resolves any ID the gateway serves.
const (
	chatModel      = "deepseek/deepseek-v4-flash"
	visionModel    = "anthropic/claude-haiku-4.5"
	reasoningModel = "deepseek/deepseek-v4-flash"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("ORCAROUTER_API_KEY") == "" {
		t.Skip("ORCAROUTER_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&orcarouter.OrcaRouter{}),
		genkit.WithDefaultModel("orcarouter/"+chatModel),
	)

	livetest.Run(t, g, livetest.Suite{
		Model: orcarouter.ModelRef(chatModel, nil),
		ReasoningModel: orcarouter.ModelRef(reasoningModel, &orcarouter.ChatConfig{
			MaxOutputTokens: 1024,
			ReasoningEffort: orcarouter.ReasoningEffortLow,
		}),
		ReasoningContent: false,
		VisionModel:      orcarouter.ModelRef(visionModel, nil),
		// Forced tool choice (tool_choice_none and tool_choice_required) is a
		// per-model property the gateway does not enforce for the chat model
		// this suite spends on; the tool-calling checks above already cover the
		// model's real tool support. Leave the flag off to keep the suite green
		// on any model swapped in for credit reasons.
		ExtraConfig: map[string]any{
			"extra": map[string]any{"user": "genkit-livetest"},
		},
	})
}
