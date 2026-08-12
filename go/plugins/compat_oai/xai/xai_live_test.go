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

package xai_test

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("XAI_API_KEY") == "" {
		t.Skip("XAI_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&xai.XAI{}),
		genkit.WithDefaultModel("xai/grok-4.5"),
	)

	livetest.Run(t, g, livetest.Suite{
		Model: xai.ModelRef("grok-4.5", nil),
		// Grok takes the effort knob but keeps the reasoning content
		// server-side.
		ReasoningModel: xai.ModelRef("grok-4.3", &xai.ChatConfig{
			MaxOutputTokens: 512,
			ReasoningEffort: xai.ReasoningEffortLow,
		}),
		VisionModel: xai.ModelRef("grok-4.5", nil),
		ToolChoice:  true,
		ExtraConfig: map[string]any{
			"extra": map[string]any{"user": "genkit-livetest"},
		},
	})
}
