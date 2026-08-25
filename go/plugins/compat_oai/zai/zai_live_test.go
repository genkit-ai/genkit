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

package zai_test

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/zai"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("ZAI_API_KEY") == "" {
		t.Skip("ZAI_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&zai.ZAI{}),
		genkit.WithDefaultModel("zai/glm-5.1"),
	)

	// Thinking is on by default, so the cheap checks turn it off and the
	// reasoning checks turn it back on.
	livetest.Run(t, g, livetest.Suite{
		Model: zai.ModelRef("glm-5.1", &zai.ChatConfig{
			Thinking: &zai.ThinkingConfig{Type: "disabled"},
		}),
		ReasoningModel: zai.ModelRef("glm-5.1", &zai.ChatConfig{
			Thinking: &zai.ThinkingConfig{Type: "enabled"},
		}),
		ReasoningContent: true,
		VisionModel:      zai.ModelRef("glm-4.6v-flash", nil),
		ExtraConfig: map[string]any{
			"thinking": map[string]any{"type": "disabled"},
			"extra":    map[string]any{"user_id": "genkit-livetest"},
		},
	})
}
