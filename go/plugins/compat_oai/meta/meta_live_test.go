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

package meta_test

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/meta"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("MODEL_API_KEY") == "" {
		t.Skip("MODEL_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&meta.Meta{}),
		genkit.WithDefaultModel("meta/muse-spark-1.2"),
	)

	model := meta.ModelRef("muse-spark-1.2", &meta.ChatConfig{
		ReasoningEffort: meta.ReasoningEffortMinimal,
	})
	livetest.Run(t, g, livetest.Suite{
		Model: model,
		ReasoningModel: meta.ModelRef("muse-spark-1.2", &meta.ChatConfig{
			ReasoningEffort: meta.ReasoningEffortLow,
		}),
		VisionModel: model,
		ToolChoice:  true,
		ExtraConfig: map[string]any{
			"reasoningEffort": string(meta.ReasoningEffortMinimal),
			"extra": map[string]any{
				"prompt_cache_retention": "24h",
			},
		},
	})
}
