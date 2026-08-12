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

package kimi_test

import (
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("KIMI_API_KEY") == "" && os.Getenv("MOONSHOT_API_KEY") == "" {
		t.Skip("KIMI_API_KEY and MOONSHOT_API_KEY are not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&kimi.Kimi{}),
		genkit.WithDefaultModel("kimi/kimi-k3"),
	)

	livetest.Run(t, g, livetest.Suite{
		Model:            kimi.ModelRef("kimi-k3", nil),
		ReasoningModel:   kimi.ModelRef("kimi-k2.6", nil),
		ReasoningContent: true,
		VisionModel:      kimi.ModelRef("kimi-k3", nil),
		ToolChoice:       true,
		ExtraConfig: map[string]any{
			"extra": map[string]any{"thinking": map[string]any{"type": "disabled"}},
		},
	})
}
