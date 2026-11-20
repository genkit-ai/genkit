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
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/meta"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "META_API_KEY")
	g := livetest.Init(t, &meta.Meta{})

	// Meta redacts reasoning_content for external callers, so the reasoning
	// cases check only that the effort is accepted.
	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model:          meta.ModelRef("muse-wonder-1.0", nil),
			ReasoningModel: meta.ModelRef("muse-wonder-1.0", &meta.ChatConfig{ReasoningEffort: meta.ReasoningEffortLow}),
			AudioModel:     meta.ModelRef("muse-wonder-1.0", nil),
			DocumentModel:  meta.ModelRef("muse-wonder-1.0", nil),
			LimitConfig:    &meta.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin:   &meta.Meta{APIKey: "invalid"},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"safety_identifier": "genkit-live-test"},
		},
	})
}
