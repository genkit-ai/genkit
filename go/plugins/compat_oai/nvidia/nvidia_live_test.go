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

package nvidia_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/nvidia"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "NVIDIA_API_KEY")
	g := livetest.Init(t, &nvidia.NVIDIA{})

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model:            nvidia.ModelRef("nvidia/nemotron-3.5-super-vl-120b-a12b", nil),
			ReasoningModel:   nvidia.ModelRef("nvidia/nemotron-3.5-lightning-30b-a3b", nil),
			ReasoningContent: true,
			LimitConfig:      &nvidia.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin:     &nvidia.NVIDIA{APIKey: "invalid"},
			Skip: map[string]string{
				"generate/array output streaming": "NIM buffers a json_schema-constrained array and streams it in one chunk",
			},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"seed": 7},
		},
	})
}
