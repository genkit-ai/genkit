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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func TestMistralLive(t *testing.T) {
	vertexEnv(t)
	g := livetest.Init(t, &modelgarden.Mistral{})

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: ai.NewModelRef("vertexai/mistral-small-2503", nil),
		},
	})

	for name, id := range map[string]string{
		// The publisher prefix is trimmed, so both forms find the model.
		"publisher-prefixed id": "mistralai/mistral-small-2503",
		"codestral":             "codestral-2",
	} {
		t.Run(name, func(t *testing.T) {
			m := modelgarden.MistralModel(g, id)
			if m == nil {
				t.Fatalf("MistralModel(%q) = nil, want the registered model", id)
			}
			resp, err := genkit.Generate(t.Context(), g,
				ai.WithModel(m),
				ai.WithPrompt("Reply with the single word: ok."))
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if strings.TrimSpace(resp.Text()) == "" {
				t.Error("Text() is empty")
			}
		})
	}
}
