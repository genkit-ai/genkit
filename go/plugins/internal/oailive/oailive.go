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

// Package oailive is the live checklist tier for the OpenAI-compatible
// plugins. It runs the shared [livetest] checklist and then the checks every
// plugin built on the compat_oai base owes: the extra config passthrough that
// carries provider fields the typed config does not model.
package oailive

import (
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

// Suite describes how to drive one OpenAI-compatible provider.
type Suite struct {
	livetest.Suite
	// ExtraConfig, when set, is the whole request config for the
	// passthrough check and must route at least one field through the
	// config's extra map. Plugins that speak the SDK's own request type
	// have no extra map and leave it nil.
	ExtraConfig map[string]any
}

// Run walks the plugin registered on g through the shared checklist and then
// the OpenAI-compatible one, under a "compat_oai" subtest. See [livetest.Run]
// for what it defines on g.
func Run(t *testing.T, g *genkit.Genkit, s Suite) {
	t.Helper()
	livetest.Run(t, g, s.Suite, livetest.Group{Name: "compat_oai", Cases: []livetest.Case{
		{
			Name: "extra config passthrough",
			Needs: func() string {
				if s.ExtraConfig == nil {
					return "Suite.ExtraConfig is not set"
				}
				return ""
			},
			Run: func(t *testing.T) {
				resp, err := genkit.Generate(t.Context(), g,
					ai.WithModel(s.Model),
					ai.WithConfig(s.ExtraConfig),
					ai.WithPrompt("What is the capital of France? Reply with just the city name."),
				)
				if err != nil {
					t.Fatalf("Generate() error = %v", err)
				}
				if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
					t.Errorf("Text() = %q, want it to contain %q", resp.Text(), "Paris")
				}
			},
		},
	}})
}
