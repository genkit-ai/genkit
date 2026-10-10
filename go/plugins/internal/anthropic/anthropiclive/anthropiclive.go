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

// Package anthropiclive is the live checklist tier for the plugins built on
// the shared Anthropic Messages API code: the anthropic plugin and the Model
// Garden Claude models. It runs the shared [livetest] checklist with the gaps
// that code has whatever the plugin, so both plugins record them once.
package anthropiclive

import (
	"maps"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

// familyGaps are the cases the shared Anthropic code cannot pass yet,
// whatever the plugin.
var familyGaps = map[string]string{
	// The code constrains only the json format. The models claim
	// constrained output, so the framework leaves the format instructions
	// out, and an array or enum request reaches the API with neither a
	// constraint nor instructions.
	"generate/array output":               "no output_format is sent for the array format",
	"generate/array output streaming":     "no output_format is sent for the array format",
	"generate/enum output":                "no output_format is sent for the enum format",
	"generate/enum output is constrained": "no output_format is sent for the enum format",
}

// Run walks the plugin registered on g through the shared checklist. See
// [livetest.Run] for what it defines on g.
func Run(t *testing.T, g *genkit.Genkit, s livetest.Suite) {
	t.Helper()
	skip := maps.Clone(familyGaps)
	maps.Copy(skip, s.Skip)
	s.Skip = skip
	livetest.Run(t, g, s)
}
