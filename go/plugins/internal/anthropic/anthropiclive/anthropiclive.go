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
// Garden Claude models. It runs the shared [livetest] checklist, and is where
// a gap or a check that code has whatever the plugin is recorded once for
// both plugins.
package anthropiclive

import (
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

// Run walks the plugin registered on g through the shared checklist. See
// [livetest.Run] for what it defines on g.
func Run(t *testing.T, g *genkit.Genkit, s livetest.Suite) {
	t.Helper()
	livetest.Run(t, g, s)
}
