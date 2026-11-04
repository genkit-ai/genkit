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

package main

import "github.com/firebase/genkit/go/genkit"

// registerModelCases covers the two "Model runner" audit sections.
//
// Config editor: ANT-9 (adaptive thinking via JSON), ANT-10 (server tools via
// JSON), ANT-15 (string-form system), ANT-25 (no value constraints), ANT-49
// (config.version), ANT-54/GGA-34 (backend caveats), GGA-2 (timeout unit),
// GGA-50 (serviceTier description).
//
// Request/response: ANT-3 (media URL corruption), ANT-4 (raw response),
// ANT-5/ANT-8 (streaming), ANT-6 (finish reasons), GGA-1/GGA-28 (Imagen),
// GGA-7 (media truncation), GGA-23 (TTS), GGA-42 (resource parts).
//
// Most of these need only the curated models the plugins already register;
// add model refs here for uncurated IDs (GGA-5 *-tts fallback) and any
// capability-override cases.
func registerModelCases(g *genkit.Genkit) {
	// TODO: uncurated model refs, TTS, Imagen, and media-input fixtures.
	_ = g
}
