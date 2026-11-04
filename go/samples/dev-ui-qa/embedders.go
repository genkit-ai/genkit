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

// registerEmbedderCases covers the embedder rows of "Discovery and action
// list" and "Model runner": cross-backend catalog leak (GGA-40), typo'd ID
// resolving as a live embedder (GGA-60), and multimodal input hitting an SDK
// refusal or process panic (GGA-32).
func registerEmbedderCases(g *genkit.Genkit) {
	// TODO: embedder refs, including a deliberately misspelled ID and a
	// multimodal input fixture.
	_ = g
}
