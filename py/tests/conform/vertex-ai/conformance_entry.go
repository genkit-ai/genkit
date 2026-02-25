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
//
// SPDX-License-Identifier: Apache-2.0

// Minimal entry point for vertex-ai model conformance testing (Go).
//
// Note: The Go plugin directory is "vertexai" (no hyphen) and registers
// models under the "vertexai/" prefix. The Python plugin uses "vertex-ai"
// and tests Model Garden models under the "modelgarden/" prefix.
//
// Usage:
//
//	genkit dev:test-model --from-file model-conformance.yaml -- go run conformance_entry.go
//
// Env:
//
//	GOOGLE_CLOUD_PROJECT: Required. GCP project ID.
package main

import (
	"context"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai"
)

func main() {
	g, err := genkit.Init(context.Background(), genkit.WithPlugins(&vertexai.VertexAI{}))
	if err != nil {
		panic(err)
	}
	_ = g
	select {} // Block forever.
}
