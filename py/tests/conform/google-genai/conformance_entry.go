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

// Minimal entry point for google-genai model conformance testing (Go).
//
// Usage:
//
//	genkit dev:test-model --from-file model-conformance.yaml -- go run conformance_entry.go
//
// Env:
//
//	GEMINI_API_KEY: Required. Google AI API key.
package main

import (
	"context"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	g, err := genkit.Init(context.Background(), genkit.WithPlugins(&googlegenai.GoogleAI{}))
	if err != nil {
		panic(err)
	}
	_ = g
	select {} // Block forever.
}
