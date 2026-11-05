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

package exp

import (
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core/status"
)

// Ollama is a local Ollama server, which serves the decision models it has
// pulled, such as Clef, and takes no key. Its models are
// ollama-decisions/<name> by Ollama's name, such as
// ollama-decisions/clef. The name is not ollama, which the plugin for
// Ollama's chat models has, so both serve one app. The server is
// http://localhost:11434 unless BaseURL says otherwise. The Dev UI lists
// Clef, and Models adds the other models pulled.
//
// Ollama takes images as raw base64 rather than data URLs, which this
// server converts, and requires text in the state, so a request with no
// text, such as one of images alone, is refused before it is sent.
func Ollama() *SystemOne {
	return &SystemOne{
		Provider: "ollama-decisions",
		Route:    ollamaRoute,
		Images:   MediaSupported,
		Models:   map[string]ModelSpec{"clef": {}},
		preset: preset{
			label:   "Ollama",
			baseURL: "http://localhost:11434",
		},
	}
}

// ollamaRoute sends the native body with the images as raw base64.
func ollamaRoute(model string, body map[string]any) (string, any, error) {
	urls, _ := body["images"].([]string)
	if !hasText(body["state"]) {
		msg := "systemone: Ollama requires text in the state"
		if len(urls) > 0 {
			msg += ", so a request with images needs text too"
		}
		return "", nil, status.Errorf(status.ErrInvalidArgument, "%s", msg)
	}
	if len(urls) > 0 {
		images := make([]string, len(urls))
		for i, image := range urls {
			_, images[i], _ = strings.Cut(image, ";base64,")
		}
		body["images"] = images
	}
	return "", body, nil
}

// hasText reports whether a state carries any text: a nonempty string, a
// turn or document with text, or any other value, such as the object of a
// data part.
func hasText(state any) bool {
	switch v := state.(type) {
	case nil:
		return false
	case string:
		return v != ""
	case []map[string]any:
		return slices.ContainsFunc(v, func(turn map[string]any) bool { return hasText(turn["content"]) })
	case map[string]any:
		if turns, ok := v["messages"].([]map[string]any); ok {
			docs, _ := v["context"].([]any)
			return hasText(turns) || slices.ContainsFunc(docs, hasText)
		}
		return len(v) > 0
	}
	return true
}
