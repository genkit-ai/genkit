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

package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// APIVersionStable and APIVersionBeta select the Anthropic Messages API surface.
const (
	APIVersionStable = "stable"
	APIVersionBeta   = "beta"
)

// serverToolUseToPart mirrors the JS serverToolUseBlockToPart /
// betaServerToolUseBlockToPart converters: a text part plus
// metadata.anthropicServerToolUse.
func serverToolUseToPart(id, name string, input any) *ai.Part {
	if name == "" {
		name = "unknown_tool"
	}
	inputJSON, err := json.Marshal(input)
	if err != nil {
		inputJSON = []byte(fmt.Sprintf("%v", input))
	}
	p := ai.NewTextPart(fmt.Sprintf("[Anthropic server tool %s] input: %s", name, string(inputJSON)))
	p.Metadata = map[string]any{
		"anthropicServerToolUse": map[string]any{
			"id":    id,
			"name":  name,
			"input": input,
		},
	}
	return p
}

// webSearchToolResultToPart mirrors the JS webSearchToolResultBlockToPart
// converter: a text part plus metadata.anthropicServerToolResult.
func webSearchToolResultToPart(toolUseID string, content any) *ai.Part {
	contentJSON, err := json.Marshal(content)
	if err != nil {
		contentJSON = []byte(fmt.Sprintf("%v", content))
	}
	p := ai.NewTextPart(fmt.Sprintf(
		"[Anthropic server tool result %s] %s",
		toolUseID,
		string(contentJSON),
	))
	p.Metadata = map[string]any{
		"anthropicServerToolResult": map[string]any{
			"type":      "web_search_tool_result",
			"toolUseId": toolUseID,
			"content":   content,
		},
	}
	return p
}

// parseJSONAny unmarshals raw JSON into a generic value for metadata
// passthrough. Empty input yields nil.
func parseJSONAny(raw string) any {
	if raw == "" {
		return nil
	}
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return raw
	}
	return v
}

func unsupportedServerToolError(blockType string) error {
	return status.Errorf(
		ai.ErrUnsupportedByModel,
		"unsupported Anthropic server tool block %q; only server_tool_use and web_search_tool_result are supported",
		blockType,
	)
}
