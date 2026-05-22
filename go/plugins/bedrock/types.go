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

package bedrock

import (
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

// Config is the per-call configuration for Bedrock Converse models. Pass it
// via [ai.WithConfig].
type Config struct {
	// MaxTokens is the upper bound on the generated response length. Bedrock
	// rejects requests with MaxTokens <= 0 for most models; if unset, a
	// sensible per-family default is used by the plugin (4096).
	MaxTokens int `json:"maxTokens,omitempty"`

	// Temperature controls sampling randomness. Range [0, 1] for most models;
	// nil leaves it to the model's default.
	Temperature *float32 `json:"temperature,omitempty"`

	// TopP is nucleus-sampling cutoff. nil leaves it to the model default.
	TopP *float32 `json:"topP,omitempty"`

	// StopSequences are strings that, when generated, halt generation.
	StopSequences []string `json:"stopSequences,omitempty"`

	// ToolChoice selects how the model should pick tools:
	//   - "" (empty) — model decides; equivalent to auto
	//   - "auto"     — model decides
	//   - "any"      — model must call one of the provided tools
	//   - "<name>"   — model must call exactly the named tool
	ToolChoice string `json:"toolChoice,omitempty"`

	// AdditionalModelRequestFields is forwarded verbatim as the Converse
	// API's AdditionalModelRequestFields document. Use for model-specific
	// knobs (e.g. Claude `thinking`, Nova reasoning levels) that are not
	// covered by the inference-config surface.
	AdditionalModelRequestFields map[string]any `json:"additionalModelRequestFields,omitempty"`
}

// ToolChoice constants for [Config.ToolChoice]. Pass a tool name (any other
// string) to force the model to call exactly that tool.
const (
	ToolChoiceAuto = "auto"
	ToolChoiceAny  = "any"
)

const cachePointMetadataKey = "bedrockCachePoint"

// NewCachePointPart returns an [ai.Part] that becomes a Bedrock cache point
// marker in the Converse request. Insert it between content blocks (typically
// after a long system prompt or document) to opt into prompt caching for
// everything before it.
//
// Bedrock requires a minimum amount of cacheable content (currently ~1024
// tokens per model family) before the cache point takes effect. If the
// content is too small the request still succeeds but no cache is created;
// the plugin does not validate this client-side.
func NewCachePointPart() *ai.Part {
	return ai.NewCustomPart(map[string]any{cachePointMetadataKey: true})
}

// isCachePoint reports whether p is the marker produced by NewCachePointPart.
func isCachePoint(p *ai.Part) bool {
	if p == nil || !p.IsCustom() {
		return false
	}
	v, ok := p.Custom[cachePointMetadataKey]
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
}

// configSchema returns the JSON schema for [Config], used as the per-call
// ConfigSchema on every defined model.
func configSchema() map[string]any { return core.InferSchemaMap(Config{}) }
