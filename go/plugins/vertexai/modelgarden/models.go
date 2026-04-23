// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package modelgarden

import (
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
)

// provider is the shared registration namespace for every Vertex AI Model
// Garden plugin in this package (Anthropic, Llama, Mistral). Their models
// are all registered as "vertexai/<model>" so users see a single namespace.
// Each plugin's own Name() ("vertex-model-garden-<plugin>") is distinct from
// this registration prefix.
//
// Note that this collides with the main googlegenai.VertexAI plugin, whose
// Name() is also "vertexai" — model-name prefixes overlap, but plugin names
// do not. Consequence: dynamic resolution of an unknown "vertexai/<name>" is
// routed to googlegenai.VertexAI, not to the modelgarden plugins.
//
// TODO: in the next major version, switch to per-plugin prefixes
// (e.g. "vertex-model-garden-anthropic/<model>") so the model namespace
// matches the plugin that owns it.
const provider = "vertexai"

// AnthropicModels is a list of models supported in VertexAI
// Keep this list updated since models cannot be dynamically listed
// if we are authenticating with Google Credentials
var AnthropicModels = map[string]ai.ModelOptions{
	"claude-3-5-sonnet-v2@20241022": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-3-5-sonnet@20240620": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-3-sonnet@20240229": {
		Label:    "Claude 3 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-3-haiku@20240307": {
		Label:    "Claude 3 Haiku",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-opus@20240229": {
		Label:    "Claude 3 Opus",
		Supports: &internal.Multimodal,
	},
	"claude-3-7-sonnet@20250219": {
		Label:    "Claude 3.7 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4@20250514": {
		Label:    "Claude Opus 4",
		Supports: &internal.Multimodal,
	},
	"claude-sonnet-4@20250514": {
		Label:    "Claude Sonnet 4",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4-1-20250805": {
		Label:    "Claude 4.1 Opus",
		Supports: &internal.Multimodal,
	},
	"claude-sonnet-4-5-20250929": {
		Label:    "Claude 4.5 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-haiku-4-5-20251001": {
		Label:    "Claude 4.5 Haiku",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4-5@20251101": {
		Label:    "Claude Opus 4.5",
		Supports: &internal.Multimodal,
	},
}

// LlamaModels lists the Meta Llama models available through Vertex AI Model Garden
// as Model-as-a-Service (MaaS) endpoints. These models are served via an
// OpenAI-compatible API.
var LlamaModels = map[string]ai.ModelOptions{
	"meta/llama-4-maverick-17b-128e-instruct-maas": {
		Label:    "Llama 4 Maverick 17B 128E Instruct",
		Supports: &internal.Multimodal,
	},
	"meta/llama-4-scout-17b-16e-instruct-maas": {
		Label:    "Llama 4 Scout 17B 16E Instruct",
		Supports: &internal.Multimodal,
	},
	"meta/llama-3.3-70b-instruct-maas": {
		Label:    "Llama 3.3 70B Instruct",
		Supports: &internal.Multimodal,
	},
}

// MistralModels lists the Mistral and Codestral models available through Vertex
// AI Model Garden as Model-as-a-Service (MaaS) endpoints. Capabilities match
// the JS parity target (media: false, tools: true, systemRole: true).
var MistralModels = map[string]ai.ModelOptions{
	"mistral-medium-3": {
		Label:    "Mistral Medium 3",
		Supports: &internal.BasicText,
	},
	"mistral-ocr-2505": {
		Label:    "Mistral OCR 2505",
		Supports: &internal.BasicText,
	},
	"mistral-small-2503": {
		Label:    "Mistral Small 2503",
		Supports: &internal.BasicText,
	},
	"codestral-2": {
		Label:    "Codestral 2",
		Supports: &internal.BasicText,
	},
}
