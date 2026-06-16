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
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
)

// resolveVertexMaasEnv resolves project and location from explicit arguments
// with fallback to the conventional environment variables. Panics if neither a
// value nor a fallback env var is set. Shared by all Vertex AI Model Garden
// plugins in this package.
func resolveVertexMaasEnv(projectID, location string) (string, string) {
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if projectID == "" {
			projectID = os.Getenv("GCLOUD_PROJECT")
		}
		if projectID == "" {
			panic("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT in the environment. You can get a project ID at https://console.cloud.google.com/home/dashboard")
		}
	}
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_REGION")
		}
		if location == "" {
			panic("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION in the environment. You can get a location at https://cloud.google.com/vertex-ai/docs/general/locations")
		}
	}
	return projectID, location
}

// provider is the shared registration namespace for every Vertex AI Model
// Garden plugin in this package (Anthropic, Llama). Their models are all
// registered as "vertexai/<model>" so users see a single namespace.
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

// AnthropicModels is a list of models supported in VertexAI.
// Keep this list updated since models cannot be dynamically listed if we are
// authenticating with Google Credentials. Keys must be Vertex AI publisher
// model IDs, not Anthropic API date-versioned IDs.
var AnthropicModels = map[string]ai.ModelOptions{
	"claude-opus-4-7": {
		Label:    "Claude Opus 4.7",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4-6": {
		Label:    "Claude Opus 4.6",
		Supports: &internal.Multimodal,
	},
	"claude-sonnet-4-6": {
		Label:    "Claude Sonnet 4.6",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4-5": {
		Label:    "Claude Opus 4.5",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4-1": {
		Label:    "Claude Opus 4.1",
		Supports: &internal.Multimodal,
	},
	"claude-opus-4": {
		Label:    "Claude Opus 4",
		Supports: &internal.Multimodal,
	},
	"claude-sonnet-4-5": {
		Label:    "Claude Sonnet 4.5",
		Supports: &internal.Multimodal,
	},
	"claude-sonnet-4": {
		Label:    "Claude Sonnet 4",
		Supports: &internal.Multimodal,
	},
	"claude-3-7-sonnet": {
		Label:    "Claude 3.7 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-3-5-sonnet-v2": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-haiku-4-5": {
		Label:    "Claude Haiku 4.5",
		Supports: &internal.Multimodal,
	},
	"claude-3-5-haiku": {
		Label:    "Claude 3.5 Haiku",
		Supports: &internal.Multimodal,
	},
	"claude-3-5-sonnet": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-3-sonnet": {
		Label:    "Claude 3 Sonnet",
		Supports: &internal.Multimodal,
	},
	"claude-3-haiku": {
		Label:    "Claude 3 Haiku",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-opus": {
		Label:    "Claude 3 Opus",
		Supports: &internal.Multimodal,
	},

	// Date-versioned aliases kept for backwards compatibility with callers
	// pinned to the IDs that shipped before the undated rename. Vertex AI
	// accepts both forms for the same underlying model.
	"claude-opus-4@20250514": {
		Label:    "Claude Opus 4",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-sonnet-4@20250514": {
		Label:    "Claude Sonnet 4",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-7-sonnet@20250219": {
		Label:    "Claude 3.7 Sonnet",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-5-sonnet-v2@20241022": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-5-sonnet@20240620": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-sonnet@20240229": {
		Label:    "Claude 3 Sonnet",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-haiku@20240307": {
		Label:    "Claude 3 Haiku",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
	},
	"claude-3-opus@20240229": {
		Label:    "Claude 3 Opus",
		Supports: &internal.Multimodal,
		Stage:    ai.ModelStageDeprecated,
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
		Supports: &internal.BasicText,
	},
}
