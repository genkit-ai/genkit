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

package cohere

import "github.com/firebase/genkit/go/ai"

// modelInfo holds the static metadata for a curated Cohere chat model.
type modelInfo struct {
	Label    string
	Supports *ai.ModelSupports
	Versions []string
	Stage    ai.ModelStage
}

// defaultChatSupports describes the capabilities shared by the ChatV2 models.
// Media is left disabled for v1; vision models (command-a-vision) are deferred.
var defaultChatSupports = &ai.ModelSupports{
	Multiturn:   true,
	Tools:       true,
	ToolChoice:  true,
	SystemRole:  true,
	Media:       false,
	Constrained: ai.ConstrainedSupportAll,
}

// cohereChatModels is the curated catalogue of Cohere ChatV2 models. The map
// key is the Genkit action name; Versions lists dated aliases that resolve to
// the same family on Cohere's side.
var cohereChatModels = map[string]modelInfo{
	"command-a-03-2025": {
		Label:    cohereLabelPrefix + " - Command A (03-2025)",
		Supports: defaultChatSupports,
		Stage:    ai.ModelStageStable,
	},
	"command-r-plus": {
		Label:    cohereLabelPrefix + " - Command R+",
		Supports: defaultChatSupports,
		Versions: []string{"command-r-plus-08-2024", "command-r-plus-04-2024"},
		Stage:    ai.ModelStageStable,
	},
	"command-r": {
		Label:    cohereLabelPrefix + " - Command R",
		Supports: defaultChatSupports,
		Versions: []string{"command-r-08-2024", "command-r-03-2024"},
		Stage:    ai.ModelStageStable,
	},
	"command-r7b-12-2024": {
		Label:    cohereLabelPrefix + " - Command R7B (12-2024)",
		Supports: defaultChatSupports,
		Stage:    ai.ModelStageStable,
	},
}

// embedderInfo holds the static metadata for a curated Cohere embedder.
type embedderInfo struct {
	Label      string
	Dimensions int
}

// cohereEmbedders is the curated catalogue of Cohere embedding models.
var cohereEmbedders = map[string]embedderInfo{
	"embed-v4.0": {
		Label:      cohereLabelPrefix + " - Embed v4.0",
		Dimensions: 1536,
	},
	"embed-english-v3.0": {
		Label:      cohereLabelPrefix + " - Embed English v3.0",
		Dimensions: 1024,
	},
	"embed-multilingual-v3.0": {
		Label:      cohereLabelPrefix + " - Embed Multilingual v3.0",
		Dimensions: 1024,
	},
}

// GetModelOptions returns the curated metadata for a chat model, falling back
// to sensible defaults for an unrecognized name so callers can still target
// newly released models.
func GetModelOptions(name string) modelInfo {
	if info, ok := cohereChatModels[name]; ok {
		return info
	}
	return modelInfo{
		Label:    cohereLabelPrefix + " - " + name,
		Supports: defaultChatSupports,
		Stage:    ai.ModelStageStable,
	}
}

// GetEmbedderOptions returns the curated metadata for an embedder, falling back
// to defaults for an unrecognized name.
func GetEmbedderOptions(name string) embedderInfo {
	if info, ok := cohereEmbedders[name]; ok {
		return info
	}
	return embedderInfo{Label: cohereLabelPrefix + " - " + name}
}
