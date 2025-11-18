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

package ollamacloud

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const (
	provider = "ollamacloud"
	baseURL  = "https://ollama.com"
	version  = "/v1"
)

// supportedModels defines a curated set of Ollama Cloud models.
// Model IDs are aligned with https://ollama.com/v1/models.
var supportedModels = map[string]ai.ModelOptions{
	// Large Language Models (text-only)
	"gpt-oss:20b": {
		Label:    "GPT-OSS 20B",
		Supports: &compat_oai.BasicText,
		Versions: []string{"gpt-oss:20b"},
	},
	"gpt-oss:120b": {
		Label:    "GPT-OSS 120B",
		Supports: &compat_oai.BasicText,
		Versions: []string{"gpt-oss:120b"},
	},
	"qwen3-coder:480b": {
		Label:    "Qwen3 Coder 480B",
		Supports: &compat_oai.BasicText,
		Versions: []string{"qwen3-coder:480b"},
	},
	"deepseek-v3.1:671b": {
		Label:    "DeepSeek v3.1 671B",
		Supports: &compat_oai.BasicText,
		Versions: []string{"deepseek-v3.1:671b"},
	},
	"glm-4.6": {
		Label:    "GLM-4.6",
		Supports: &compat_oai.BasicText,
		Versions: []string{"glm-4.6"},
	},
	"minimax-m2": {
		Label:    "MiniMax M2",
		Supports: &compat_oai.BasicText,
		Versions: []string{"minimax-m2"},
	},
	"kimi-k2:1t": {
		Label:    "Kimi K2 1T",
		Supports: &compat_oai.BasicText,
		Versions: []string{"kimi-k2:1t"},
	},
	"kimi-k2-thinking": {
		Label:    "Kimi K2 Thinking",
		Supports: &compat_oai.BasicText,
		Versions: []string{"kimi-k2-thinking"},
	},

	// Multimodal Models (Vision + Text)
	"qwen3-vl:235b-instruct": {
		Label:    "Qwen3 VL 235B Instruct",
		Supports: &compat_oai.Multimodal,
		Versions: []string{"qwen3-vl:235b-instruct"},
	},
	"qwen3-vl:235b": {
		Label:    "Qwen3 VL 235B",
		Supports: &compat_oai.Multimodal,
		Versions: []string{"qwen3-vl:235b"},
	},
}

// OllamaCloud represents the Ollama Cloud plugin
type OllamaCloud struct {
	APIKey string
	Opts   []option.RequestOption

	openAICompatible *compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (o *OllamaCloud) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (o *OllamaCloud) Init(ctx context.Context) []api.Action {
	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OLLAMACLOUD_API_KEY")
	}

	if apiKey == "" {
		panic("ollamacloud plugin initialization failed: API key is required")
	}

	if o.openAICompatible == nil {
		o.openAICompatible = &compat_oai.OpenAICompatible{}
	}

	// Configure OpenAI-compatible client with Ollama Cloud settings
	o.openAICompatible.Opts = []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL + version),
	}
	if len(o.Opts) > 0 {
		o.openAICompatible.Opts = append(o.openAICompatible.Opts, o.Opts...)
	}

	o.openAICompatible.Provider = provider
	compatActions := o.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// Define available models
	for model, opts := range supportedModels {
		actions = append(actions, o.DefineModel(model, opts).(api.Action))
	}

	return actions
}

// Model returns the ai.Model with the given name.
func (o *OllamaCloud) Model(g *genkit.Genkit, name string) ai.Model {
	return o.openAICompatible.Model(g, api.NewName(provider, name))
}

// DefineModel defines a model with the given ID and options.
func (o *OllamaCloud) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return o.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions implements genkit.Plugin.
func (o *OllamaCloud) ListActions(ctx context.Context) []api.ActionDesc {
	return o.openAICompatible.ListActions(ctx)
}

// ResolveAction implements genkit.Plugin.
func (o *OllamaCloud) ResolveAction(atype api.ActionType, name string) api.Action {
	return o.openAICompatible.ResolveAction(atype, name)
}
