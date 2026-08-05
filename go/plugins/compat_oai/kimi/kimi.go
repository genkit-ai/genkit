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

// Package kimi provides a Genkit plugin for Moonshot AI's Kimi models.
package kimi

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
	provider       = "kimi"
	defaultBaseURL = "https://api.moonshot.ai/v1"

	// ModelKimiK3 is the latest Kimi flagship multimodal model.
	ModelKimiK3 = "kimi-k3"
	// ModelKimiK25 is the Kimi K2.5 multimodal model.
	ModelKimiK25 = "kimi-k2.5"
	// ModelKimiK26 is the Kimi K2.6 multimodal model.
	ModelKimiK26 = "kimi-k2.6"
	// ModelKimiK27Code is the Kimi K2.7 coding model.
	ModelKimiK27Code = "kimi-k2.7-code"
	// ModelKimiK27CodeHighspeed is the high-speed Kimi K2.7 coding model.
	ModelKimiK27CodeHighspeed = "kimi-k2.7-code-highspeed"
)

var supportedModels = map[string]ai.ModelOptions{
	ModelKimiK3: {
		Label: "Kimi K3",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{ModelKimiK3},
	},
	ModelKimiK25: {
		Label: "Kimi K2.5 (Deprecated)",
		Stage: ai.ModelStageDeprecated,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{ModelKimiK25},
	},
	ModelKimiK26: {
		Label: "Kimi K2.6",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{ModelKimiK26},
	},
	ModelKimiK27Code: {
		Label: "Kimi K2.7 Code",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{ModelKimiK27Code},
	},
	ModelKimiK27CodeHighspeed: {
		Label: "Kimi K2.7 Code Highspeed",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{ModelKimiK27CodeHighspeed},
	},
}

// Kimi configures the Moonshot AI Kimi plugin.
type Kimi struct {
	// APIKey is the Moonshot API key. If empty, KIMI_API_KEY and then
	// MOONSHOT_API_KEY are consulted.
	APIKey string
	// BaseURL overrides the Moonshot API endpoint. If empty, KIMI_BASE_URL,
	// MOONSHOT_BASE_URL, and then the default international endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (k *Kimi) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (k *Kimi) Init(ctx context.Context) []api.Action {
	baseURL := k.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("KIMI_BASE_URL")
	}
	if baseURL == "" {
		baseURL = os.Getenv("MOONSHOT_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := k.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("KIMI_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("MOONSHOT_API_KEY")
	}
	if apiKey == "" {
		panic("kimi plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, k.Opts...)

	k.openAICompatible.Provider = provider
	k.openAICompatible.Opts = opts
	actions := k.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, k.DefineModel(model, modelOpts).(api.Action))
	}
	return actions
}

// Model returns a registered Kimi model.
func (k *Kimi) Model(g *genkit.Genkit, id string) ai.Model {
	return k.openAICompatible.Model(g, api.NewName(provider, id))
}

// DefineModel registers a Kimi model, including models not in the built-in list.
func (k *Kimi) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return k.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions lists models exposed by the configured Kimi endpoint.
func (k *Kimi) ListActions(ctx context.Context) []api.ActionDesc {
	return k.openAICompatible.ListActions(ctx)
}

// ResolveAction dynamically registers a model exposed by the Kimi endpoint.
func (k *Kimi) ResolveAction(atype api.ActionType, name string) api.Action {
	return k.openAICompatible.ResolveAction(atype, name)
}
