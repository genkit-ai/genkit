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

// Package meta provides a Genkit plugin for models hosted by Meta Model API.
package meta

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
	provider       = "meta"
	defaultBaseURL = "https://api.meta.ai/v1"

	// ModelMuseSpark11 is Meta's Muse Spark 1.1 multimodal reasoning model.
	ModelMuseSpark11 = "muse-spark-1.1"
)

var supportedModels = map[string]ai.ModelOptions{
	ModelMuseSpark11: {
		Label: "Meta Muse Spark 1.1",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{ModelMuseSpark11},
	},
}

// Meta configures the Meta Model API plugin.
type Meta struct {
	// APIKey is the Meta Model API key. If empty, MODEL_API_KEY and then
	// META_API_KEY are consulted.
	APIKey string
	// BaseURL overrides the Meta Model API endpoint. If empty, META_BASE_URL
	// and then the default endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (m *Meta) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (m *Meta) Init(ctx context.Context) []api.Action {
	baseURL := m.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("META_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := m.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("MODEL_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("META_API_KEY")
	}
	if apiKey == "" {
		panic("meta plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, m.Opts...)

	m.openAICompatible.Provider = provider
	m.openAICompatible.Opts = opts
	actions := m.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, m.DefineModel(model, modelOpts).(api.Action))
	}
	return actions
}

// Model returns a registered Meta model.
func (m *Meta) Model(g *genkit.Genkit, id string) ai.Model {
	return m.openAICompatible.Model(g, api.NewName(provider, id))
}

// DefineModel registers a Meta model, including models not in the built-in list.
func (m *Meta) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return m.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions lists models exposed by the configured Meta endpoint.
func (m *Meta) ListActions(ctx context.Context) []api.ActionDesc {
	return m.openAICompatible.ListActions(ctx)
}

// ResolveAction dynamically registers a model exposed by the Meta endpoint.
func (m *Meta) ResolveAction(atype api.ActionType, name string) api.Action {
	return m.openAICompatible.ResolveAction(atype, name)
}
