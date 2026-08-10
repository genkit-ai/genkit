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

package anthropic

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
	provider = "anthropic"
	baseURL  = "https://api.anthropic.com/v1"
)

// Supported models: https://docs.anthropic.com/en/docs/about-claude/models/all-models
var supportedModels = map[string]ai.ModelOptions{
	"claude-opus-4-1-20250805": {
		Label: "Claude 4.1 Opus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-opus-4-1-latest", "claude-opus-4-1-20250805"},
	},
	"claude-sonnet-4-5-20250929": {
		Label: "Claude 4.5 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-sonnet-4-5-latest", "claude-sonnet-4-5-20250929"},
	},
	"claude-haiku-4-5-20251001": {
		Label: "Claude 4.5 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-haiku-4-5-latest", "claude-haiku-4-5-20251001"},
	},
	"claude-3-7-sonnet-20250219": {
		Label: "Claude 3.7 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"},
	},
	"claude-3-5-haiku-20241022": {
		Label: "Claude 3.5 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-3-5-haiku-latest", "claude-3-5-haiku-20241022"},
	},
	"claude-3-5-sonnet-20240620": {
		Label: "Claude 3.5 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: false, // NOTE: This model does not support system role
			Media:      true,
		},
		Versions: []string{"claude-3-5-sonnet-20240620"},
	},
	"claude-3-opus-20240229": {
		Label: "Claude 3 Opus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: false, // NOTE: This model does not support system role
			Media:      true,
		},
		Versions: []string{"claude-3-opus-latest", "claude-3-opus-20240229"},
	},
	"claude-3-haiku-20240307": {
		Label: "Claude 3 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      false, // NOTE: Anthropic supports tool use, but it's not compatible with the OpenAI API
			SystemRole: false, // NOTE: This model does not support system role
			Media:      true,
		},
		Versions: []string{"claude-3-haiku-20240307"},
	},
}

type Anthropic struct {
	Opts             []option.RequestOption
	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (a *Anthropic) Name() string {
	return provider
}

func (a *Anthropic) Init(ctx context.Context) []api.Action {
	url := os.Getenv("ANTHROPIC_BASE_URL")
	if url == "" {
		url = baseURL
	}
	a.Opts = append([]option.RequestOption{option.WithBaseURL(url)}, a.Opts...)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey != "" {
		a.Opts = append([]option.RequestOption{option.WithAPIKey(apiKey)}, a.Opts...)
	}

	// initialize OpenAICompatible
	a.openAICompatible.Opts = a.Opts
	compatActions := a.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// define default models
	for model, opts := range supportedModels {
		actions = append(actions, a.DefineModel(model, opts).(api.Action))
	}

	return actions
}

func (a *Anthropic) Model(g *genkit.Genkit, id string) ai.Model {
	return a.openAICompatible.Model(g, api.NewName(provider, id))
}

func (a *Anthropic) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return a.openAICompatible.DefineModel(provider, id, opts)
}
