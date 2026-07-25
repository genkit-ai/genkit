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

package xai

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
	provider = "xai"
	baseURL  = "https://api.x.ai/v1"
)

var supportedModels = map[string]ai.ModelOptions{
	"grok-code-fast-1": {
		Label:    "Grok Code Fast 1",
		Supports: &compat_oai.BasicText,
		Versions: []string{"grok-code-fast-1"},
	},
	"grok-4-fast-reasoning": {
		Label:    "Grok 4 Fast",
		Supports: &compat_oai.Multimodal,
		Versions: []string{"grok-4-fast-reasoning", "grok-4-fast", "grok-4-fast-reasoning-latest"},
	},
	"grok-4-fast-non-reasoning": {
		Label:    "Grok 4 Fast (Non-Reasoning)",
		Supports: &compat_oai.Multimodal,
		Versions: []string{"grok-4-fast-non-reasoning", "grok-4-fast-non-reasoning-latest"},
	},
	"grok-4-0709": {
		Label:    "Grok 4",
		Supports: &compat_oai.Multimodal,
		Versions: []string{"grok-4-0709", "grok-4", "grok-4-latest"},
	},
	"grok-3": {
		Label:    "Grok 3",
		Supports: &compat_oai.BasicText,
		Versions: []string{"grok-3"},
	},
	"grok-3-mini": {
		Label:    "Grok 3 Mini",
		Supports: &compat_oai.BasicText,
		Versions: []string{"grok-3-mini"},
	},
	"grok-2-vision": {
		Label: "Grok 2 Vision",
		Supports: &ai.ModelSupports{
			Multiturn:  false,
			Tools:      true,
			SystemRole: false,
			Media:      true,
		},
		Versions: []string{"grok-2-vision", "grok-2-vision-1212", "grok-2-vision-latest"},
	},
}

type XAi struct {
	Opts             []option.RequestOption
	openAICompatible *compat_oai.OpenAICompatible
}

func (x *XAi) Name() string {
	return provider
}

func (x *XAi) Init(ctx context.Context) []api.Action {
	url := os.Getenv("XAI_BASE_URL")
	if url == "" {
		url = baseURL
	}
	x.Opts = append([]option.RequestOption{option.WithBaseURL(url)}, x.Opts...)

	apiKey := os.Getenv("XAI_API_KEY")
	if apiKey != "" {
		x.Opts = append([]option.RequestOption{option.WithAPIKey(apiKey)}, x.Opts...)
	}

	if x.openAICompatible == nil {
		x.openAICompatible = &compat_oai.OpenAICompatible{}
	}

	x.openAICompatible.Opts = x.Opts
	compatActions := x.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// define default models
	for model, opts := range supportedModels {
		actions = append(actions, x.DefineModel(model, opts).(api.Action))
	}

	return actions
}

func (x *XAi) Model(g *genkit.Genkit, id string) ai.Model {
	return x.openAICompatible.Model(g, api.NewName(provider, id))
}

func (x *XAi) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return x.openAICompatible.DefineModel(provider, id, opts)
}

func (x *XAi) ListActions(ctx context.Context) []api.ActionDesc {
	return x.openAICompatible.ListActions(ctx)
}

func (x *XAi) ResolveAction(atype api.ActionType, name string) api.Action {
	return x.openAICompatible.ResolveAction(atype, name)
}
