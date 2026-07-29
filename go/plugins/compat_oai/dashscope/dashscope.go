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

package dashscope

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
	provider = "dashscope"
	// defaultBaseURL is the shared international endpoint, used as a fallback
	// when neither BaseURL nor DASHSCOPE_BASE_URL is set. Works for standard
	// API keys; mainland-China accounts or workspace-dedicated domains
	// (Alibaba's recommended production setup) should override via BaseURL or
	// DASHSCOPE_BASE_URL. See https://help.aliyun.com/en/model-studio/base-url
	// and the package README.
	defaultBaseURL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

// Supported models: https://www.alibabacloud.com/help/en/model-studio/models
// Confirmed against the live GET /compatible-mode/v1/models response for this workspace.
// Dated snapshots are folded into Versions rather than registered as separate models,
// mirroring the anthropic and openai subplugins.
var supportedModels = map[string]ai.ModelOptions{
	"qwen-flash": {
		Label: "Qwen Flash",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen-flash", "qwen-flash-2025-07-28"},
	},
	"qwen-plus": {
		Label: "Qwen Plus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen-plus", "qwen-plus-2025-07-28", "qwen-plus-2025-09-11", "qwen-plus-2025-12-01"},
	},
	"qwen3.5-flash": {
		Label: "Qwen 3.5 Flash",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3.5-flash", "qwen3.5-flash-2026-02-23"},
	},
	"qwen3.5-plus": {
		Label: "Qwen 3.5 Plus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3.5-plus", "qwen3.5-plus-2026-02-15"},
	},
	"qwen3.6-flash": {
		Label: "Qwen 3.6 Flash",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3.6-flash", "qwen3.6-flash-2026-04-16"},
	},
	"qwen3.6-plus": {
		Label: "Qwen 3.6 Plus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3.6-plus", "qwen3.6-plus-2026-04-02"},
	},
	"qwen3.7-plus": {
		Label: "Qwen 3.7 Plus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3.7-plus", "qwen3.7-plus-2026-05-26"},
	},
	"qwen3.7-max": {
		Label: "Qwen 3.7 Max",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3.7-max", "qwen3.7-max-2026-06-08", "qwen3.7-max-2026-05-20"},
	},
	"qwen3-max": {
		Label: "Qwen 3 Max",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3-max", "qwen3-max-2026-01-23", "qwen3-max-2025-09-23", "qwen3-max-preview"},
	},
	"qwen3-vl-plus": {
		Label: "Qwen 3 VL Plus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3-vl-plus", "qwen3-vl-plus-2025-12-19", "qwen3-vl-plus-2025-09-23"},
	},
	"qwen3-coder-plus": {
		Label: "Qwen 3 Coder Plus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{"qwen3-coder-plus", "qwen3-coder-plus-2025-07-22", "qwen3-coder-plus-2025-09-23"},
	},
}

// DashScope configures the Alibaba Cloud DashScope (Qwen) plugin.
type DashScope struct {
	// APIKey is the DashScope API key. If empty, DASHSCOPE_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the DashScope API endpoint. If empty, DASHSCOPE_BASE_URL,
	// and then the default international endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible *compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DashScope) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DashScope) Init(ctx context.Context) []api.Action {
	baseURL := d.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("DASHSCOPE_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := d.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DASHSCOPE_API_KEY")
	}
	if apiKey == "" {
		panic("dashscope plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, d.Opts...)

	if d.openAICompatible == nil {
		d.openAICompatible = &compat_oai.OpenAICompatible{}
	}

	d.openAICompatible.Provider = provider
	d.openAICompatible.Opts = opts
	compatActions := d.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// define default models
	for model, modelOpts := range supportedModels {
		actions = append(actions, d.DefineModel(model, modelOpts).(api.Action))
	}

	return actions
}

func (d *DashScope) Model(g *genkit.Genkit, id string) ai.Model {
	return d.openAICompatible.Model(g, api.NewName(provider, id))
}

func (d *DashScope) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return d.openAICompatible.DefineModel(provider, id, opts)
}
