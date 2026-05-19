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
	"fmt"
	"os"
	"sort"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const (
	provider   = "ollamacloud"
	apiBaseURL = "https://ollama.com"
	apiVersion = "v1"
)

// supportedModels defines a curated set of Ollama Cloud models.
// Model IDs are aligned with https://ollama.com/v1/models.
var (
	textOnly = ai.ModelSupports{
		Multiturn:  true,
		Tools:      false,
		SystemRole: true,
		Media:      false,
	}

	visionOnly = ai.ModelSupports{
		Multiturn:  true,
		Tools:      false,
		SystemRole: true,
		Media:      true,
	}
)

func modelOptions(label, version string, supports *ai.ModelSupports) ai.ModelOptions {
	return ai.ModelOptions{
		Label:    label,
		Supports: supports,
		Versions: []string{version},
	}
}

func sortedSupportedModelIDs() []string {
	models := make([]string, 0, len(supportedModels))
	for model := range supportedModels {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

var supportedModels = map[string]ai.ModelOptions{
	// Text models without Ollama tool tags.
	"cogito-2.1:671b": modelOptions("Cogito 2.1 671B", "cogito-2.1:671b", &textOnly),

	// Text models with Ollama tool tags.
	"deepseek-v3.1:671b":  modelOptions("DeepSeek V3.1 671B", "deepseek-v3.1:671b", &compat_oai.BasicText),
	"deepseek-v3.2":       modelOptions("DeepSeek V3.2", "deepseek-v3.2", &compat_oai.BasicText),
	"deepseek-v4-flash":   modelOptions("DeepSeek V4 Flash", "deepseek-v4-flash", &compat_oai.BasicText),
	"deepseek-v4-pro":     modelOptions("DeepSeek V4 Pro", "deepseek-v4-pro", &compat_oai.BasicText),
	"devstral-2:123b":     modelOptions("Devstral 2 123B", "devstral-2:123b", &compat_oai.BasicText),
	"glm-4.6":             modelOptions("GLM-4.6", "glm-4.6", &compat_oai.BasicText),
	"glm-4.7":             modelOptions("GLM-4.7", "glm-4.7", &compat_oai.BasicText),
	"glm-5":               modelOptions("GLM-5", "glm-5", &compat_oai.BasicText),
	"glm-5.1":             modelOptions("GLM-5.1", "glm-5.1", &compat_oai.BasicText),
	"gpt-oss:20b":         modelOptions("GPT-OSS 20B", "gpt-oss:20b", &compat_oai.BasicText),
	"gpt-oss:120b":        modelOptions("GPT-OSS 120B", "gpt-oss:120b", &compat_oai.BasicText),
	"kimi-k2:1t":          modelOptions("Kimi K2 1T", "kimi-k2:1t", &compat_oai.BasicText),
	"kimi-k2-thinking":    modelOptions("Kimi K2 Thinking", "kimi-k2-thinking", &compat_oai.BasicText),
	"minimax-m2":          modelOptions("MiniMax M2", "minimax-m2", &compat_oai.BasicText),
	"minimax-m2.1":        modelOptions("MiniMax M2.1", "minimax-m2.1", &compat_oai.BasicText),
	"minimax-m2.5":        modelOptions("MiniMax M2.5", "minimax-m2.5", &compat_oai.BasicText),
	"minimax-m2.7":        modelOptions("MiniMax M2.7", "minimax-m2.7", &compat_oai.BasicText),
	"nemotron-3-nano:30b": modelOptions("Nemotron 3 Nano 30B", "nemotron-3-nano:30b", &compat_oai.BasicText),
	"nemotron-3-super":    modelOptions("Nemotron 3 Super", "nemotron-3-super", &compat_oai.BasicText),
	"qwen3-coder:480b":    modelOptions("Qwen3 Coder 480B", "qwen3-coder:480b", &compat_oai.BasicText),
	"qwen3-coder-next":    modelOptions("Qwen3 Coder Next", "qwen3-coder-next", &compat_oai.BasicText),
	"qwen3-next:80b":      modelOptions("Qwen3 Next 80B", "qwen3-next:80b", &compat_oai.BasicText),
	"rnj-1:8b":            modelOptions("RNJ-1 8B", "rnj-1:8b", &compat_oai.BasicText),

	// Vision models without Ollama tool tags.
	"gemma3:4b":  modelOptions("Gemma 3 4B", "gemma3:4b", &visionOnly),
	"gemma3:12b": modelOptions("Gemma 3 12B", "gemma3:12b", &visionOnly),
	"gemma3:27b": modelOptions("Gemma 3 27B", "gemma3:27b", &visionOnly),

	// Vision models with Ollama tool tags.
	"devstral-small-2:24b":   modelOptions("Devstral Small 2 24B", "devstral-small-2:24b", &compat_oai.Multimodal),
	"gemini-3-flash-preview": modelOptions("Gemini 3 Flash Preview", "gemini-3-flash-preview", &compat_oai.Multimodal),
	"gemma4:31b":             modelOptions("Gemma 4 31B", "gemma4:31b", &compat_oai.Multimodal),
	"kimi-k2.5":              modelOptions("Kimi K2.5", "kimi-k2.5", &compat_oai.Multimodal),
	"kimi-k2.6":              modelOptions("Kimi K2.6", "kimi-k2.6", &compat_oai.Multimodal),
	"ministral-3:3b":         modelOptions("Ministral 3 3B", "ministral-3:3b", &compat_oai.Multimodal),
	"ministral-3:8b":         modelOptions("Ministral 3 8B", "ministral-3:8b", &compat_oai.Multimodal),
	"ministral-3:14b":        modelOptions("Ministral 3 14B", "ministral-3:14b", &compat_oai.Multimodal),
	"mistral-large-3:675b":   modelOptions("Mistral Large 3 675B", "mistral-large-3:675b", &compat_oai.Multimodal),
	"qwen3-vl:235b":          modelOptions("Qwen3 VL 235B", "qwen3-vl:235b", &compat_oai.Multimodal),
	"qwen3-vl:235b-instruct": modelOptions("Qwen3 VL 235B Instruct", "qwen3-vl:235b-instruct", &compat_oai.Multimodal),
	"qwen3.5:397b":           modelOptions("Qwen3.5 397B", "qwen3.5:397b", &compat_oai.Multimodal),
}

// OllamaCloud represents the Ollama Cloud plugin
type OllamaCloud struct {
	APIKey string
	Opts   []option.RequestOption

	mu      sync.Mutex
	initted bool
	actions []api.Action

	openAICompatible *compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (o *OllamaCloud) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (o *OllamaCloud) Init(ctx context.Context) []api.Action {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.initted {
		return append([]api.Action(nil), o.actions...)
	}

	var compatActions []api.Action
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
		option.WithBaseURL(fmt.Sprintf("%s/%s", apiBaseURL, apiVersion)),
	}
	if len(o.Opts) > 0 {
		o.openAICompatible.Opts = append(o.openAICompatible.Opts, o.Opts...)
	}

	o.openAICompatible.Provider = provider
	compatActions = o.openAICompatible.Init(ctx)

	actions := make([]api.Action, 0, len(supportedModels)+len(compatActions))
	actions = append(actions, compatActions...)

	// Define available models
	for _, model := range sortedSupportedModelIDs() {
		actions = append(actions, o.defineModelAction(model, supportedModels[model]))
	}

	o.actions = actions
	o.initted = true
	return append([]api.Action(nil), o.actions...)
}

// Model returns the ai.Model with the given name.
func (o *OllamaCloud) Model(g *genkit.Genkit, name string) ai.Model {
	return o.openAICompatible.Model(g, api.NewName(provider, name))
}

// DefineModel defines a model with the given ID and options.
func (o *OllamaCloud) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return o.openAICompatible.DefineModel(provider, id, opts)
}

func (o *OllamaCloud) defineModelAction(id string, opts ai.ModelOptions) api.Action {
	action, ok := o.DefineModel(id, opts).(api.Action)
	if !ok {
		panic(fmt.Sprintf("ollamacloud model %q does not implement api.Action", id))
	}
	return action
}

// ListActions implements genkit.Plugin.
func (o *OllamaCloud) ListActions(ctx context.Context) []api.ActionDesc {
	return o.openAICompatible.ListActions(ctx)
}

// ResolveAction implements genkit.Plugin.
func (o *OllamaCloud) ResolveAction(atype api.ActionType, name string) api.Action {
	return o.openAICompatible.ResolveAction(atype, name)
}
