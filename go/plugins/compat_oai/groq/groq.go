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

// Package groq provides a Genkit plugin for Groq's OpenAI-compatible API.
package groq

import (
	"context"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const (
	provider       = "groq"
	defaultBaseURL = "https://api.groq.com/openai/v1"

	ModelLlama318bInstant    = "llama-3.1-8b-instant"
	ModelLlama3370bVersatile = "llama-3.3-70b-versatile"
	ModelGPTOss120b          = "openai/gpt-oss-120b"
	ModelGPTOss20b           = "openai/gpt-oss-20b"
	ModelCompound            = "groq/compound"
	ModelCompoundMini        = "groq/compound-mini"
	ModelQwen3627b           = "qwen/qwen3.6-27b"
)

var supportedModels = map[string]ai.ModelOptions{
	ModelLlama318bInstant:    textModel("Groq Llama 3.1 8B Instant"),
	ModelLlama3370bVersatile: textModel("Groq Llama 3.3 70B Versatile"),
	ModelGPTOss120b:          textModel("Groq GPT-OSS 120B"),
	ModelGPTOss20b:           textModel("Groq GPT-OSS 20B"),
	ModelCompound:            textModel("Groq Compound"),
	ModelCompoundMini:        textModel("Groq Compound Mini"),
	ModelQwen3627b:           mediaModel("Groq Qwen 3.6 27B"),
}

func textModel(label string) ai.ModelOptions {
	return ai.ModelOptions{
		Label: label,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
	}
}

func mediaModel(label string) ai.ModelOptions {
	opts := textModel(label)
	opts.Supports.Media = true
	return opts
}

// Groq configures the Groq OpenAI-compatible plugin.
type Groq struct {
	// APIKey is the Groq API key. If empty, GROQ_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the Groq API endpoint. If empty, GROQ_BASE_URL and
	// then https://api.groq.com/openai/v1 are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (g *Groq) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (g *Groq) Init(ctx context.Context) []api.Action {
	baseURL := g.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("GROQ_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := g.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GROQ_API_KEY")
	}
	if apiKey == "" {
		panic("groq plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, g.Opts...)

	g.openAICompatible.Provider = provider
	g.openAICompatible.Opts = opts
	actions := g.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, g.DefineModel(model, modelOpts).(api.Action))
	}
	return actions
}

// Model returns a registered Groq model.
func (g *Groq) Model(gk *genkit.Genkit, id string) ai.Model {
	return g.openAICompatible.Model(gk, api.NewName(provider, id))
}

// DefineModel registers a Groq model, including models not in the built-in list.
func (g *Groq) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return g.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions lists chat models exposed by the configured Groq endpoint,
// filtering out Whisper / TTS / embedding ids that are not chat-completion models.
func (g *Groq) ListActions(ctx context.Context) []api.ActionDesc {
	actions := g.openAICompatible.ListActions(ctx)
	if len(actions) == 0 {
		return actions
	}
	filtered := make([]api.ActionDesc, 0, len(actions))
	for _, action := range actions {
		_, id := api.ParseName(action.Name)
		if isNonChatGroqModel(id) {
			continue
		}
		filtered = append(filtered, action)
	}
	return filtered
}

// ResolveAction dynamically registers a model exposed by the Groq endpoint.
func (g *Groq) ResolveAction(atype api.ActionType, name string) api.Action {
	if isNonChatGroqModel(name) {
		return nil
	}
	return g.openAICompatible.ResolveAction(atype, name)
}

func isNonChatGroqModel(id string) bool {
	lid := strings.ToLower(id)
	return strings.HasPrefix(lid, "whisper") ||
		strings.Contains(lid, "orpheus") ||
		strings.Contains(lid, "embed")
}
