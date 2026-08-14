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
//
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	oa "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	oai "github.com/firebase/genkit/go/plugins/internal/openai"
)

const (
	provider          = "openai"
	openAILabelPrefix = "OpenAI"
)

// OpenAI is a Genkit plugin for interacting with OpenAI services.
type OpenAI struct {
	APIKey  string                 // If not provided, defaults to OPENAI_API_KEY.
	BaseURL string                 // Optional. If not provided, defaults to OPENAI_BASE_URL.
	Opts    []option.RequestOption // Optional SDK request options appended after API key and base URL.

	client      oa.Client
	mu          sync.Mutex
	initted     bool
	models      []string
	lastUpdated time.Time
}

// Name returns the name of the plugin.
func (o *OpenAI) Name() string {
	return provider
}

// Init initializes the OpenAI plugin.
func (o *OpenAI) Init(ctx context.Context) []api.Action {
	if o == nil {
		o = &OpenAI{}
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	if o.initted {
		panic("plugin already initialized")
	}

	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		panic("OpenAI requires setting OPENAI_API_KEY in the environment")
	}

	opts := []option.RequestOption{option.WithAPIKey(apiKey)}
	baseURL := o.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("OPENAI_BASE_URL")
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if len(o.Opts) > 0 {
		opts = append(opts, o.Opts...)
	}

	o.client = oa.NewClient(opts...)
	o.initted = true
	return []api.Action{}
}

// DefineModel defines an unknown model with the given name.
// The second argument describes the capability of the model.
// Use [IsDefinedModel] to determine if a model is already defined.
// After [Init] is called, only the known models are defined.
func (o *OpenAI) DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	return oai.DefineModel(o.client, provider, name, *opts), nil
}

// ListActions lists all actions supported by the OpenAI plugin.
func (o *OpenAI) ListActions(ctx context.Context) []api.ActionDesc {
	actions := []api.ActionDesc{}

	models, err := o.getModels(ctx)
	if err != nil {
		slog.Error("unable to list openai models from OpenAI API", "error", err)
		return nil
	}

	for _, name := range models {
		model := newModel(o.client, name, name, defaultOpenAIOpts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}
	return actions
}

// Model returns a previously registered model.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// IsDefinedModel returns whether a model is already defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// ResolveAction resolves an action with the given name.
func (o *OpenAI) ResolveAction(atype api.ActionType, id string) api.Action {
	switch atype {
	case api.ActionTypeModel:
		return newModel(o.client, id, id, ai.ModelOptions{
			Label:    fmt.Sprintf("%s - %s", openAILabelPrefix, id),
			Stage:    ai.ModelStageStable,
			Versions: []string{},
			Supports: defaultOpenAIOpts.Supports,
		}).(api.Action)
	}
	return nil
}

// getModels returns the list of available models, using a cache if available.
func (o *OpenAI) getModels(ctx context.Context) ([]string, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if !o.lastUpdated.IsZero() && time.Since(o.lastUpdated) < time.Hour {
		return o.models, nil
	}

	models, err := listModels(ctx, &o.client)
	if err != nil {
		return nil, err
	}

	o.models = models
	o.lastUpdated = time.Now()
	return models, nil
}

// newModel creates a model without registering it.
func newModel(client oa.Client, name, apiModelName string, opts ai.ModelOptions) ai.Model {
	config := &responses.ResponseNewParams{}

	meta := &ai.ModelOptions{
		Label:        opts.Label,
		Supports:     opts.Supports,
		Versions:     opts.Versions,
		ConfigSchema: oai.ConfigSchema(config),
		Stage:        opts.Stage,
	}

	targetModel := name
	if apiModelName != "" {
		targetModel = apiModelName
	}

	fn := func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return oai.Generate(ctx, client, targetModel, input, cb)
	}

	return ai.NewModel(api.NewName(provider, name), meta, fn)
}
