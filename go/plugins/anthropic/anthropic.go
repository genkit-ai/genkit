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

package anthropic

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"
)

const (
	provider             = "anthropic"
	anthropicLabelPrefix = "Anthropic"
)

var dateSuffix = regexp.MustCompile(`-\d{8}$`)

// Anthropic is a Genkit plugin for interacting with the Anthropic services
type Anthropic struct {
	APIKey  string // If not provided, defaults to ANTHROPIC_API_KEY
	BaseURL string // Optional. If not provided, defaults to ANTHROPIC_BASE_URL

	aclient     anthropic.Client // Anthropic client
	mu          sync.Mutex       // Mutex to control access
	initted     bool             // Whether the plugin has been initialized
	models      []string         // Cached list of models
	lastUpdated time.Time        // When the cache was last updated
}

// Name returns the name of the plugin
func (a *Anthropic) Name() string {
	return provider
}

// Init initializes the Anthropic plugin and all known models
func (a *Anthropic) Init(ctx context.Context) []api.Action {
	if a == nil {
		a = &Anthropic{}
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.initted {
		panic("plugin already initialized")
	}

	apiKey := a.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		panic("Anthropic requires setting ANTHROPIC_API_KEY in the environment")
	}

	opts := []option.RequestOption{option.WithAPIKey(apiKey)}

	baseURL := a.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("ANTHROPIC_BASE_URL")
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}

	ac := anthropic.NewClient(opts...)
	a.aclient = ac
	a.initted = true

	return []api.Action{}
}

// buildModel builds an unregistered Claude model. A nil opts takes the
// capabilities the plugin resolves for that name, and name is the model ID,
// bare or provider-prefixed.
func (a *Anthropic) buildModel(name string, opts *ai.ModelOptions) *ai.ModelAction {
	// Trim before resolving, so a prefixed name still hits knownModels.
	name = strings.TrimPrefix(name, provider+"/")

	var modelOpts ai.ModelOptions
	if opts != nil {
		modelOpts = *opts
	} else {
		modelOpts = modelOptions(name)
	}

	return newModel(a.aclient, name, name, modelOpts)
}

// RegisterModel registers a Claude model with g and returns it. The plugin
// supplies the implementation; opts describes what the model supports, and a
// nil opts takes the capabilities the plugin resolves for that name, curated
// for a known model and the Claude defaults for the rest.
//
// Most applications never need this. Every Claude model resolves on demand,
// so naming one that was never registered is enough:
//
//	genkit.Generate(ctx, g, ai.WithModelName("anthropic/claude-opus-4-5"), ...)
//
// Reach for RegisterModel only to pin capabilities that differ from the ones
// the plugin resolves, which is what opts is for.
//
// Registering a name that is already registered panics, and generating with a
// name registers it, so register a model before its first use or guard with
// [IsDefinedModel]. name is the model ID, bare or provider-prefixed.
func (a *Anthropic) RegisterModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	model := a.buildModel(name, opts)
	genkit.RegisterAction(g, model)
	return model, nil
}

// DefineModel builds a Claude model and returns it, without registering it
// with g.
//
// Deprecated: use [Anthropic.RegisterModel]. This method builds the model and
// ignores g. Generation resolves a model from its name, so passing the result
// to ai.WithModel contributes only that name and serves the request with a
// model resolved from it instead; registering it with [genkit.RegisterAction]
// is what makes these capabilities the ones used.
func (a *Anthropic) DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	return a.buildModel(name, opts), nil
}

// modelOptions returns the ModelOptions for a Claude model name. Known models
// (see knownModels) carry curated capabilities and labels; any other model
// falls back to defaultClaudeOpts, whose label newModel fills in from the
// name. This is the single source of model capabilities shared by ListActions
// and ResolveAction, mirroring the JS plugin's claudeModelReference.
func modelOptions(name string) ai.ModelOptions {
	opts, ok := knownModels[baseModelName(name)]
	if !ok {
		opts = defaultClaudeOpts
	}
	return opts
}

// ListActions lists all the actions supported by the Anthropic plugin
func (a *Anthropic) ListActions(ctx context.Context) []api.ActionDesc {
	actions := []api.ActionDesc{}

	models, err := a.getModels(ctx)
	if err != nil {
		slog.Error("unable to list anthropic models from Anthropic API", "error", err)
		return nil
	}

	for _, name := range models {
		// When listing discovered models, the Genkit action name and the
		// Anthropic API model ID are identical.
		actions = append(actions, newModel(a.aclient, name, name, modelOptions(name)).Desc())
	}

	return actions
}

// Model returns a previously registered model.
//
// Deprecated: Generation resolves a model from its name, so looking one up
// first is rarely necessary: pass ai.WithModelName("anthropic/claude-opus-4-5")
// or, to carry config with it, [ModelRef]. Use [genkit.LookupModel] when the
// action itself is what you need.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, modelName(name))
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [Anthropic.RegisterModel]). The lookup
// deliberately does not resolve dynamically: a resolving lookup would ask the
// plugin to resolve the very model the caller is checking for, registering it
// and answering true for any name the Anthropic API can serve.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupAction(g, fmt.Sprintf("/%s/%s", api.ActionTypeModel, modelName(name))) != nil
}

// modelName builds the action name for a Claude model ID, taking the ID either
// bare or already provider-prefixed. The prefix is applied by concatenation,
// so without the trim an already-prefixed name would double up and name a
// model that resolves nowhere.
func modelName(name string) string {
	return api.NewName(provider, strings.TrimPrefix(name, provider+"/"))
}

// ResolveAction resolves an action with the given name
func (a *Anthropic) ResolveAction(atype api.ActionType, id string) api.Action {
	switch atype {
	case api.ActionTypeModel:
		models, err := a.getModels(context.Background())
		if err != nil {
			slog.Error("unable to list anthropic models from Anthropic API", "error", err)
			return nil
		}

		realID, ok := resolveModelID(id, models)
		if !ok {
			// If not found, fall back to using id as is (legacy behavior, or for models not in list)
			realID = id
		}

		// We register the model using the ID requested by the user, but
		// use the resolved 'realID' (e.g. versioned) for actual API calls.
		return newModel(a.aclient, id, realID, modelOptions(id))
	}
	return nil
}

// getModels returns the list of available models, using a cache if available.
func (a *Anthropic) getModels(ctx context.Context) ([]string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.lastUpdated.IsZero() && time.Since(a.lastUpdated) < time.Hour {
		return a.models, nil
	}

	models, err := listModels(ctx, &a.aclient)
	if err != nil {
		return nil, err
	}

	a.models = models
	a.lastUpdated = time.Now()
	return models, nil
}

// newModel creates a model without registering it. name is the Genkit action
// name and apiModelName is the model ID sent to the API, which differ when the
// name is an alias for a dated release.
func newModel(client anthropic.Client, name, apiModelName string, opts ai.ModelOptions) *ai.ModelAction {
	return ant.NewModel(client, provider, name, apiModelName, opts)
}

func baseModelName(name string) string {
	return dateSuffix.ReplaceAllString(name, "")
}

func resolveModelID(id string, availableModels []string) (string, bool) {
	// First check for exact match
	for _, m := range availableModels {
		if m == id {
			return m, true
		}
	}

	var bestMatch string
	prefix := id + "-"

	for _, m := range availableModels {
		if strings.HasPrefix(m, prefix) && baseModelName(m) == id {
			if m > bestMatch {
				bestMatch = m
			}
		}
	}

	if bestMatch != "" {
		return bestMatch, true
	}

	return "", false
}
