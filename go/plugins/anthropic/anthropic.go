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
	"errors"
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
	"github.com/firebase/genkit/go/plugins/internal"
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

	// Models overrides what the plugin knows about a Claude model, keyed by
	// model ID, bare or provider-prefixed. Every Claude model already works
	// without an entry here: known IDs carry curated capabilities and the rest
	// take the Claude defaults. Supply an entry only to correct or extend what
	// the plugin resolves, most often to describe a model released after this
	// version of the plugin.
	//
	//	&anthropic.Anthropic{Models: map[string]ai.ModelOptions{
	//		"claude-opus-4-5": {Supports: &ai.ModelSupports{Tools: true, Multiturn: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the config
	// schema. Entries apply everywhere a model is described: the actions
	// [Anthropic.ListActions] advertises and the ones
	// [Anthropic.ResolveAction] builds to serve a request.
	Models map[string]ai.ModelOptions

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

// DefineModel builds a Claude model and returns it, without registering it
// with g.
//
// Deprecated: describe the model through [Anthropic.Models] instead. This
// method builds the model and ignores g, so the result carries only the
// model's name: generation resolves a model from that name and serves the
// request with the capabilities the plugin resolves, not the ones passed
// here. An entry in Models reaches both paths.
func (a *Anthropic) DefineModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.initted {
		return nil, errors.New("anthropic plugin not initialized")
	}

	// Trim before resolving, so a prefixed id still hits supportedModels.
	id = strings.TrimPrefix(id, provider+"/")

	modelOpts := a.modelOptions(id)
	if opts != nil {
		modelOpts = *opts
	}

	return newModel(a.aclient, id, id, modelOpts), nil
}

// modelOptions returns the ModelOptions for a Claude model ID. Known models
// (see supportedModels) carry curated capabilities and labels; any other model
// falls back to dynamicModelOptions, whose label newModel fills in from the
// ID. An entry in [Anthropic.Models] overlays whichever of the two applies.
//
// This is the single source of model capabilities shared by ListActions and
// ResolveAction, mirroring the JS plugin's claudeModelReference, which is what
// makes a caller's override authoritative no matter which path describes the
// model first.
//
// The caller is responsible for trimming the provider prefix off id; Models is
// keyed either way, so both forms are accepted there.
func (a *Anthropic) modelOptions(id string) ai.ModelOptions {
	opts, ok := supportedModels[baseModelName(id)]
	if !ok {
		opts = dynamicModelOptions
	}
	if override, ok := a.modelOverride(id); ok {
		opts = internal.OverlayModelOptions(opts, override)
	}
	return opts
}

// modelOverride returns the caller's entry for a bare model ID, accepting the
// key in either the bare or the provider-prefixed form the rest of the package
// takes.
func (a *Anthropic) modelOverride(id string) (ai.ModelOptions, bool) {
	if opts, ok := a.Models[id]; ok {
		return opts, true
	}
	opts, ok := a.Models[provider+"/"+id]
	return opts, ok
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
		actions = append(actions, newModel(a.aclient, name, name, a.modelOptions(name)).Desc())
	}

	return actions
}

// Model returns a previously registered model.
//
// Deprecated: Generation resolves a model from its name, so looking one up
// first is rarely necessary: pass ai.WithModelName("anthropic/claude-opus-4-5")
// or, to carry config with it, [ModelRef]. Use [genkit.LookupModel] when the
// action itself is what you need.
func Model(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, modelName(id))
}

// IsDefinedModel reports whether a model is already registered. The lookup
// deliberately does not resolve dynamically: a resolving lookup would ask the
// plugin to resolve the very model the caller is checking for, registering it
// and answering true for any ID the Anthropic API can serve.
//
// Deprecated: this existed to guard a registration call that could panic on a
// duplicate. Capabilities now come from [Anthropic.Models], which nothing has
// to register and which no ordering can defeat, leaving this a question about
// registry state that applications do not need to ask.
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return genkit.LookupAction(g, fmt.Sprintf("/%s/%s", api.ActionTypeModel, modelName(id))) != nil
}

// modelName builds the action name for a Claude model ID, taking the ID either
// bare or already provider-prefixed. The prefix is applied by concatenation,
// so without the trim an already-prefixed ID would double up and name a
// model that resolves nowhere.
func modelName(id string) string {
	return api.NewName(provider, strings.TrimPrefix(id, provider+"/"))
}

// ResolveAction resolves an action with the given ID
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
		return newModel(a.aclient, id, realID, a.modelOptions(id))
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

func baseModelName(id string) string {
	return dateSuffix.ReplaceAllString(id, "")
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
