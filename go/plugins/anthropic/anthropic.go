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
	"log/slog"
	"os"
	"regexp"
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

// provider prefixes the action name of every model this plugin serves.
const provider = "anthropic"

// anthropicLabelPrefix opens the dev UI label of every model this plugin
// describes. It comes from [ant.DisplayName] so the curated labels cannot
// drift from the ones NewModel defaults.
var anthropicLabelPrefix = ant.DisplayName(provider)

// modelCacheTTL is how long a discovered model list is reused before the API
// is asked again. Anthropic's catalog changes on the order of weeks, so this
// trades staleness that resolves itself for a request on every listing.
const modelCacheTTL = time.Hour

// dateSuffix matches the -YYYYMMDD release date the API appends to a dated
// model ID. It is what separates claude-opus-4-5-20251101 from the
// claude-opus-4-5 alias pointing at it.
var dateSuffix = regexp.MustCompile(`-\d{8}$`)

// Anthropic is a Genkit plugin for interacting with the Anthropic API.
type Anthropic struct {
	// APIKey is the key requests are authenticated with. When empty, the
	// ANTHROPIC_API_KEY environment variable is used, and [Anthropic.Init]
	// panics unless authentication arrives another way: an auth token in
	// ANTHROPIC_AUTH_TOKEN, or a non-empty Opts.
	APIKey string

	// BaseURL overrides the API endpoint requests are sent to. When empty, the
	// ANTHROPIC_BASE_URL environment variable is used, and failing that the
	// SDK's own default.
	BaseURL string

	// Opts are anthropic-sdk-go request options applied to every request the
	// client sends. They open the SDK's full client configuration to the
	// plugin: [option.WithMiddleware], [option.WithMaxRetries],
	// [option.WithRequestTimeout], extra headers, and the SDK's bedrock and
	// vertex packages, whose routing helpers are request options too.
	//
	// They are applied after the options derived from APIKey and BaseURL, and
	// the SDK applies options in order, so an option here wins over those
	// fields when it sets the same setting. Distinct settings do not displace
	// each other: the API key rides an X-Api-Key header and an auth token an
	// authorization header, so a configured key (here, in APIKey, or in
	// ANTHROPIC_API_KEY, which the SDK reads on its own) is still sent
	// alongside whatever Opts configure. A key-less setup such as Bedrock or
	// Vertex routing therefore needs the key unset everywhere, or an
	// [option.WithHeaderDel] for X-Api-Key here, which applies after the key
	// options and so strips the header.
	//
	// Options are opaque, so a non-empty Opts is trusted to carry
	// authentication when no API key is configured.
	Opts []option.RequestOption

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

	client      anthropic.Client // set once by Init, read without the lock after
	mu          sync.Mutex       // guards the three fields below
	initted     bool             // whether Init has already run
	models      []string         // model IDs from the most recent discovery
	lastUpdated time.Time        // when models was last refreshed
}

// Name returns the plugin's name, which is also the provider prefix on the
// action name of every model it serves.
func (a *Anthropic) Name() string {
	return provider
}

// Init prepares the plugin to serve models and registers none: every Claude
// model arrives through [Anthropic.ResolveAction] on first use.
//
// It panics when no authentication is configured (an API key in APIKey or
// ANTHROPIC_API_KEY, an auth token in ANTHROPIC_AUTH_TOKEN, or a non-empty
// Opts, which is trusted to carry its own) and when called twice, both being
// setup mistakes rather than conditions an application can recover from.
func (a *Anthropic) Init(ctx context.Context) []api.Action {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.initted {
		panic("plugin already initialized")
	}

	apiKey := a.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" && os.Getenv("ANTHROPIC_AUTH_TOKEN") == "" && len(a.Opts) == 0 {
		panic("Anthropic requires authentication: set APIKey, set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in the environment, or carry it in Opts")
	}

	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	baseURL := a.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("ANTHROPIC_BASE_URL")
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}

	// The caller's options come last so they win over the fields above.
	opts = append(opts, a.Opts...)

	a.client = anthropic.NewClient(opts...)
	a.initted = true

	return []api.Action{}
}

// ListActions describes every model the API currently advertises. A discovery
// failure is reported as an empty catalog rather than an error, since the
// interface has nowhere to return one.
func (a *Anthropic) ListActions(ctx context.Context) []api.ActionDesc {
	models, err := a.discoveredModels(ctx)
	if err != nil {
		slog.Error("unable to list anthropic models from Anthropic API", "error", err)
		return nil
	}

	actions := []api.ActionDesc{}
	for _, id := range models {
		// A discovered model is named by the same ID the API serves it under,
		// so the action name and the API model ID are identical here.
		actions = append(actions, newModel(a.client, id, a.modelOptions(id)).Desc())
	}
	return actions
}

// ResolveAction builds the model named by a request. Models are the only
// action type this plugin serves.
//
// The ID is passed through to the API untouched. Anthropic resolves an alias
// like claude-opus-4-5 to its current dated release itself, so there is
// nothing to look up and nothing to validate here: the API is the authority on
// whether a model exists, and it answers when the request is made. Building an
// action is local work either way.
func (a *Anthropic) ResolveAction(atype api.ActionType, id string) api.Action {
	if atype != api.ActionTypeModel {
		return nil
	}
	return newModel(a.client, id, a.modelOptions(id))
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

	// Trim before resolving, so a prefixed ID still hits supportedModels.
	id = internal.TrimProvider(provider, id)

	modelOpts := a.modelOptions(id)
	if opts != nil {
		modelOpts = *opts
	}
	return newModel(a.client, id, modelOpts), nil
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
	return genkit.LookupAction(g, api.KeyFromName(api.ActionTypeModel, modelName(id))) != nil
}

// modelOptions returns the capabilities to describe a Claude model with. A
// known ID (see supportedModels) carries curated capabilities and a label; any
// other falls back to dynamicModelOptions, whose label newModel fills in from
// the ID. An entry in [Anthropic.Models] overlays whichever of the two applies.
//
// This is the one source of model capabilities, shared by ListActions and
// ResolveAction, which is what makes a caller's entry authoritative no matter
// which path describes the model first.
func (a *Anthropic) modelOptions(id string) ai.ModelOptions {
	opts, ok := supportedModels[baseModelName(id)]
	if !ok {
		opts = dynamicModelOptions
	}
	if override, ok := internal.LookupOverride(a.Models, provider, id); ok {
		opts = opts.Overlay(override)
	}
	return opts
}

// discoveredModels returns the IDs the API advertises, from cache when it is
// still fresh. Callers are on the request path, so a refresh is worth at most
// one API round trip per modelCacheTTL.
func (a *Anthropic) discoveredModels(ctx context.Context) ([]string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.lastUpdated.IsZero() && time.Since(a.lastUpdated) < modelCacheTTL {
		return a.models, nil
	}

	models, err := listModels(ctx, &a.client)
	if err != nil {
		return nil, err
	}

	a.models = models
	a.lastUpdated = time.Now()
	return models, nil
}

// newModel builds a model action without registering it. Anthropic resolves an
// alias like claude-opus-4-5 to its current dated release itself, so the action
// name is also the model ID requests are sent to.
func newModel(client anthropic.Client, name string, opts ai.ModelOptions) *ai.ModelAction {
	return ant.NewModel(client, provider, name, opts)
}

// modelName builds the action name for a Claude model ID, taking the ID either
// bare or already provider-prefixed. The prefix is applied by concatenation,
// so without the trim an already-prefixed ID would double up and name a model
// that resolves nowhere.
func modelName(id string) string {
	return api.NewName(provider, internal.TrimProvider(provider, id))
}

// baseModelName strips the release date off a dated model ID, mapping it onto
// the alias the curated catalog is keyed by.
func baseModelName(id string) string {
	return dateSuffix.ReplaceAllString(id, "")
}
