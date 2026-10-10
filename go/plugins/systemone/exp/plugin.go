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

package exp

import (
	"cmp"
	"context"
	"maps"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/plugins/internal"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
)

// listingTimeout bounds a model listing, which serves the Dev UI and must
// not hold it up when the server is slow. A listing is kept for
// listingTTL, and a failed one is tried again after listingRetry.
const (
	listingTimeout = 5 * time.Second
	listingTTL     = 5 * time.Minute
	listingRetry   = time.Minute
)

// SystemOne is a plugin for any server that speaks System One. It serves
// the server's decision models under Provider, by the IDs the server uses:
// a model the server calls d1 is Provider/d1, and one a gateway calls
// liquid/d1 is Provider/liquid/d1. Any ID resolves, so a model the server
// adds later works without a change here.
//
// For a server with no constructor of its own, set the fields:
//
//	&systemonex.SystemOne{
//		Provider:   "liquid",
//		BaseURL:    "https://api.liquid.ai/decisions",
//		ModelsPath: "/v1/models",
//		APIKey:     os.Getenv("LIQUID_API_KEY"),
//	}
//
// The constructors, such as [TypeSafe], return one set up for a known
// server; its fields can still be changed before the plugin is used. The
// fields are read once, by Init: a change after it has no effect.
type SystemOne struct {
	// Provider names the plugin and prefixes its model names. It is
	// required, and must differ from every other plugin's name, so a host
	// that serves chat models through another plugin needs a name of its
	// own here, such as openrouter-decisions.
	Provider string

	// BaseURL is the server's root, which Path and ModelsPath are joined
	// to. It is required, unless a constructor gives a default.
	BaseURL string

	// Path is the request path. Empty means /v1/systemone, and "/" means
	// BaseURL itself, for a BaseURL that is the whole endpoint.
	Path string

	// ModelsPath is the path of the server's model listing. When empty, no
	// listing is asked for, and the Dev UI shows the models in Models.
	ModelsPath string

	// APIKey is sent as a bearer token. A constructor reads it from the
	// environment variable it names when this is empty; a server that takes
	// no key, such as a local one, is reached with none.
	APIKey string

	// Models describes the models known ahead, keyed by the ID the server
	// uses or by the full model name, Provider/ID; a server ID that itself
	// starts with Provider/ takes the full name. They are listed in the Dev
	// UI whether or not the server has a listing. Any other ID resolves
	// too, as a model with no description.
	Models map[string]ModelSpec

	// HTTPClient sends the requests. It is the escape hatch to transport
	// settings: timeouts, proxies, and client middleware. When nil, a
	// client with a 30-second timeout is used.
	HTTPClient *http.Client

	// Headers are sent on every request, after the authorization header.
	Headers http.Header

	// Route, for a server whose wire is not the native one, builds the
	// request it takes. It gets the model ID and the native body, which it
	// may change, and returns the path to append to Path, which is joined
	// with a slash when it does not start with one, and the body to send.
	// When nil, the native body goes to Path.
	Route func(model string, body map[string]any) (path string, out any, err error)

	// Unwrap, for a server that wraps its responses, returns the native
	// response from the body the server sent, such as by taking it out of
	// an envelope. An error it returns is reported under Provider, as
	// UNKNOWN unless it carries a Genkit status. When nil, the body is read
	// as the native response.
	Unwrap func(body []byte) ([]byte, error)

	// preset is what a constructor sets that a caller cannot.
	preset preset

	mu     sync.Mutex
	client *systemone.Client
	// models is Models as Init read it, keyed by the server's ID.
	models map[string]ModelSpec

	// listMu serializes model listings, so concurrent ones share a fetch;
	// listed is the last list of actions, and listedUntil when it goes
	// stale.
	listMu      sync.Mutex
	listed      []api.ActionDesc
	listedUntil time.Time
}

// ModelSpec describes one decision model.
type ModelSpec struct {
	// Label names the model in the Dev UI. Empty means one built from the
	// ID.
	Label string
}

// preset is what a constructor sets for its server beyond the public
// fields: where its key and base URL come from.
type preset struct {
	// label names the server in model labels; empty means Provider.
	label string
	// apiKeyEnv names the variable the key is read from. A server that
	// names one requires a key.
	apiKeyEnv string
	// baseURL is the default base URL, and baseURLEnv the variable that
	// overrides it.
	baseURL    string
	baseURLEnv string
}

// Config is the per-request configuration of a decision model: StateJSON
// parses each text part of the state as JSON, so a prompt template that
// renders a JSON document produces an object state, and Extra merges
// fields into the request body that this package does not model, such as
// a gateway's session_id or trace.
type Config = systemone.Config

// Name implements [api.Plugin].
func (s *SystemOne) Name() string { return s.provider() }

// Init implements [api.Plugin]. It builds the client, and panics without a
// provider name or a base URL, or without a key for a server that requires
// one, since every request would fail, and when Models names one model
// twice. No actions are registered up front:
// models resolve by name.
func (s *SystemOne) Init(ctx context.Context) []api.Action {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Provider == "" {
		panic("systemone: set Provider, the plugin's name and the prefix of its models")
	}
	if s.client != nil {
		panic(s.Provider + ": plugin already initialized")
	}
	apiKey := s.APIKey
	if env := s.preset.apiKeyEnv; env != "" {
		if apiKey = cmp.Or(apiKey, os.Getenv(env)); apiKey == "" {
			panic(s.Provider + ": set APIKey or the " + env + " environment variable")
		}
	}
	baseURL := cmp.Or(s.BaseURL, s.preset.baseURL)
	if env := s.preset.baseURLEnv; env != "" && s.BaseURL == "" {
		baseURL = cmp.Or(os.Getenv(env), baseURL)
	}
	if baseURL == "" {
		panic(s.Provider + ": set BaseURL, the root of the server's System One API")
	}
	path := rooted(cmp.Or(s.Path, "/v1/systemone"))
	if path == "/" {
		path = ""
	}
	s.models = make(map[string]ModelSpec, len(s.Models))
	for key, spec := range s.Models {
		id := internal.TrimProvider(s.Provider, key)
		if _, ok := s.models[id]; ok {
			panic(s.Provider + ": Models names " + id + " twice, by its ID and by its full name")
		}
		s.models[id] = spec
	}
	s.client = &systemone.Client{
		HTTP:    cmp.Or(s.HTTPClient, &http.Client{Timeout: systemone.RequestTimeout}),
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		APIKey:  apiKey,
		Headers: s.Headers.Clone(),
		Endpoint: &systemone.Endpoint{
			Name:       s.Provider,
			Path:       path,
			ModelsPath: rooted(s.ModelsPath),
			Route:      s.Route,
			Unwrap:     s.Unwrap,
		},
	}
	return nil
}

// rooted gives a non-empty path the leading slash that joins it to the
// base URL.
func rooted(path string) string {
	if path == "" || strings.HasPrefix(path, "/") {
		return path
	}
	return "/" + path
}

// ListActions implements [api.DynamicPlugin]. It lists the models in
// Models together with the ones the server's listing names. A listing that
// fails or takes too long leaves the models in Models, so the Dev UI still
// shows them while offline. A listing is kept for a few minutes, and a
// failed one is not tried again for a minute, so the Dev UI does not wait
// on the server each time it lists actions.
func (s *SystemOne) ListActions(ctx context.Context) []api.ActionDesc {
	c := s.initialized()
	if c == nil {
		return nil
	}
	s.listMu.Lock()
	defer s.listMu.Unlock()
	if s.listed != nil && time.Now().Before(s.listedUntil) {
		return slices.Clone(s.listed)
	}
	listed, fresh := s.listing(ctx, c)
	ids := append(slices.Collect(maps.Keys(s.models)), listed...)
	slices.Sort(ids)
	ids = slices.Compact(ids)
	actions := make([]api.ActionDesc, 0, len(ids))
	for _, id := range ids {
		actions = append(actions, s.newModel(c, id).Desc())
	}
	s.listed, s.listedUntil = actions, time.Now().Add(fresh)
	return slices.Clone(actions)
}

// listing returns the IDs the server's listing names and how long they
// stay fresh. The fetch outlives a caller that gives up, since its result
// serves every caller until it goes stale.
func (s *SystemOne) listing(ctx context.Context, c *systemone.Client) ([]string, time.Duration) {
	if c.Endpoint.ModelsPath == "" {
		return nil, listingTTL
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), listingTimeout)
	defer cancel()
	models, err := c.ListModels(ctx)
	if err != nil {
		logger.Debug(ctx, c.Endpoint.Name+": model listing failed, advertising the configured models", "error", err)
		return nil, listingRetry
	}
	ids := make([]string, 0, len(models))
	for _, m := range models {
		if id := m.Model(); id != "" {
			ids = append(ids, id)
		}
	}
	return ids, listingTTL
}

// ResolveAction implements [api.DynamicPlugin]. Models are the only action
// type served. The ID is forwarded as given; whether it exists is for the
// server to say when the first request is made.
func (s *SystemOne) ResolveAction(atype api.ActionType, id string) api.Action {
	if atype != api.ActionTypeModel {
		return nil
	}
	c := s.initialized()
	if c == nil {
		return nil
	}
	return s.newModel(c, id)
}

// ModelRef returns a reference to one of the plugin's models with a
// config, for the places that take a reference rather than a name, such as
// a fallback list. The ID is the server's or the full model name, as a key
// of Models is: ModelRef("jev-1.13.0", cfg) and
// ModelRef("typesafe/jev-1.13.0", cfg) on [TypeSafe] are both
// typesafe/jev-1.13.0. With no config to attach, [ai.WithModelName] with
// the full name is the usual way to pick the model.
func (s *SystemOne) ModelRef(id string, config *Config) ai.ModelRef {
	provider := s.provider()
	name := api.NewName(provider, internal.TrimProvider(provider, id))
	if config == nil {
		return ai.NewModelRef(name, nil)
	}
	return ai.NewModelRef(name, config)
}

// provider is Provider as Init read it, or as it is until Init runs.
func (s *SystemOne) provider() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.client != nil {
		return s.client.Endpoint.Name
	}
	return s.Provider
}

func (s *SystemOne) initialized() *systemone.Client {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.client
}

// newModel builds the model action for one ID.
func (s *SystemOne) newModel(c *systemone.Client, id string) *ai.ModelAction {
	label := s.models[id].Label
	if label == "" {
		label = internal.ProviderLabel(cmp.Or(s.preset.label, c.Endpoint.Name), id)
	}
	return systemone.NewModel(c, api.NewName(c.Endpoint.Name, id), id, label)
}

// TypeSafe is TypeSafe AI's own API, which serves jev, the first System One
// model. Its models are typesafe/<id>, such as typesafe/jev-1.13.0. The
// API key comes from TYPESAFE_API_KEY and the base URL from
// TYPESAFE_BASE_URL, the variables TypeSafe's SDKs read. Pin a version in
// production: confidence thresholds tuned against one release do not carry
// over to the next.
func TypeSafe() *SystemOne {
	return &SystemOne{
		Provider:   "typesafe",
		ModelsPath: "/v1/models",
		Models:     map[string]ModelSpec{"jev-latest": {}, "jev-1.13.0": {}},
		preset: preset{
			label:      "TypeSafe",
			apiKeyEnv:  "TYPESAFE_API_KEY",
			baseURL:    "https://api.typesafe.ai",
			baseURLEnv: "TYPESAFE_BASE_URL",
		},
	}
}

// OpenRouter is OpenRouter's Decisions API, which serves several vendors'
// decision models under one key. Its models are openrouter-decisions/<ID>
// by OpenRouter's ID, such as openrouter-decisions/liquid/d1 and
// openrouter-decisions/typesafe/jev-1.13. The name is not openrouter, which
// the plugin for OpenRouter's chat models has, so both serve one app. The
// API key comes from OPENROUTER_API_KEY. The Dev UI lists the models
// OpenRouter lists as decision models, and the best known of them when the
// listing fails, and a request's cost is reported in the response's usage.
func OpenRouter() *SystemOne {
	return &SystemOne{
		Provider:   "openrouter-decisions",
		Path:       "/api/alpha/decisions",
		ModelsPath: "/api/v1/models?output_modalities=decisions",
		Models: map[string]ModelSpec{
			"cloudflare/clef":       {},
			"cloudflare/clef-flash": {},
			"liquid/d1":             {},
			"typesafe/jev-1.13":     {},
			"~typesafe/jev-latest":  {},
		},
		preset: preset{
			label:     "OpenRouter",
			apiKeyEnv: "OPENROUTER_API_KEY",
			baseURL:   "https://openrouter.ai",
		},
	}
}
