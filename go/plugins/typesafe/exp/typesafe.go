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

// Package exp provides a Genkit plugin for TypeSafe AI's System One models,
// of which jev is the first. A System One model does not generate text. It
// evaluates a state against typed questions and returns one typed answer
// per question, with calibrated probabilities, in one round trip of a few
// hundred milliseconds.
//
// The plugin serves jev as a model that speaks only constrained JSON, which
// is the subset of the generate API the model fits exactly. The questions
// are the fields of the output type, declared with the question types of
// go/plugins/systemone/exp, so a decision is one typed generate call with
// nothing more than the model and the state:
//
//	import systemonex "github.com/firebase/genkit/go/plugins/systemone/exp"
//
//	type Triage struct {
//		Department systemonex.Choice[Dept] `json:"department" jsonschema_description:"Which team should handle this?"`
//		IsUrgent   systemonex.Noul         `json:"is_urgent"  jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
//	}
//
//	out, resp, err := genkit.GenerateData[Triage](ctx, g,
//		ai.WithModelName("typesafe/jev-1.13.0"),
//		ai.WithPrompt(ticket))
//	if out.Department.Confidence < 0.6 {
//		// route to a human
//	}
//
// The state is built from the user and model messages: one message is sent
// as its value, a string for a text part or the JSON of a data part;
// several messages are sent as an array of {role, content} records; with
// documents attached the state is {messages, context}. A system message is
// never state: it is instructions, put in front of every question, and for
// the enum format it is the question. Answers come back as one JSON text
// part shaped like the output type, with the response's resolved model
// version and raw answers read with systemonex.ResponseInfo.
//
// The same questions reach jev through TypeSafe's own API or through a
// gateway; see [Endpoint].
//
// This package is a preview: its API may change in any minor release.
package exp

import (
	"cmp"
	"context"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/plugins/internal"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
)

const provider = "typesafe"

// TypeSafe is the plugin. Models are resolved on demand by ID, under the
// typesafe prefix: typesafe/jev-latest, typesafe/jev-1.13.0. Any ID is
// forwarded, since the API accepts a versioned ID it does not list; pin a
// version in production, because confidence thresholds tuned against one
// release do not carry over to the next.
type TypeSafe struct {
	// APIKey authenticates requests. When empty, the environment variable
	// the endpoint names is read: TYPESAFE_API_KEY for TypeSafe's own API,
	// OPENROUTER_API_KEY for OpenRouter, CLOUDFLARE_API_TOKEN for Cloudflare.
	APIKey string

	// BaseURL overrides the endpoint's origin, for a proxy or a private
	// deployment. A proxy that forwards the native protocol under a prefix,
	// such as LiteLLM, is reached by including the prefix here. For the
	// direct endpoint, TYPESAFE_BASE_URL is read when this is empty.
	BaseURL string

	// Endpoint selects the server, [Direct] when nil. See [OpenRouter] and
	// [Cloudflare] for the gateways.
	Endpoint *Endpoint

	// HTTPClient sends the requests. It is the escape hatch to transport
	// settings: timeouts, proxies, and client middleware. When nil, a
	// client with a 30-second timeout is used.
	HTTPClient *http.Client

	// Headers are sent on every request, after the authorization header.
	Headers http.Header

	mu      sync.Mutex
	client  *systemone.Client
	initted bool
}

// Config is the per-request model configuration.
type Config struct {
	// StateJSON parses each text part of the state as JSON, so a prompt
	// template that renders a JSON document produces an object state
	// rather than a string one. A text that is not JSON is an error.
	StateJSON bool `json:"stateJSON,omitzero" jsonschema_description:"Parse each text part of the state as JSON, so a template that renders JSON produces an object state."`

	// Extra is merged over the top-level fields of the request body, last
	// write wins. It reaches fields this package does not model, such as
	// OpenRouter's provider, session_id, and trace. It wins over the fields
	// the plugin builds too, model included, so it is the way to send a
	// model ID the endpoint's translation would refuse. On Cloudflare it
	// merges into input, the native body, not the envelope around it.
	Extra map[string]any `json:"extra,omitempty" jsonschema_description:"Extra top-level request fields, merged over the ones the plugin builds."`
}

// Name implements [api.Plugin].
func (t *TypeSafe) Name() string { return provider }

// Init implements [api.Plugin]. It builds the client and panics without a
// key, or without an account ID on an endpoint that needs one, since every
// request would fail. No actions are registered up front: models resolve
// by name.
func (t *TypeSafe) Init(ctx context.Context) []api.Action {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.initted {
		panic("typesafe: plugin already initialized")
	}

	ep, err := cmp.Or(t.Endpoint, Direct()).ep.WithAccount()
	if err != nil {
		panic("typesafe: " + err.Error())
	}
	apiKey := cmp.Or(t.APIKey, os.Getenv(ep.APIKeyEnv))
	if apiKey == "" {
		panic("typesafe: set APIKey or the " + ep.APIKeyEnv + " environment variable")
	}
	t.client = &systemone.Client{
		HTTP:     cmp.Or(t.HTTPClient, &http.Client{Timeout: systemone.RequestTimeout}),
		BaseURL:  strings.TrimSuffix(cmp.Or(t.BaseURL, os.Getenv(ep.BaseURLEnv), ep.BaseURL), "/"),
		APIKey:   apiKey,
		Headers:  t.Headers,
		Endpoint: ep,
	}
	t.initted = true
	return nil
}

// ListActions implements [api.DynamicPlugin]. TypeSafe's own API lists its
// models; a gateway has no such listing, so the known IDs are advertised.
// A listing failure falls back to the known IDs too, so the Dev UI still
// shows the models while offline.
func (t *TypeSafe) ListActions(ctx context.Context) []api.ActionDesc {
	t.mu.Lock()
	c := t.client
	t.mu.Unlock()
	if c == nil {
		return nil
	}
	ids := c.Endpoint.Models
	if c.Endpoint.ModelsPath != "" {
		if models, err := c.ListModels(ctx); err != nil {
			logger.Debug(ctx, "typesafe: model listing failed, advertising the known models", "error", err)
		} else if len(models) > 0 {
			ids = make([]string, 0, len(models))
			for _, m := range models {
				ids = append(ids, m.Name)
			}
		}
	}
	actions := make([]api.ActionDesc, 0, len(ids))
	for _, id := range ids {
		actions = append(actions, newModel(c, id).Desc())
	}
	return actions
}

// ResolveAction implements [api.DynamicPlugin]. Models are the only action
// type served. The ID is forwarded as given; whether it exists is for the
// endpoint to say when the first request is made.
func (t *TypeSafe) ResolveAction(atype api.ActionType, id string) api.Action {
	if atype != api.ActionTypeModel {
		return nil
	}
	t.mu.Lock()
	c := t.client
	t.mu.Unlock()
	if c == nil {
		return nil
	}
	return newModel(c, id)
}

// ModelRef returns a reference to a jev model with a config, for the places
// that take a reference rather than a name, such as a fallback list. The ID
// may carry the typesafe/ prefix or not. With no config to attach,
// [ai.WithModelName] with the full name is the usual way to pick the model.
func ModelRef(id string, config *Config) ai.ModelRef {
	if config == nil {
		return ai.NewModelRef(modelName(id), nil)
	}
	return ai.NewModelRef(modelName(id), config)
}

func modelName(id string) string {
	return api.NewName(provider, internal.TrimProvider(provider, id))
}

// newModel builds the model action for one ID.
func newModel(c *systemone.Client, id string) *ai.ModelAction {
	return systemone.NewModel(c, modelName(id), id, internal.ProviderLabel("TypeSafe", id), func(cfg *Config) systemone.RequestOptions {
		return systemone.RequestOptions{StateJSON: cfg.StateJSON, Extra: cfg.Extra}
	})
}
