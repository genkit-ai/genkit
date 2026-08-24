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

// Package cohere provides a Genkit plugin for Cohere's Chat v2 and Embed APIs,
// built on the official github.com/cohere-ai/cohere-go/v2 SDK.
package cohere

import (
	"context"
	"os"
	"sync"

	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const (
	provider          = "cohere"
	cohereLabelPrefix = "Cohere"
)

// Cohere implements the dynamic plugin interface: models and embedders are
// resolved on demand via ResolveAction rather than registered up front.
var _ api.DynamicPlugin = (*Cohere)(nil)

// Cohere is a Genkit plugin for interacting with the Cohere API.
type Cohere struct {
	APIKey  string // If not provided, defaults to COHERE_API_KEY, then CO_API_KEY.
	BaseURL string // Optional. If not provided, defaults to COHERE_BASE_URL.

	client  *cohereclient.Client // Cohere client.
	mu      sync.Mutex           // Mutex to control access.
	initted bool                 // Whether the plugin has been initialized.
}

// Name returns the name of the plugin.
func (c *Cohere) Name() string {
	return provider
}

// Init initializes the Cohere plugin. Models and embedders are discovered
// dynamically via [Cohere.ListActions] and [Cohere.ResolveAction].
func (c *Cohere) Init(ctx context.Context) []api.Action {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initted {
		panic("cohere.Init: plugin already initialized")
	}

	apiKey := resolveAPIKey(c.APIKey)
	if apiKey == "" {
		panic("cohere.Init: requires APIKey, COHERE_API_KEY, or CO_API_KEY")
	}

	opts := []option.RequestOption{option.WithToken(apiKey)}

	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("COHERE_BASE_URL")
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}

	c.client = cohereclient.NewClient(opts...)
	c.initted = true

	return []api.Action{}
}

// resolveAPIKey follows the plugin's documented precedence while retaining
// compatibility with the environment variable used natively by cohere-go.
func resolveAPIKey(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if apiKey := os.Getenv("COHERE_API_KEY"); apiKey != "" {
		return apiKey
	}
	return os.Getenv("CO_API_KEY")
}

// ListActions lists the chat models and embedders exposed by the plugin.
func (c *Cohere) ListActions(ctx context.Context) []api.ActionDesc {
	actions := []api.ActionDesc{}

	for name := range cohereChatModels {
		if action, ok := c.newModel(name).(api.Action); ok {
			actions = append(actions, action.Desc())
		}
	}
	for name := range cohereEmbedders {
		if action, ok := c.newEmbedder(name).(api.Action); ok {
			actions = append(actions, action.Desc())
		}
	}

	return actions
}

// ResolveAction resolves a model or embedder action by name. Unknown names are
// resolved with default metadata so newly released models can still be used.
func (c *Cohere) ResolveAction(atype api.ActionType, name string) api.Action {
	switch atype {
	case api.ActionTypeModel:
		if action, ok := c.newModel(name).(api.Action); ok {
			return action
		}
	case api.ActionTypeEmbedder:
		if action, ok := c.newEmbedder(name).(api.Action); ok {
			return action
		}
	}
	return nil
}

// Model returns a previously registered Cohere model.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// IsDefinedModel reports whether a Cohere model is already defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// newModel creates a chat model without registering it.
func (c *Cohere) newModel(name string) ai.Model {
	info := GetModelOptions(name)
	meta := &ai.ModelOptions{
		Label:        info.Label,
		Supports:     info.Supports,
		Versions:     info.Versions,
		Stage:        info.Stage,
		ConfigSchema: core.InferSchemaMap(ChatOptions{}),
	}

	client := c.client
	fn := func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return generate(ctx, client, name, input, cb)
	}

	return ai.NewModel(api.NewName(provider, name), meta, fn)
}
