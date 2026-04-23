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

package modelgarden

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"

	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"
)

const pluginName = "vertex-model-garden-anthropic"

// Anthropic is a Genkit plugin for interacting with Anthropic models in Vertex AI Model Garden
type Anthropic struct {
	ProjectID string // Google Cloud project to use for Vertex AI. If empty, the value of the environment variable GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT will be consulted in that order
	Location  string // Location of the Vertex AI service. If empty, the value of the environment variable GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION will be consulted in that order

	client  anthropic.Client // Client for the model garden service
	mu      sync.Mutex       // Mutex to control access
	initted bool             // Whether the plugin has been initialized
}

// Name returns the name of the plugin
func (a *Anthropic) Name() string {
	return pluginName
}

// Init initializes the VertexAI Model Garden for Anthropic plugin and all its known models.
// After calling Init, you may call [DefineModel] to create and register any additional models.
func (a *Anthropic) Init(ctx context.Context) []api.Action {
	if a == nil {
		a = &Anthropic{}
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.initted {
		panic("plugin already initialized")
	}

	projectID := a.ProjectID
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if projectID == "" {
			projectID = os.Getenv("GCLOUD_PROJECT")
			if projectID == "" {
				panic("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT in the environment. You can get a project ID at https://console.cloud.google.com/home/dashboard")
			}
		}
	}

	location := a.Location
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_REGION")
		}
		if location == "" {
			panic("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION in the environment. You can get a location at https://cloud.google.com/vertex-ai/docs/general/locations")
		}
	}

	c := anthropic.NewClient(
		vertex.WithGoogleAuth(ctx, location, projectID, "https://www.googleapis.com/auth/cloud-platform"),
	)
	a.client = c
	a.initted = true

	// Claude models in VertexAI cannot be listed using the Anthropic SDK
	// Models must be defined manually
	var actions []api.Action
	for name, opts := range AnthropicModels {
		model := ant.DefineModel(a.client, provider, name, opts)
		actions = append(actions, model.(api.Action))
	}

	return actions
}

// AnthropicModel returns the [ai.Model] with the given id.
// It returns nil if the model was not defined
func AnthropicModel(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, id))
}

// DefineModel adds the model to the registry
func (a *Anthropic) DefineModel(name string, opts *ai.ModelOptions) (ai.Model, error) {
	if opts == nil {
		return nil, fmt.Errorf("DefineModel called with nil ai.ModelOptions")
	}
	return ant.DefineModel(a.client, provider, name, *opts), nil
}
