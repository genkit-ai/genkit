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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	llamaPluginName = "vertex-model-garden-llama"
)

// Llama is a Genkit plugin for interacting with Meta Llama MaaS models in
// Vertex AI Model Garden. Llama models on Vertex are served via an
// OpenAI-compatible endpoint (`.../endpoints/openapi`) and authenticated with
// a Google OAuth2 access token.
type Llama struct {
	// ProjectID is the Google Cloud project to use for Vertex AI. If empty,
	// the value of the environment variable GOOGLE_CLOUD_PROJECT or
	// GCLOUD_PROJECT will be consulted in that order.
	ProjectID string
	// Location is the Vertex AI location (e.g. "us-central1"). If empty, the
	// value of GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION will be
	// consulted.
	Location string

	mu      sync.Mutex
	initted bool
	oai     compat_oai.OpenAICompatible
}

// Name returns the name of the plugin.
func (l *Llama) Name() string { return llamaPluginName }

// Init initializes the Vertex AI Model Garden Llama plugin and registers all
// known Llama MaaS models. After calling Init you may call DefineModel to
// register additional Llama models.
func (l *Llama) Init(ctx context.Context) []api.Action {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.initted {
		panic("plugin already initialized")
	}

	projectID, location := resolveVertexMaasEnv(l.ProjectID, l.Location)

	ts, err := google.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		panic(fmt.Errorf("modelgarden llama: obtaining default Google token source: %w", err))
	}
	httpClient := oauth2.NewClient(ctx, ts)

	baseURL := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/endpoints/openapi",
		location, projectID, location,
	)

	l.oai.Provider = provider
	l.oai.Opts = []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithHTTPClient(httpClient),
	}

	var actions []api.Action
	actions = append(actions, l.oai.Init(ctx)...)

	for name, opts := range LlamaModels {
		actions = append(actions, l.oai.DefineModel(provider, name, opts).(api.Action))
	}

	l.initted = true
	return actions
}

// LlamaModel returns the Llama [ai.Model] with the given id, or nil if it was
// not defined.
func LlamaModel(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, id))
}

// DefineModel adds a Llama model to the registry.
func (l *Llama) DefineModel(name string, opts *ai.ModelOptions) (ai.Model, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.initted {
		return nil, fmt.Errorf("modelgarden llama: plugin not initialized")
	}
	if opts == nil {
		return nil, fmt.Errorf("DefineModel called with nil ai.ModelOptions")
	}
	return l.oai.DefineModel(provider, name, *opts), nil
}

// resolveVertexMaasEnv resolves project and location from explicit arguments
// with fallback to the conventional environment variables. Panics if neither a
// value nor a fallback env var is set, matching the behaviour of the existing
// Anthropic plugin.
func resolveVertexMaasEnv(projectID, location string) (string, string) {
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if projectID == "" {
			projectID = os.Getenv("GCLOUD_PROJECT")
		}
		if projectID == "" {
			panic("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT in the environment. You can get a project ID at https://console.cloud.google.com/home/dashboard")
		}
	}
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_REGION")
		}
		if location == "" {
			panic("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION in the environment. You can get a location at https://cloud.google.com/vertex-ai/docs/general/locations")
		}
	}
	return projectID, location
}
