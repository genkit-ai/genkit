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
	mistralPluginName = "vertex-model-garden-mistral"
)

// Mistral is a Genkit plugin for interacting with Mistral and Codestral MaaS
// models in Vertex AI Model Garden. The JS equivalent uses the native
// @mistralai/mistralai-gcp SDK; no maintained Mistral-GCP SDK exists for Go,
// so this plugin routes requests through the Vertex MaaS OpenAI-compatible
// endpoint (https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral#openai-compatible).
type Mistral struct {
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
func (m *Mistral) Name() string { return mistralPluginName }

// Init initializes the Vertex AI Model Garden Mistral plugin and registers
// all known Mistral/Codestral MaaS models.
func (m *Mistral) Init(ctx context.Context) []api.Action {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.initted {
		panic("plugin already initialized")
	}

	projectID, location := resolveVertexMaasEnv(m.ProjectID, m.Location)

	ts, err := google.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		panic(fmt.Errorf("modelgarden mistral: obtaining default Google token source: %w", err))
	}
	httpClient := oauth2.NewClient(ctx, ts)

	baseURL := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/endpoints/openapi",
		location, projectID, location,
	)

	m.oai.Provider = provider
	m.oai.Opts = []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithHTTPClient(httpClient),
	}

	var actions []api.Action
	actions = append(actions, m.oai.Init(ctx)...)

	for name, opts := range MistralModels {
		actions = append(actions, m.oai.DefineModel(provider, name, opts).(api.Action))
	}

	m.initted = true
	return actions
}

// MistralModel returns the Mistral/Codestral [ai.Model] with the given id,
// or nil if it was not defined.
func MistralModel(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, id))
}

// DefineModel adds a Mistral model to the registry.
func (m *Mistral) DefineModel(name string, opts *ai.ModelOptions) (ai.Model, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.initted {
		return nil, fmt.Errorf("modelgarden mistral: plugin not initialized")
	}
	if opts == nil {
		return nil, fmt.Errorf("DefineModel called with nil ai.ModelOptions")
	}
	return m.oai.DefineModel(provider, name, *opts), nil
}
