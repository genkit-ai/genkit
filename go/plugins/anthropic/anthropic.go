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
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"
	"github.com/invopop/jsonschema"
)

const (
	provider             = "anthropic"
	anthropicLabelPrefix = "Anthropic"
)

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

// DefineModel defines an unknown model with the given name.
// The second argument describes the capability of the model.
// Use [IsDefinedModel] to determine if a model is already defined.
// After [Init] is called, only the known models are defined.
func (a *Anthropic) DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	return ant.DefineModel(a.aclient, provider, name, *opts), nil
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
		model := newModel(a.aclient, name, name, defaultClaudeOpts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	return actions
}

// Model returns a previously registered model
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// IsDefinedModel returns whether a model is already defined
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
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
		return newModel(a.aclient, id, realID, ai.ModelOptions{
			Label:    fmt.Sprintf("%s - %s", anthropicLabelPrefix, id),
			Stage:    ai.ModelStageStable,
			Versions: []string{},
			Supports: defaultClaudeOpts.Supports,
		}).(api.Action)
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

// newModel creates a model wihout registering it
func newModel(client anthropic.Client, name, apiModelName string, opts ai.ModelOptions) ai.Model {
	config := &anthropic.MessageNewParams{}

	meta := &ai.ModelOptions{
		Label:        opts.Label,
		Supports:     opts.Supports,
		Versions:     opts.Versions,
		ConfigSchema: configToMap(config),
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
		return ant.Generate(ctx, client, targetModel, input, cb)
	}

	return ai.NewModel(api.NewName(provider, name), meta, fn)
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
	// Suffix must be exactly 8 digits (YYYYMMDD)
	dateSuffix := regexp.MustCompile(`^\d{8}$`)

	for _, m := range availableModels {
		if strings.HasPrefix(m, prefix) {
			suffix := strings.TrimPrefix(m, prefix)
			if dateSuffix.MatchString(suffix) {
				if m > bestMatch {
					bestMatch = m
				}
			}
		}
	}

	if bestMatch != "" {
		return bestMatch, true
	}

	return "", false
}

// configToMap converts a config struct to a map[string]any.
func configToMap(config any) map[string]any {
	r := jsonschema.Reflector{
		DoNotReference:             true, // Prevent $ref usage
		AllowAdditionalProperties:  false,
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: true,
	}
	// The anthropic SDK uses a number of wrapper types for float, int, etc.
	// By default, jsonschema will treat these as objects, but we want to
	// treat them as their underlying primitive types.
	r.Mapper = func(r reflect.Type) *jsonschema.Schema {
		if r.Name() == "Opt[float64]" {
			return &jsonschema.Schema{
				Type: "number",
			}
		}
		if r.Name() == "Opt[int64]" {
			return &jsonschema.Schema{
				Type: "integer",
			}
		}
		if r.Name() == "Opt[string]" {
			return &jsonschema.Schema{
				Type: "string",
			}
		}
		if r.Name() == "Opt[bool]" {
			return &jsonschema.Schema{
				Type: "boolean",
			}
		}
		return nil
	}
	schema := r.Reflect(config)
	result := base.SchemaAsMap(schema)

	return result
}
