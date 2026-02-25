// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/httptransport"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"google.golang.org/genai"
)

const (
	googleAIProvider = "googleai"
	vertexAIProvider = "vertexai"

	googleAILabelPrefix = "Google AI"
	vertexAILabelPrefix = "Vertex AI"
)

var (
	defaultGeminiOpts = ai.ModelOptions{
		Supports: &Multimodal,
		Versions: []string{},
		Stage:    ai.ModelStageUnstable,
	}

	defaultImagenOpts = ai.ModelOptions{
		Supports: &Media,
		Versions: []string{},
		Stage:    ai.ModelStageUnstable,
	}

	defaultEmbedOpts = ai.EmbedderOptions{
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
		Dimensions: 768,
	}
)

// GoogleAI is a Genkit plugin for interacting with the Google AI service.
type GoogleAI struct {
	APIKey string // API key to access the service. If empty, the values of the environment variables GEMINI_API_KEY or GOOGLE_API_KEY will be consulted, in that order.

	gclient *genai.Client // Client for the Google AI service.
	mu      sync.Mutex    // Mutex to control access.
	initted bool          // Whether the plugin has been initialized.
}

// VertexAI is a Genkit plugin for interacting with the Google Vertex AI service.
type VertexAI struct {
	ProjectID string // Google Cloud project to use for Vertex AI. If empty, the value of the environment variable GOOGLE_CLOUD_PROJECT will be consulted.
	Location  string // Location of the Vertex AI service. If empty, GOOGLE_CLOUD_LOCATION and GOOGLE_CLOUD_REGION environment variables will be consulted, in that order.

	gclient *genai.Client // Client for the Vertex AI service.
	mu      sync.Mutex    // Mutex to control access.
	initted bool          // Whether the plugin has been initialized.
}

// Name returns the name of the plugin.
func (ga *GoogleAI) Name() string {
	return googleAIProvider
}

// Name returns the name of the plugin.
func (v *VertexAI) Name() string {
	return vertexAIProvider
}

// Init initializes the Google AI plugin and all known models and embedders.
// After calling Init, you may call [DefineModel] and [DefineEmbedder] to create
// and register any additional generative models and embedders
func (ga *GoogleAI) Init(ctx context.Context) []api.Action {
	if ga == nil {
		ga = &GoogleAI{}
	}
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if ga.initted {
		panic("plugin already initialized")
	}

	apiKey := ga.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GOOGLE_API_KEY")
		}
		if apiKey == "" {
			panic("Google AI requires setting GEMINI_API_KEY or GOOGLE_API_KEY in the environment. You can get an API key at https://ai.google.dev")
		}
	}

	gc := genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Transport: otelhttp.NewTransport(http.DefaultTransport),
		},
		HTTPOptions: genai.HTTPOptions{
			Headers: genkitClientHeader,
		},
	}

	client, err := genai.NewClient(ctx, &gc)
	if err != nil {
		panic(fmt.Errorf("GoogleAI.Init: %w", err))
	}
	ga.gclient = client
	ga.initted = true

	return []api.Action{}
}

// Init initializes the VertexAI plugin and all known models and embedders.
// After calling Init, you may call [DefineModel] and [DefineEmbedder] to create
// and register any additional generative models and embedders
func (v *VertexAI) Init(ctx context.Context) []api.Action {
	if v == nil {
		v = &VertexAI{}
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.initted {
		panic("plugin already initialized")
	}

	projectID := v.ProjectID
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if projectID == "" {
			panic("Vertex AI requires setting GOOGLE_CLOUD_PROJECT in the environment. You can get a project ID at https://console.cloud.google.com/home/dashboard")
		}
	}

	location := v.Location
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_REGION")
		}
		if location == "" {
			panic("Vertex AI requires setting GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION in the environment. You can get a location at https://cloud.google.com/vertex-ai/docs/general/locations")
		}
	}
	cred, err := credentials.DetectDefault(&credentials.DetectOptions{
		Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
	})
	if err != nil {
		panic(fmt.Errorf("failed to find default credentials: %w", err))
	}
	quotaProjectID, err := cred.QuotaProjectID(ctx)
	if err != nil {
		panic(fmt.Errorf("failed to get quota project ID: %v", quotaProjectID))
	}
	httpClient, err := httptransport.NewClient(&httptransport.Options{
		Credentials:      cred,
		BaseRoundTripper: otelhttp.NewTransport(http.DefaultTransport),
		Headers: http.Header{
			"X-Goog-User-Project": []string{quotaProjectID},
		},
	})
	if err != nil {
		panic(fmt.Errorf("failed to create http client: %w", err))
	}

	// Project and Region values gets validated by genai SDK upon client creation
	gc := genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    projectID,
		Location:   location,
		HTTPClient: httpClient,
		HTTPOptions: genai.HTTPOptions{
			Headers: genkitClientHeader,
		},
	}

	client, err := genai.NewClient(ctx, &gc)
	if err != nil {
		panic(fmt.Errorf("VertexAI.Init: %w", err))
	}
	v.gclient = client
	v.initted = true

	return []api.Action{}
}

// DefineModel defines an unknown model with the given name.
// The second argument describes the capability of the model.
// Use [IsDefinedModel] to determine if a model is already defined.
// After [Init] is called, only the known models are defined.
func (ga *GoogleAI) DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if !ga.initted {
		return nil, errors.New("GoogleAI plugin not initialized")
	}
	models, err := listModels(googleAIProvider)
	if err != nil {
		return nil, err
	}

	if opts == nil {
		var ok bool
		modelOpts, ok := models[name]
		if !ok {
			return nil, fmt.Errorf("GoogleAI.DefineModel: called with unknown model %q and nil ModelOptions", name)
		}
		opts = &modelOpts
	}

	return newModel(ga.gclient, name, *opts), nil
}

// DefineModel defines an unknown model with the given name.
// The second argument describes the capability of the model.
// Use [IsDefinedModel] to determine if a model is already defined.
// After [Init] is called, only the known models are defined.
func (v *VertexAI) DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}
	models, err := listModels(vertexAIProvider)
	if err != nil {
		return nil, err
	}

	if opts == nil {
		var ok bool
		modelOpts, ok := models[name]
		if !ok {
			return nil, fmt.Errorf("VertexAI.DefineModel: called with unknown model %q and nil ModelOptions", name)
		}
		opts = &modelOpts
	}

	return newModel(v.gclient, name, *opts), nil
}

// DefineEmbedder defines an embedder with a given name.
func (ga *GoogleAI) DefineEmbedder(g *genkit.Genkit, name string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if !ga.initted {
		return nil, errors.New("GoogleAI plugin not initialized")
	}
	return newEmbedder(ga.gclient, name, embedOpts), nil
}

// DefineEmbedder defines an embedder with a given name.
func (v *VertexAI) DefineEmbedder(g *genkit.Genkit, name string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}
	return newEmbedder(v.gclient, name, embedOpts), nil
}

// IsDefinedEmbedder reports whether the named [Embedder] is defined by this plugin.
func (ga *GoogleAI) IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, api.NewName(googleAIProvider, name)) != nil
}

// IsDefinedEmbedder reports whether the named [Embedder] is defined by this plugin.
func (v *VertexAI) IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, api.NewName(vertexAIProvider, name)) != nil
}

// ModelRef creates a new ModelRef for a Google Gen AI model with the given name and configuration.
func ModelRef(name string, config *genai.GenerateContentConfig) ai.ModelRef {
	return ai.NewModelRef(name, config)
}

// GoogleAIModelRef creates a new ModelRef for a Google AI model with the given ID and configuration.
func GoogleAIModelRef(id string, config *genai.GenerateContentConfig) ai.ModelRef {
	return ai.NewModelRef(googleAIProvider+"/"+id, config)
}

// VertexAIModelRef creates a new ModelRef for a Vertex AI model with the given ID and configuration.
func VertexAIModelRef(id string, config *genai.GenerateContentConfig) ai.ModelRef {
	return ai.NewModelRef(vertexAIProvider+"/"+id, config)
}

// GoogleAIModel returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func GoogleAIModel(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(googleAIProvider, name))
}

// VertexAIModel returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func VertexAIModel(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(vertexAIProvider, name))
}

// GoogleAIEmbedder returns the [ai.Embedder] with the given name.
// It returns nil if the embedder was not defined.
func GoogleAIEmbedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(googleAIProvider, name))
}

// VertexAIEmbedder returns the [ai.Embedder] with the given name.
// It returns nil if the embedder was not defined.
func VertexAIEmbedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(vertexAIProvider, name))
}

// ListActions lists all the actions supported by the Google AI plugin.
func (ga *GoogleAI) ListActions(ctx context.Context) []api.ActionDesc {
	models, err := listGenaiModels(ctx, ga.gclient)
	if err != nil {
		return nil
	}

	actions := []api.ActionDesc{}

	// Generative models.
	for _, name := range models.gemini {
		var opts ai.ModelOptions
		if knownOpts, ok := supportedGeminiModels[name]; ok {
			opts = knownOpts
			opts.Label = fmt.Sprintf("%s - %s", googleAILabelPrefix, opts.Label)
		} else {
			opts = defaultGeminiOpts
			opts.Label = fmt.Sprintf("%s - %s", googleAILabelPrefix, name)
		}

		model := newModel(ga.gclient, name, opts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Imagen models.
	for _, name := range models.imagen {
		var opts ai.ModelOptions
		if knownOpts, ok := supportedImagenModels[name]; ok {
			opts = knownOpts
			opts.Label = fmt.Sprintf("%s - %s", googleAILabelPrefix, opts.Label)
		} else {
			opts = defaultImagenOpts
			opts.Label = fmt.Sprintf("%s - %s", googleAILabelPrefix, name)
		}

		model := newModel(ga.gclient, name, opts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Embedders.
	for _, e := range models.embedders {
		var embedOpts ai.EmbedderOptions
		if knownOpts, ok := googleAIEmbedderConfig[e]; ok {
			embedOpts = knownOpts
		} else {
			embedOpts = defaultEmbedOpts
			embedOpts.Label = fmt.Sprintf("%s - %s", googleAILabelPrefix, e)
		}

		embedder := newEmbedder(ga.gclient, e, &embedOpts)
		if actionDef, ok := embedder.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	return actions
}

// ResolveAction resolves an action with the given name.
func (ga *GoogleAI) ResolveAction(atype api.ActionType, name string) api.Action {
	switch atype {
	case api.ActionTypeEmbedder:
		return newEmbedder(ga.gclient, name, &ai.EmbedderOptions{}).(api.Action)
	case api.ActionTypeModel:
		var supports *ai.ModelSupports
		var config any

		// TODO: Add veo case.
		switch {
		case strings.Contains(name, "imagen"):
			supports = &Media
			config = &genai.GenerateImagesConfig{}
		default:
			supports = &Multimodal
			config = &genai.GenerateContentConfig{}
		}

		return newModel(ga.gclient, name, ai.ModelOptions{
			Label:        fmt.Sprintf("%s - %s", googleAILabelPrefix, name),
			Stage:        ai.ModelStageStable,
			Versions:     []string{},
			Supports:     supports,
			ConfigSchema: configToMap(config),
		}).(api.Action)
	case api.ActionTypeBackgroundModel:
		// Handle VEO models as background models
		if strings.HasPrefix(name, "veo") {
			veoModel := newVeoModel(ga.gclient, name, ai.ModelOptions{
				Label:    fmt.Sprintf("%s - %s", googleAILabelPrefix, name),
				Stage:    ai.ModelStageStable,
				Versions: []string{},
				Supports: &ai.ModelSupports{
					Media:       true,
					Multiturn:   false,
					Tools:       false,
					SystemRole:  false,
					Output:      []string{"media"},
					LongRunning: true,
				},
			})
			actionName := fmt.Sprintf("%s/%s", googleAIProvider, name)
			return core.NewAction(actionName, api.ActionTypeBackgroundModel, nil, nil,
				func(ctx context.Context, input *ai.ModelRequest) (*core.Operation[*ai.ModelResponse], error) {
					op, err := veoModel.Start(ctx, input)
					if err != nil {
						return nil, err
					}
					op.Action = api.KeyFromName(api.ActionTypeBackgroundModel, actionName)
					return op, nil
				})
		}
		return nil
	case api.ActionTypeCheckOperation:
		// Handle VEO model check operations
		if strings.HasPrefix(name, "veo") {
			veoModel := newVeoModel(ga.gclient, name, ai.ModelOptions{
				Label:    fmt.Sprintf("%s - %s", googleAILabelPrefix, name),
				Stage:    ai.ModelStageStable,
				Versions: []string{},
				Supports: &ai.ModelSupports{
					Media:       true,
					Multiturn:   false,
					Tools:       false,
					SystemRole:  false,
					Output:      []string{"media"},
					LongRunning: true,
				},
			})

			actionName := fmt.Sprintf("%s/%s", googleAIProvider, name)
			return core.NewAction(actionName, api.ActionTypeCheckOperation,
				map[string]any{"description": fmt.Sprintf("Check status of %s operation", name)}, nil,
				func(ctx context.Context, op *core.Operation[*ai.ModelResponse]) (*core.Operation[*ai.ModelResponse], error) {
					updatedOp, err := veoModel.Check(ctx, op)
					if err != nil {
						return nil, err
					}
					updatedOp.Action = api.KeyFromName(api.ActionTypeBackgroundModel, actionName)
					return updatedOp, nil
				})
		}
		return nil
	}
	return nil
}

// ListActions lists all the actions supported by the Vertex AI plugin.
func (v *VertexAI) ListActions(ctx context.Context) []api.ActionDesc {
	models, err := listGenaiModels(ctx, v.gclient)
	if err != nil {
		return nil
	}

	actions := []api.ActionDesc{}

	// Gemini generative models.
	for _, name := range models.gemini {
		var opts ai.ModelOptions
		if knownOpts, ok := supportedGeminiModels[name]; ok {
			opts = knownOpts
			opts.Label = fmt.Sprintf("%s - %s", vertexAILabelPrefix, opts.Label)
		} else {
			opts = defaultGeminiOpts
			opts.Label = fmt.Sprintf("%s - %s", vertexAILabelPrefix, name)
		}

		model := newModel(v.gclient, name, opts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Imagen models.
	for _, name := range models.imagen {
		var opts ai.ModelOptions
		if knownOpts, ok := supportedImagenModels[name]; ok {
			opts = knownOpts
			opts.Label = fmt.Sprintf("%s - %s", vertexAILabelPrefix, opts.Label)
		} else {
			opts = defaultImagenOpts
			opts.Label = fmt.Sprintf("%s - %s", vertexAILabelPrefix, name)
		}

		model := newModel(v.gclient, name, opts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Embedders.
	for _, e := range models.embedders {
		var embedOpts ai.EmbedderOptions
		if knownOpts, ok := googleAIEmbedderConfig[e]; ok {
			embedOpts = knownOpts
		} else {
			embedOpts = defaultEmbedOpts
			embedOpts.Label = fmt.Sprintf("%s - %s", vertexAILabelPrefix, e)
		}

		embedder := newEmbedder(v.gclient, e, &embedOpts)
		if actionDef, ok := embedder.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	return actions
}

// ResolveAction resolves an action with the given name.
func (v *VertexAI) ResolveAction(atype api.ActionType, id string) api.Action {
	switch atype {
	case api.ActionTypeEmbedder:
		return newEmbedder(v.gclient, id, &ai.EmbedderOptions{}).(api.Action)
	case api.ActionTypeModel:
		var supports *ai.ModelSupports
		var config any

		// TODO: Add veo case.
		switch {
		case strings.Contains(id, "imagen"):
			supports = &Media
			config = &genai.GenerateImagesConfig{}
		default:
			supports = &Multimodal
			config = &genai.GenerateContentConfig{}
		}

		return newModel(v.gclient, id, ai.ModelOptions{
			Label:        fmt.Sprintf("%s - %s", vertexAILabelPrefix, id),
			Stage:        ai.ModelStageStable,
			Versions:     []string{},
			Supports:     supports,
			ConfigSchema: configToMap(config),
		}).(api.Action)
	}
	return nil
}
