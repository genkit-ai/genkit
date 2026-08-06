// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sync"

	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/httptransport"
	"github.com/firebase/genkit/go/ai"
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

// GoogleAI is a Genkit plugin for interacting with the Google AI service.
type GoogleAI struct {
	APIKey string // API key to access the service. If empty, the values of the environment variables GEMINI_API_KEY or GOOGLE_API_KEY will be consulted, in that order.

	gclient *genai.Client // Client for the Google AI service.
	mu      sync.Mutex    // Mutex to control access.
	initted bool          // Whether the plugin has been initialized.
}

// VertexAI is a Genkit plugin for interacting with the Google Vertex AI service.
type VertexAI struct {
	ProjectID  string // Google Cloud project to use for Vertex AI. If empty, the value of the environment variable GOOGLE_CLOUD_PROJECT will be consulted.
	Location   string // Location of the Vertex AI service. If empty, GOOGLE_CLOUD_LOCATION and GOOGLE_CLOUD_REGION environment variables will be consulted, in that order. Accepts a regional location (e.g. "us-central1"), a multi-region location ("us" or "eu"), or "global".
	APIVersion string // API version to use ("v1" or "v1beta1"). If empty, the genai SDK default (v1beta1) is used. Can be overridden per-request via config.HTTPOptions.APIVersion.

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

	switch v.APIVersion {
	case "", "v1", "v1beta1":
	default:
		panic(fmt.Sprintf("Vertex AI APIVersion must be %q or %q, got %q", "v1", "v1beta1", v.APIVersion))
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
			Headers:    genkitClientHeader,
			APIVersion: v.APIVersion,
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

// buildModel builds an unregistered Gemini model. A nil opts takes the
// capabilities the plugin knows for that ID.
func (ga *GoogleAI) buildModel(id string, opts *ai.ModelOptions) (*ai.ModelAction, error) {
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
		modelOpts, ok := models[id]
		if !ok {
			return nil, fmt.Errorf("GoogleAI: called with unknown model %q and nil ModelOptions", id)
		}
		opts = &modelOpts
	}

	return newModel(ga.gclient, id, *opts), nil
}

// RegisterModel registers a model with g and returns it. The plugin supplies
// the implementation; opts describes what the model supports, and a nil opts
// takes the capabilities the plugin knows for that ID, which makes an
// unknown ID with a nil opts an error.
//
// Registering an ID that is already registered panics; [GoogleAI.Init]
// registers every known model, so register a model before its first use or
// guard with [GoogleAI.IsDefinedModel].
func (ga *GoogleAI) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	model, err := ga.buildModel(id, opts)
	if err != nil {
		return nil, err
	}
	genkit.RegisterAction(g, model)
	return model, nil
}

// DefineModel defines an unknown model with the given ID.
// The second argument describes the capability of the model.
//
// Deprecated: use [GoogleAI.RegisterModel]. This method builds the model and
// ignores g. Generation resolves a model from its name, so passing the result
// to ai.WithModel contributes only that name and serves the request with a
// model resolved from it instead; registering it with [genkit.RegisterAction]
// is what makes these capabilities the ones used.
func (ga *GoogleAI) DefineModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return ga.buildModel(id, opts)
}

// buildModel builds an unregistered Gemini model. A nil opts takes the
// capabilities the plugin knows for that ID.
//
// Tuned Gemini endpoints are accepted in either the short form
// `endpoints/ID` or the full resource path
// `projects/PROJECT/locations/LOCATION/endpoints/ID`. When opts is nil the
// caller gets the default Gemini capability set.
func (v *VertexAI) buildModel(id string, opts *ai.ModelOptions) (*ai.ModelAction, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}

	if opts == nil {
		if isTunedGeminiName(id) {
			defaults := GetModelOptions(id, vertexAIProvider)
			opts = &defaults
		} else {
			models, err := listModels(vertexAIProvider)
			if err != nil {
				return nil, err
			}
			modelOpts, ok := models[id]
			if !ok {
				return nil, fmt.Errorf("VertexAI: called with unknown model %q and nil ModelOptions", id)
			}
			opts = &modelOpts
		}
	}

	return newModel(v.gclient, id, *opts), nil
}

// RegisterModel registers a model with g and returns it; see
// [GoogleAI.RegisterModel]. Tuned Gemini endpoints are accepted in either the
// short form `endpoints/ID` or the full resource path
// `projects/PROJECT/locations/LOCATION/endpoints/ID`, and take the default
// Gemini capability set when opts is nil.
func (v *VertexAI) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	model, err := v.buildModel(id, opts)
	if err != nil {
		return nil, err
	}
	genkit.RegisterAction(g, model)
	return model, nil
}

// DefineModel defines an unknown model with the given ID.
// The second argument describes the capability of the model.
//
// Deprecated: use [VertexAI.RegisterModel]. This method builds the model and
// ignores g. Generation resolves a model from its name, so passing the result
// to ai.WithModel contributes only that name and serves the request with a
// model resolved from it instead; registering it with [genkit.RegisterAction]
// is what makes these capabilities the ones used.
func (v *VertexAI) DefineModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return v.buildModel(id, opts)
}

// buildEmbedder builds an unregistered embedder.
func (ga *GoogleAI) buildEmbedder(id string, embedOpts *ai.EmbedderOptions) (*ai.EmbedderAction, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if !ga.initted {
		return nil, errors.New("GoogleAI plugin not initialized")
	}
	return newEmbedder(ga.gclient, id, embedOpts), nil
}

// RegisterEmbedder registers an embedder with g and returns it. Registering
// an ID that is already registered panics, so guard with
// [GoogleAI.IsDefinedEmbedder] when in doubt.
func (ga *GoogleAI) RegisterEmbedder(g *genkit.Genkit, id string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	embedder, err := ga.buildEmbedder(id, embedOpts)
	if err != nil {
		return nil, err
	}
	genkit.RegisterAction(g, embedder)
	return embedder, nil
}

// DefineEmbedder defines an embedder with a given ID.
//
// Deprecated: use [GoogleAI.RegisterEmbedder]. Like [GoogleAI.DefineModel],
// this method builds the embedder and ignores g, so embedding by that name
// serves the request with a different one unless the caller registers it with
// [genkit.RegisterAction].
func (ga *GoogleAI) DefineEmbedder(g *genkit.Genkit, id string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	return ga.buildEmbedder(id, embedOpts)
}

// buildEmbedder builds an unregistered embedder.
func (v *VertexAI) buildEmbedder(id string, embedOpts *ai.EmbedderOptions) (*ai.EmbedderAction, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}
	return newEmbedder(v.gclient, id, embedOpts), nil
}

// RegisterEmbedder registers an embedder with g and returns it; see
// [GoogleAI.RegisterEmbedder].
func (v *VertexAI) RegisterEmbedder(g *genkit.Genkit, id string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	embedder, err := v.buildEmbedder(id, embedOpts)
	if err != nil {
		return nil, err
	}
	genkit.RegisterAction(g, embedder)
	return embedder, nil
}

// DefineEmbedder defines an embedder with a given ID.
//
// Deprecated: use [VertexAI.RegisterEmbedder]. Like [VertexAI.DefineModel],
// this method builds the embedder and ignores g, so embedding by that name
// serves the request with a different one unless the caller registers it with
// [genkit.RegisterAction].
func (v *VertexAI) DefineEmbedder(g *genkit.Genkit, id string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	return v.buildEmbedder(id, embedOpts)
}

// isDefined reports whether an action of atype is registered with g under the
// provider-prefixed ID. The lookup deliberately does not resolve
// dynamically: these plugins resolve actions on demand, so a resolving lookup
// would register the very action the caller is checking for and answer true
// for any ID.
func isDefined(g *genkit.Genkit, atype api.ActionType, provider, id string) bool {
	return genkit.LookupAction(g, fmt.Sprintf("/%s/%s", atype, api.NewName(provider, id))) != nil
}

// IsDefinedModel reports whether the model is defined, which is the
// guard against registering one twice (see [GoogleAI.RegisterModel]).
func (ga *GoogleAI) IsDefinedModel(g *genkit.Genkit, id string) bool {
	return isDefined(g, api.ActionTypeModel, googleAIProvider, id)
}

// IsDefinedModel reports whether the model is defined, which is the
// guard against registering one twice (see [VertexAI.RegisterModel]).
func (v *VertexAI) IsDefinedModel(g *genkit.Genkit, id string) bool {
	return isDefined(g, api.ActionTypeModel, vertexAIProvider, id)
}

// IsDefinedEmbedder reports whether the [Embedder] is defined by this plugin.
func (ga *GoogleAI) IsDefinedEmbedder(g *genkit.Genkit, id string) bool {
	return isDefined(g, api.ActionTypeEmbedder, googleAIProvider, id)
}

// IsDefinedEmbedder reports whether the [Embedder] is defined by this plugin.
func (v *VertexAI) IsDefinedEmbedder(g *genkit.Genkit, id string) bool {
	return isDefined(g, api.ActionTypeEmbedder, vertexAIProvider, id)
}

// GoogleAIModel returns the [ai.Model] with the given ID.
// It returns nil if the model was not defined.
//
// Deprecated: Use genkit.LookupModel instead.
func GoogleAIModel(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, api.NewName(googleAIProvider, id))
}

// VertexAIModel returns the [ai.Model] with the given ID.
// It returns nil if the model was not defined.
//
// Deprecated: Use genkit.LookupModel instead.
func VertexAIModel(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, api.NewName(vertexAIProvider, id))
}

// GoogleAIEmbedder returns the [ai.Embedder] with the given ID.
// It returns nil if the embedder was not defined.
//
// Deprecated: Use genkit.LookupEmbedder instead.
func GoogleAIEmbedder(g *genkit.Genkit, id string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(googleAIProvider, id))
}

// VertexAIEmbedder returns the [ai.Embedder] with the given ID.
// It returns nil if the embedder was not defined.
//
// Deprecated: Use genkit.LookupEmbedder instead.
func VertexAIEmbedder(g *genkit.Genkit, id string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(vertexAIProvider, id))
}
