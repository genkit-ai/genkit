// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"sync"

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/httptransport"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"google.golang.org/genai"
)

const (
	googleAIProvider = "googleai"
	vertexAIProvider = "vertexai"

	googleAILabelPrefix = "Google AI"
	vertexAILabelPrefix = "Vertex AI"
)

// displayName is how a provider is spelled in a model's dev UI label.
func displayName(provider string) string {
	if provider == vertexAIProvider {
		return vertexAILabelPrefix
	}
	return googleAILabelPrefix
}

// rejectBackgroundModel refuses IDs whose modality is served by a background
// action. They deserialize a different config type and speak a different API
// method, so building one as a plain model yields an action that advertises
// video fields and can only fail at the API.
func rejectBackgroundModel(provider, id string) error {
	if ClassifyModel(id).ActionType() == api.ActionTypeBackgroundModel {
		return fmt.Errorf("%s: %q is a background model; generate with it directly instead of defining it as a model", provider, id)
	}
	return nil
}

// GoogleAI is a Genkit plugin for interacting with the Google AI service.
type GoogleAI struct {
	APIKey     string // API key to access the service. If empty, the values of the environment variables GEMINI_API_KEY or GOOGLE_API_KEY will be consulted, in that order.
	APIVersion string // API version to use ("v1", "v1beta", or "v1alpha"). If empty, the genai SDK default (v1beta) is used. Can be overridden per-request via config.HTTPOptions.APIVersion.

	// BaseURL overrides the default API endpoint
	// (https://generativelanguage.googleapis.com), e.g. to point at a proxy
	// or an API gateway. Optional.
	BaseURL string
	// Headers are additional HTTP headers to send with every request. They
	// are merged over the plugin's default headers, so a header set here wins
	// on collision. Optional.
	Headers http.Header
	// HTTPClient is the HTTP client used for API requests. When set, it is
	// used verbatim: the plugin does not install its default OpenTelemetry
	// instrumented transport, so wrap your own transport with otelhttp if you
	// want tracing. Optional.
	HTTPClient *http.Client

	// Models overrides what the plugin knows about a model, keyed by model ID.
	// Every model the backend serves already works without an entry here:
	// known IDs carry curated capabilities and the rest take the defaults for
	// their kind. Supply an entry only to correct or extend what the plugin
	// resolves, most often to describe a model released after this version of
	// the plugin.
	//
	//	&googlegenai.GoogleAI{Models: map[string]ai.ModelOptions{
	//		"gemini-flash-latest": {Supports: &ai.ModelSupports{Tools: true, Multiturn: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the config
	// schema. Gemini, Imagen and Veo IDs are all keyed the same way, and
	// entries apply to the actions [GoogleAI.ListActions] advertises as well
	// as the ones [GoogleAI.ResolveAction] builds to serve a request.
	Models map[string]ai.ModelOptions

	// Embedders overrides what the plugin knows about an embedder, keyed by
	// embedder ID. It works exactly as Models does.
	Embedders map[string]ai.EmbedderOptions

	gclient *genai.Client // Client for the Google AI service.
	mu      sync.Mutex    // Mutex to control access.
	initted bool          // Whether the plugin has been initialized.
}

// VertexAI is a Genkit plugin for interacting with the Google Vertex AI service.
type VertexAI struct {
	ProjectID  string // Google Cloud project to use for Vertex AI. If empty, the value of the environment variable GOOGLE_CLOUD_PROJECT will be consulted.
	Location   string // Location of the Vertex AI service. If empty, GOOGLE_CLOUD_LOCATION and GOOGLE_CLOUD_REGION environment variables will be consulted, in that order. Accepts a regional location (e.g. "us-central1"), a multi-region location ("us" or "eu"), or "global".
	APIVersion string // API version to use ("v1" or "v1beta1"). If empty, the genai SDK default (v1beta1) is used. Can be overridden per-request via config.HTTPOptions.APIVersion.

	// APIKey enables Vertex AI Express Mode: API key authentication with no
	// Google Cloud project, location, or Application Default Credentials
	// involved. See
	// https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview.
	// Mutually exclusive with ProjectID, Location, and Credentials. Optional.
	APIKey string
	// Credentials overrides the Google Cloud credentials used to authenticate.
	// If nil, Application Default Credentials are detected. Mutually exclusive
	// with APIKey and HTTPClient. Optional.
	Credentials *auth.Credentials
	// BaseURL overrides the default location-derived API endpoint, e.g. to
	// point at a proxy or an API gateway. Optional.
	BaseURL string
	// Headers are additional HTTP headers to send with every request. They
	// are merged over the plugin's default headers, so a header set here wins
	// on collision. Optional.
	Headers http.Header
	// HTTPClient is the HTTP client used for API requests. When set, it is
	// used verbatim and must handle authentication itself (unless APIKey is
	// set, which rides on a request header): the plugin does not install its
	// default credential-carrying, OpenTelemetry instrumented transport.
	// Optional.
	HTTPClient *http.Client

	// Models overrides what the plugin knows about a model, keyed by model ID;
	// see [GoogleAI.Models]. Tuned Gemini endpoints are keyed in either the
	// short form `endpoints/ID` or the full resource path
	// `projects/PROJECT/locations/LOCATION/endpoints/ID`, whichever form the
	// request names them by.
	Models map[string]ai.ModelOptions

	// Embedders overrides what the plugin knows about an embedder, keyed by
	// embedder ID; see [GoogleAI.Embedders].
	Embedders map[string]ai.EmbedderOptions

	gclient *genai.Client // Client for the Vertex AI service.
	mu      sync.Mutex    // Mutex to control access.
	initted bool          // Whether the plugin has been initialized.
}

// firstEnv returns the value of the first environment variable in names that
// is set to a non-empty value.
func firstEnv(names ...string) string {
	for _, n := range names {
		if v := os.Getenv(n); v != "" {
			return v
		}
	}
	return ""
}

// vertexLocationFromEnv returns the Vertex AI location named by the
// environment, or "".
func vertexLocationFromEnv() string {
	return firstEnv("GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION")
}

// vertexAPIKeyFromEnv returns the Vertex AI Express Mode API key named by
// the environment and the variable it came from, or two empty strings.
// GEMINI_API_KEY is deliberately not consulted, matching the JS plugin: it
// names a Gemini Developer API key, which Vertex AI does not accept, and a
// process that also uses the Google AI plugin commonly has one set.
func vertexAPIKeyFromEnv() (key, source string) {
	for _, name := range []string{"VERTEX_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"} {
		if k := os.Getenv(name); k != "" {
			return k, name
		}
	}
	return "", ""
}

// mergedHeaders returns the plugin's default headers merged with extra.
// Extra values win on key collisions.
func mergedHeaders(extra http.Header) http.Header {
	h := genkitClientHeader.Clone()
	for k, vs := range extra {
		h[http.CanonicalHeaderKey(k)] = slices.Clone(vs)
	}
	return h
}

// httpClientOrDefault returns c, or the plugin's default OpenTelemetry
// instrumented client when c is nil.
func httpClientOrDefault(c *http.Client) *http.Client {
	if c != nil {
		return c
	}
	return &http.Client{
		Transport: otelhttp.NewTransport(http.DefaultTransport),
	}
}

// Name returns the name of the plugin.
func (ga *GoogleAI) Name() string {
	return googleAIProvider
}

// Name returns the name of the plugin.
func (v *VertexAI) Name() string {
	return vertexAIProvider
}

// expressAPIKey returns the API key that selects Express Mode, or "".
// An explicit APIKey always selects Express Mode, and combining it with
// project, location, or credentials panics. Explicit project, location, or
// credentials always select credential authentication instead. With nothing
// configured, an environment key applies only when the environment names
// neither a project nor a location, the same precedence rule the genai SDK
// applies to its variables (the key names follow the JS plugin; see
// vertexAPIKeyFromEnv). For a key from the environment, source names the
// variable it came from; for an explicit key, source is "".
func (v *VertexAI) expressAPIKey() (key, source string) {
	if v.APIKey != "" {
		if v.ProjectID != "" || v.Location != "" {
			panic("Vertex AI: APIKey (Express Mode) is mutually exclusive with ProjectID and Location")
		}
		if v.Credentials != nil {
			panic("Vertex AI: APIKey (Express Mode) is mutually exclusive with Credentials")
		}
		return v.APIKey, ""
	}
	if v.ProjectID != "" || v.Location != "" || v.Credentials != nil {
		return "", ""
	}
	if os.Getenv("GOOGLE_CLOUD_PROJECT") != "" || vertexLocationFromEnv() != "" {
		return "", ""
	}
	return vertexAPIKeyFromEnv()
}

// customEndpointOnly reports whether the only configuration present, here or
// in the environment, is a custom base URL: the BaseURL field, or
// GOOGLE_VERTEX_BASE_URL, which the SDK reads on its own. The SDK treats
// that as a valid standalone setup (the endpoint owns authentication), so
// the plugin must not demand a project for it. Call after expressAPIKey has
// returned "".
func (v *VertexAI) customEndpointOnly() bool {
	return (v.BaseURL != "" || os.Getenv("GOOGLE_VERTEX_BASE_URL") != "") &&
		v.ProjectID == "" && v.Location == "" && v.Credentials == nil &&
		os.Getenv("GOOGLE_CLOUD_PROJECT") == "" && vertexLocationFromEnv() == ""
}

// Init initializes the Google AI plugin and all known models and embedders.
// After calling Init, you may call [DefineModel] and [DefineEmbedder] to create
// and register any additional generative models and embedders
func (ga *GoogleAI) Init(ctx context.Context) []api.Action {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if ga.initted {
		panic("plugin already initialized")
	}

	switch ga.APIVersion {
	case "", "v1", "v1beta", "v1alpha":
	default:
		panic(fmt.Sprintf("Google AI APIVersion must be %q, %q, or %q, got %q", "v1", "v1beta", "v1alpha", ga.APIVersion))
	}

	apiKey := ga.APIKey
	if apiKey == "" {
		apiKey = firstEnv("GEMINI_API_KEY", "GOOGLE_API_KEY")
		if apiKey == "" {
			panic("Google AI requires setting GEMINI_API_KEY or GOOGLE_API_KEY in the environment. You can get an API key at https://ai.google.dev")
		}
	}

	gc := genai.ClientConfig{
		Backend:    genai.BackendGeminiAPI,
		APIKey:     apiKey,
		HTTPClient: httpClientOrDefault(ga.HTTPClient),
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    ga.BaseURL,
			Headers:    mergedHeaders(ga.Headers),
			APIVersion: ga.APIVersion,
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
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.initted {
		panic("plugin already initialized")
	}

	switch v.APIVersion {
	case "", "v1", "v1beta1":
	default:
		panic(fmt.Sprintf("Vertex AI APIVersion must be %q or %q, got %q", "v1", "v1beta1", v.APIVersion))
	}

	gc := genai.ClientConfig{
		Backend: genai.BackendVertexAI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    v.BaseURL,
			Headers:    mergedHeaders(v.Headers),
			APIVersion: v.APIVersion,
		},
	}

	apiKey, keySource := v.expressAPIKey()
	switch {
	case apiKey != "":
		// Express Mode: the API key alone authenticates; no project,
		// location, or credentials are involved.
		if keySource != "" {
			// Breadcrumb for the ambient-key case: a Gemini Developer API
			// key in these variables is not valid for Vertex AI and fails
			// each request with PERMISSION_DENIED, so name where the key
			// came from.
			logger.FromContext(ctx).Info("Vertex AI: using Express Mode API key from environment", "variable", keySource)
		}
		gc.APIKey = apiKey
		gc.HTTPClient = httpClientOrDefault(v.HTTPClient)
	case v.customEndpointOnly():
		// A base URL with no project, location, key, or credentials routes
		// everything to the endpoint as-is; the endpoint owns
		// authentication. Mirrors the SDK, which accepts a custom base URL
		// as sufficient configuration on its own.
		gc.HTTPClient = httpClientOrDefault(v.HTTPClient)
	default:
		projectID := v.ProjectID
		if projectID == "" {
			projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
			if projectID == "" {
				panic("Vertex AI requires setting GOOGLE_CLOUD_PROJECT in the environment, or an Express Mode API key in VERTEX_API_KEY or GOOGLE_API_KEY. You can get a project ID at https://console.cloud.google.com/home/dashboard")
			}
		}

		location := v.Location
		if location == "" {
			location = vertexLocationFromEnv()
			if location == "" {
				panic("Vertex AI requires setting GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION in the environment. You can get a location at https://cloud.google.com/vertex-ai/docs/general/locations")
			}
		}

		// Project and Region values gets validated by genai SDK upon client creation
		gc.Project = projectID
		gc.Location = location

		if v.HTTPClient != nil {
			if v.Credentials != nil {
				panic("Vertex AI: HTTPClient and Credentials are mutually exclusive; a custom HTTP client must handle authentication itself")
			}
			gc.HTTPClient = v.HTTPClient
		} else {
			cred := v.Credentials
			if cred == nil {
				var err error
				cred, err = credentials.DetectDefault(&credentials.DetectOptions{
					Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
				})
				if err != nil {
					panic(fmt.Errorf("failed to find default credentials: %w", err))
				}
			}
			quotaProjectID, err := cred.QuotaProjectID(ctx)
			if err != nil {
				panic(fmt.Errorf("failed to get quota project ID: %w", err))
			}
			var headers http.Header
			if quotaProjectID != "" {
				headers = http.Header{
					"X-Goog-User-Project": []string{quotaProjectID},
				}
			}
			httpClient, err := httptransport.NewClient(&httptransport.Options{
				Credentials:      cred,
				BaseRoundTripper: otelhttp.NewTransport(http.DefaultTransport),
				Headers:          headers,
			})
			if err != nil {
				panic(fmt.Errorf("failed to create http client: %w", err))
			}
			gc.HTTPClient = httpClient
		}
	}

	client, err := genai.NewClient(ctx, &gc)
	if err != nil {
		panic(fmt.Errorf("VertexAI.Init: %w", err))
	}
	v.gclient = client
	v.initted = true

	return []api.Action{}
}

// Client returns the underlying Google GenAI SDK client used by the plugin.
// It gives access to service features that Genkit does not wrap, such as the
// Files, Caches, Batches, and Tunings APIs. It returns an error if the plugin
// has not been initialized.
func (ga *GoogleAI) Client() (*genai.Client, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if !ga.initted {
		return nil, errors.New("GoogleAI plugin not initialized")
	}
	return ga.gclient, nil
}

// Client returns the underlying Google GenAI SDK client used by the plugin;
// see [GoogleAI.Client].
func (v *VertexAI) Client() (*genai.Client, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}
	return v.gclient, nil
}

// DefineModel defines an unknown model with the given ID.
// The second argument describes the capability of the model.
//
// Deprecated: describe the model through [GoogleAI.Models] instead. This
// method builds the model and ignores g, so the result carries only the
// model's name: generation resolves a model from that name and serves the
// request with the capabilities the plugin resolves, not the ones passed
// here. An entry in Models reaches both paths.
func (ga *GoogleAI) DefineModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if !ga.initted {
		return nil, errors.New("GoogleAI plugin not initialized")
	}
	id = internal.TrimProvider(googleAIProvider, id)
	if err := rejectBackgroundModel(googleAIProvider, id); err != nil {
		return nil, err
	}
	if opts != nil {
		return newModel(ga.gclient, id, *opts), nil
	}

	c := ga.catalog()
	models, err := listModels(googleAIProvider)
	if err != nil {
		return nil, err
	}
	if _, known := models[id]; !known && !c.modelOverridden(id) {
		return nil, fmt.Errorf("GoogleAI: called with unknown model %q and nil ModelOptions", id)
	}

	return newModel(ga.gclient, id, c.modelOptions(id)), nil
}

// DefineModel defines an unknown model with the given ID.
// The second argument describes the capability of the model.
//
// Tuned Gemini endpoints are accepted in either the short form `endpoints/ID`
// or the full resource path
// `projects/PROJECT/locations/LOCATION/endpoints/ID`, and take the default
// Gemini capability set when opts is nil.
//
// Deprecated: describe the model through [VertexAI.Models] instead; see
// [GoogleAI.DefineModel] for why the result of this method is not what serves
// the request.
func (v *VertexAI) DefineModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}
	id = internal.TrimProvider(vertexAIProvider, id)
	if err := rejectBackgroundModel(vertexAIProvider, id); err != nil {
		return nil, err
	}
	if opts != nil {
		return newModel(v.gclient, id, *opts), nil
	}

	c := v.catalog()
	if !isTunedGeminiName(id) && !c.modelOverridden(id) {
		models, err := listModels(vertexAIProvider)
		if err != nil {
			return nil, err
		}
		if _, known := models[id]; !known {
			return nil, fmt.Errorf("VertexAI: called with unknown model %q and nil ModelOptions", id)
		}
	}

	return newModel(v.gclient, id, c.modelOptions(id)), nil
}

// DefineEmbedder defines an embedder with a given ID.
//
// Deprecated: describe the embedder through [GoogleAI.Embedders] instead.
// Like [GoogleAI.DefineModel], this method builds the embedder and ignores g,
// so embedding by that name serves the request with the capabilities the
// plugin resolves rather than the ones passed here.
func (ga *GoogleAI) DefineEmbedder(g *genkit.Genkit, id string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	if !ga.initted {
		return nil, errors.New("GoogleAI plugin not initialized")
	}
	id = internal.TrimProvider(googleAIProvider, id)
	if embedOpts == nil {
		opts := ga.catalog().embedderOptions(id)
		embedOpts = &opts
	}
	return newEmbedder(ga.gclient, id, embedOpts), nil
}

// DefineEmbedder defines an embedder with a given ID.
//
// Deprecated: describe the embedder through [VertexAI.Embedders] instead; see
// [GoogleAI.DefineEmbedder].
func (v *VertexAI) DefineEmbedder(g *genkit.Genkit, id string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("VertexAI plugin not initialized")
	}
	id = internal.TrimProvider(vertexAIProvider, id)
	if embedOpts == nil {
		opts := v.catalog().embedderOptions(id)
		embedOpts = &opts
	}
	return newEmbedder(v.gclient, id, embedOpts), nil
}

// isDefined reports whether an action of atype is registered with g under the
// provider-prefixed ID. The lookup deliberately does not resolve
// dynamically: these plugins resolve actions on demand, so a resolving lookup
// would register the very action the caller is checking for and answer true
// for any ID.
func isDefined(g *genkit.Genkit, atype api.ActionType, provider, id string) bool {
	return genkit.LookupAction(g, api.NewKey(atype, provider, internal.TrimProvider(provider, id))) != nil
}

// IsDefinedEmbedder reports whether the [Embedder] is defined by this plugin.
//
// Deprecated: this existed to guard a registration call that could panic on a
// duplicate. Capabilities now come from [GoogleAI.Embedders], which nothing
// has to register and which no ordering can defeat, leaving this a question
// about registry state that applications do not need to ask.
func (ga *GoogleAI) IsDefinedEmbedder(g *genkit.Genkit, id string) bool {
	return isDefined(g, api.ActionTypeEmbedder, googleAIProvider, id)
}

// IsDefinedEmbedder reports whether the [Embedder] is defined by this plugin.
//
// Deprecated: see [GoogleAI.IsDefinedEmbedder]; capabilities now come from
// [VertexAI.Embedders].
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
