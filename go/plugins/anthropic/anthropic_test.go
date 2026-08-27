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

package anthropic

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
)

// TestModelOptionsKnownModels verifies the curated Claude models resolve through
// the shared modelOptions helper (used by both ListActions and ResolveAction)
// with JS ADVANCED_MODEL_INFO-equivalent supports (JSON output) and a stable
// stage. The set mirrors the JS plugin's ADVANCED entries in KNOWN_MODELS.
func TestModelOptionsKnownModels(t *testing.T) {
	advancedModels := []string{
		"claude-fable-5",
		"claude-opus-5",
		"claude-sonnet-5",
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-sonnet-4-6",
		"claude-sonnet-4-5",
		"claude-haiku-4-5",
	}
	for _, name := range advancedModels {
		opts := (&Anthropic{}).modelOptions(name)
		if opts.Supports == nil {
			t.Errorf("modelOptions(%q): Supports is nil", name)
			continue
		}
		if !slices.Contains(opts.Supports.Output, "json") {
			t.Errorf("modelOptions(%q): Output = %v, want it to include \"json\"", name, opts.Supports.Output)
		}
		if !opts.Supports.Tools || !opts.Supports.SystemRole {
			t.Errorf("modelOptions(%q): expected Tools and SystemRole supported, got %+v", name, opts.Supports)
		}
		if opts.Stage != ai.ModelStageStable {
			t.Errorf("modelOptions(%q): Stage = %q, want Stable", name, opts.Stage)
		}
		if opts.Label == "" {
			t.Errorf("modelOptions(%q): Label is empty", name)
		}
	}
}

func TestModelOptionsKnownVersionedModels(t *testing.T) {
	advancedModels := []string{
		"claude-opus-4-5-20251101",
		"claude-sonnet-4-5-20250929",
		"claude-haiku-4-5-20251001",
	}
	for _, name := range advancedModels {
		opts := (&Anthropic{}).modelOptions(name)
		if opts.Supports == nil {
			t.Errorf("modelOptions(%q): Supports is nil", name)
			continue
		}
		if !slices.Contains(opts.Supports.Output, "json") {
			t.Errorf("modelOptions(%q): Output = %v, want it to include \"json\"", name, opts.Supports.Output)
		}
		if !opts.Supports.Tools || !opts.Supports.SystemRole {
			t.Errorf("modelOptions(%q): expected Tools and SystemRole supported, got %+v", name, opts.Supports)
		}
	}
}

// TestModelOptionsUnknownFallback verifies models not in supportedModels fall back
// to dynamicModelOptions (no JSON output).
func TestModelOptionsUnknownFallback(t *testing.T) {
	const name = "claude-something-unreleased"
	opts := (&Anthropic{}).modelOptions(name)

	if opts.Supports == nil {
		t.Fatalf("modelOptions(%q): Supports is nil", name)
	}
	if slices.Contains(opts.Supports.Output, "json") {
		t.Errorf("modelOptions(%q): unknown model should use default supports without JSON output, got %v", name, opts.Supports.Output)
	}
}

// TestNewModelDescriptor covers what a built model advertises: a curated label
// for known models and a name-derived one for the rest, plus the config schema
// the framework validates every request against.
func TestNewModelDescriptor(t *testing.T) {
	tests := []struct {
		name      string
		wantLabel string
	}{
		{"claude-opus-4-5", anthropicLabelPrefix + " - Claude Opus 4.5"},
		{"claude-opus-4-5-20251101", anthropicLabelPrefix + " - Claude Opus 4.5"},
		{"claude-something-unreleased", anthropicLabelPrefix + " - claude-something-unreleased"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			desc := newModel(anthropic.Client{}, tt.name, (&Anthropic{}).modelOptions(tt.name)).Desc()

			model, ok := desc.Metadata["model"].(map[string]any)
			if !ok {
				t.Fatalf("model metadata missing, got %v", desc.Metadata)
			}
			if got := model["label"]; got != tt.wantLabel {
				t.Errorf("label = %v, want %q", got, tt.wantLabel)
			}

			schema, ok := model["customOptions"].(map[string]any)
			if !ok {
				t.Fatalf("customOptions missing, got %v", model["customOptions"])
			}
			props, ok := schema["properties"].(map[string]any)
			if !ok || props["max_tokens"] == nil {
				t.Errorf("config schema is not the Anthropic message params schema, got %v", schema)
			}
		})
	}
}

// TestModelsOverlaysCuratedCapabilities pins the merge rule: an entry replaces
// only the fields it sets, so pinning one capability keeps the curated label
// and config schema the model needs to work at all.
func TestModelsOverlaysCuratedCapabilities(t *testing.T) {
	a := &Anthropic{Models: map[string]ai.ModelOptions{
		"claude-opus-4-5": {Supports: &ai.ModelSupports{Multiturn: true}},
	}}

	opts := a.modelOptions("claude-opus-4-5")
	if opts.Supports == nil || opts.Supports.Tools {
		t.Errorf("Supports = %+v, want the entry's value to replace the curated one wholesale", opts.Supports)
	}
	if want := anthropicLabelPrefix + " - Claude Opus 4.5"; opts.Label != want {
		t.Errorf("Label = %q, want the curated %q kept by an entry that does not set one", opts.Label, want)
	}
	if opts.Stage != ai.ModelStageStable {
		t.Errorf("Stage = %q, want the curated stage kept", opts.Stage)
	}

	// An unknown ID starts from the Claude defaults rather than nothing, so an
	// entry describing a model this version never heard of is still complete.
	b := &Anthropic{Models: map[string]ai.ModelOptions{
		"claude-opus-9": {Label: "Claude Opus 9"},
	}}
	unknown := b.modelOptions("claude-opus-9")
	if unknown.Label != "Claude Opus 9" {
		t.Errorf("Label = %q, want the entry's", unknown.Label)
	}
	if unknown.Supports == nil {
		t.Error("Supports = nil, want the Claude defaults kept for a model the entry does not describe")
	}
}

// TestModelConfigIsValidated pins that the config schema reaches the request
// input schema, so the framework rejects a config the SDK type cannot hold
// before it reaches the model function.
func TestModelConfigIsValidated(t *testing.T) {
	const name = "claude-opus-4-5"
	inputSchema := newModel(anthropic.Client{}, name, (&Anthropic{}).modelOptions(name)).Desc().InputSchema

	req := func(config any) *ai.ModelRequest {
		return &ai.ModelRequest{
			Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hello"))},
			Config:   config,
		}
	}

	if err := base.ValidateValue(req(map[string]any{"max_tokens": 100, "temperature": 0.4}), inputSchema); err != nil {
		t.Errorf("config rejected at the action boundary: %v", err)
	}
	if err := base.ValidateValue(req(map[string]any{"max_tokens": "lots"}), inputSchema); err == nil {
		t.Error("expected a mistyped max_tokens to be rejected")
	}
}

// TestResolveActionSendsTheIDItWasGiven pins that resolving a model is local
// work. It used to list the catalog over the network to map an alias onto a
// dated release, which put an API call on the path that answers a lookup, took
// a context.Background() with no deadline or cancellation to make it, and
// failed the lookup outright whenever that call failed. Anthropic resolves its
// own aliases, so none of it bought anything.
//
// A dated ID still finds the curated entry for its alias, which is what
// baseModelName is for and the reason this is not simply a passthrough.
func TestResolveActionSendsTheIDItWasGiven(t *testing.T) {
	a := &Anthropic{}
	for _, id := range []string{
		"claude-opus-4-5",          // an alias the API lists only in dated form
		"claude-opus-4-5-20251101", // the dated release itself
		"claude-not-yet-released",  // an ID no catalog knows
	} {
		t.Run(id, func(t *testing.T) {
			action := a.ResolveAction(api.ActionTypeModel, id)
			if action == nil {
				t.Fatal("ResolveAction() = nil, want a model")
			}
			if got := action.Desc().Name; got != provider+"/"+id {
				t.Errorf("action name = %q, want the ID as given", got)
			}
		})
	}

	if got := a.ResolveAction(api.ActionTypeEmbedder, "anything"); got != nil {
		t.Errorf("ResolveAction(embedder) = %v, want nil (models are all this plugin serves)", got)
	}
}

// TestCuratedCapabilitiesSurviveADatedID pins the one thing the deleted
// alias resolution has to keep doing: a request naming a dated release must
// still be described by the curated entry keyed under its alias.
func TestCuratedCapabilitiesSurviveADatedID(t *testing.T) {
	a := &Anthropic{}
	alias := a.modelOptions("claude-opus-4-5")
	dated := a.modelOptions("claude-opus-4-5-20251101")

	if alias.Label == "" {
		t.Fatal("claude-opus-4-5 has no curated label; the test is checking nothing")
	}
	if dated.Label != alias.Label {
		t.Errorf("dated label = %q, want the alias's %q", dated.Label, alias.Label)
	}
	if dated.Supports == nil || alias.Supports == nil || !reflect.DeepEqual(dated.Supports, alias.Supports) {
		t.Errorf("dated supports = %+v, want the alias's %+v", dated.Supports, alias.Supports)
	}
}

// modelsListServer serves the Anthropic models list so ResolveAction is
// reachable without a real endpoint.
func modelsListServer(t *testing.T) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-opus-4-5-20251101","type":"model"}],"has_more":false}`)
	}))
	t.Cleanup(server.Close)
	return server.URL
}

// recordingModelsServer is [modelsListServer] with the headers of the most
// recent request captured, so a test can observe what the client sent.
func recordingModelsServer(t *testing.T) (url string, lastHeader func() http.Header) {
	t.Helper()
	var mu sync.Mutex
	var header http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		header = r.Header.Clone()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-opus-4-5-20251101","type":"model"}],"has_more":false}`)
	}))
	t.Cleanup(server.Close)
	return server.URL, func() http.Header {
		mu.Lock()
		defer mu.Unlock()
		return header
	}
}

// TestInitRequiresAuth pins the panic when no authentication is configured
// anywhere: no APIKey, no key or token in the environment, and no Opts.
func TestInitRequiresAuth(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	defer func() {
		if recover() == nil {
			t.Fatal("Init did not panic with no authentication configured")
		}
	}()
	(&Anthropic{}).Init(context.Background())
}

// TestInitAuthTokenSuffices covers the bearer-token setups the SDK serves
// from ANTHROPIC_AUTH_TOKEN on its own: the plugin must not demand an API
// key on top of one.
func TestInitAuthTokenSuffices(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
	(&Anthropic{}).Init(context.Background())
}

// TestInitOptsCarryAuth covers the key-less setups Opts exists for (the SDK's
// Bedrock and Vertex routing options sign requests themselves): a non-empty
// Opts waives the API-key requirement, and its options reach every request
// the client sends.
func TestInitOptsCarryAuth(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	url, lastHeader := recordingModelsServer(t)
	a := &Anthropic{Opts: []option.RequestOption{
		option.WithBaseURL(url),
		option.WithHeader("X-Test-Opt", "carried"),
	}}
	a.Init(context.Background())

	if actions := a.ListActions(context.Background()); len(actions) == 0 {
		t.Fatal("ListActions() = empty, want the served model list")
	}
	header := lastHeader()
	if header == nil {
		t.Fatal("the server saw no request")
	}
	if got := header.Get("X-Test-Opt"); got != "carried" {
		t.Errorf("X-Test-Opt = %q, want %q", got, "carried")
	}
	if got := header.Get("X-Api-Key"); got != "" {
		t.Errorf("x-api-key = %q, want none with no key configured", got)
	}
}

// TestInitOptsWinOverFields pins the precedence contract: the options derived
// from APIKey and BaseURL are applied first and the SDK applies options in
// order, so an explicit option in Opts overrides both fields.
func TestInitOptsWinOverFields(t *testing.T) {
	fieldURL, fieldHeader := recordingModelsServer(t)
	optsURL, optsHeader := recordingModelsServer(t)

	a := &Anthropic{
		APIKey:  "field-key",
		BaseURL: fieldURL,
		Opts: []option.RequestOption{
			option.WithBaseURL(optsURL),
			option.WithAPIKey("opts-key"),
		},
	}
	a.Init(context.Background())

	if actions := a.ListActions(context.Background()); len(actions) == 0 {
		t.Fatal("ListActions() = empty, want the served model list")
	}
	if fieldHeader() != nil {
		t.Error("the BaseURL field's server saw a request, want the Opts base URL to win")
	}
	header := optsHeader()
	if header == nil {
		t.Fatal("the Opts base URL's server saw no request")
	}
	if got := header.Get("X-Api-Key"); got != "opts-key" {
		t.Errorf("x-api-key = %q, want the Opts key %q", got, "opts-key")
	}
}

// TestInitAPIKeyRidesAlongsideOptsAuth pins the limit of "wins over those
// fields": the API key and an auth token are distinct settings riding
// distinct headers, so an auth token in Opts displaces nothing and a
// configured key still goes out beside it.
func TestInitAPIKeyRidesAlongsideOptsAuth(t *testing.T) {
	url, lastHeader := recordingModelsServer(t)
	a := &Anthropic{
		APIKey: "field-key",
		Opts: []option.RequestOption{
			option.WithBaseURL(url),
			option.WithAuthToken("opts-token"),
		},
	}
	a.Init(context.Background())

	if actions := a.ListActions(context.Background()); len(actions) == 0 {
		t.Fatal("ListActions() = empty, want the served model list")
	}
	header := lastHeader()
	if got := header.Get("Authorization"); got != "Bearer opts-token" {
		t.Errorf("authorization = %q, want the Opts token", got)
	}
	if got := header.Get("X-Api-Key"); got != "field-key" {
		t.Errorf("x-api-key = %q, want the configured key still sent beside the token", got)
	}
}

// TestInitHeaderDelStripsAPIKey pins the documented escape hatch for setups
// that cannot keep a key out of the environment: a WithHeaderDel in Opts
// applies after the options derived from APIKey, so it removes the key
// header before the request goes out.
func TestInitHeaderDelStripsAPIKey(t *testing.T) {
	url, lastHeader := recordingModelsServer(t)
	a := &Anthropic{
		APIKey: "field-key",
		Opts: []option.RequestOption{
			option.WithBaseURL(url),
			option.WithHeaderDel("X-Api-Key"),
		},
	}
	a.Init(context.Background())

	if actions := a.ListActions(context.Background()); len(actions) == 0 {
		t.Fatal("ListActions() = empty, want the served model list")
	}
	if got := lastHeader().Get("X-Api-Key"); got != "" {
		t.Errorf("x-api-key = %q, want it stripped by WithHeaderDel", got)
	}
}

// TestModelsOverrideReachesResolution is the reason capabilities live in plugin
// config. Nothing registers the model up front: the first lookup drives the
// plugin's ResolveAction, and the caller's entry is what describes what comes
// back. No ordering makes this miss, which is what a registration call could
// not promise, since resolving a name registers it and a later registration of
// the same name would panic.
func TestModelsOverrideReachesResolution(t *testing.T) {
	const name = "claude-opus-4-5"
	a := &Anthropic{
		APIKey:  "test-key",
		BaseURL: modelsListServer(t),
		Models: map[string]ai.ModelOptions{
			name: {Label: "Custom Claude", Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		},
	}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	if IsDefinedModel(g, name) {
		t.Fatalf("IsDefinedModel(%q) = true before anything resolved it", name)
	}

	m := genkit.LookupModel(g, "anthropic/"+name)
	if m == nil {
		t.Fatal("LookupModel() = nil, want the plugin to resolve the model")
	}
	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if model["label"] != "Custom Claude" {
		t.Errorf("label = %v, want the entry's capabilities to describe the resolved model", model["label"])
	}
}

// TestModelsOverrideReachesListActions pins the other half: the actions the
// plugin advertises carry the caller's entry too, so what the dev UI lists and
// what serves a request agree.
func TestModelsOverrideReachesListActions(t *testing.T) {
	a := &Anthropic{
		APIKey:  "test-key",
		BaseURL: modelsListServer(t),
		Models: map[string]ai.ModelOptions{
			"claude-opus-4-5-20251101": {Label: "Custom Claude"},
		},
	}
	genkit.Init(context.Background(), genkit.WithPlugins(a))

	actions := a.ListActions(context.Background())
	if len(actions) == 0 {
		t.Fatal("ListActions() = empty, want the served model list")
	}
	model, ok := actions[0].Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing: %v", actions[0].Metadata)
	}
	if model["label"] != "Custom Claude" {
		t.Errorf("label = %v, want the entry's label in the advertised action", model["label"])
	}
}

// TestModelRef pins the name a ref carries and that the typed config rides
// along, since the ref is how an application supplies config at the call site.
// Both the bare ID and the already-prefixed name resolve to the same model,
// so passing the name a sibling plugin would take is not a silent miss.
func TestModelRef(t *testing.T) {
	cfg := &anthropic.MessageNewParams{MaxTokens: 1024}

	for _, name := range []string{"claude-opus-4-5", "anthropic/claude-opus-4-5"} {
		ref := ModelRef(name, cfg)
		if want := "anthropic/claude-opus-4-5"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	// A nil config rides along as a typed nil rather than an untyped one. The
	// config slot tolerates that: it marshals to JSON null and deserializes to
	// the zero MessageNewParams, the same as googlegenai's refs.
	if got := ModelRef("claude-opus-4-5", nil).Config(); got != (*anthropic.MessageNewParams)(nil) {
		t.Errorf("Config() = %v for a nil config, want a typed nil", got)
	}
}

// TestPrefixedNamesAreEquivalent pins that the exported entry points take a
// model ID either bare or provider-prefixed. The prefix is applied by
// concatenation, so an untrimmed name would double up and name a model that
// resolves nowhere.
func TestPrefixedNamesAreEquivalent(t *testing.T) {
	a := &Anthropic{APIKey: "test-key", BaseURL: modelsListServer(t)}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	for _, name := range []string{"claude-opus-4-5", "anthropic/claude-opus-4-5"} {
		if Model(g, name) == nil {
			t.Errorf("Model(%q) = nil, want the model resolved under either form", name)
		}
		if !IsDefinedModel(g, name) {
			t.Errorf("IsDefinedModel(%q) = false, want the resolved model found under either form", name)
		}
	}

	// Resolving by the prefixed name must find the curated capabilities, not
	// the unknown-model defaults, which is why the trim precedes the lookup.
	m := Model(g, "claude-opus-4-5")
	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if want := anthropicLabelPrefix + " - Claude Opus 4.5"; model["label"] != want {
		t.Errorf("label = %v, want %q", model["label"], want)
	}
}

// TestIsDefinedModelDoesNotResolve pins that asking whether a model is defined
// must not itself resolve and register one. The plugin resolves any name the
// Anthropic API can serve, so a resolving lookup would answer true for every
// name. The fake endpoint serves the models list to make resolution reachable,
// which is exactly what this must not trigger.
func TestIsDefinedModelDoesNotResolve(t *testing.T) {
	a := &Anthropic{APIKey: "test-key", BaseURL: modelsListServer(t)}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	if IsDefinedModel(g, "claude-opus-4-5") {
		t.Fatal("IsDefinedModel() = true for a model nothing has resolved yet")
	}
	if genkit.LookupModel(g, "anthropic/claude-opus-4-5") == nil {
		t.Fatal("LookupModel() = nil, want the plugin to resolve the model")
	}
	if !IsDefinedModel(g, "claude-opus-4-5") {
		t.Error("IsDefinedModel() = false after the resolving lookup registered it")
	}
}

// TestDefineModelDoesNotRegister pins the deprecated builder: it hands back a
// model without touching the registry, which is why capabilities passed to it
// never reach the model that serves a request.
func TestDefineModelDoesNotRegister(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")

	a := &Anthropic{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	const name = "claude-opus-4-5"
	m, err := a.DefineModel(g, name, nil)
	if err != nil {
		t.Fatalf("DefineModel() error = %v", err)
	}
	if m == nil {
		t.Fatal("DefineModel() = nil, want the built model")
	}
	if IsDefinedModel(g, name) {
		t.Errorf("IsDefinedModel(%q) = true after DefineModel(), want the deprecated builder to leave the registry alone", name)
	}
}

// TestDefineModelRequiresInit pins the guard that was missing while
// capabilities came from a registration call: an uninitialized plugin has no
// client, so building a model from it would hand back one that fails much
// later with an error pointing nowhere near the cause.
func TestDefineModelRequiresInit(t *testing.T) {
	a := &Anthropic{}
	g := genkit.Init(context.Background())

	if _, err := a.DefineModel(g, "claude-opus-4-5", nil); err == nil {
		t.Error("DefineModel() error = nil on an uninitialized plugin, want it refused")
	}
}

// TestOverrideKeysAcceptProviderPrefix mirrors the googlegenai test: both
// spellings of a model ID must reach the same entry in Models.
func TestOverrideKeysAcceptProviderPrefix(t *testing.T) {
	for _, key := range []string{"claude-opus-4-5", "anthropic/claude-opus-4-5"} {
		a := &Anthropic{Models: map[string]ai.ModelOptions{key: {Label: "custom"}}}
		for _, id := range []string{"claude-opus-4-5", "anthropic/claude-opus-4-5"} {
			if got := a.modelOptions(id).Label; got != "custom" {
				t.Errorf("Models[%q] did not apply to %q: label = %q", key, id, got)
			}
		}
	}
}
