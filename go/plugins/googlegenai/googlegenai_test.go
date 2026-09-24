// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai_test

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/auth"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// TestVertexAIInit_InvalidAPIVersion verifies that a bad APIVersion is
// rejected before any credential detection or network access is attempted,
// so the failure is immediate and clear rather than a confusing 404 at
// request time.
func TestVertexAIInit_InvalidAPIVersion(t *testing.T) {
	recovered := mustPanic(t, func() {
		(&googlegenai.VertexAI{
			ProjectID:  "test-project",
			Location:   "us-central1",
			APIVersion: "v1beta", // invalid: valid values are "v1" and "v1beta1"
		}).Init(context.Background())
	})
	msg, ok := recovered.(string)
	if !ok {
		t.Fatalf("expected panic value to be a string, got %T: %v", recovered, recovered)
	}
	if !strings.Contains(msg, "v1beta") {
		t.Errorf("panic message %q does not mention the invalid value", msg)
	}
}

// TestGoogleAIInit_InvalidAPIVersion verifies that a bad APIVersion is
// rejected up front, mirroring the Vertex AI check. The invalid value is
// Vertex AI's valid one, so the test also documents that the two backends
// name their versions differently.
func TestGoogleAIInit_InvalidAPIVersion(t *testing.T) {
	recovered := mustPanic(t, func() {
		(&googlegenai.GoogleAI{
			APIKey:     "test-api-key",
			APIVersion: "v1beta1", // invalid: valid values are "v1", "v1beta", and "v1alpha"
		}).Init(context.Background())
	})
	msg, ok := recovered.(string)
	if !ok {
		t.Fatalf("expected panic value to be a string, got %T: %v", recovered, recovered)
	}
	if !strings.Contains(msg, "v1beta1") {
		t.Errorf("panic message %q does not mention the invalid value", msg)
	}
}

// mustPanic runs f and returns the recovered panic value, failing the test
// if f does not panic.
func mustPanic(t *testing.T, f func()) any {
	t.Helper()
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		f()
	}()
	if recovered == nil {
		t.Fatal("expected a panic, but there was none")
	}
	return recovered
}

func TestGoogleAIInit_HTTPOptions(t *testing.T) {
	custom := &http.Client{}
	ga := &googlegenai.GoogleAI{
		APIKey:     "test-api-key",
		APIVersion: "v1alpha",
		BaseURL:    "https://gateway.example.com",
		Headers:    http.Header{"X-Test-Header": {"yes"}},
		HTTPClient: custom,
	}
	ga.Init(context.Background())

	client, err := ga.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.HTTPOptions.BaseURL != "https://gateway.example.com" {
		t.Errorf("BaseURL = %q, want the configured gateway", cc.HTTPOptions.BaseURL)
	}
	if cc.HTTPOptions.APIVersion != "v1alpha" {
		t.Errorf("APIVersion = %q, want %q", cc.HTTPOptions.APIVersion, "v1alpha")
	}
	if got := cc.HTTPOptions.Headers.Get("X-Test-Header"); got != "yes" {
		t.Errorf("custom header = %q, want %q", got, "yes")
	}
	if got := cc.HTTPOptions.Headers.Get("x-goog-api-client"); !strings.Contains(got, "genkit-go") {
		t.Errorf("attribution header = %q, want it to carry genkit-go", got)
	}
	if cc.HTTPClient != custom {
		t.Error("HTTPClient was not used verbatim")
	}
}

func TestGoogleAIInit_DefaultHTTPClient(t *testing.T) {
	ga := &googlegenai.GoogleAI{
		APIKey: "test-api-key",
	}
	ga.Init(context.Background())

	client, err := ga.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.HTTPClient != http.DefaultClient {
		t.Errorf("HTTPClient = %v, want http.DefaultClient", cc.HTTPClient)
	}
}

// clearVertexEnv blanks every environment variable that Vertex AI
// initialization consults, so a test starts from a known state regardless of
// what the developer or CI has exported.
func clearVertexEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION",
		"VERTEX_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "GEMINI_API_KEY",
		"GOOGLE_VERTEX_BASE_URL",
	} {
		t.Setenv(name, "")
	}
}

// fakeADC points Application Default Credentials at a fabricated
// authorized-user credential file. Detection and quota-project lookup read
// the file only, so the credential-based paths resolve identically on a
// machine with no gcloud login.
func fakeADC(t *testing.T) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "adc.json")
	const cred = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com",` +
		`"client_secret":"secret","refresh_token":"refresh","quota_project_id":"quota-project"}`
	if err := os.WriteFile(path, []byte(cred), 0o600); err != nil {
		t.Fatalf("writing fake ADC file: %v", err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", path)
}

func TestVertexAIInit_ExpressMode(t *testing.T) {
	// Express mode must work with no project, location, or credentials, in
	// the environment or otherwise.
	clearVertexEnv(t)

	v := &googlegenai.VertexAI{APIKey: "express-key"}
	v.Init(context.Background())

	client, err := v.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.Backend != genai.BackendVertexAI {
		t.Errorf("Backend = %v, want BackendVertexAI", cc.Backend)
	}
	if cc.APIKey != "express-key" {
		t.Errorf("APIKey = %q, want the express key", cc.APIKey)
	}
	if cc.HTTPClient != http.DefaultClient {
		t.Errorf("HTTPClient = %v, want http.DefaultClient", cc.HTTPClient)
	}
}

// TestVertexAIInit_CustomEndpointOnly covers the gateway setup the SDK
// accepts as sufficient on its own: a base URL with no project, location,
// key, or credentials.
func TestVertexAIInit_CustomEndpointOnly(t *testing.T) {
	clearVertexEnv(t)

	v := &googlegenai.VertexAI{BaseURL: "https://my-gateway.example.com"}
	v.Init(context.Background())

	client, err := v.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.HTTPOptions.BaseURL != "https://my-gateway.example.com" {
		t.Errorf("BaseURL = %q, want the configured gateway", cc.HTTPOptions.BaseURL)
	}
	if cc.Project != "" || cc.APIKey != "" {
		t.Errorf("Project = %q, APIKey = %q, want both empty in custom-endpoint mode", cc.Project, cc.APIKey)
	}
}

// TestVertexAIInit_CustomEndpointFromEnv covers the same mode selected by
// GOOGLE_VERTEX_BASE_URL, which the SDK reads on its own: the plugin must
// recognize it as sufficient configuration rather than demand a project.
func TestVertexAIInit_CustomEndpointFromEnv(t *testing.T) {
	clearVertexEnv(t)
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "https://my-gateway.example.com")

	v := &googlegenai.VertexAI{}
	v.Init(context.Background())

	client, err := v.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.HTTPOptions.BaseURL != "https://my-gateway.example.com" {
		t.Errorf("BaseURL = %q, want the env gateway resolved by the SDK", cc.HTTPOptions.BaseURL)
	}
	if cc.Project != "" || cc.APIKey != "" {
		t.Errorf("Project = %q, APIKey = %q, want both empty in custom-endpoint mode", cc.Project, cc.APIKey)
	}
}

func TestVertexAIInit_ExpressModeConflicts(t *testing.T) {
	msg := mustPanic(t, func() {
		(&googlegenai.VertexAI{APIKey: "k", ProjectID: "p"}).Init(context.Background())
	})
	if !strings.Contains(fmt.Sprint(msg), "mutually exclusive") {
		t.Errorf("panic = %v, want a mutual-exclusion explanation", msg)
	}
}

// TestVertexAIInit_ExpressModeFromEnv covers the case that used to be a hard
// panic: no project anywhere, but an Express Mode key in the environment.
func TestVertexAIInit_ExpressModeFromEnv(t *testing.T) {
	for _, name := range []string{"VERTEX_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"} {
		t.Run(name, func(t *testing.T) {
			clearVertexEnv(t)
			t.Setenv(name, "env-express-key")

			v := &googlegenai.VertexAI{}
			v.Init(context.Background())

			client, err := v.Client()
			if err != nil {
				t.Fatalf("Client: %v", err)
			}
			cc := client.ClientConfig()
			if cc.APIKey != "env-express-key" {
				t.Errorf("APIKey = %q, want the key from %s", cc.APIKey, name)
			}
			if cc.Project != "" {
				t.Errorf("Project = %q, want empty in Express Mode", cc.Project)
			}
		})
	}
}

func TestVertexAIInit_ExpressModeEnvPrecedence(t *testing.T) {
	clearVertexEnv(t)
	t.Setenv("VERTEX_API_KEY", "vertex-key")
	t.Setenv("GOOGLE_API_KEY", "google-key")
	t.Setenv("GOOGLE_GENAI_API_KEY", "genai-key")

	v := &googlegenai.VertexAI{}
	v.Init(context.Background())

	client, err := v.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	if got := client.ClientConfig().APIKey; got != "vertex-key" {
		t.Errorf("APIKey = %q, want VERTEX_API_KEY to win", got)
	}
}

// TestVertexAIInit_GeminiAPIKeyIgnored guards the footgun of a process that
// also uses the Google AI plugin: GEMINI_API_KEY names a Gemini Developer API
// key, which Vertex AI does not accept, so it must not silently select
// Express Mode.
func TestVertexAIInit_GeminiAPIKeyIgnored(t *testing.T) {
	clearVertexEnv(t)
	t.Setenv("GEMINI_API_KEY", "developer-api-key")

	msg := mustPanic(t, func() {
		(&googlegenai.VertexAI{}).Init(context.Background())
	})
	if !strings.Contains(fmt.Sprint(msg), "GOOGLE_CLOUD_PROJECT") {
		t.Errorf("panic = %v, want it to ask for a project", msg)
	}
}

// TestVertexAIInit_EnvProjectBeatsEnvAPIKey is the regression guard for
// existing deployments: an ambient API key must not take a
// project-configured deployment into Express Mode.
func TestVertexAIInit_EnvProjectBeatsEnvAPIKey(t *testing.T) {
	clearVertexEnv(t)
	fakeADC(t)
	t.Setenv("GOOGLE_CLOUD_PROJECT", "env-project")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
	t.Setenv("GOOGLE_API_KEY", "ambient-key")

	v := &googlegenai.VertexAI{}
	v.Init(context.Background())

	client, err := v.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.APIKey != "" {
		t.Errorf("APIKey = %q, want the ambient key ignored", cc.APIKey)
	}
	if cc.Project != "env-project" {
		t.Errorf("Project = %q, want env-project", cc.Project)
	}
}

// TestVertexAIInit_EnvLocationSuppressesEnvAPIKey mirrors the genai SDK: a
// location in the environment is enough to mean "not Express Mode", so the
// caller gets the actionable missing-project message instead of an
// authentication failure at request time.
func TestVertexAIInit_EnvLocationSuppressesEnvAPIKey(t *testing.T) {
	clearVertexEnv(t)
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
	t.Setenv("GOOGLE_API_KEY", "ambient-key")

	msg := mustPanic(t, func() {
		(&googlegenai.VertexAI{}).Init(context.Background())
	})
	if !strings.Contains(fmt.Sprint(msg), "GOOGLE_CLOUD_PROJECT") {
		t.Errorf("panic = %v, want it to ask for a project", msg)
	}
}

func TestVertexAIInit_ExplicitConfigBeatsEnvAPIKey(t *testing.T) {
	clearVertexEnv(t)
	fakeADC(t)
	t.Setenv("GOOGLE_API_KEY", "ambient-key")

	v := &googlegenai.VertexAI{ProjectID: "explicit-project", Location: "us-central1"}
	v.Init(context.Background())

	client, err := v.Client()
	if err != nil {
		t.Fatalf("Client: %v", err)
	}
	cc := client.ClientConfig()
	if cc.APIKey != "" {
		t.Errorf("APIKey = %q, want the ambient key ignored", cc.APIKey)
	}
	if cc.Project != "explicit-project" {
		t.Errorf("Project = %q, want explicit-project", cc.Project)
	}
}

// TestVertexAIInit_CredentialsBeatEnvAPIKey verifies that supplying
// credentials opts out of Express Mode entirely, even with a key in the
// environment and nothing else configured.
func TestVertexAIInit_CredentialsBeatEnvAPIKey(t *testing.T) {
	clearVertexEnv(t)
	t.Setenv("GOOGLE_API_KEY", "ambient-key")

	creds := auth.NewCredentials(&auth.CredentialsOptions{
		TokenProvider: auth.NewCachedTokenProvider(tokenProviderFunc(func(ctx context.Context) (*auth.Token, error) {
			return &auth.Token{Value: "token", Expiry: time.Now().Add(time.Hour)}, nil
		}), nil),
	})

	msg := mustPanic(t, func() {
		(&googlegenai.VertexAI{Credentials: creds}).Init(context.Background())
	})
	if !strings.Contains(fmt.Sprint(msg), "GOOGLE_CLOUD_PROJECT") {
		t.Errorf("panic = %v, want it to ask for a project rather than using the ambient key", msg)
	}
}

type tokenProviderFunc func(context.Context) (*auth.Token, error)

func (f tokenProviderFunc) Token(ctx context.Context) (*auth.Token, error) { return f(ctx) }

func TestClient_RequiresInit(t *testing.T) {
	if _, err := (&googlegenai.GoogleAI{}).Client(); err == nil {
		t.Error("GoogleAI.Client before Init = nil error, want error")
	}
	if _, err := (&googlegenai.VertexAI{}).Client(); err == nil {
		t.Error("VertexAI.Client before Init = nil error, want error")
	}
}
