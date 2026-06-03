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

// Package checks provides a Genkit plugin for Google Checks AI Safety. It
// exposes a "checks/guardrails" evaluator for batch evaluation and an exported
// [Guardrails] client (plus [GuardrailMiddleware]) for synchronous, in-flight
// content classification. It is a port of the JS @genkit-ai/checks plugin built
// directly on net/http + ADC against the Checks v1alpha REST API.
package checks

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/firebase/genkit/go/core/api"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	provider = "checks"

	// classifyEndpoint is the Checks AI Safety classifyContent REST endpoint.
	classifyEndpoint = "https://checks.googleapis.com/v1alpha/aisafety:classifyContent"

	cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
	checksScope        = "https://www.googleapis.com/auth/checks"
)

// PolicyType is a Checks AI safety policy.
type PolicyType string

const (
	DangerousContent      PolicyType = "DANGEROUS_CONTENT"
	PIISolicitingReciting PolicyType = "PII_SOLICITING_RECITING"
	Harassment            PolicyType = "HARASSMENT"
	SexuallyExplicit      PolicyType = "SEXUALLY_EXPLICIT"
	HateSpeech            PolicyType = "HATE_SPEECH"
	MedicalInfo           PolicyType = "MEDICAL_INFO"
	ViolenceAndGore       PolicyType = "VIOLENCE_AND_GORE"
	ObscenityAndProfanity PolicyType = "OBSCENITY_AND_PROFANITY"
)

// ChecksMetricConfig configures a single policy to evaluate against. Threshold
// is optional; when nil the API uses its default for that policy.
type ChecksMetricConfig struct {
	Type      PolicyType
	Threshold *float64
}

// Checks is the Genkit plugin for Google Checks AI Safety.
type Checks struct {
	// ProjectID is the Google Cloud project with Checks API quota. If empty,
	// resolved from GOOGLE_CLOUD_PROJECT then Application Default Credentials.
	ProjectID string
	// Metrics is the set of policies the checks/guardrails evaluator scores.
	Metrics []ChecksMetricConfig

	mu        sync.Mutex
	initted   bool
	ts        oauth2.TokenSource
	projectID string
	endpoint  string // overridable classifyContent URL; defaults to classifyEndpoint (tests).
}

// classifyURL returns the configured endpoint or the default Checks endpoint.
func (c *Checks) classifyURL() string {
	if c.endpoint != "" {
		return c.endpoint
	}
	return classifyEndpoint
}

var _ api.Plugin = (*Checks)(nil)

// Name returns the plugin provider name.
func (c *Checks) Name() string { return provider }

// Init resolves credentials and project ID, then registers the
// checks/guardrails evaluator (when at least one metric is configured).
func (c *Checks) Init(ctx context.Context) []api.Action {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initted {
		panic("checks.Init already called")
	}

	creds, err := google.FindDefaultCredentials(ctx, cloudPlatformScope, checksScope)
	if err != nil {
		panic(fmt.Sprintf("checks: unable to find default credentials: %v", err))
	}
	if creds.TokenSource == nil {
		panic("checks: missing or invalid credentials")
	}
	c.ts = creds.TokenSource

	c.projectID = resolveProjectID(c.ProjectID, creds)
	if c.projectID == "" {
		panic("checks: missing project ID; set Checks.ProjectID or the GOOGLE_CLOUD_PROJECT environment variable")
	}

	c.initted = true

	if len(c.Metrics) == 0 {
		return []api.Action{}
	}
	return []api.Action{newEvaluator(c).(api.Action)}
}

// resolveProjectID applies the precedence: explicit > GOOGLE_CLOUD_PROJECT > ADC.
func resolveProjectID(explicit string, creds *google.Credentials) string {
	if explicit != "" {
		return explicit
	}
	if v := os.Getenv("GOOGLE_CLOUD_PROJECT"); v != "" {
		return v
	}
	if creds != nil {
		return creds.ProjectID
	}
	return ""
}
