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
	"errors"
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
	// initErr records a credential/project-resolution failure from Init. It is
	// returned when the evaluator runs, rather than panicking at startup.
	initErr error
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
// checks/guardrails evaluator (when at least one metric is configured). A
// credential or project-resolution failure is recorded in c.initErr and
// surfaced when the evaluator runs, so a misconfigured plugin doesn't crash the
// whole application at startup.
func (c *Checks) Init(ctx context.Context) []api.Action {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initted {
		panic("checks.Init already called")
	}
	c.initted = true

	// No metrics means no evaluator to register, so skip credential and project
	// resolution entirely — registering the plugin shouldn't require ADC when the
	// evaluator isn't used (the exported Guardrails client resolves its own creds).
	if len(c.Metrics) == 0 {
		return []api.Action{}
	}

	creds, err := google.FindDefaultCredentials(ctx, cloudPlatformScope, checksScope)
	if err != nil {
		c.initErr = fmt.Errorf("checks: unable to find default credentials: %w", err)
	} else if creds.TokenSource == nil {
		c.initErr = errors.New("checks: missing or invalid credentials")
	} else {
		c.ts = creds.TokenSource
		c.projectID = resolveProjectID(c.ProjectID, creds)
		if c.projectID == "" {
			c.initErr = errors.New("checks: missing project ID; set Checks.ProjectID or the GOOGLE_CLOUD_PROJECT environment variable")
		}
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
