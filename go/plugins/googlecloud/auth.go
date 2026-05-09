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

package googlecloud

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"golang.org/x/oauth2/google"
)

// GcpPrincipal represents the current GCP principal information
type GcpPrincipal struct {
	ProjectID           string `json:"project_id,omitempty"`
	ServiceAccountEmail string `json:"service_account_email,omitempty"`
}

// GcpAuthConfig represents authentication configuration for GCP
type GcpAuthConfig struct {
	ProjectID   string
	Credentials *google.Credentials
}

// CredentialsFromEnvironment allows Google Cloud credentials to be passed in "raw" as an environment
// variable. This is helpful in environments where the developer has limited
// ability to configure their compute environment, but does have the ability to
// set environment variables.
//
// This is different from the GOOGLE_APPLICATION_CREDENTIALS used by ADC, which
// represents a path to a credential file on disk. In *most* cases, even for
// 3rd party cloud providers, developers *should* attempt to use ADC, which
// searches for credential files in standard locations, before using this
// method.
//
// See also: https://cloud.google.com/docs/authentication/provide-credentials-adc
func CredentialsFromEnvironment() (*GcpAuthConfig, error) {
	ctx := context.Background()
	config := &GcpAuthConfig{}

	if credsJSON := os.Getenv("GCLOUD_SERVICE_ACCOUNT_CREDS"); credsJSON != "" {
		var serviceAccountCreds map[string]interface{}
		if err := json.Unmarshal([]byte(credsJSON), &serviceAccountCreds); err != nil {
			return nil, fmt.Errorf("failed to parse GCLOUD_SERVICE_ACCOUNT_CREDS: %w", err)
		}

		creds, err := google.CredentialsFromJSON(ctx, []byte(credsJSON))
		if err != nil {
			return nil, fmt.Errorf("failed to create credentials from service account: %w", err)
		}

		config.Credentials = creds

		if projectID, ok := serviceAccountCreds["project_id"].(string); ok && projectID != "" {
			config.ProjectID = projectID
		}
	} else {
		creds, err := google.FindDefaultCredentials(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to find default credentials: %w", err)
		}

		config.Credentials = creds
		config.ProjectID = creds.ProjectID
	}
	if config.ProjectID == "" && config.Credentials != nil {
		if config.Credentials.ProjectID != "" {
			config.ProjectID = config.Credentials.ProjectID
		}
	}

	return config, nil
}

// ResolveCurrentPrincipal resolves the currently configured principal, either from the Genkit specific
// GCLOUD_SERVICE_ACCOUNT_CREDS environment variable, or from ADC.
//
// Since the Google Cloud Telemetry Exporter will discover credentials on its
// own, we don't immediately have access to the current principal. This method
// can be handy to get access to the current credential for logging debugging
// information or other purposes.
func ResolveCurrentPrincipal() (*GcpPrincipal, error) {
	ctx := context.Background()
	principal := &GcpPrincipal{}

	envConfig, err := CredentialsFromEnvironment()
	if err != nil {
		adcCreds, adcErr := google.FindDefaultCredentials(ctx)
		if adcErr != nil {
			return principal, fmt.Errorf("could not resolve credentials from environment or ADC: %w", err)
		}

		principal.ProjectID = adcCreds.ProjectID
		if email := extractServiceAccountEmail(adcCreds); email != "" {
			principal.ServiceAccountEmail = email
		}

		return principal, nil
	}
	principal.ProjectID = envConfig.ProjectID
	if email := extractServiceAccountEmail(envConfig.Credentials); email != "" {
		principal.ServiceAccountEmail = email
	}

	return principal, nil
}

// extractServiceAccountEmail extracts the service account email from Google credentials
func extractServiceAccountEmail(creds *google.Credentials) string {
	if creds == nil {
		return ""
	}

	if creds.JSON != nil {
		var serviceAccount map[string]interface{}
		if err := json.Unmarshal(creds.JSON, &serviceAccount); err == nil {
			if email, ok := serviceAccount["client_email"].(string); ok {
				return email
			}
		}
	}

	return ""
}
