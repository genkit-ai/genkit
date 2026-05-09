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

package firebase

import (
	"log/slog"
	"os"

	"github.com/firebase/genkit/go/plugins/googlecloud"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"golang.org/x/oauth2/google"
)

// FirebaseTelemetryOptions provides comprehensive configuration for Firebase telemetry.
// This matches the Google Cloud plugin options for full compatibility, with Firebase-specific
// project ID resolution that checks FIREBASE_PROJECT_ID first.
//
// Environment Variables:
// - GENKIT_ENV: Set to "dev" to disable export unless ForceDevExport is true
// - FIREBASE_PROJECT_ID: Project ID for telemetry if not provided inline
// - GOOGLE_CLOUD_PROJECT: Project ID for telemetry if not provided inline
// - GCLOUD_PROJECT: Project ID for telemetry if not provided inline
type FirebaseTelemetryOptions struct {
	// ProjectID is the Firebase/Google Cloud project ID.
	// If empty, will be auto-detected from environment variables in priority order:
	// 1. FIREBASE_PROJECT_ID, 2. GOOGLE_CLOUD_PROJECT, 3. GCLOUD_PROJECT
	ProjectID string

	// Credentials for authenticating with Google Cloud.
	// If nil, uses Application Default Credentials (ADC).
	// Primarily intended for use in environments outside of GCP.
	// On GCP, credentials will typically be inferred from the environment via ADC.
	Credentials *google.Credentials

	// Sampler controls trace sampling. If nil, uses AlwaysOnSampler.
	// Examples: AlwaysOnSampler, AlwaysOffSampler, TraceIdRatioBasedSampler
	Sampler sdktrace.Sampler

	// MetricExportIntervalMillis controls metrics export frequency.
	// Google Cloud requires minimum 5000ms. Defaults to 5000 (dev) or 300000 (prod).
	MetricExportIntervalMillis *int

	// MetricExportTimeoutMillis controls metrics export timeout.
	// Defaults to match MetricExportIntervalMillis.
	MetricExportTimeoutMillis *int

	// DisableMetrics disables metric export to Google Cloud.
	// Traces and logs may still be exported. Defaults to false.
	DisableMetrics bool

	// DisableTraces disables trace export to Google Cloud.
	// Metrics and logs may still be exported. Defaults to false.
	DisableTraces bool

	// DisableLoggingInputAndOutput disables input/output logging.
	// Defaults to false (input/output logging enabled).
	DisableLoggingInputAndOutput bool

	// ForceDevExport forces telemetry export even in development environment.
	// Defaults to false (only exports in production unless forced).
	ForceDevExport bool
}

// EnableFirebaseTelemetry enables comprehensive telemetry export to Genkit Monitoring,
// backed by Google Cloud Observability (Cloud Logging, Metrics, and Trace).
//
// Example usage:
//
//	// Basic usage - uses environment variables for project ID
//	firebase.EnableFirebaseTelemetry(nil)
//	g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
//
//	// With full configuration
//	firebase.EnableFirebaseTelemetry(&firebase.FirebaseTelemetryOptions{
//		ProjectID:                  "my-firebase-project",
//		ForceDevExport:             true,
//		DisableLoggingInputAndOutput: false,
//		DisableMetrics:             false,
//		DisableTraces:              false,
//		MetricExportIntervalMillis: &[]int{10000}[0], // 10 seconds
//	})
//	g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
func EnableFirebaseTelemetry(options *FirebaseTelemetryOptions) {
	if options == nil {
		options = &FirebaseTelemetryOptions{}
	}
	initializeTelemetry(options)
}

// initializeTelemetry is the internal function that sets up Firebase telemetry.
func initializeTelemetry(options *FirebaseTelemetryOptions) {
	slog.Debug("Initializing Firebase Genkit Monitoring.")

	if options == nil {
		options = &FirebaseTelemetryOptions{}
	}

	projectID := resolveFirebaseProjectID(options.ProjectID)
	if projectID == "" {
		slog.Warn("Firebase project ID not found. Set FIREBASE_PROJECT_ID, GOOGLE_CLOUD_PROJECT, or GCLOUD_PROJECT environment variable, or pass ProjectID in options.")
	}

	gcOptions := &googlecloud.GoogleCloudTelemetryOptions{
		ProjectID:                    projectID,
		Credentials:                  options.Credentials,
		Sampler:                      options.Sampler,
		MetricExportIntervalMillis:   options.MetricExportIntervalMillis,
		MetricExportTimeoutMillis:    options.MetricExportTimeoutMillis,
		DisableMetrics:               options.DisableMetrics,
		DisableTraces:                options.DisableTraces,
		DisableLoggingInputAndOutput: options.DisableLoggingInputAndOutput,
		ForceDevExport:               options.ForceDevExport,
	}
	googlecloud.EnableGoogleCloudTelemetry(gcOptions)
}

// resolveFirebaseProjectID resolves the Firebase project ID from various sources.
// Priority: 1) Provided projectID, 2) FIREBASE_PROJECT_ID, 3) GOOGLE_CLOUD_PROJECT, 4) GCLOUD_PROJECT
func resolveFirebaseProjectID(projectID string) string {
	if projectID != "" {
		return projectID
	}

	if envID := os.Getenv("FIREBASE_PROJECT_ID"); envID != "" {
		return envID
	}

	if envID := os.Getenv("GOOGLE_CLOUD_PROJECT"); envID != "" {
		return envID
	}

	return os.Getenv("GCLOUD_PROJECT")
}
