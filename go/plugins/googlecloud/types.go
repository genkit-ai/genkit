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
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"golang.org/x/oauth2/google"
)

// Telemetry interface that all telemetry modules implement
type Telemetry interface {
	Tick(span sdktrace.ReadOnlySpan, logInputOutput bool, projectID string)
}

// SharedDimensions contains common metric dimensions used across telemetry modules
type SharedDimensions struct {
	FeatureName   string
	Path          string
	Status        string
	Source        string
	SourceVersion string
}

// GoogleCloudTelemetryOptions provides comprehensive configuration for Google Cloud telemetry.
// Matches the JavaScript Google Cloud plugin options for full compatibility.
//
// Environment Variables:
// - GENKIT_ENV: Set to "dev" to disable export unless ForceDevExport is true
// - GOOGLE_CLOUD_PROJECT: Auto-detected project ID if ProjectID is not set
// - GCLOUD_PROJECT: Auto-detected project ID if ProjectID is not set
type GoogleCloudTelemetryOptions struct {
	// ProjectID is the Google Cloud project ID.
	// If empty, will be auto-detected from environment.
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
