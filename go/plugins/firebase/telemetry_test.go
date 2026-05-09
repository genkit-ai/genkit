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

package firebase

import (
	"os"
	"testing"
)

func TestEnableFirebaseTelemetry(t *testing.T) {
	// Test zero config doesn't crash
	EnableFirebaseTelemetry(nil)

	// Test with options doesn't crash
	options := &FirebaseTelemetryOptions{
		ProjectID:      "test-project",
		ForceDevExport: true,
	}
	EnableFirebaseTelemetry(options)
}

func TestProjectIDResolution(t *testing.T) {
	tests := []struct {
		input    string
		fbEnv    string
		gcpEnv   string
		expected string
	}{
		{"explicit", "firebase", "gcp", "explicit"}, // Explicit wins
		{"", "firebase", "gcp", "firebase"},         // Firebase env wins
		{"", "", "gcp", "gcp"},                      // GCP fallback
		{"", "", "", ""},                            // Empty fallback
	}

	for _, tt := range tests {
		if tt.fbEnv != "" {
			os.Setenv("FIREBASE_PROJECT_ID", tt.fbEnv)
		} else {
			os.Unsetenv("FIREBASE_PROJECT_ID")
		}
		if tt.gcpEnv != "" {
			os.Setenv("GOOGLE_CLOUD_PROJECT", tt.gcpEnv)
		} else {
			os.Unsetenv("GOOGLE_CLOUD_PROJECT")
		}

		result := resolveFirebaseProjectID(tt.input)
		if result != tt.expected {
			t.Errorf("resolveFirebaseProjectID(%q) = %q, want %q", tt.input, result, tt.expected)
		}

		// Cleanup
		os.Unsetenv("FIREBASE_PROJECT_ID")
		os.Unsetenv("GOOGLE_CLOUD_PROJECT")
	}
}
