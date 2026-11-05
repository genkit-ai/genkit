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

package internal

import (
	"runtime/debug"
	"testing"
)

func TestVersionFromBuildInfo(t *testing.T) {
	tests := []struct {
		name string
		info *debug.BuildInfo
		want string
	}{
		{
			name: "no build info",
			info: nil,
			want: "dev",
		},
		{
			name: "genkit is the main module",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: modulePath, Version: "(devel)"},
			},
			want: "dev",
		},
		{
			name: "genkit is a dependency",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: "example.com/app"},
				Deps: []*debug.Module{
					{Path: "example.com/other", Version: "v2.0.0"},
					{Path: modulePath, Version: "v1.11.0"},
				},
			},
			want: "1.11.0",
		},
		{
			name: "genkit is a dependency at a pseudo-version",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: "example.com/app"},
				Deps: []*debug.Module{
					{Path: modulePath, Version: "v1.11.1-0.20260810123456-abcdef123456"},
				},
			},
			want: "1.11.1-0.20260810123456-abcdef123456",
		},
		{
			name: "genkit replaced by a source checkout",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: "example.com/app"},
				Deps: []*debug.Module{
					{
						Path:    modulePath,
						Version: "v1.11.0",
						Replace: &debug.Module{Path: "../genkit/go", Version: "(devel)"},
					},
				},
			},
			want: "dev",
		},
		{
			name: "genkit replaced by another module version",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: "example.com/app"},
				Deps: []*debug.Module{
					{
						Path:    modulePath,
						Version: "v1.11.0",
						Replace: &debug.Module{Path: "example.com/fork/genkit/go", Version: "v1.2.3"},
					},
				},
			},
			want: "1.2.3",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := versionFromBuildInfo(tt.info); got != tt.want {
				t.Errorf("versionFromBuildInfo() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestVersionInRepo(t *testing.T) {
	// Within this repository Genkit is the main module, so the resolved
	// version must be the source fallback.
	if Version != "dev" {
		t.Errorf("Version = %q, want %q when built from source", Version, "dev")
	}
}
