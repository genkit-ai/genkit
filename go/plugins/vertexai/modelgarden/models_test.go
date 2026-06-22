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

package modelgarden

import (
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestResolveVertexMaasEnv_ExplicitArgsWin(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "from-env")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "from-env")

	p, l := resolveVertexMaasEnv("explicit-proj", "explicit-loc")
	if p != "explicit-proj" {
		t.Errorf("project = %q, want explicit-proj", p)
	}
	if l != "explicit-loc" {
		t.Errorf("location = %q, want explicit-loc", l)
	}
}

func TestResolveVertexMaasEnv_FallsBackToPrimaryEnv(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "primary-proj")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "primary-loc")
	t.Setenv("GCLOUD_PROJECT", "secondary-proj")
	t.Setenv("GOOGLE_CLOUD_REGION", "secondary-loc")

	p, l := resolveVertexMaasEnv("", "")
	if p != "primary-proj" {
		t.Errorf("project = %q, want primary-proj", p)
	}
	if l != "primary-loc" {
		t.Errorf("location = %q, want primary-loc", l)
	}
}

func TestResolveVertexMaasEnv_FallsBackToSecondaryEnv(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GCLOUD_PROJECT", "secondary-proj")
	t.Setenv("GOOGLE_CLOUD_REGION", "secondary-loc")

	p, l := resolveVertexMaasEnv("", "")
	if p != "secondary-proj" {
		t.Errorf("project = %q, want secondary-proj", p)
	}
	if l != "secondary-loc" {
		t.Errorf("location = %q, want secondary-loc", l)
	}
}

func TestResolveVertexMaasEnv_PanicsWithoutProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic when no project env is set")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "GOOGLE_CLOUD_PROJECT") {
			t.Fatalf("panic = %v, want message mentioning GOOGLE_CLOUD_PROJECT", r)
		}
	}()
	resolveVertexMaasEnv("", "")
}

// TestAnthropicModels_DeprecatedAliases pins the backwards-compat surface:
// the date-suffixed IDs that shipped before the undated rename must remain
// in AnthropicModels and must be marked deprecated, so callers pinned to
// those keys keep resolving and get a warning via model_middleware.
func TestAnthropicModels_DeprecatedAliases(t *testing.T) {
	aliases := []string{
		"claude-opus-4@20250514",
		"claude-sonnet-4@20250514",
		"claude-3-7-sonnet@20250219",
		"claude-3-5-sonnet-v2@20241022",
		"claude-3-5-sonnet@20240620",
		"claude-3-sonnet@20240229",
		"claude-3-haiku@20240307",
		"claude-3-opus@20240229",
	}
	for _, id := range aliases {
		opts, ok := AnthropicModels[id]
		if !ok {
			t.Errorf("alias %q missing from AnthropicModels", id)
			continue
		}
		if opts.Stage != ai.ModelStageDeprecated {
			t.Errorf("alias %q stage = %q, want %q", id, opts.Stage, ai.ModelStageDeprecated)
		}
	}
}

func TestResolveVertexMaasEnv_PanicsWithoutLocation(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "some-proj")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_REGION", "")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic when no location env is set")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "GOOGLE_CLOUD_LOCATION") {
			t.Fatalf("panic = %v, want message mentioning GOOGLE_CLOUD_LOCATION", r)
		}
	}()
	resolveVertexMaasEnv("", "")
}
