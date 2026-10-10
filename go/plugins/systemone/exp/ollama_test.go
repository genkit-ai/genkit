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

package exp

import (
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/systemone/systemonetest"
)

func TestOllama(t *testing.T) {
	fake := &systemonetest.Server{}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	p := Ollama()
	p.BaseURL = srv.URL
	g := genkit.Init(t.Context(), genkit.WithPlugins(p))
	const png = "iVBORw0KGgo="

	t.Run("images as raw base64", func(t *testing.T) {
		if _, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("ollama-decisions/clef"),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Is the parcel damaged?"), ai.NewMediaPart("image/png", "data:image/png;base64,"+png)))); err != nil {
			t.Fatal(err)
		}
		req, body := fake.Last(t)
		if req.URL.Path != "/v1/systemone" || body["model"] != "clef" {
			t.Errorf("request = %s model=%v, want /v1/systemone naming clef", req.URL.Path, body["model"])
		}
		if !reflect.DeepEqual(body["images"], []any{png}) {
			t.Errorf("images = %v, want the base64 without the data URL prefix", body["images"])
		}
	})

	t.Run("a request needs text", func(t *testing.T) {
		image := ai.NewMediaPart("image/png", "data:image/png;base64,"+png)
		for _, tt := range []struct {
			name string
			opt  ai.GenerateOption
			want string
		}{
			{"images alone", ai.WithMessages(ai.NewUserMessage(image)), "images needs text"},
			{"turns of images alone", ai.WithMessages(ai.NewUserMessage(image), ai.NewModelMessage(image)), "images needs text"},
			{"empty prompt", ai.WithPrompt(""), "requires text"},
		} {
			calls := fake.Calls()
			_, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("ollama-decisions/clef"), tt.opt)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("%s: error = %v, want it refused with %q", tt.name, err, tt.want)
			}
			if fake.Calls() != calls {
				t.Errorf("%s: the request was sent", tt.name)
			}
		}
	})

	t.Run("lists clef", func(t *testing.T) {
		descs := p.ListActions(t.Context())
		if len(descs) != 1 || descs[0].Name != "ollama-decisions/clef" {
			t.Errorf("listed %d actions, want ollama-decisions/clef alone", len(descs))
		}
	})
}
