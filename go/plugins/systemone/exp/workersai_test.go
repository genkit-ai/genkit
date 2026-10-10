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
	"errors"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/systemone/systemonetest"
)

// newWorkersAI starts a fake endpoint that answers in the Workers AI
// envelope, and a Genkit with the plugin pointed at it under the path the
// account's API has.
func newWorkersAI(t *testing.T, fake *systemonetest.Server) *genkit.Genkit {
	t.Helper()
	fake.Wrap = func(reply map[string]any) any {
		return map[string]any{"result": reply, "success": true, "errors": []any{}, "messages": []any{}}
	}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	p := WorkersAI("acct/1")
	p.APIKey = "test-token"
	p.BaseURL = srv.URL + "/client/v4/accounts/{account}"
	return genkit.Init(t.Context(), genkit.WithPlugins(p))
}

func TestWorkersAI(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newWorkersAI(t, fake)
	const png = "data:image/png;base64,iVBORw0KGgo="

	t.Run("Cloudflare's own model", func(t *testing.T) {
		out, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef"),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Is the parcel damaged?"), ai.NewMediaPart("image/png", png))))
		if err != nil {
			t.Fatal(err)
		}
		req, body := fake.Last(t)
		// The account is escaped into the path.
		if req.URL.EscapedPath() != "/client/v4/accounts/acct%2F1/ai/run/@cf/cloudflare/clef" {
			t.Errorf("path = %s, want the model's own route", req.URL.EscapedPath())
		}
		if body["model"] != "clef" || body["state"] != "Is the parcel damaged?" || !reflect.DeepEqual(body["images"], []any{png}) {
			t.Errorf("body = %v, want the native body naming clef, with the image", body)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("the envelope was not unwrapped: %+v", out)
		}
	})

	t.Run("Clef-omni hears and sees", func(t *testing.T) {
		const wav, mp4 = "data:audio/wav;base64,UklGRg==", "data:video/mp4;base64,AAAAIGZ0eXA="
		clip := ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Is the fan running?"), ai.NewMediaPart("audio/wav", wav), ai.NewMediaPart("video/mp4", mp4)))
		if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef-omni"), clip); err != nil {
			t.Fatal(err)
		}
		if _, body := fake.Last(t); body["model"] != "clef-omni" || !reflect.DeepEqual(body["audio"], []any{wav}) || !reflect.DeepEqual(body["videos"], []any{mp4}) {
			t.Errorf("body = %v, want the audio and the video beside the state", body)
		}
		// Clef reads images only, so it refuses the clip before sending.
		calls := fake.Calls()
		_, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef"), clip)
		if err == nil || !strings.Contains(err.Error(), "audio/wav media") || fake.Calls() != calls {
			t.Errorf("error = %v, want the audio refused before sending", err)
		}
	})

	t.Run("partner's model", func(t *testing.T) {
		if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/typesafe/jev"), ai.WithPrompt("hi")); err != nil {
			t.Fatal(err)
		}
		req, body := fake.Last(t)
		if req.URL.EscapedPath() != "/client/v4/accounts/acct%2F1/ai/run" || body["model"] != "typesafe/jev" {
			t.Errorf("request = %s model=%v, want /ai/run naming typesafe/jev", req.URL.EscapedPath(), body["model"])
		}
		input, _ := body["input"].(map[string]any)
		if input["state"] != "hi" || input["questions"] == nil || input["model"] != nil || body["state"] != nil {
			t.Errorf("body = %v, want the native body under input, without a model", body)
		}
	})

	t.Run("text-only partner's model", func(t *testing.T) {
		_, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("cloudflare-decisions/typesafe/jev"),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("hi"), ai.NewMediaPart("image/png", png))))
		if err == nil || !strings.Contains(err.Error(), "media") {
			t.Errorf("error = %v, want the image refused", err)
		}
	})

	t.Run("model ID escaped into the path", func(t *testing.T) {
		if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/x?y"), ai.WithPrompt("hi")); err != nil {
			t.Fatal(err)
		}
		if req, _ := fake.Last(t); req.URL.EscapedPath() != "/client/v4/accounts/acct%2F1/ai/run/@cf/x%3Fy" || req.URL.RawQuery != "" {
			t.Errorf("request = %s?%s, want the ID escaped in the path", req.URL.EscapedPath(), req.URL.RawQuery)
		}
		calls := fake.Calls()
		_, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/../../tokens"), ai.WithPrompt("hi"))
		if !errors.Is(err, status.ErrInvalidArgument) || fake.Calls() != calls {
			t.Errorf("error = %v, want a dot segment refused before sending", err)
		}
	})

	t.Run("bare output", func(t *testing.T) {
		wrap := fake.Wrap
		fake.Wrap = nil
		t.Cleanup(func() { fake.Wrap = wrap })
		out, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef-flash"), ai.WithPrompt("hi"))
		if err != nil {
			t.Fatal(err)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("the bare output was not read: %+v", out)
		}
	})

	t.Run("failed envelope", func(t *testing.T) {
		fake.Wrap = func(map[string]any) any {
			return map[string]any{"result": nil, "success": false, "errors": []any{map[string]any{"code": 7000, "message": "No route for that URI"}}}
		}
		_, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef-flash"), ai.WithPrompt("hi"))
		if !errors.Is(err, status.ErrUnknown) || !strings.Contains(err.Error(), "cloudflare-decisions: No route") {
			t.Errorf("error = %v, want UNKNOWN with Cloudflare's message under the plugin's name", err)
		}
	})
}

func TestWorkersAIRequiresAnAccount(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	defer func() {
		if r := recover(); r == nil || !strings.Contains(r.(string), "CLOUDFLARE_ACCOUNT_ID") {
			t.Errorf("Init without an account ID: recovered %v, want a panic naming the variable", r)
		}
	}()
	p := WorkersAI("")
	p.APIKey = "test-token"
	p.Init(t.Context())
}

func TestWorkersAIBehindAProxy(t *testing.T) {
	// A base URL with no {account} segment needs no account ID.
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	fake := &systemonetest.Server{Wrap: func(reply map[string]any) any {
		return map[string]any{"result": reply, "success": true}
	}}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	p := WorkersAI("")
	p.APIKey = "test-token"
	p.BaseURL = srv.URL + "/proxy"
	g := genkit.Init(t.Context(), genkit.WithPlugins(p))

	if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef"), ai.WithPrompt("hi")); err != nil {
		t.Fatal(err)
	}
	if req, _ := fake.Last(t); req.URL.Path != "/proxy/ai/run/@cf/cloudflare/clef" {
		t.Errorf("path = %s, want the model's route under the proxy", req.URL.Path)
	}
}
