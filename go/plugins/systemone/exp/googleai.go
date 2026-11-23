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
	"context"
	"net/http"

	genkitinternal "github.com/firebase/genkit/go/internal"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
	"google.golang.org/genai"
)

// GoogleAI is the Decisions API of the Gemini API, which serves Gemini
// Decision 1, reached through the genai SDK. Its models are
// googleai-decisions/<ID>, such as googleai-decisions/gemini-decision-1.
// The name is not googleai, which the googlegenai plugin has for Gemini's
// chat models, so both serve one app. The API key comes from
// GEMINI_API_KEY or GOOGLE_API_KEY, in that order, as it does for the
// googlegenai plugin.
//
// Its models read images, audio, video, and PDFs, inline or by an HTTPS
// URL the API fetches, such as a Files API URI, and Config.MediaResolution
// sets how closely images and video are read. Guidance and structured
// instructions go to the model as their JSON text, since the API takes
// text only for both, and a choice's options go in key order rather than
// the order declared. An input the API refuses as unsafe is a response
// with [ai.FinishReasonBlocked], not an error, and the response names no
// model version, so [ResponseInfo] reports the model the request named.
//
// The SDK builds the whole request and retries it, so Config.Extra is
// refused, and Path, ModelsPath, Route, and Unwrap do not apply. BaseURL
// can point at a proxy that forwards the Gemini API, and HTTPClient and
// Headers reach the SDK's transport.
func GoogleAI() *SystemOne {
	return &SystemOne{
		Provider: "googleai-decisions",
		Images:   MediaSupported,
		Audio:    MediaSupported,
		Video:    MediaSupported,
		Models:   map[string]ModelSpec{"gemini-decision-1": {}},
		preset: preset{
			label:           "Google AI",
			apiKeyEnvs:      []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"},
			baseURL:         "https://generativelanguage.googleapis.com",
			transport:       newGemini,
			headers:         http.Header{"X-Goog-Api-Client": {"genkit-go/" + genkitinternal.Version}},
			documents:       true,
			urls:            true,
			mediaResolution: true,
		},
	}
}

// newGemini builds the genai SDK client the Decisions API is reached on.
func newGemini(ctx context.Context, c transportConfig) (systemone.Transport, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:     c.apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: c.http,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: c.baseURL,
			Headers: c.headers,
		},
	})
	if err != nil {
		return nil, err
	}
	return &systemone.Gemini{Name: c.name, Client: client}, nil
}
