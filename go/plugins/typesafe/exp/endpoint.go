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
	"encoding/json"
	"strings"

	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
)

// Endpoint is one server that serves jev. TypeSafe's own API, OpenRouter,
// and Cloudflare Workers AI all take the same questions and return the
// same answers; they differ in the URL, the model IDs, and the envelope
// around the body, which is what an Endpoint captures.
//
// Endpoints are built by [Direct], [OpenRouter], and [Cloudflare]. A proxy
// that forwards the native protocol, such as LiteLLM, is [Direct] with
// [TypeSafe.BaseURL] pointed at it.
type Endpoint struct {
	ep *systemone.Endpoint
}

// Direct is TypeSafe's own API. The API key comes from TYPESAFE_API_KEY and
// the base URL from TYPESAFE_BASE_URL, the variables the vendor SDKs read.
func Direct() *Endpoint {
	return &Endpoint{&systemone.Endpoint{
		Name:       "typesafe",
		BaseURL:    "https://api.typesafe.ai",
		BaseURLEnv: "TYPESAFE_BASE_URL",
		Path:       "/v1/systemone",
		APIKeyEnv:  "TYPESAFE_API_KEY",
		ModelsPath: "/v1/models",
		Models:     []string{"jev-latest", "jev-1.13.0"},
	}}
}

// OpenRouter serves jev through its Decisions API, an alpha endpoint that
// takes the native body. The API key comes from OPENROUTER_API_KEY. Model
// IDs are translated: jev-latest is ~typesafe/jev-latest there, and a
// release is named by minor version, jev-1.13 for typesafe/jev-1.13. A
// patch version such as jev-1.13.0 is refused, since OpenRouter cannot
// pin one.
func OpenRouter() *Endpoint {
	return &Endpoint{&systemone.Endpoint{
		Name:      "openrouter",
		BaseURL:   "https://openrouter.ai",
		Path:      "/api/alpha/decisions",
		APIKeyEnv: "OPENROUTER_API_KEY",
		Models:    []string{"jev-latest", "jev-1.13"},
		Route: func(model string, body map[string]any) (string, any, error) {
			id, err := openRouterModelID(model)
			body["model"] = id
			return "", body, err
		},
	}}
}

// Cloudflare serves jev on Workers AI under the alias typesafe/jev only, so
// a versioned model ID is rejected rather than silently served by whatever
// the alias points at. The API token comes from CLOUDFLARE_API_TOKEN, and
// the account ID from CLOUDFLARE_ACCOUNT_ID when accountID is empty.
func Cloudflare(accountID string) *Endpoint {
	return &Endpoint{&systemone.Endpoint{
		Name:       "cloudflare",
		BaseURL:    "https://api.cloudflare.com",
		Path:       "/client/v4/accounts/{account}/ai/run",
		Account:    accountID,
		AccountEnv: "CLOUDFLARE_ACCOUNT_ID",
		APIKeyEnv:  "CLOUDFLARE_API_TOKEN",
		Models:     []string{"jev-latest"},
		// jev takes the native body under input, with the model beside it.
		Route: func(model string, body map[string]any) (string, any, error) {
			id, err := cloudflareModelID(model)
			delete(body, "model")
			return "", map[string]any{"model": id, "input": body}, err
		},
		Unwrap: cloudflareUnwrap,
	}}
}

// openRouterModelID maps an ID onto OpenRouter's names: an alias to its
// ~typesafe/ form, a release to typesafe/jev-<major.minor>. OpenRouter
// names a release by its minor version and serves dated snapshots under
// it, so a patch version cannot be pinned there and is refused rather than
// widened to whatever the minor version serves.
func openRouterModelID(id string) (string, error) {
	switch id {
	case "jev", "jev-latest":
		return "~typesafe/jev-latest", nil
	case "jev-preview":
		return "~typesafe/jev-preview", nil
	}
	if strings.HasPrefix(id, "typesafe/") || strings.HasPrefix(id, "~typesafe/") {
		return id, nil
	}
	if version, ok := strings.CutPrefix(id, "jev-"); ok {
		if parts := strings.Split(version, "."); len(parts) > 2 {
			return "", status.Errorf(status.ErrInvalidArgument, "typesafe: OpenRouter names jev releases by minor version, so %q cannot be pinned there; use jev-%s.%s", id, parts[0], parts[1])
		}
		return "typesafe/" + id, nil
	}
	return id, nil
}

func cloudflareModelID(id string) (string, error) {
	switch id {
	case "jev", "jev-latest", "typesafe/jev":
		return "typesafe/jev", nil
	}
	return "", status.Errorf(status.ErrInvalidArgument, "typesafe: Cloudflare serves jev under the alias typesafe/jev only, so %q cannot be pinned there", id)
}

// cloudflareUnwrap handles both response shapes the Workers AI REST API is
// documented with: the bare model output, and the {result, success, errors}
// envelope the rest of the API uses.
func cloudflareUnwrap(body []byte) ([]byte, error) {
	var envelope struct {
		Result  json.RawMessage `json:"result"`
		Success *bool           `json:"success"`
		Errors  []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Success == nil {
		return body, nil
	}
	if !*envelope.Success {
		msg := "request failed"
		if len(envelope.Errors) > 0 {
			msg = envelope.Errors[0].Message
		}
		return nil, status.Errorf(status.ErrInternal, "cloudflare: %s", msg)
	}
	return envelope.Result, nil
}
