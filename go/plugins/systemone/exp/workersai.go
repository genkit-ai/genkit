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
	"errors"
	"net/url"
	"strings"

	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
)

// WorkersAI is Cloudflare Workers AI for one account. It serves
// Cloudflare's own decision models, Clef and Clef-flash, which read images,
// and Clef-omni, which reads images, audio, and video, and partners' models
// such as jev. Its models are cloudflare-decisions/<ID> by the Workers AI
// ID: cloudflare-decisions/@cf/cloudflare/clef,
// cloudflare-decisions/@cf/cloudflare/clef-flash,
// cloudflare-decisions/@cf/cloudflare/clef-omni, and
// cloudflare-decisions/typesafe/jev. The API token comes from
// CLOUDFLARE_API_TOKEN, and the account ID from CLOUDFLARE_ACCOUNT_ID when
// accountID is empty. A BaseURL with no {account} segment, such as a
// proxy's, needs no account ID.
//
// Workers AI names one model per release, with no versions, so a
// threshold tuned against a model can move when Cloudflare updates it.
func WorkersAI(accountID string) *SystemOne {
	return &SystemOne{
		Provider: "cloudflare-decisions",
		Path:     "/ai/run",
		Route:    workersAIRoute,
		Unwrap:   unwrapWorkersAI,
		Images:   MediaSupported,
		Models: map[string]ModelSpec{
			"@cf/cloudflare/clef":       {},
			"@cf/cloudflare/clef-flash": {},
			"@cf/cloudflare/clef-omni":  {Audio: MediaSupported, Video: MediaSupported},
			"typesafe/jev":              {Images: MediaUnsupported},
		},
		preset: preset{
			label:      "Workers AI",
			apiKeyEnv:  "CLOUDFLARE_API_TOKEN",
			baseURL:    "https://api.cloudflare.com/client/v4/accounts/{account}",
			account:    accountID,
			accountEnv: "CLOUDFLARE_ACCOUNT_ID",
		},
	}
}

// workersAIRoute builds a request the way Workers AI runs the model. Its
// own models, under @cf/, run at /ai/run/<ID> with the native body, which
// names the model by its last segment, as clef. A partner's model runs at
// /ai/run with the native body under input and the model beside it. A
// model ID goes into the path one escaped segment at a time, and one with
// an empty or dot segment is refused, so no ID can change the query or
// reach another path of the account's API with its token.
func workersAIRoute(model string, body map[string]any) (string, any, error) {
	if !strings.HasPrefix(model, "@cf/") {
		delete(body, "model")
		return "", map[string]any{"model": model, "input": body}, nil
	}
	segments := strings.Split(model, "/")
	for i, segment := range segments {
		if segment == "" || segment == "." || segment == ".." {
			return "", nil, status.Errorf(status.ErrInvalidArgument, "systemone: %q is not a Workers AI model ID", model)
		}
		segments[i] = url.PathEscape(segment)
	}
	body["model"] = model[strings.LastIndex(model, "/")+1:]
	return "/" + strings.Join(segments, "/"), body, nil
}

// unwrapWorkersAI handles both response shapes the Workers AI REST API is
// documented with: the bare model output, and the {result, success, errors}
// envelope the rest of the API uses. A failed envelope's messages are read
// the way an error response's are.
func unwrapWorkersAI(body []byte) ([]byte, error) {
	var envelope struct {
		Result  json.RawMessage `json:"result"`
		Success *bool           `json:"success"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Success == nil {
		return body, nil
	}
	if !*envelope.Success {
		return nil, errors.New(systemone.ErrorMessage(body))
	}
	return envelope.Result, nil
}
