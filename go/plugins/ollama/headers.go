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

package ollama

import (
	"context"
	"net/http"

	"github.com/firebase/genkit/go/ai"
)

// HeaderParams carries context passed to [RequestHeaderFunc] when resolving
// dynamic auth or proxy headers for an Ollama HTTP request.
type HeaderParams struct {
	ServerAddress string
	Model         *ModelDefinition
	ModelRequest  *ai.ModelRequest
	EmbedRequest  *ai.EmbedRequest
}

// RequestHeaderFunc generates request headers dynamically (for example, to
// fetch a short-lived auth token against a hosted or proxied Ollama deployment).
// Returning a nil map is treated as no extra headers.
type RequestHeaderFunc func(ctx context.Context, params HeaderParams) (map[string]string, error)

// resolveHeaders returns static or dynamic headers for an Ollama request.
// When RequestHeaderFunc is set it takes precedence over RequestHeaders,
// matching the JS plugin's static-map-or-function union.
func (o *Ollama) resolveHeaders(ctx context.Context, params HeaderParams) (map[string]string, error) {
	if o == nil {
		return nil, nil
	}
	if o.RequestHeaderFunc != nil {
		headers, err := o.RequestHeaderFunc(ctx, params)
		if err != nil {
			return nil, err
		}
		return headers, nil
	}
	return o.RequestHeaders, nil
}

func applyHeaders(req *http.Request, headers map[string]string) {
	for k, v := range headers {
		req.Header.Set(k, v)
	}
}
