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

package ollama

import (
	"context"
	"net/http"

	"github.com/firebase/genkit/go/ai"
)

// HeaderParams carries context passed to [RequestHeaderFunc] when resolving
// dynamic auth or proxy headers for an Ollama HTTP request.
//
// Optional fields may be nil depending on the call site — for example,
// /api/tags discovery only sets ServerAddress. Implementations must nil-check
// before dereferencing Model, ModelRequest, or EmbedRequest.
type HeaderParams struct {
	ServerAddress string
	Model         *ModelDefinition // may be nil (e.g. /api/tags discovery)
	ModelRequest  *ai.ModelRequest // may be nil (e.g. embed or /api/tags)
	EmbedRequest  *ai.EmbedRequest // may be nil (e.g. generate or /api/tags)
}

// RequestHeaderFunc generates request headers dynamically (for example, to
// fetch a short-lived auth token against a hosted or proxied Ollama deployment).
// Returning a nil map is treated as no extra headers.
type RequestHeaderFunc func(ctx context.Context, params HeaderParams) (map[string]string, error)

// resolveHeaders returns static or dynamic headers for an Ollama request.
// When headerFunc is set it takes precedence over staticHeaders, matching the
// JS plugin's static-map-or-function union.
func resolveHeaders(ctx context.Context, staticHeaders map[string]string, headerFunc RequestHeaderFunc, params HeaderParams) (map[string]string, error) {
	if headerFunc != nil {
		return headerFunc(ctx, params)
	}
	return staticHeaders, nil
}

func (o *Ollama) resolveHeaders(ctx context.Context, params HeaderParams) (map[string]string, error) {
	if o == nil {
		return nil, nil
	}
	return resolveHeaders(ctx, o.RequestHeaders, o.RequestHeaderFunc, params)
}

func (g *generator) resolveHeaders(ctx context.Context, params HeaderParams) (map[string]string, error) {
	return resolveHeaders(ctx, g.requestHeaders, g.requestHeaderFunc, params)
}

func applyHeaders(req *http.Request, headers map[string]string) {
	for k, v := range headers {
		req.Header.Set(k, v)
	}
}
