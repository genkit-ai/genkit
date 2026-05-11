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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// mistralVertexTransport rewrites outbound OpenAI-shaped chat completion
// requests to Vertex's per-model Mistral rawPredict (or streamRawPredict)
// URLs. Vertex's OpenAI-compatible /endpoints/openapi/chat/completions does
// not serve Mistral; only the publisher-qualified rawPredict path does. The
// request and response bodies are already in OpenAI shape, so only the URL
// needs to change. The inner RoundTripper (typically an oauth2.Transport)
// continues to add the Bearer token.
type mistralVertexTransport struct {
	inner    http.RoundTripper
	project  string
	location string
}

// RoundTrip rewrites any …/chat/completions request to the Mistral
// rawPredict URL for the model named in the request body. The body itself
// is preserved byte-for-byte so the openai-go SDK's framing (stream flag,
// stream_options, tools, response_format) reaches Vertex unchanged.
func (t *mistralVertexTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// The openai-go SDK emits POST .../chat/completions for chat requests.
	// Any other path (probes, future SDK changes) is passed through.
	if !strings.HasSuffix(req.URL.Path, "/chat/completions") {
		return t.inner.RoundTrip(req)
	}

	var bodyBytes []byte
	if req.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("modelgarden mistral transport: read body: %w", err)
		}
		req.Body.Close()
	}

	model, stream, err := peekModelAndStream(bodyBytes)
	if err != nil {
		return nil, fmt.Errorf("modelgarden mistral transport: %w", err)
	}
	// Defensive: openai-go may receive a publisher-qualified model id from
	// callers that bypassed MistralModel/DefineModel. rawPredict expects
	// the bare id; the publisher already lives in the URL.
	model = strings.TrimPrefix(model, mistralPublisher+"/")
	if model == "" {
		return nil, fmt.Errorf("modelgarden mistral transport: request body has no model field")
	}

	suffix := "rawPredict"
	if stream {
		suffix = "streamRawPredict"
	}

	newURL, err := url.Parse(fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/%s/models/%s:%s",
		t.location, t.project, t.location, mistralPublisher, model, suffix,
	))
	if err != nil {
		return nil, fmt.Errorf("modelgarden mistral transport: build url: %w", err)
	}

	// Clone to avoid mutating the caller's request. Body and GetBody are
	// reset so openai-go's retry path can replay the request.
	rewritten := req.Clone(req.Context())
	rewritten.URL = newURL
	rewritten.Host = newURL.Host
	rewritten.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	rewritten.ContentLength = int64(len(bodyBytes))
	rewritten.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(bodyBytes)), nil
	}

	return t.inner.RoundTrip(rewritten)
}

// peekModelAndStream extracts the "model" and "stream" fields from an
// OpenAI-shaped chat completion request body without losing other fields.
func peekModelAndStream(body []byte) (model string, stream bool, err error) {
	if len(body) == 0 {
		return "", false, nil
	}
	var peek struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &peek); err != nil {
		return "", false, fmt.Errorf("decode body: %w", err)
	}
	return peek.Model, peek.Stream, nil
}
