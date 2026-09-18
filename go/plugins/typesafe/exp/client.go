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
	"bytes"
	"context"
	"encoding/json"
	"io"
	"maps"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/core/status"
)

// Endpoint is one server that answers System One questions. TypeSafe's own
// API, OpenRouter, and Cloudflare Workers AI all take the same questions
// and return the same answers; they differ in the URL, the model IDs, and
// the envelope around the body, which is what an Endpoint captures.
//
// Endpoints are built by [Direct], [OpenRouter], and [Cloudflare]. A proxy
// that forwards the native protocol, such as LiteLLM, is [Direct] with
// [TypeSafe.BaseURL] pointed at it.
type Endpoint struct {
	name       string
	baseURL    string
	baseURLEnv string
	path       string
	apiKeyEnv  string
	modelsPath string
	models     []string
	// modelID maps a registered model ID to the one the endpoint serves.
	modelID func(id string) (string, error)
	// body wraps the native request in the endpoint's envelope.
	body func(model string, req *request) any
	// unwrap extracts the native response from the endpoint's envelope.
	unwrap func(body []byte) ([]byte, error)
}

// Direct is TypeSafe's own API. The API key comes from TYPESAFE_API_KEY and
// the base URL from TYPESAFE_BASE_URL, the variables the vendor SDKs read.
func Direct() *Endpoint {
	return &Endpoint{
		name:       "typesafe",
		baseURL:    "https://api.typesafe.ai",
		baseURLEnv: "TYPESAFE_BASE_URL",
		path:       "/v1/systemone",
		apiKeyEnv:  "TYPESAFE_API_KEY",
		modelsPath: "/v1/models",
		models:     []string{"jev-latest", "jev-1.13.0"},
	}
}

// OpenRouter serves jev through its Decisions API, an alpha endpoint that
// takes the native body. The API key comes from OPENROUTER_API_KEY. Model
// IDs are translated: jev-latest is ~typesafe/jev-latest there, and a
// release such as jev-1.13.0 is typesafe/jev-1.13, named by minor version.
func OpenRouter() *Endpoint {
	return &Endpoint{
		name:      "openrouter",
		baseURL:   "https://openrouter.ai",
		path:      "/api/alpha/decisions",
		apiKeyEnv: "OPENROUTER_API_KEY",
		models:    []string{"jev-latest", "jev-1.13.0"},
		modelID:   openRouterModelID,
	}
}

// Cloudflare serves jev on Workers AI under the alias typesafe/jev only, so
// a versioned model ID is rejected rather than silently served by whatever
// the alias points at. The API token comes from CLOUDFLARE_API_TOKEN.
func Cloudflare(accountID string) *Endpoint {
	return &Endpoint{
		name:      "cloudflare",
		baseURL:   "https://api.cloudflare.com",
		path:      "/client/v4/accounts/" + accountID + "/ai/run",
		apiKeyEnv: "CLOUDFLARE_API_TOKEN",
		models:    []string{"jev-latest"},
		modelID:   cloudflareModelID,
		body: func(model string, req *request) any {
			return map[string]any{"model": model, "input": req.body("")}
		},
		unwrap: cloudflareUnwrap,
	}
}

func openRouterModelID(id string) (string, error) {
	switch id {
	case "jev", "jev-latest", "jev-preview":
		return "~typesafe/jev-latest", nil
	}
	if strings.HasPrefix(id, "typesafe/") || strings.HasPrefix(id, "~typesafe/") {
		return id, nil
	}
	if version, ok := strings.CutPrefix(id, "jev-"); ok {
		// Keep major.minor: OpenRouter names a release by its minor version.
		parts := strings.SplitN(version, ".", 3)
		return "typesafe/jev-" + strings.Join(parts[:min(2, len(parts))], "."), nil
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

// request is the native request body, before the endpoint's envelope.
type request struct {
	State     any
	Questions map[string]question
	// Extra is merged over the top-level fields, last write wins, which is
	// the escape hatch to a field this package does not model.
	Extra map[string]any
}

// body builds the JSON body. An empty model leaves the field out, for an
// endpoint that names the model elsewhere.
func (r *request) body(model string) map[string]any {
	body := map[string]any{"state": r.State, "questions": r.Questions}
	if model != "" {
		body["model"] = model
	}
	maps.Copy(body, r.Extra)
	return body
}

// response is the native response body. Provider, ID, and the cost are
// gateway additions, absent from TypeSafe's own responses.
type response struct {
	Model    string                    `json:"model"`
	Provider string                    `json:"provider,omitempty"`
	ID       string                    `json:"id,omitempty"`
	Answers  map[string]map[string]any `json:"answers"`
	Usage    struct {
		InputTokens  int      `json:"input_tokens"`
		OutputTokens int      `json:"output_tokens"`
		Cost         *float64 `json:"cost,omitempty"`
	} `json:"usage"`
}

// modelInfo is one entry of the models listing.
type modelInfo struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	ReleaseDate string `json:"release_date,omitempty"`
}

// client posts questions to one endpoint.
type client struct {
	http    *http.Client
	baseURL string
	apiKey  string
	headers http.Header
	ep      *Endpoint
}

// Retry policy, matching the vendor SDK's defaults: two retries after the
// first attempt, half a second doubling to five, a quarter of jitter, and
// the server's Retry-After honored when it sends one.
const (
	maxRetries     = 2
	retryBase      = 500 * time.Millisecond
	retryCap       = 5 * time.Second
	retryAfterCap  = 30 * time.Second
	maxBodyBytes   = 8 << 20
	requestTimeout = 30 * time.Second
)

// decide posts one request and decodes the answers.
func (c *client) decide(ctx context.Context, model string, req *request) (*response, error) {
	id := model
	if c.ep.modelID != nil {
		var err error
		if id, err = c.ep.modelID(model); err != nil {
			return nil, err
		}
	}
	var body any = req.body(id)
	if c.ep.body != nil {
		body = c.ep.body(id, req)
	}
	raw, err := c.do(ctx, http.MethodPost, c.baseURL+c.ep.path, body)
	if err != nil {
		return nil, err
	}
	if c.ep.unwrap != nil {
		if raw, err = c.ep.unwrap(raw); err != nil {
			return nil, err
		}
	}
	var resp response
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, status.Errorf(status.ErrInternal, "%s: response is not a System One answer: %w", c.ep.name, err)
	}
	if resp.Answers == nil {
		return nil, status.Errorf(status.ErrInternal, "%s: response carries no answers", c.ep.name)
	}
	return &resp, nil
}

// listModels fetches the endpoint's model listing. It is sent once, with
// no retries: the listing serves the Dev UI, and the caller falls back to
// the known models when it fails. The listing's envelope is not pinned
// down by the API reference, so a bare array and the usual wrapping keys
// are all accepted.
func (c *client) listModels(ctx context.Context) ([]modelInfo, error) {
	if c.ep.modelsPath == "" {
		return nil, status.Errorf(status.ErrUnimplemented, "%s: no model listing", c.ep.name)
	}
	raw, _, _, err := c.once(ctx, http.MethodGet, c.baseURL+c.ep.modelsPath, nil)
	if err != nil {
		return nil, err
	}
	var models []modelInfo
	if err := json.Unmarshal(raw, &models); err == nil {
		return models, nil
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, status.Errorf(status.ErrInternal, "%s: model listing is not JSON: %w", c.ep.name, err)
	}
	for _, key := range []string{"models", "data"} {
		if list, ok := envelope[key]; ok {
			if err := json.Unmarshal(list, &models); err == nil {
				return models, nil
			}
		}
	}
	return nil, status.Errorf(status.ErrInternal, "%s: model listing has no models array", c.ep.name)
}

// do sends one request with retries and returns the response body. A
// request that fails to connect, times out, is rate limited, or hits a
// server error is retried; anything else is the caller's problem.
func (c *client) do(ctx context.Context, method, url string, body any) ([]byte, error) {
	var payload []byte
	if body != nil {
		var err error
		if payload, err = json.Marshal(body); err != nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "typesafe: request is not JSON: %w", err)
		}
	}
	for attempt := 0; ; attempt++ {
		data, retry, retryAfter, err := c.once(ctx, method, url, payload)
		if err == nil {
			return data, nil
		}
		if attempt >= maxRetries || !retry || ctx.Err() != nil {
			return nil, err
		}
		delay := backoff(attempt)
		if retryAfter >= 0 {
			delay = min(retryAfter, retryAfterCap)
		}
		select {
		case <-ctx.Done():
			return nil, err
		case <-time.After(delay):
		}
	}
}

// once sends one attempt. It reports whether a failure is worth a retry,
// which is a failure to reach the endpoint, a timeout, a rate limit, or a
// server error, and the Retry-After the server asked for; that is negative
// when the server did not send one, since zero is a request to retry at
// once.
func (c *client) once(ctx context.Context, method, url string, payload []byte) (data []byte, retry bool, retryAfter time.Duration, err error) {
	retryAfter = -1
	var reader io.Reader
	if payload != nil {
		reader = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return nil, false, retryAfter, status.Errorf(status.ErrInvalidArgument, "typesafe: %w", err)
	}
	maps.Copy(req.Header, c.headers)
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, true, retryAfter, status.Errorf(status.ErrUnavailable, "%s: %w", c.ep.name, err)
	}
	defer resp.Body.Close()
	data, err = io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return nil, true, retryAfter, status.Errorf(status.ErrUnavailable, "%s: reading response: %w", c.ep.name, err)
	}
	if resp.StatusCode/100 == 2 {
		return data, false, retryAfter, nil
	}
	code := resp.StatusCode
	retry = code == http.StatusRequestTimeout || code == http.StatusTooManyRequests || code >= 500
	return nil, retry, parseRetryAfter(resp.Header.Get("Retry-After")), httpError(c.ep.name, code, data)
}

// httpError maps an error response onto a Genkit status: the canonical
// HTTP mapping, with the codes the endpoints give their own meaning to
// handled first. The body is searched for the message under the shapes the
// three endpoints use.
func httpError(endpoint string, code int, body []byte) error {
	var sentinel *status.Sentinel
	switch {
	case code == http.StatusPaymentRequired:
		// Out of credit at a gateway: a quota, not a bad request.
		sentinel = status.ErrResourceExhausted
	case code == http.StatusRequestTimeout, code == 524:
		// 524 is Cloudflare's origin timeout.
		sentinel = status.ErrDeadlineExceeded
	case code > http.StatusInternalServerError:
		// Every other server-side failure reads as transient.
		sentinel = status.ErrUnavailable
	default:
		sentinel = status.Base(status.FromHTTPCode(code))
		if sentinel == status.ErrUnknown {
			// A client error the mapping does not name, such as 422, is
			// the request's fault.
			sentinel = status.ErrInvalidArgument
		}
	}
	return status.Errorf(sentinel, "%s: HTTP %d: %s", endpoint, code, errorMessage(body))
}

func errorMessage(body []byte) string {
	var envelope struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  json.RawMessage `json:"detail"`
	}
	if err := json.Unmarshal(body, &envelope); err == nil {
		if len(envelope.Error) > 0 {
			var nested struct {
				Message string `json:"message"`
			}
			if json.Unmarshal(envelope.Error, &nested) == nil && nested.Message != "" {
				return nested.Message
			}
			var flat string
			if json.Unmarshal(envelope.Error, &flat) == nil && flat != "" {
				return flat
			}
		}
		if envelope.Message != "" {
			return envelope.Message
		}
		if len(envelope.Detail) > 0 {
			return string(envelope.Detail)
		}
	}
	msg := strings.TrimSpace(string(body))
	if len(msg) > 200 {
		msg = msg[:200] + "..."
	}
	if msg == "" {
		msg = http.StatusText(http.StatusInternalServerError)
	}
	return msg
}

func backoff(attempt int) time.Duration {
	delay := min(retryBase<<attempt, retryCap)
	jitter := time.Duration(rand.Float64() * 0.25 * float64(delay))
	return delay - jitter
}

// parseRetryAfter reads a Retry-After header, given either as seconds or as
// an HTTP date. It is negative when there is no usable header, and zero
// when the server asks for an immediate retry.
func parseRetryAfter(header string) time.Duration {
	if header == "" {
		return -1
	}
	if seconds, err := strconv.Atoi(header); err == nil {
		return max(time.Duration(seconds)*time.Second, 0)
	}
	if at, err := http.ParseTime(header); err == nil {
		return max(time.Until(at), 0)
	}
	return -1
}
