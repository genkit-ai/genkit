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

package systemone

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/core/status"
)

// Endpoint is one server that answers System One questions. The servers
// take the same questions and return the same answers; they differ in the
// URL, the model IDs, and the envelope around the body, which is what an
// Endpoint captures.
type Endpoint struct {
	// Name names the endpoint in errors.
	Name string
	// BaseURL is the origin requests go to, and BaseURLEnv the environment
	// variable that overrides it, if any.
	BaseURL    string
	BaseURLEnv string
	// Path is the request path. An {account} segment is filled from
	// Account; see [Endpoint.URLPath].
	Path string
	// APIKeyEnv is the environment variable the API key is read from.
	APIKeyEnv string
	// ModelsPath is the path of the model listing, empty when the endpoint
	// has none.
	ModelsPath string
	// Models are the IDs advertised when there is no listing, or it fails.
	Models []string
	// Account fills the {account} segment of Path, from AccountEnv when it
	// is empty; see [Endpoint.WithAccount].
	Account    string
	AccountEnv string
	// ModelID maps a registered model ID to the one the endpoint serves.
	// When nil, the ID is sent as it is.
	ModelID func(id string) (string, error)
	// Body wraps the native request in the endpoint's envelope. When nil,
	// the native body is sent.
	Body func(model string, req *Request) any
	// Unwrap extracts the native response from the endpoint's envelope.
	// When nil, the response is the native body.
	Unwrap func(body []byte) ([]byte, error)
}

// WithAccount returns the endpoint with its account ID resolved, from the
// environment when none was given, or an error naming what to set. An
// endpoint whose path has no account is returned as it is.
func (ep *Endpoint) WithAccount() (*Endpoint, error) {
	if !strings.Contains(ep.Path, "{account}") {
		return ep, nil
	}
	resolved := *ep
	resolved.Account = cmp.Or(ep.Account, os.Getenv(ep.AccountEnv))
	if resolved.Account == "" {
		return nil, fmt.Errorf("%s needs an account ID; pass one or set %s", ep.Name, ep.AccountEnv)
	}
	return &resolved, nil
}

// URLPath is the request path, with the account ID escaped into it.
func (ep *Endpoint) URLPath() string {
	return strings.ReplaceAll(ep.Path, "{account}", url.PathEscape(ep.Account))
}

// Request is the native request body, before the endpoint's envelope.
type Request struct {
	State     any
	Questions map[string]Question
	// Extra is merged over the top-level fields, last write wins, which is
	// the escape hatch to a field this package does not model.
	Extra map[string]any
}

// Body builds the JSON body. An empty model leaves the field out, for an
// endpoint that names the model elsewhere.
func (r *Request) Body(model string) map[string]any {
	body := map[string]any{"state": r.State, "questions": r.Questions}
	if model != "" {
		body["model"] = model
	}
	maps.Copy(body, r.Extra)
	return body
}

// Response is the native response body. Provider, ID, and the cost are
// gateway additions, absent from a model vendor's own responses.
type Response struct {
	Info
	Usage struct {
		InputTokens  int      `json:"input_tokens"`
		OutputTokens int      `json:"output_tokens"`
		Cost         *float64 `json:"cost,omitempty"`
	} `json:"usage"`
}

// ModelInfo is one entry of the models listing.
type ModelInfo struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	ReleaseDate string `json:"release_date,omitempty"`
}

// Client posts questions to one endpoint.
type Client struct {
	// HTTP sends the requests.
	HTTP *http.Client
	// BaseURL is the origin the endpoint's paths are joined to.
	BaseURL string
	// APIKey is sent as a bearer token.
	APIKey string
	// Headers are sent on every request, after the authorization header.
	Headers  http.Header
	Endpoint *Endpoint
}

// Retry policy, matching the vendor SDK's defaults: two retries after the
// first attempt, half a second doubling to five, a quarter of jitter, and
// the server's Retry-After honored when it sends one.
const (
	maxRetries    = 2
	retryBase     = 500 * time.Millisecond
	retryCap      = 5 * time.Second
	retryAfterCap = 30 * time.Second
	maxBodyBytes  = 8 << 20
)

// RequestTimeout bounds one attempt when the plugin builds the HTTP client.
const RequestTimeout = 30 * time.Second

// Decide posts one request and decodes the answers.
func (c *Client) Decide(ctx context.Context, model string, req *Request) (*Response, error) {
	id := model
	if c.Endpoint.ModelID != nil {
		var err error
		if id, err = c.Endpoint.ModelID(model); err != nil {
			return nil, err
		}
	}
	var body any = req.Body(id)
	if c.Endpoint.Body != nil {
		body = c.Endpoint.Body(id, req)
	}
	raw, err := c.do(ctx, http.MethodPost, c.BaseURL+c.Endpoint.URLPath(), body)
	if err != nil {
		return nil, err
	}
	if c.Endpoint.Unwrap != nil {
		if raw, err = c.Endpoint.Unwrap(raw); err != nil {
			return nil, err
		}
	}
	var resp Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, status.Errorf(status.ErrInternal, "%s: response is not a System One answer: %w", c.Endpoint.Name, err)
	}
	if resp.Answers == nil {
		return nil, status.Errorf(status.ErrInternal, "%s: response carries no answers", c.Endpoint.Name)
	}
	return &resp, nil
}

// ListModels fetches the endpoint's model listing. It is sent once, with
// no retries: the listing serves the Dev UI, and the caller falls back to
// the known models when it fails. The listing's envelope is not pinned
// down by the API reference, so a bare array and the usual wrapping keys
// are all accepted.
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if c.Endpoint.ModelsPath == "" {
		return nil, status.Errorf(status.ErrUnimplemented, "%s: no model listing", c.Endpoint.Name)
	}
	raw, _, _, err := c.once(ctx, http.MethodGet, c.BaseURL+c.Endpoint.ModelsPath, nil)
	if err != nil {
		return nil, err
	}
	var models []ModelInfo
	if err := json.Unmarshal(raw, &models); err == nil {
		return models, nil
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, status.Errorf(status.ErrInternal, "%s: model listing is not JSON: %w", c.Endpoint.Name, err)
	}
	for _, key := range []string{"models", "data"} {
		if list, ok := envelope[key]; ok {
			if err := json.Unmarshal(list, &models); err == nil {
				return models, nil
			}
		}
	}
	return nil, status.Errorf(status.ErrInternal, "%s: model listing has no models array", c.Endpoint.Name)
}

// do sends one request with retries and returns the response body. A
// request that fails to connect, times out, is rate limited, or hits a
// server error other than 501 and 505 is retried; anything else is the
// caller's problem.
func (c *Client) do(ctx context.Context, method, url string, body any) ([]byte, error) {
	var payload []byte
	if body != nil {
		var err error
		if payload, err = json.Marshal(body); err != nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "%s: request is not JSON: %w", c.Endpoint.Name, err)
		}
	}
	for attempt := 0; ; attempt++ {
		data, retry, retryAfter, err := c.once(ctx, method, url, payload)
		if err == nil {
			return data, nil
		}
		if ctx.Err() != nil {
			return nil, stopped(ctx, err)
		}
		if attempt >= maxRetries || !retry {
			return nil, err
		}
		delay := backoff(attempt)
		if retryAfter >= 0 {
			delay = min(retryAfter, retryAfterCap)
		}
		select {
		case <-ctx.Done():
			return nil, stopped(ctx, err)
		case <-time.After(delay):
		}
	}
}

// stopped is the error for a request whose context ended: the context's
// cause, so the caller sees a cancellation or a deadline rather than the
// retryable failure it interrupted. The last attempt's failure is kept as
// text only, since wrapping its status would make the call read as worth
// retrying.
func stopped(ctx context.Context, last error) error {
	return fmt.Errorf("%w (last attempt: %v)", context.Cause(ctx), last)
}

// once sends one attempt. It reports whether a failure is worth a retry,
// which is a failure to reach the endpoint, a timeout, a rate limit, or a
// server error, and the Retry-After the server asked for; that is negative
// when the server did not send one, since zero is a request to retry at
// once.
func (c *Client) once(ctx context.Context, method, url string, payload []byte) (data []byte, retry bool, retryAfter time.Duration, err error) {
	retryAfter = -1
	var reader io.Reader
	if payload != nil {
		reader = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return nil, false, retryAfter, status.Errorf(status.ErrInvalidArgument, "%s: %w", c.Endpoint.Name, err)
	}
	maps.Copy(req.Header, c.Headers)
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, true, retryAfter, status.Errorf(status.ErrUnavailable, "%s: %w", c.Endpoint.Name, err)
	}
	defer resp.Body.Close()
	data, err = io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return nil, true, retryAfter, status.Errorf(status.ErrUnavailable, "%s: reading response: %w", c.Endpoint.Name, err)
	}
	if resp.StatusCode/100 == 2 {
		return data, false, retryAfter, nil
	}
	code := resp.StatusCode
	retry = code == http.StatusRequestTimeout || code == http.StatusTooManyRequests ||
		code >= 500 && code != http.StatusNotImplemented && code != http.StatusHTTPVersionNotSupported
	return nil, retry, parseRetryAfter(resp.Header.Get("Retry-After")), httpError(c.Endpoint.Name, code, data)
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
	case code == http.StatusRequestTimeout, code == http.StatusGatewayTimeout, code == 524:
		// 524 is Cloudflare's origin timeout.
		sentinel = status.ErrDeadlineExceeded
	case code == http.StatusNotImplemented, code == http.StatusHTTPVersionNotSupported:
		// A server without the path or the protocol: a setup to fix, not
		// a failure to wait out.
		sentinel = status.ErrUnimplemented
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
	return status.Errorf(sentinel, "%s: HTTP %d: %s", endpoint, code, cmp.Or(errorMessage(body), http.StatusText(code), "empty response"))
}

// errorMessage reads the message out of an error body. The endpoints use
// three shapes: {error: {message}} or {error: "..."}, a {message}, and a
// validation list, which TypeSafe's API sends under detail as {loc, msg}
// records and OpenRouter sends as the bare array of {path, message}
// records. A body in none of these shapes is quoted as it came, and an
// empty one gives "".
func errorMessage(body []byte) string {
	if msg := validationErrors(body); msg != "" {
		return msg
	}
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
			if msg := validationErrors(envelope.Detail); msg != "" {
				return msg
			}
			var flat string
			if json.Unmarshal(envelope.Detail, &flat) == nil && flat != "" {
				return flat
			}
			return string(envelope.Detail)
		}
	}
	msg := strings.TrimSpace(string(body))
	if len(msg) > 200 {
		msg = msg[:200] + "..."
	}
	return msg
}

// validationErrors joins a list of validation records into one line, each
// as its path and message, or returns "" when the body is not such a list.
func validationErrors(body []byte) string {
	var records []struct {
		Message string `json:"message"`
		Msg     string `json:"msg"`
		Path    []any  `json:"path"`
		Loc     []any  `json:"loc"`
	}
	if json.Unmarshal(body, &records) != nil || len(records) == 0 {
		return ""
	}
	var msgs []string
	for _, r := range records {
		msg := cmp.Or(r.Message, r.Msg)
		if msg == "" {
			continue
		}
		var path []string
		for _, p := range append(r.Path, r.Loc...) {
			path = append(path, fmt.Sprint(p))
		}
		if len(path) > 0 {
			msg = strings.Join(path, ".") + ": " + msg
		}
		msgs = append(msgs, msg)
	}
	return strings.Join(msgs, "; ")
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
