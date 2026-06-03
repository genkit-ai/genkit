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

package checks

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"golang.org/x/oauth2"
)

const (
	violative  = "VIOLATIVE"
	maxRetries = 3
)

// --- REST request/response shapes (Checks v1alpha aisafety:classifyContent) ---

type classifyRequest struct {
	Input    classifyInput    `json:"input"`
	Policies []classifyPolicy `json:"policies"`
}

type classifyInput struct {
	TextInput textInput `json:"text_input"`
}

type textInput struct {
	Content string `json:"content"`
}

type classifyPolicy struct {
	PolicyType string   `json:"policy_type"`
	Threshold  *float64 `json:"threshold,omitempty"`
}

type classifyResponse struct {
	PolicyResults []policyResult `json:"policyResults"`
}

type policyResult struct {
	PolicyType string `json:"policyType"`
	// Score is optional in the API schema; nil when omitted.
	Score           *float64 `json:"score,omitempty"`
	ViolationResult string   `json:"violationResult"`
}

// Guardrails is a synchronous Checks AI-safety classifier. It is exported for
// use as in-flight guardrail logic (see [GuardrailMiddleware]).
type Guardrails struct {
	ts         oauth2.TokenSource
	projectID  string
	endpoint   string
	httpClient *http.Client
}

// NewGuardrails creates a Guardrails client. The token source must carry the
// Checks/cloud-platform scopes; projectID is sent as the x-goog-user-project
// quota project.
func NewGuardrails(ts oauth2.TokenSource, projectID string) *Guardrails {
	return newGuardrails(ts, projectID, classifyEndpoint)
}

// newGuardrails is the internal constructor with an overridable endpoint (used
// by the plugin and by tests pointing at an httptest server).
func newGuardrails(ts oauth2.TokenSource, projectID, endpoint string) *Guardrails {
	return &Guardrails{
		ts:         ts,
		projectID:  projectID,
		endpoint:   endpoint,
		httpClient: http.DefaultClient,
	}
}

// ClassifyContent classifies content against the given policies, retrying on
// 429/503 with exponential backoff (honoring Retry-After).
func (g *Guardrails) ClassifyContent(ctx context.Context, content string, policies []ChecksMetricConfig) (*classifyResponse, error) {
	payload, err := json.Marshal(classifyRequest{
		Input:    classifyInput{TextInput: textInput{Content: content}},
		Policies: toPolicies(policies),
	})
	if err != nil {
		return nil, fmt.Errorf("checks: marshal request: %w", err)
	}

	var lastErr error
	delay := 500 * time.Millisecond
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			wait := delay
			var he *httpError
			if errors.As(lastErr, &he) && he.retryAfter > wait {
				wait = he.retryAfter
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
			delay *= 2
		}

		resp, err := g.do(ctx, payload)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		var he *httpError
		if !errors.As(err, &he) || !he.retryable {
			return nil, err
		}
	}
	return nil, fmt.Errorf("checks: classifyContent failed after %d retries: %w", maxRetries, lastErr)
}

// httpError carries a non-2xx response and whether it is retryable.
type httpError struct {
	status     int
	body       string
	retryable  bool
	retryAfter time.Duration
}

func (e *httpError) Error() string {
	return fmt.Sprintf("checks: API returned status %d: %s", e.status, e.body)
}

func (g *Guardrails) do(ctx context.Context, payload []byte) (*classifyResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("checks: build request: %w", err)
	}
	tok, err := g.ts.Token()
	if err != nil {
		return nil, fmt.Errorf("checks: get token: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+tok.AccessToken)
	req.Header.Set("x-goog-user-project", g.projectID)
	req.Header.Set("Content-Type", "application/json")

	httpResp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("checks: request failed: %w", err)
	}
	defer httpResp.Body.Close()
	body, _ := io.ReadAll(httpResp.Body)

	if httpResp.StatusCode != http.StatusOK {
		he := &httpError{status: httpResp.StatusCode, body: string(body)}
		if httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode == http.StatusServiceUnavailable {
			he.retryable = true
			he.retryAfter = parseRetryAfter(httpResp.Header.Get("Retry-After"))
		}
		return nil, he
	}

	var out classifyResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("checks: parse response: %w", err)
	}
	return &out, nil
}

func toPolicies(metrics []ChecksMetricConfig) []classifyPolicy {
	policies := make([]classifyPolicy, 0, len(metrics))
	for _, m := range metrics {
		policies = append(policies, classifyPolicy{
			PolicyType: string(m.Type),
			Threshold:  m.Threshold,
		})
	}
	return policies
}

// parseRetryAfter parses a Retry-After header value (delay-seconds or HTTP-date).
func parseRetryAfter(v string) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// --- Guardrail middleware ---

// GuardrailMiddleware returns an [ai.Middleware] (usable via ai.WithUse) that
// classifies model input and output against the given policies and blocks
// generation when any policy is VIOLATIVE.
func GuardrailMiddleware(g *Guardrails, policies []ChecksMetricConfig) ai.Middleware {
	return &guardrailMiddleware{g: g, policies: policies}
}

type guardrailMiddleware struct {
	g        *Guardrails
	policies []ChecksMetricConfig
}

func (m *guardrailMiddleware) Name() string { return provider + "/guardrail" }

func (m *guardrailMiddleware) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{WrapModel: m.wrapModel}, nil
}

func (m *guardrailMiddleware) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	// Check input before calling the model.
	violated, err := m.classify(ctx, messagesText(params.Request.Messages))
	if err != nil {
		return nil, err
	}
	if len(violated) > 0 {
		return blockedResponse("input", violated), nil
	}

	resp, err := next(ctx, params)
	if err != nil {
		return nil, err
	}

	// Check output after generation.
	if resp != nil && resp.Message != nil {
		violated, err := m.classify(ctx, partsText(resp.Message.Content))
		if err != nil {
			return nil, err
		}
		if len(violated) > 0 {
			return blockedResponse("output", violated), nil
		}
	}
	return resp, nil
}

// classify runs each non-empty text through the classifier and returns the
// names of any VIOLATIVE policies found.
func (m *guardrailMiddleware) classify(ctx context.Context, texts []string) ([]string, error) {
	var violated []string
	for _, t := range texts {
		if strings.TrimSpace(t) == "" {
			continue
		}
		resp, err := m.g.ClassifyContent(ctx, t, m.policies)
		if err != nil {
			return nil, err
		}
		for _, pr := range resp.PolicyResults {
			if pr.ViolationResult == violative {
				violated = append(violated, pr.PolicyType)
			}
		}
	}
	return violated, nil
}

func blockedResponse(stage string, policies []string) *ai.ModelResponse {
	return &ai.ModelResponse{
		FinishReason:  ai.FinishReasonBlocked,
		FinishMessage: fmt.Sprintf("Model %s violated Checks policies: [%s], further processing blocked.", stage, strings.Join(policies, " ")),
	}
}

func messagesText(messages []*ai.Message) []string {
	var texts []string
	for _, msg := range messages {
		if msg == nil {
			continue
		}
		texts = append(texts, partsText(msg.Content)...)
	}
	return texts
}

func partsText(parts []*ai.Part) []string {
	var texts []string
	for _, p := range parts {
		if p != nil && p.IsText() {
			texts = append(texts, p.Text)
		}
	}
	return texts
}
