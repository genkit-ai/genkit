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
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

// captureRoundTripper records the last request it received and returns a
// canned 200 OK response. The test mutates inner.last to inspect what the
// outer transport actually sent.
type captureRoundTripper struct {
	last    *http.Request
	lastRaw []byte
}

func (c *captureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	c.last = req
	if req.Body != nil {
		raw, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		req.Body.Close()
		c.lastRaw = raw
		// restore for any downstream reader
		req.Body = io.NopCloser(bytes.NewReader(raw))
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(`{}`)),
		Header:     make(http.Header),
		Request:    req,
	}, nil
}

func newChatRequest(t *testing.T, body string) *http.Request {
	t.Helper()
	u, err := url.Parse("https://us-central1-aiplatform.googleapis.com/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	req := &http.Request{
		Method: http.MethodPost,
		URL:    u,
		Host:   u.Host,
		Header: http.Header{"Content-Type": []string{"application/json"}},
		Body:   io.NopCloser(strings.NewReader(body)),
	}
	return req
}

func TestMistralTransport_RewritesRawPredict(t *testing.T) {
	inner := &captureRoundTripper{}
	rt := &mistralVertexTransport{
		inner:    inner,
		project:  "test-proj",
		location: "us-central1",
	}

	body := `{"model":"mistral-small-2503","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := newChatRequest(t, body)

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	resp.Body.Close()

	want := "https://us-central1-aiplatform.googleapis.com/v1/projects/test-proj/locations/us-central1/publishers/mistralai/models/mistral-small-2503:rawPredict"
	if got := inner.last.URL.String(); got != want {
		t.Errorf("URL = %q, want %q", got, want)
	}
	if inner.last.Host != "us-central1-aiplatform.googleapis.com" {
		t.Errorf("Host = %q, want us-central1-aiplatform.googleapis.com", inner.last.Host)
	}
	if string(inner.lastRaw) != body {
		t.Errorf("body mutated:\n got  %s\n want %s", inner.lastRaw, body)
	}
	if inner.last.ContentLength != int64(len(body)) {
		t.Errorf("ContentLength = %d, want %d", inner.last.ContentLength, len(body))
	}
	if inner.last.GetBody == nil {
		t.Error("GetBody is nil; openai-go retries would send empty body")
	} else {
		rc, err := inner.last.GetBody()
		if err != nil {
			t.Fatalf("GetBody: %v", err)
		}
		raw, _ := io.ReadAll(rc)
		rc.Close()
		if string(raw) != body {
			t.Errorf("GetBody body = %q, want %q", raw, body)
		}
	}
}

func TestMistralTransport_RewritesStreamRawPredict(t *testing.T) {
	inner := &captureRoundTripper{}
	rt := &mistralVertexTransport{
		inner:    inner,
		project:  "test-proj",
		location: "us-central1",
	}

	body := `{"model":"codestral-2","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := newChatRequest(t, body)

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	resp.Body.Close()

	want := "https://us-central1-aiplatform.googleapis.com/v1/projects/test-proj/locations/us-central1/publishers/mistralai/models/codestral-2:streamRawPredict"
	if got := inner.last.URL.String(); got != want {
		t.Errorf("URL = %q, want %q", got, want)
	}
}

func TestMistralTransport_StripsPublisherPrefixFromURL(t *testing.T) {
	inner := &captureRoundTripper{}
	rt := &mistralVertexTransport{
		inner:    inner,
		project:  "test-proj",
		location: "us-central1",
	}

	body := `{"model":"mistralai/codestral-2","messages":[{"role":"user","content":"hi"}]}`
	req := newChatRequest(t, body)

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	resp.Body.Close()

	// URL must use bare id even though body had publisher prefix.
	if !strings.HasSuffix(inner.last.URL.Path, "/publishers/mistralai/models/codestral-2:rawPredict") {
		t.Errorf("URL path did not strip publisher prefix: %s", inner.last.URL.Path)
	}
}

func TestMistralTransport_PassThroughNonChat(t *testing.T) {
	inner := &captureRoundTripper{}
	rt := &mistralVertexTransport{
		inner:    inner,
		project:  "test-proj",
		location: "us-central1",
	}

	u, _ := url.Parse("https://us-central1-aiplatform.googleapis.com/v1/models")
	req := &http.Request{
		Method: http.MethodGet,
		URL:    u,
		Host:   u.Host,
		Header: http.Header{},
	}

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	resp.Body.Close()

	if inner.last.URL.String() != u.String() {
		t.Errorf("non-chat URL was rewritten: got %s, want %s", inner.last.URL, u)
	}
}

func TestMistralTransport_MissingModelErrors(t *testing.T) {
	inner := &captureRoundTripper{}
	rt := &mistralVertexTransport{
		inner:    inner,
		project:  "test-proj",
		location: "us-central1",
	}

	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := newChatRequest(t, body)

	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("expected error for missing model field, got nil")
	}
	if !strings.Contains(err.Error(), "no model") {
		t.Errorf("error = %v, want one mentioning missing model", err)
	}
}

func TestMistralTransport_PreservesExtraBodyFields(t *testing.T) {
	inner := &captureRoundTripper{}
	rt := &mistralVertexTransport{
		inner:    inner,
		project:  "test-proj",
		location: "us-central1",
	}

	body := `{"model":"mistral-small-2503","tools":[{"type":"function","function":{"name":"x"}}],"response_format":{"type":"json_object"}}`
	req := newChatRequest(t, body)

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	resp.Body.Close()

	var got map[string]any
	if err := json.Unmarshal(inner.lastRaw, &got); err != nil {
		t.Fatalf("decode forwarded body: %v", err)
	}
	if _, ok := got["tools"]; !ok {
		t.Error("tools field missing from forwarded body")
	}
	if _, ok := got["response_format"]; !ok {
		t.Error("response_format field missing from forwarded body")
	}
}
