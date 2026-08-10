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

package vectorsearch

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

// --- test helpers ---

type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func newHTTPResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

type fakeTokenSource struct {
	tok *oauth2.Token
	err error
}

func (f fakeTokenSource) Token() (*oauth2.Token, error) {
	if f.err != nil {
		return nil, f.err
	}
	// Return a copy to simulate real behavior and avoid accidental mutation.
	cp := *f.tok
	return &cp, nil
}

func newVSWithToken(token string, tokenErr error) *VertexAIVectorSearch {
	return &VertexAIVectorSearch{
		// We assume VertexAIVectorSearch has an unexported field named `client *client`,
		// as used in the methods being tested.
		client: &client{
			TokenSource: fakeTokenSource{
				tok: &oauth2.Token{
					AccessToken: token,
					TokenType:   "Bearer",
					Expiry:      time.Now().Add(time.Hour),
				},
				err: tokenErr,
			},
		},
	}
}

// withTransport temporarily swaps http.DefaultTransport for the duration
// of the callback. Restores it afterwards.
func withTransport(t *testing.T, rt http.RoundTripper, fn func()) {
	t.Helper()
	orig := http.DefaultTransport
	http.DefaultTransport = rt
	defer func() { http.DefaultTransport = orig }()
	fn()
}

// readReqBody reads and restores the request body for assertions.
func readReqBody(t *testing.T, r *http.Request) []byte {
	t.Helper()
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	r.Body = io.NopCloser(bytes.NewReader(b)) // restore
	return b
}

// --- tests for UpsertDatapoints ---

func TestUpsertDatapoints_SendsCorrectRequestAndSucceeds(t *testing.T) {
	v := newVSWithToken("tok123", nil)

	params := UpsertDatapointsParams{
		ProjectID: "my-proj",
		Location:  "us-central1",
		IndexID:   "idx-1",
		Datapoints: []IndexDatapoint{
			{
				DatapointID:   "dp1",
				FeatureVector: []float32{0.1, 0.2},
				Restricts: []Restrict{
					{Namespace: "ns", AllowList: []string{"a", "b"}},
				},
			},
		},
	}

	var capturedReq *http.Request

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		capturedReq = r
		// Return OK with empty JSON.
		return newHTTPResponse(http.StatusOK, `{}`), nil
	}), func() {
		if err := v.UpsertDatapoints(params); err != nil {
			t.Fatalf("UpsertDatapoints returned error: %v", err)
		}
	})

	if capturedReq == nil {
		t.Fatal("expected request to be captured")
	}

	wantURL := "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/indexes/idx-1:upsertDatapoints"
	if got := capturedReq.URL.String(); got != wantURL {
		t.Fatalf("wrong URL:\n got: %s\nwant: %s", got, wantURL)
	}

	if capturedReq.Method != http.MethodPost {
		t.Fatalf("wrong method: got %s want POST", capturedReq.Method)
	}

	if got := capturedReq.Header.Get("Content-Type"); got != "application/json" {
		t.Fatalf("wrong Content-Type: %q", got)
	}
	if got := capturedReq.Header.Get("Authorization"); got != "Bearer tok123" {
		t.Fatalf("wrong Authorization: %q", got)
	}

	// Check body shape.
	body := readReqBody(t, capturedReq)
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("request body not JSON: %v", err)
	}
	_, ok := payload["datapoints"]
	if !ok {
		t.Fatalf("request body missing 'datapoints' field: %s", string(body))
	}
}

func TestUpsertDatapoints_TokenError(t *testing.T) {
	v := newVSWithToken("", errors.New("no token"))
	params := UpsertDatapointsParams{
		ProjectID: "p",
		Location:  "loc",
		IndexID:   "idx",
	}

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		t.Fatalf("should not reach transport when token fails")
		return nil, nil
	}), func() {
		err := v.UpsertDatapoints(params)
		if err == nil {
			t.Fatalf("expected token error, got: %v", err)
		}
	})
}

func TestUpsertDatapoints_HTTPErrorPropagatesBody(t *testing.T) {
	v := newVSWithToken("tok123", nil)
	params := UpsertDatapointsParams{
		ProjectID: "p",
		Location:  "loc",
		IndexID:   "idx",
	}

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusInternalServerError, `boom`), nil
	}), func() {
		err := v.UpsertDatapoints(params)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

// --- tests for FindNeighbors ---

func TestFindNeighbors_SendsCorrectRequestAndParsesResponse(t *testing.T) {
	v := newVSWithToken("tokXYZ", nil)

	params := FindNeighborsParams{
		FeatureVector:    []float32{0.9, 1.1, -0.2},
		NeighborCount:    3,
		ProjectNumber:    "1234567890",
		Location:         "europe-west4",
		IndexEndpointID:  "ep-1",
		PublicDomainName: "public.test-domain.example", // host part only; code prefixes https://
		DeployedIndexID:  "deployed-1",
		Restricts: []Restrict{
			{Namespace: "color", AllowList: []string{"red"}},
		},
		NumericRestricts: []NumericRestrict{
			{Namespace: "price", ValueInt: 10, Op: "LESS_EQUAL"},
		},
	}

	respJSON := `{
		"nearestNeighbors": [
			{
				"neighbors": [
					{
						"datapoint": {
							"datapointId": "doc1",
							"featureVector": [0.1, 0.2]
						},
						"distance": 0.42
					}
				]
			}
		]
	}`

	var capturedReq *http.Request

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		capturedReq = r
		return newHTTPResponse(http.StatusOK, respJSON), nil
	}), func() {
		out, err := v.FindNeighbors(params)
		if err != nil {
			t.Fatalf("FindNeighbors returned error: %v", err)
		}
		if out == nil || len(out.NearestNeighbors) != 1 || len(out.NearestNeighbors[0].Neighbors) != 1 {
			t.Fatalf("unexpected parsed response: %#v", out)
		}
		if got := out.NearestNeighbors[0].Neighbors[0].Datapoint.DatapointId; got != "doc1" {
			t.Fatalf("wrong datapoint id: %q", got)
		}
	})

	if capturedReq == nil {
		t.Fatal("expected request to be captured")
	}

	wantURL := "https://public.test-domain.example/v1/projects/1234567890/locations/europe-west4/indexEndpoints/ep-1:findNeighbors"
	if got := capturedReq.URL.String(); got != wantURL {
		t.Fatalf("wrong URL:\n got: %s\nwant: %s", got, wantURL)
	}

	if capturedReq.Method != http.MethodPost {
		t.Fatalf("wrong method: got %s want POST", capturedReq.Method)
	}
	if got := capturedReq.Header.Get("Content-Type"); got != "application/json" {
		t.Fatalf("wrong Content-Type: %q", got)
	}
	if got := capturedReq.Header.Get("Authorization"); got != "Bearer tokXYZ" {
		t.Fatalf("wrong Authorization: %q", got)
	}

	body := readReqBody(t, capturedReq)
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("request body not JSON: %v", err)
	}
	if payload["deployedIndexId"] != "deployed-1" {
		t.Fatalf("deployedIndexId missing or wrong: %v", payload["deployedIndexId"])
	}
	queries, ok := payload["queries"].([]any)
	if !ok || len(queries) != 1 {
		t.Fatalf("queries missing or wrong: %T %#v", payload["queries"], payload["queries"])
	}
}

func TestFindNeighbors_TokenError(t *testing.T) {
	v := newVSWithToken("", errors.New("token fail"))
	params := FindNeighborsParams{
		FeatureVector:    []float32{1},
		NeighborCount:    1,
		ProjectNumber:    "1",
		Location:         "loc",
		IndexEndpointID:  "ep",
		PublicDomainName: "host",
		DeployedIndexID:  "dep",
	}

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		t.Fatalf("should not reach transport when token fails")
		return nil, nil
	}), func() {
		_, err := v.FindNeighbors(params)
		if err == nil {
			t.Fatalf("expected token error, got: %v", err)
		}
	})
}

func TestFindNeighbors_HTTPErrorPropagatesBody(t *testing.T) {
	v := newVSWithToken("tok", nil)
	params := FindNeighborsParams{
		FeatureVector:    []float32{1},
		NeighborCount:    1,
		ProjectNumber:    "1",
		Location:         "loc",
		IndexEndpointID:  "ep",
		PublicDomainName: "host",
		DeployedIndexID:  "dep",
	}

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusBadGateway, `upstream bad`), nil
	}), func() {
		_, err := v.FindNeighbors(params)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestFindNeighbors_InvalidJSONInResponse(t *testing.T) {
	v := newVSWithToken("tok", nil)
	params := FindNeighborsParams{
		FeatureVector:    []float32{1},
		NeighborCount:    1,
		ProjectNumber:    "1",
		Location:         "loc",
		IndexEndpointID:  "ep",
		PublicDomainName: "host",
		DeployedIndexID:  "dep",
	}

	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusOK, `{"nearestNeighbors": [`), nil // broken JSON
	}), func() {
		_, err := v.FindNeighbors(params)
		if err == nil {
			t.Fatalf("expected JSON decode error, got: %v", err)
		}
	})
}
