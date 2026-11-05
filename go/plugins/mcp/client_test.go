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

package mcp

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// TestCreateTransportHonorsStreamableHTTPClient verifies that a custom
// http.Client set on StreamableHTTPConfig is actually used by the Streamable
// HTTP transport, rather than being silently ignored.
func TestCreateTransportHonorsStreamableHTTPClient(t *testing.T) {
	var mu sync.Mutex
	var requests int

	customClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			mu.Lock()
			requests++
			mu.Unlock()
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05"}}`,
				)),
				Request: req,
			}, nil
		}),
	}

	c := &GenkitMCPClient{}
	tr, err := c.createTransport(MCPClientOptions{
		StreamableHTTP: &StreamableHTTPConfig{
			BaseURL:    "http://example.com/mcp",
			HTTPClient: customClient,
			Timeout:    7 * time.Second,
		},
	})
	if err != nil {
		t.Fatalf("createTransport() error = %v", err)
	}

	// The caller's client must NOT be mutated: the transport receives a shallow
	// copy, so a timeout configured here must not leak back onto customClient.
	if customClient.Timeout != 0 {
		t.Errorf("custom client Timeout = %v, want 0; the transport must not mutate the caller's client", customClient.Timeout)
	}

	// Drive one request through the transport to prove it uses the custom client.
	ctx := context.Background()
	if err := tr.Start(ctx); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if _, err := tr.SendRequest(ctx, transport.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      mcp.NewRequestId(1),
		Method:  string(mcp.MethodInitialize),
	}); err != nil {
		t.Fatalf("SendRequest() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if requests == 0 {
		t.Error("custom HTTP client transport was not used; expected at least one request")
	}
}

// TestCreateTransportAppliesTimeoutToCustomClient verifies that
// StreamableHTTPConfig.Timeout still takes effect when a custom http.Client is
// supplied, and that it is applied to the copy rather than depending on the
// order in which transport options happen to be assembled.
func TestCreateTransportAppliesTimeoutToCustomClient(t *testing.T) {
	customClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			select {
			case <-req.Context().Done():
				// The client's Timeout cancelled the request, as expected.
				return nil, req.Context().Err()
			case <-time.After(2 * time.Second):
				// The timeout never reached this client. Answer normally so the
				// assertion below reports it rather than hanging the test.
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body: io.NopCloser(strings.NewReader(
						`{"jsonrpc":"2.0","id":1,"result":{}}`,
					)),
					Request: req,
				}, nil
			}
		}),
	}

	c := &GenkitMCPClient{}
	tr, err := c.createTransport(MCPClientOptions{
		StreamableHTTP: &StreamableHTTPConfig{
			BaseURL:    "http://example.com/mcp",
			HTTPClient: customClient,
			Timeout:    50 * time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("createTransport() error = %v", err)
	}
	if customClient.Timeout != 0 {
		t.Errorf("custom client Timeout = %v, want 0; the transport must not mutate the caller's client", customClient.Timeout)
	}

	ctx := context.Background()
	if err := tr.Start(ctx); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if _, err := tr.SendRequest(ctx, transport.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      mcp.NewRequestId(1),
		Method:  string(mcp.MethodInitialize),
	}); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("SendRequest() error = %v, want context.DeadlineExceeded; the configured Timeout was not applied to the custom client", err)
	}
}
