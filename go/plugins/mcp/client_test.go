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

	// Ordering guard: WithHTTPTimeout mutates the client currently installed,
	// so the custom client must have been applied before it. If it weren't, the
	// timeout would land on an internal default client and this field would
	// remain zero.
	if customClient.Timeout != 7*time.Second {
		t.Errorf("custom client Timeout = %v, want 7s; WithHTTPBasicClient must be applied before WithHTTPTimeout", customClient.Timeout)
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
