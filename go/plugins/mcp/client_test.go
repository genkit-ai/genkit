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
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestNewGenkitMCPClientReturnsInitializationError(t *testing.T) {
	sentinel := errors.New("test MCP handshake failure")
	client, err := NewGenkitMCPClient(MCPClientOptions{
		StreamableHTTP: &StreamableHTTPConfig{
			BaseURL: "http://example.com/mcp",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, sentinel
			})},
		},
	})
	if client != nil {
		t.Cleanup(func() { client.Disconnect() })
		t.Errorf("NewGenkitMCPClient() client = %v, want nil", client)
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("NewGenkitMCPClient() error = %v, want wrapped %v", err, sentinel)
	}
}

func TestHelperStdioRejectInit(t *testing.T) {
	if os.Getenv("GENKIT_MCP_TEST_HELPER") != "reject-init" {
		return
	}
	if pidFile := os.Getenv("GENKIT_MCP_TEST_PID_FILE"); pidFile != "" {
		if err := os.WriteFile(pidFile, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			os.Exit(1)
		}
	}

	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		os.Exit(2)
	}
	var req struct {
		ID json.RawMessage `json:"id"`
	}
	if err := json.Unmarshal(line, &req); err != nil {
		os.Exit(3)
	}
	if len(req.ID) == 0 {
		req.ID = json.RawMessage("1")
	}

	if _, err := fmt.Fprintf(os.Stdout, `{"jsonrpc":"2.0","id":%s,"error":{"code":-32600,"message":"test stdio handshake rejected"}}`+"\n", req.ID); err != nil {
		os.Exit(4)
	}

	// Drain stdin to EOF and stay alive afterward so Stdio.Close's cmd.Wait
	// would hang unless the client cancels the subprocess context.
	_, _ = io.Copy(io.Discard, reader)
	time.Sleep(30 * time.Second)
	os.Exit(0)
}

func TestNewGenkitMCPClientTerminatesStdioSubprocessOnInitError(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable(): %v", err)
	}
	pidFile := filepath.Join(t.TempDir(), "child.pid")

	type result struct {
		client *GenkitMCPClient
		err    error
	}
	done := make(chan result, 1)
	go func() {
		client, err := NewGenkitMCPClient(MCPClientOptions{
			Name: "rejecting-stdio",
			Stdio: &StdioConfig{
				Command: exe,
				Args:    []string{"-test.run=^TestHelperStdioRejectInit$"},
				Env: []string{
					"GENKIT_MCP_TEST_HELPER=reject-init",
					"GENKIT_MCP_TEST_PID_FILE=" + pidFile,
				},
			},
		})
		done <- result{client: client, err: err}
	}()

	var res result
	select {
	case res = <-done:
	case <-time.After(5 * time.Second):
		if data, err := os.ReadFile(pidFile); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil {
				if proc, err := os.FindProcess(pid); err == nil {
					_ = proc.Kill()
				}
			}
		}
		t.Fatal("NewGenkitMCPClient() hung waiting for stdio subprocess that stays alive after EOF")
	}

	if res.client != nil {
		t.Cleanup(func() { res.client.Disconnect() })
		t.Errorf("NewGenkitMCPClient() client = %v, want nil", res.client)
	}
	if res.err == nil || !strings.Contains(res.err.Error(), "test stdio handshake rejected") {
		t.Fatalf("NewGenkitMCPClient() error = %v, want stdio handshake rejection", res.err)
	}

	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("reading child pid file: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatalf("parsing child pid %q: %v", string(data), err)
	}
	assertProcessReaped(t, pid)
}

func TestReenableLeavesClientDisabledAfterInitializationError(t *testing.T) {
	client, err := NewGenkitMCPClient(MCPClientOptions{
		Disabled: true,
		StreamableHTTP: &StreamableHTTPConfig{
			BaseURL: "http://example.com/mcp",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("test MCP handshake failure")
			})},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Disconnect() })

	client.Reenable()
	if client.IsEnabled() {
		t.Error("Reenable() left a failed client enabled")
	}
	if client.server != nil {
		t.Error("Reenable() retained a failed connection")
	}
}

func TestMCPHostReconnectAfterFailedStartup(t *testing.T) {
	var available atomic.Bool
	httpClient := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if !available.Load() {
			return nil, errors.New("server unavailable")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"test","version":"1.0.0"}}}`,
			)),
			Request: req,
		}, nil
	})}

	host, err := NewMCPHost(nil, MCPHostOptions{
		MCPServers: []MCPServerConfig{{
			Name: "recovering-server",
			Config: MCPClientOptions{StreamableHTTP: &StreamableHTTPConfig{
				BaseURL: "http://example.com/mcp", HTTPClient: httpClient,
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { host.Disconnect(context.Background(), "recovering-server") })
	if host.clients["recovering-server"].server != nil {
		t.Fatal("failed startup retained a live server connection")
	}

	available.Store(true)
	if err := host.Reconnect(context.Background(), "recovering-server"); err != nil {
		t.Fatalf("Reconnect() after server recovery: %v", err)
	}
	if host.clients["recovering-server"].server == nil {
		t.Error("Reconnect() did not establish a server connection")
	}
}

func TestMCPHostRetainsDefaultOptionsAfterFailedConnect(t *testing.T) {
	host, err := NewMCPHost(nil, MCPHostOptions{})
	if err != nil {
		t.Fatal(err)
	}

	err = host.Connect(context.Background(), nil, "", MCPClientOptions{
		StreamableHTTP: &StreamableHTTPConfig{
			BaseURL: "http://example.com/mcp",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("server unavailable")
			})},
		},
	})
	if err == nil {
		t.Fatal("Connect() error = nil, want failure")
	}

	retained := host.clients[""]
	if retained == nil {
		t.Fatal("Connect() did not retain failed client config")
	}
	if got := retained.GetToolNameWithNamespace("echo"); got != "unnamed_echo" {
		t.Errorf("GetToolNameWithNamespace() = %q, want %q", got, "unnamed_echo")
	}
	if retained.options.Version != defaultMCPVersion {
		t.Errorf("retained client version = %q, want %q", retained.options.Version, defaultMCPVersion)
	}
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
