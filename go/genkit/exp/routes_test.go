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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
)

// routeKey is a compact "METHOD path" identity for asserting on a route set
// without depending on order.
func routeKey(r Route) string { return r.Method + " " + r.Path }

func routeKeys(routes []Route) []string {
	keys := make([]string, len(routes))
	for i, r := range routes {
		keys[i] = routeKey(r)
	}
	slices.Sort(keys)
	return keys
}

// newRouteTestGenkit defines a server-managed agent (with abortable store),
// a client-managed agent, and a flow, the mix the route builders must
// distinguish.
func newRouteTestGenkit(t *testing.T) *genkit.Genkit {
	t.Helper()
	g := genkit.Init(context.Background())

	genkit.DefineModel(g, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Message:      ai.NewModelTextMessage(fmt.Sprintf("echo %d", len(req.Messages))),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})

	store, err := localstore.NewFileSessionStore[any](t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	DefineAgent(g, "serverChat", aix.InlinePrompt{ai.WithModelName("test/echo")},
		aix.WithSessionStore(store),
	)
	DefineAgent[any](g, "clientChat", aix.InlinePrompt{ai.WithModelName("test/echo")})
	genkit.DefineFlow(g, "greet", func(ctx context.Context, name string) (string, error) {
		return "hi " + name, nil
	})

	return g
}

func TestAllAgentRoutes(t *testing.T) {
	g := newRouteTestGenkit(t)

	got := routeKeys(AllAgentRoutes(g))
	want := []string{
		"POST /agents/clientChat",
		"POST /agents/serverChat",
		"POST /agents/serverChat/abort",
		"POST /agents/serverChat/getSnapshot",
	}
	if !slices.Equal(got, want) {
		t.Errorf("AllAgentRoutes layout =\n  %v\nwant\n  %v", got, want)
	}

	// Every route is a POST carrying a non-nil action; companions are plain
	// subpaths served with the same enveloped Handler as the turn route.
	for _, r := range AllAgentRoutes(g) {
		if r.Action == nil {
			t.Errorf("route %q has nil Action", routeKey(r))
		}
		if r.Method != http.MethodPost {
			t.Errorf("route %q method = %q, want POST", routeKey(r), r.Method)
		}
	}
}

func TestAgentRoutes_PicksOneAgentAndMirrorsCapabilities(t *testing.T) {
	g := genkit.Init(context.Background())
	genkit.DefineModel(g, "test/echo", &ai.ModelOptions{},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Message: ai.NewModelTextMessage("ok"), FinishReason: ai.FinishReasonStop}, nil
		})
	store, err := localstore.NewFileSessionStore[any](t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	server := DefineAgent(g, "srv", aix.InlinePrompt{ai.WithModelName("test/echo")}, aix.WithSessionStore(store))
	client := DefineAgent[any](g, "cli", aix.InlinePrompt{ai.WithModelName("test/echo")})

	if got, want := routeKeys(AgentRoutes(server)), []string{
		"POST /agents/srv",
		"POST /agents/srv/abort",
		"POST /agents/srv/getSnapshot",
	}; !slices.Equal(got, want) {
		t.Errorf("AgentRoutes(server) = %v, want %v", got, want)
	}

	// Client-managed: just the turn route, no companions.
	if got, want := routeKeys(AgentRoutes(client)), []string{"POST /agents/cli"}; !slices.Equal(got, want) {
		t.Errorf("AgentRoutes(client) = %v, want %v", got, want)
	}
}

func TestAllFlowRoutes(t *testing.T) {
	g := newRouteTestGenkit(t)

	got := routeKeys(AllFlowRoutes(g))
	if want := []string{"POST /flows/greet"}; !slices.Equal(got, want) {
		t.Errorf("AllFlowRoutes = %v, want %v", got, want)
	}

	greet := genkit.NewFlow("standalone", func(ctx context.Context, s string) (string, error) { return s, nil })
	single := FlowRoutes(greet)
	if len(single) != 1 || routeKey(single[0]) != "POST /flows/standalone" {
		t.Errorf("FlowRoutes = %v, want one POST /flows/standalone", routeKeys(single))
	}
}

// TestRoutesServedOverServeMux exercises the full path: build the all-agents
// layout, wire it onto a ServeMux, and drive the resulting endpoints. It
// proves every route speaks the same enveloped Handler transport (the turn
// and the getSnapshot companion alike) and that a client-managed agent has
// only its turn route.
func TestRoutesServedOverServeMux(t *testing.T) {
	g := newRouteTestGenkit(t)

	mux := http.NewServeMux()
	for _, route := range AllAgentRoutes(g) {
		mux.HandleFunc(route.Pattern(), route.Handler())
	}

	do := func(t *testing.T, method, path, body string) (int, string) {
		t.Helper()
		var rdr io.Reader
		if body != "" {
			rdr = strings.NewReader(body)
		}
		req := httptest.NewRequest(method, path, rdr)
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		b, _ := io.ReadAll(w.Result().Body)
		return w.Result().StatusCode, string(b)
	}

	// A turn on the server-managed agent goes through the enveloped Handler
	// transport and yields a snapshot.
	code, body := do(t, "POST", "/agents/serverChat",
		`{"data":{"message":{"role":"user","content":[{"text":"hi"}]}}}`)
	if code != http.StatusOK {
		t.Fatalf("turn status = %d, body = %s", code, body)
	}
	var env struct {
		Result struct {
			SnapshotID string `json:"snapshotId"`
		} `json:"result"`
	}
	if err := json.Unmarshal([]byte(body), &env); err != nil {
		t.Fatalf("turn not enveloped as expected: %v; body = %s", err, body)
	}
	if env.Result.SnapshotID == "" {
		t.Fatalf("no snapshotId from turn; body = %s", body)
	}

	// The mounted getSnapshot route is a POST taking the snapshot ID in the
	// {"data": ...} body and returning the snapshot in the {"result": ...}
	// envelope, exactly like the turn route.
	code, body = do(t, "POST", "/agents/serverChat/getSnapshot",
		`{"data":{"snapshotId":"`+env.Result.SnapshotID+`"}}`)
	if code != http.StatusOK {
		t.Fatalf("getSnapshot status = %d, body = %s", code, body)
	}
	var snapEnv struct {
		Result struct {
			SnapshotID string `json:"snapshotId"`
		} `json:"result"`
	}
	if err := json.Unmarshal([]byte(body), &snapEnv); err != nil {
		t.Fatalf("getSnapshot not enveloped as expected: %v; body = %s", err, body)
	}
	if snapEnv.Result.SnapshotID != env.Result.SnapshotID {
		t.Errorf("snapshot id = %q, want %q", snapEnv.Result.SnapshotID, env.Result.SnapshotID)
	}

	// The client-managed agent is reachable...
	code, body = do(t, "POST", "/agents/clientChat",
		`{"data":{"message":{"role":"user","content":[{"text":"hi"}]}}}`)
	if code != http.StatusOK {
		t.Fatalf("client turn status = %d, body = %s", code, body)
	}
	// ...but has no companion route mounted.
	code, _ = do(t, "POST", "/agents/clientChat/getSnapshot", `{"data":{"snapshotId":"whatever"}}`)
	if code != http.StatusNotFound {
		t.Errorf("client-managed agent should have no getSnapshot route; status = %d, want 404", code)
	}
}
