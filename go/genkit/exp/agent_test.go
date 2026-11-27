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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
)

func TestLookupAgent(t *testing.T) {
	t.Run("resolves and runs a defined agent", func(t *testing.T) {
		g := genkit.Init(context.Background(), genkit.WithExperimental())
		DefineCustomAgent(g, "echo",
			func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
				if err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
					sess.AddMessages(ai.NewModelTextMessage("echo: " + input.Message.Text()))
					return nil, nil
				}); err != nil {
					return nil, err
				}
				return sess.Result(), nil
			})

		h := LookupAgent(g, "echo")
		if got := h.Name(); got != "echo" {
			t.Errorf("Name() = %q, want %q", got, "echo")
		}
		out, err := h.Run(context.Background(), &aix.AgentInput{Message: ai.NewUserTextMessage("hi")})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if got, want := out.Message.Text(), "echo: hi"; got != want {
			t.Errorf("Message.Text() = %q, want %q", got, want)
		}
	})

	t.Run("unknown agent is nil", func(t *testing.T) {
		g := genkit.Init(context.Background(), genkit.WithExperimental())
		if h := LookupAgent(g, "ghost"); h != nil {
			t.Fatalf("LookupAgent(unregistered) = %+v, want nil", h)
		}
	})

	t.Run("does not require the experimental gate", func(t *testing.T) {
		// LookupAgent only reads the registry, so unlike the constructors it
		// must not panic without genkit.WithExperimental; with no agents
		// registered it simply reports nil.
		g := genkit.Init(context.Background())
		if h := LookupAgent(g, "anything"); h != nil {
			t.Fatalf("LookupAgent(unregistered) = %+v, want nil", h)
		}
	})
}

// serveAgents serves every agent registered with g over HTTP with the
// AllAgentRoutes layout and returns the server's base URL.
func serveAgents(t *testing.T, g *genkit.Genkit) string {
	t.Helper()
	mux := http.NewServeMux()
	for _, r := range AllAgentRoutes(g) {
		mux.HandleFunc(r.Pattern(), r.Handler())
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL
}

// countingEcho answers each turn with the input text and the number of
// messages the session held when the turn began, so a test can tell whether a
// turn saw the conversation before it.
func countingEcho(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
	if err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
		n := len(sess.Messages())
		sess.AddMessages(ai.NewModelTextMessage(fmt.Sprintf("%s after %d", input.Message.Text(), n)))
		return nil, nil
	}); err != nil {
		return nil, err
	}
	return sess.Result(), nil
}

func TestDefineRemoteAgent(t *testing.T) {
	// The redaction case depends on the environment: dev shows full text.
	t.Setenv("GENKIT_ENV", "prod")
	ctx := context.Background()

	server := genkit.Init(ctx, genkit.WithExperimental())
	DefineCustomAgent(server, "echo", countingEcho,
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()))
	DefineCustomAgent(server, "broken",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				return nil, errors.New("dial 10.0.0.7:5432: password rejected")
			})
			return nil, err
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()))
	base := serveAgents(t, server)

	client := genkit.Init(ctx, genkit.WithExperimental())
	meta := &aix.AgentMetadata{StateManagement: aix.AgentStateManagementServer, Abortable: true}
	DefineRemoteAgent(client, "echo", base+"/agents/echo",
		aix.WithAgentMetadata(meta), aix.WithDescription[any]("Echoes with a count."))
	DefineRemoteAgent(client, "broken", base+"/agents/broken", aix.WithAgentMetadata(meta))

	echo := LookupAgent(client, "echo")
	if echo == nil {
		t.Fatal("LookupAgent did not find the remote agent")
	}

	t.Run("resolves by name with what was declared", func(t *testing.T) {
		if got := echo.Ref(); got.Description != "Echoes with a count." {
			t.Errorf("Ref() = %+v, want the declared description", got)
		}
		if m := echo.Metadata(); m == nil || m.StateManagement != aix.AgentStateManagementServer || !m.Abortable {
			t.Errorf("Metadata() = %+v, want the declared metadata", m)
		}
	})

	t.Run("runs turns and resumes the session", func(t *testing.T) {
		first, err := echo.RunText(ctx, "hi")
		if err != nil {
			t.Fatalf("RunText: %v", err)
		}
		if got := first.Message.Text(); got != "hi after 1" || first.SnapshotID == "" {
			t.Fatalf("first turn = %q (snapshot %q), want \"hi after 1\" with a snapshot", got, first.SnapshotID)
		}
		second, err := echo.RunText(ctx, "again", aix.WithSessionID[json.RawMessage](first.SessionID))
		if err != nil {
			t.Fatalf("RunText: %v", err)
		}
		if got := second.Message.Text(); got != "again after 3" {
			t.Errorf("second turn = %q, want \"again after 3\": the session did not carry over", got)
		}
		snap, err := echo.GetSnapshot(ctx, second.SnapshotID)
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if snap.Status != aix.SnapshotStatusCompleted || len(snap.State.Messages) != 4 {
			t.Errorf("snapshot = %s with %d messages, want completed with 4", snap.Status, len(snap.State.Messages))
		}
	})

	t.Run("carries the session across the inputs of one invocation", func(t *testing.T) {
		// The registered agent action, as the Dev UI drives it: two inputs
		// on one connection, each its own request to the server.
		act := genkit.LookupAction(client, api.KeyFromName(api.ActionTypeAgent, "echo")).(api.BidiAction)
		conn, err := act.ConnectJSON(ctx, nil)
		if err != nil {
			t.Fatalf("ConnectJSON: %v", err)
		}
		drained := make(chan struct{})
		go func() {
			defer close(drained)
			for range conn.Receive() {
			}
		}()
		for _, text := range []string{"one", "two"} {
			in, _ := json.Marshal(&aix.AgentInput{Message: ai.NewUserTextMessage(text)})
			if err := conn.Send(in); err != nil {
				t.Fatalf("Send(%q): %v", text, err)
			}
		}
		conn.Close()
		<-drained
		raw, err := conn.Output()
		if err != nil {
			t.Fatalf("Output: %v", err)
		}
		var out aix.AgentOutput[json.RawMessage]
		if err := json.Unmarshal(raw, &out); err != nil {
			t.Fatal(err)
		}
		if got := out.Message.Text(); got != "two after 3" {
			t.Errorf("last turn = %q, want \"two after 3\": the second input did not see the first", got)
		}
	})

	t.Run("follows a detached run to its end", func(t *testing.T) {
		task, err := echo.RunDetached(ctx, &aix.AgentInput{Message: ai.NewUserTextMessage("later")})
		if err != nil {
			t.Fatalf("RunDetached: %v", err)
		}
		snap, err := task.Wait(ctx)
		if err != nil {
			t.Fatalf("Wait: %v", err)
		}
		if snap.Status != aix.SnapshotStatusCompleted {
			t.Errorf("settled status = %s, want completed", snap.Status)
		}
	})

	t.Run("keeps the status of a server error", func(t *testing.T) {
		_, err := echo.GetSnapshot(ctx, "no-such-snapshot")
		if s, _ := status.Classified(err); s != status.NotFound {
			t.Errorf("GetSnapshot(missing) error = %v, want NOT_FOUND", err)
		}
	})

	t.Run("hides internal error text from a remote caller", func(t *testing.T) {
		out, err := LookupAgent(client, "broken").RunText(ctx, "go")
		if err != nil {
			t.Fatalf("RunText: %v", err)
		}
		if out.FinishReason != aix.AgentFinishReasonFailed || out.Error == nil {
			t.Fatalf("output = %+v, want a failed output with an error", out)
		}
		if strings.Contains(out.Error.Message, "10.0.0.7") {
			t.Errorf("remote output error = %q, leaks the internal text", out.Error.Message)
		}
		snap, err := LookupAgent(client, "broken").GetSnapshot(ctx, out.SnapshotID)
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if snap.Error == nil || strings.Contains(snap.Error.Message, "10.0.0.7") {
			t.Errorf("remote snapshot error = %+v, want it present and redacted", snap.Error)
		}
		// The redaction is the wire's: a caller in the server's process
		// keeps the full text.
		local, err := LookupAgent(server, "broken").RunText(ctx, "go")
		if err != nil {
			t.Fatalf("local RunText: %v", err)
		}
		if local.Error == nil || !strings.Contains(local.Error.Message, "10.0.0.7") {
			t.Errorf("in-process output error = %+v, want the full text", local.Error)
		}
	})
}
