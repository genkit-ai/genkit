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

package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func TestBidiActionEcho(t *testing.T) {
	ctx := context.Background()

	// In=string (stream chunks), Out=string, Stream=string, Init=struct{} (no init data).
	action := NewBidiAction(
		"echo", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			var count int
			for input := range inCh {
				count++
				outCh <- fmt.Sprintf("echo: %s", input)
			}
			return fmt.Sprintf("processed %d messages", count), nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	// With unbuffered channels, we must send and receive concurrently.
	go func() {
		conn.Send("hello")
		conn.Send("world")
		conn.Close()
	}()

	var chunks []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %v", len(chunks), chunks)
	}
	if chunks[0] != "echo: hello" {
		t.Errorf("expected 'echo: hello', got %q", chunks[0])
	}
	if chunks[1] != "echo: world" {
		t.Errorf("expected 'echo: world', got %q", chunks[1])
	}

	output, err := conn.Output()
	if err != nil {
		t.Fatal(err)
	}
	if output != "processed 2 messages" {
		t.Errorf("expected 'processed 2 messages', got %q", output)
	}
}

func TestBidiActionWithConfig(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string
	}

	// In=string (stream chunks), Out=string, Stream=string, Init=Config.
	action := NewBidiAction(
		"prefixed", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			for input := range inCh {
				outCh <- fmt.Sprintf("%s: %s", cfg.Prefix, input)
			}
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, Config{Prefix: "INFO"})
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		conn.Send("test message")
		conn.Close()
	}()

	var chunks []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 || chunks[0] != "INFO: test message" {
		t.Errorf("unexpected chunks: %v", chunks)
	}
}

// TestRunBidi verifies the typed one-shot path: input is delivered as a
// single chunk and init configures the session.
func TestRunBidi(t *testing.T) {
	ctx := context.Background()

	type Config struct{ Prefix string }

	action := NewBidiAction(
		"prefixed-oneshot", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			var out string
			for in := range inCh {
				out = cfg.Prefix + in
			}
			return out, nil
		},
	)

	got, err := action.RunBidi(ctx, Config{Prefix: ">> "}, "hello", nil)
	if err != nil {
		t.Fatalf("RunBidi: %v", err)
	}
	if got != ">> hello" {
		t.Errorf("output = %q, want %q", got, ">> hello")
	}
}

// TestBidiActionInterfaceDetection verifies that registry lookups return
// values whose api.BidiAction conformance matches the action kind: bidi
// actions satisfy it (pinning the BidiAction.Register override, which must
// register the bidi type rather than the embedded Action) and plain actions
// do not (the basis for transports' fail-loud init handling).
func TestBidiActionInterfaceDetection(t *testing.T) {
	r := registry.New()

	DefineAction(r, "plain", api.ActionTypeCustom, nil, nil,
		func(ctx context.Context, in string) (string, error) {
			return "out:" + in, nil
		})
	DefineBidiAction(r, "bidi", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "done", nil
		})

	plain := r.LookupAction("/custom/plain")
	if plain == nil {
		t.Fatal("plain action not registered")
	}
	if _, ok := plain.(api.BidiAction); ok {
		t.Error("plain action must not satisfy api.BidiAction")
	}

	bidi := r.LookupAction("/custom/bidi")
	if bidi == nil {
		t.Fatal("bidi action not registered")
	}
	if _, ok := bidi.(api.BidiAction); !ok {
		t.Error("bidi action must satisfy api.BidiAction")
	}
}

// TestRunBidiJSON verifies the JSON one-shot path used by transports: input
// is delivered as a single chunk and opts carries the session init.
func TestRunBidiJSON(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string `json:"prefix"`
	}

	action := NewBidiAction(
		"prefixed-json", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			for in := range inCh {
				outCh <- cfg.Prefix + in
			}
			return "done", nil
		},
	)

	var chunks []string
	cb := func(_ context.Context, raw json.RawMessage) error {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return err
		}
		chunks = append(chunks, s)
		return nil
	}

	r, err := action.RunBidiJSON(ctx, json.RawMessage(`"hello"`), cb,
		&api.BidiJSONOptions{Init: json.RawMessage(`{"prefix":">> "}`)})
	if err != nil {
		t.Fatalf("RunBidiJSON: %v", err)
	}
	var got string
	if err := json.Unmarshal(r.Result, &got); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}
	if got != "done" {
		t.Errorf("output = %q, want %q", got, "done")
	}
	if len(chunks) != 1 || chunks[0] != ">> hello" {
		t.Errorf("chunks = %v, want [\">> hello\"]", chunks)
	}
	if r.TraceId == "" {
		t.Error("TraceId is empty")
	}
}

// TestRunBidiJSONInvalidInit verifies that a malformed JSON init payload
// surfaces as INVALID_ARGUMENT.
func TestRunBidiJSONInvalidInit(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string `json:"prefix"`
	}
	action := NewBidiAction(
		"bad-json-init", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			return "", nil
		},
	)

	_, err := action.RunBidiJSON(ctx, json.RawMessage(`"in"`), nil,
		&api.BidiJSONOptions{Init: json.RawMessage(`{not json`)})
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
	gerr, ok := err.(*GenkitError)
	if !ok {
		t.Fatalf("expected *GenkitError, got %T: %v", err, err)
	}
	if gerr.Status != INVALID_ARGUMENT {
		t.Errorf("status = %v, want %v", gerr.Status, INVALID_ARGUMENT)
	}
}

// TestRunBidiJSONRequiresInput verifies that a one-shot run with absent input
// is rejected up front with a message pointing at streaming sessions, rather
// than falling through to a schema validation failure. Only a streaming
// session can start up and defer its first input.
func TestRunBidiJSONRequiresInput(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string `json:"prefix"`
	}
	action := NewBidiAction(
		"input-required", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			return "ran", nil
		},
	)

	for name, input := range map[string]json.RawMessage{
		"nil input":       nil,
		"empty input":     json.RawMessage(``),
		"JSON null input": json.RawMessage(`null`),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := action.RunBidiJSON(ctx, input, nil,
				&api.BidiJSONOptions{Init: json.RawMessage(`{"prefix":">> "}`)})
			if err == nil {
				t.Fatal("expected error for absent input, got nil")
			}
			gerr, ok := err.(*GenkitError)
			if !ok {
				t.Fatalf("expected *GenkitError, got %T: %v", err, err)
			}
			if gerr.Status != INVALID_ARGUMENT {
				t.Errorf("status = %v, want %v", gerr.Status, INVALID_ARGUMENT)
			}
			if !strings.Contains(gerr.Message, "streaming session") {
				t.Errorf("message %q should point the caller at streaming sessions", gerr.Message)
			}
		})
	}
}

// TestConnectJSONNullInit verifies that nil options and a JSON-null init
// payload are both treated as no init (the zero Init value).
func TestConnectJSONNullInit(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string `json:"prefix"`
	}

	for _, opts := range []*api.BidiJSONOptions{nil, {Init: json.RawMessage(`null`)}} {
		var sawInit Config
		action := NewBidiAction(
			"null-init", api.ActionTypeCustom, nil,
			func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
				sawInit = cfg
				for range inCh {
				}
				return "done", nil
			},
		)

		conn, err := action.ConnectJSON(ctx, opts)
		if err != nil {
			t.Fatalf("ConnectJSON(%v): %v", opts, err)
		}
		conn.Close()
		if _, err := conn.Output(); err != nil {
			t.Fatalf("Output: %v", err)
		}
		if sawInit != (Config{}) {
			t.Errorf("init = %v, want zero value", sawInit)
		}
	}
}

// TestInitSchemaValidationRejectsBadInit verifies that init is validated
// against the action's InitSchema and a mismatch surfaces as INVALID_ARGUMENT.
func TestInitSchemaValidationRejectsBadInit(t *testing.T) {
	ctx := context.Background()

	// Init schema requires "prefix" to be a string.
	initSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prefix": map[string]any{"type": "string"},
		},
		"required": []any{"prefix"},
	}

	action := NewBidiAction(
		"validated-init", api.ActionTypeCustom,
		&BidiActionOptions{InitSchema: initSchema},
		func(ctx context.Context, cfg map[string]any, inCh <-chan string, outCh chan<- string) (string, error) {
			return "done", nil
		},
	)

	// Missing required "prefix" field.
	_, err := action.Connect(ctx, map[string]any{"other": 1})
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	gerr, ok := err.(*GenkitError)
	if !ok {
		t.Fatalf("expected *GenkitError, got %T: %v", err, err)
	}
	if gerr.Status != INVALID_ARGUMENT {
		t.Errorf("status = %v, want %v", gerr.Status, INVALID_ARGUMENT)
	}
}

// TestInitSchemaValidationAcceptsGoodInit verifies the matching-init path of
// init schema validation.
func TestInitSchemaValidationAcceptsGoodInit(t *testing.T) {
	ctx := context.Background()

	initSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prefix": map[string]any{"type": "string"},
		},
		"required": []any{"prefix"},
	}

	action := NewBidiAction(
		"validated-init-ok", api.ActionTypeCustom,
		&BidiActionOptions{InitSchema: initSchema},
		func(ctx context.Context, cfg map[string]any, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, map[string]any{"prefix": ">> "})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out != "done" {
		t.Errorf("output = %q, want %q", out, "done")
	}
}

// TestBidiJSONInitRejectsUnknownFields verifies that the JSON init paths
// (ConnectJSON, RunBidiJSON) validate the raw init payload against the action's
// InitSchema before unmarshaling, so a field the schema does not declare is
// rejected as INVALID_ARGUMENT. Decoding straight into a struct Init would drop
// the stray field, leaving the post-decode validateInit nothing to catch.
func TestBidiJSONInitRejectsUnknownFields(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string `json:"prefix,omitempty"`
	}

	// additionalProperties:false is what makes a stray field a violation; it is
	// also what the inferred schema for a struct Init (e.g. an agent's
	// *AgentInit) carries by default.
	initSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prefix": map[string]any{"type": "string"},
		},
		"additionalProperties": false,
	}

	action := NewBidiAction(
		"json-init-unknown-fields", api.ActionTypeCustom,
		&BidiActionOptions{InitSchema: initSchema},
		func(ctx context.Context, cfg *Config, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "done", nil
		},
	)

	assertRejected := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("expected INVALID_ARGUMENT for unknown init field, got nil")
		}
		var gerr *GenkitError
		if !errors.As(err, &gerr) || gerr.Status != INVALID_ARGUMENT {
			t.Fatalf("err = %v, want INVALID_ARGUMENT GenkitError", err)
		}
	}

	badInit := json.RawMessage(`{"prefix":">> ","bogus":true}`)

	t.Run("ConnectJSON", func(t *testing.T) {
		_, err := action.ConnectJSON(ctx, &api.BidiJSONOptions{Init: badInit})
		assertRejected(t, err)
	})

	t.Run("RunBidiJSON", func(t *testing.T) {
		_, err := action.RunBidiJSON(ctx, json.RawMessage(`"hello"`), nil,
			&api.BidiJSONOptions{Init: badInit})
		assertRejected(t, err)
	})

	// A payload with only declared fields still starts the session.
	t.Run("known fields accepted", func(t *testing.T) {
		conn, err := action.ConnectJSON(ctx, &api.BidiJSONOptions{
			Init: json.RawMessage(`{"prefix":">> "}`)})
		if err != nil {
			t.Fatalf("ConnectJSON: %v", err)
		}
		conn.Close()
		if _, err := conn.Output(); err != nil {
			t.Fatalf("Output: %v", err)
		}
	})
}

// TestBidiJSONInitNormalizedLikeInput verifies that the JSON init path runs the
// same normalization as the input path: a JSON number for an integer-typed
// field is widened to int64 rather than left as the float64 a plain decode into
// an any value would produce. This pins init handling to the input pipeline
// (base.UnmarshalAndNormalize), not just schema validation.
func TestBidiJSONInitNormalizedLikeInput(t *testing.T) {
	ctx := context.Background()

	initSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{"type": "integer"},
		},
	}

	gotCount := make(chan any, 1)
	action := NewBidiAction(
		"json-init-normalized", api.ActionTypeCustom,
		&BidiActionOptions{InitSchema: initSchema},
		func(ctx context.Context, cfg any, inCh <-chan string, outCh chan<- string) (string, error) {
			m, _ := cfg.(map[string]any)
			gotCount <- m["count"]
			for range inCh {
			}
			return "done", nil
		},
	)

	conn, err := action.ConnectJSON(ctx, &api.BidiJSONOptions{
		Init: json.RawMessage(`{"count":42}`)})
	if err != nil {
		t.Fatalf("ConnectJSON: %v", err)
	}
	conn.Close()
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output: %v", err)
	}

	if got := <-gotCount; got != int64(42) {
		t.Errorf("normalized init count = %T (%v), want int64(42)", got, got)
	}
}

// TestBidiNilInitSkipsValidation verifies that a nil init (the zero value of
// a pointer Init type) bypasses init schema validation on every no-init path.
// The inferred init schema describes the object form, which JSON null can
// never satisfy, so without the bypass an action with a pointer Init (e.g. an
// agent's *AgentInit) could not run at all through the unary surface or a
// JSON transport request that omits init.
func TestBidiNilInitSkipsValidation(t *testing.T) {
	ctx := context.Background()

	type Config struct{ Prefix string }

	r := registry.New()
	action := DefineBidiAction(r, "nil-init", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg *Config, inCh <-chan string, outCh chan<- string) (string, error) {
			prefix := "default: "
			if cfg != nil {
				prefix = cfg.Prefix
			}
			var out string
			for in := range inCh {
				out = prefix + in
			}
			return out, nil
		})

	t.Run("unary surface runs with nil init", func(t *testing.T) {
		out, err := action.RunJSON(ctx, json.RawMessage(`"hello"`), nil)
		if err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		if string(out) != `"default: hello"` {
			t.Errorf("output = %s, want %q", out, "default: hello")
		}
	})

	t.Run("JSON one-shot without init", func(t *testing.T) {
		res, err := action.RunBidiJSON(ctx, json.RawMessage(`"hello"`), nil, nil)
		if err != nil {
			t.Fatalf("RunBidiJSON: %v", err)
		}
		if string(res.Result) != `"default: hello"` {
			t.Errorf("output = %s, want %q", res.Result, "default: hello")
		}
	})

	t.Run("JSON session without init", func(t *testing.T) {
		conn, err := action.ConnectJSON(ctx, nil)
		if err != nil {
			t.Fatalf("ConnectJSON: %v", err)
		}
		if err := conn.Send(json.RawMessage(`"hello"`)); err != nil {
			t.Fatalf("Send: %v", err)
		}
		conn.Close()
		for _, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive: %v", err)
			}
		}
		out, err := conn.Output()
		if err != nil {
			t.Fatalf("Output: %v", err)
		}
		if string(out) != `"default: hello"` {
			t.Errorf("output = %s, want %q", out, "default: hello")
		}
	})

	t.Run("typed session with explicit nil init", func(t *testing.T) {
		conn, err := action.Connect(ctx, nil)
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		conn.Close()
		for _, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive: %v", err)
			}
		}
		if _, err := conn.Output(); err != nil {
			t.Fatalf("Output: %v", err)
		}
	})

	t.Run("provided init still validates", func(t *testing.T) {
		_, err := action.RunBidiJSON(ctx, json.RawMessage(`"hello"`), nil,
			&api.BidiJSONOptions{Init: json.RawMessage(`{"Prefix": 42}`)})
		if err == nil {
			t.Fatal("expected error for mistyped init, got nil")
		}
	})
}

// TestBidiZeroStructInitValidatedWithoutInit pins the struct-Init half of the
// no-init contract: running without init still validates the zero init value,
// so an init schema the zero value cannot satisfy surfaces as
// INVALID_ARGUMENT rather than silently defaulting. Authors opt into
// omissible init by choosing a pointer Init type.
func TestBidiZeroStructInitValidatedWithoutInit(t *testing.T) {
	ctx := context.Background()

	type Config struct{ Prefix string }

	initSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"Prefix": map[string]any{"type": "string", "minLength": 1},
		},
		"required": []any{"Prefix"},
	}

	action := NewBidiAction(
		"required-init", api.ActionTypeCustom,
		&BidiActionOptions{InitSchema: initSchema},
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return cfg.Prefix, nil
		},
	)

	_, err := action.RunJSON(ctx, json.RawMessage(`"hello"`), nil)
	if err == nil {
		t.Fatal("expected validation error for zero init, got nil")
	}
	var gerr *GenkitError
	if !errors.As(err, &gerr) || gerr.Status != INVALID_ARGUMENT {
		t.Errorf("err = %v, want INVALID_ARGUMENT GenkitError", err)
	}

	got, err := action.RunBidi(ctx, Config{Prefix: ">> "}, "ignored", nil)
	if err != nil {
		t.Fatalf("RunBidi: %v", err)
	}
	if got != ">> " {
		t.Errorf("output = %q, want %q", got, ">> ")
	}
}

func TestBidiConnectionSendAfterClose(t *testing.T) {
	ctx := context.Background()

	// Hold the action open so the only Send failure mode is the closed
	// input side (a completed action would race in ErrActionCompleted).
	release := make(chan struct{})
	action := NewBidiAction(
		"test", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			<-release
			for range inCh {
			}
			return "", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	conn.Close()
	if err := conn.Send("after close"); !errors.Is(err, ErrConnectionClosed) {
		t.Errorf("expected error matching ErrConnectionClosed, got %v", err)
	}

	close(release)
	<-conn.Done()
}

func TestBidiConnectionContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	action := NewBidiAction(
		"blocking", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	cancel()

	_, err = conn.Output()
	if err == nil {
		t.Error("expected error after context cancellation")
	}
}

func TestBidiActionRegistration(t *testing.T) {
	r := registry.New()

	action := DefineBidiAction(
		r, "echoAction", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for input := range inCh {
				outCh <- input
			}
			return "done", nil
		},
	)

	if action.Name() != "echoAction" {
		t.Errorf("expected name 'echoAction', got %q", action.Name())
	}

	desc := action.Desc()

	// Verify bidi metadata is set.
	if bidi, ok := desc.Metadata["bidi"].(bool); !ok || !bidi {
		t.Error("expected metadata[\"bidi\"] = true")
	}

	// Verify registered in registry.
	if r.LookupAction(desc.Key) == nil {
		t.Error("expected action to be registered")
	}
}

func TestBidiActionDone(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"quick", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "finished", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	conn.Close()
	<-conn.Done()

	output, err := conn.Output()
	if err != nil {
		t.Fatal(err)
	}
	if output != "finished" {
		t.Errorf("expected 'finished', got %q", output)
	}
}

// TestBidiRunCallbackErrorStopsAction verifies that when the streaming
// callback fails during a unary Run of a bidi action, the action's context is
// cancelled and its goroutine exits instead of leaking blocked on a stream
// write.
func TestBidiRunCallbackErrorStopsAction(t *testing.T) {
	ctx := context.Background()

	fnExited := make(chan struct{})
	action := NewBidiAction(
		"cb-error", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			defer close(fnExited)
			for i := 0; ; i++ {
				select {
				case outCh <- fmt.Sprintf("chunk %d", i):
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
		},
	)

	wantErr := errors.New("consumer failed")
	_, err := action.Run(ctx, "in", func(context.Context, string) error {
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run err = %v, want %v", err, wantErr)
	}

	select {
	case <-fnExited:
	case <-time.After(5 * time.Second):
		t.Fatal("bidi function did not exit after callback error (goroutine leak)")
	}
}

// TestBidiActionPanicRecovered verifies that a panic in a bidi function is
// recovered and reported as an error rather than crashing the process, on
// both the connection and unary Run paths.
func TestBidiActionPanicRecovered(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"panicky", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			panic("boom")
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Output(); err == nil || !strings.Contains(err.Error(), "panic in bidi action") {
		t.Errorf("Output err = %v, want panic error", err)
	}

	if _, err := action.Run(ctx, "in", nil); err == nil || !strings.Contains(err.Error(), "panic in bidi action") {
		t.Errorf("Run err = %v, want panic error", err)
	}
}

// TestBidiActionClosingOutChIsError verifies that a bidi function closing its
// output channel (which the framework owns) surfaces as an error instead of
// crashing the process, on both the connection and unary Run paths.
func TestBidiActionClosingOutChIsError(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"closer", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			close(outCh)
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Output(); err == nil || !strings.Contains(err.Error(), "closed its output channel") {
		t.Errorf("Output err = %v, want closed-output-channel error", err)
	}

	if _, err := action.Run(ctx, "in", nil); err == nil || !strings.Contains(err.Error(), "closed its output channel") {
		t.Errorf("Run err = %v, want closed-output-channel error", err)
	}
}

// TestBidiReceiveBreakDoesNotCancelSession verifies that breaking out of a
// Receive loop stops consumption without aborting the session: iteration can
// resume and the final output remains available.
func TestBidiReceiveBreakDoesNotCancelSession(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"resumable", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for i := range 3 {
				select {
				case outCh <- fmt.Sprintf("chunk %d", i):
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
			for range inCh {
			}
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	var got []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, chunk)
		break // Early break must not abort the session.
	}
	conn.Close()
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, chunk)
	}
	if len(got) != 3 {
		t.Errorf("got %d chunks total, want 3: %v", len(got), got)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out != "done" {
		t.Errorf("output = %q, want %q", out, "done")
	}
}

// TestBidiConnectionCancel verifies that Cancel aborts the session.
func TestBidiConnectionCancel(t *testing.T) {
	action := NewBidiAction(
		"cancellable", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	)

	conn, err := action.Connect(context.Background(), struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	conn.Cancel()
	if _, err := conn.Output(); err == nil {
		t.Error("expected error after Cancel")
	}
	conn.Cancel() // Safe to call again.
}

// TestBidiOutputAfterCompletionNotCancelled verifies that after a normal
// completion, Output returns the result and Receive ends cleanly even though
// the connection context is released on completion. Looped to exercise the
// completion/cancellation race.
func TestBidiOutputAfterCompletionNotCancelled(t *testing.T) {
	ctx := context.Background()

	for range 50 {
		action := NewBidiAction(
			"completes", api.ActionTypeCustom, nil,
			func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
				for range inCh {
				}
				return "done", nil
			},
		)
		conn, err := action.Connect(ctx, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		conn.Close()
		out, err := conn.Output()
		if err != nil {
			t.Fatalf("Output: %v", err)
		}
		if out != "done" {
			t.Errorf("output = %q, want %q", out, "done")
		}
		for _, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive after completion: %v", err)
			}
		}
	}
}

// TestBidiConnectionCompletionReleasesContext verifies that completion
// cancels the connection's derived context so the connection releases its
// registration on the parent context (a long-lived parent would otherwise
// accumulate one child per invocation).
func TestBidiConnectionCompletionReleasesContext(t *testing.T) {
	ctx := context.Background()

	// Capture the context the action runs under.
	ctxCh := make(chan context.Context, 1)
	action := NewBidiAction(
		"capture", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			ctxCh <- ctx
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	<-conn.Done()
	fnCtx := <-ctxCh

	// The cancel fires just after doneCh closes; poll briefly.
	deadline := time.Now().Add(5 * time.Second)
	for fnCtx.Err() == nil {
		if time.Now().After(deadline) {
			t.Fatal("connection context was not cancelled after completion")
		}
		time.Sleep(time.Millisecond)
	}
}

// TestBidiConnectionNoSpuriousErrorAfterCompletion verifies that a chunk
// still buffered when the action returns is delivered rather than lost:
// completion cancels the connection context, so the stream-closed and
// ctx-done select arms are both ready, and Receive and Output must prefer
// the stream's own terminal state over a spurious cancellation error.
func TestBidiConnectionNoSpuriousErrorAfterCompletion(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"clean", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			outCh <- "chunk"
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	<-conn.Done()

	// Iterate to defeat select randomness.
	for range 30 {
		for chunk, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive yielded error on completed connection: %v", err)
			}
			if chunk != "chunk" {
				t.Fatalf("unexpected chunk %q", chunk)
			}
		}
		output, err := conn.Output()
		if err != nil {
			t.Fatalf("Output errored on completed connection: %v", err)
		}
		if output != "done" {
			t.Fatalf("expected output 'done', got %q", output)
		}
	}
}

// TestBidiConnectionReceiveResumesAfterBreak verifies the canonical
// multi-turn pattern: send, receive one batch, break, send again. Breaking
// out of Receive must not terminate the connection, and a later Receive
// must pick up where the previous one left off.
func TestBidiConnectionReceiveResumesAfterBreak(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"echo", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for input := range inCh {
				outCh <- "echo: " + input
			}
			return "done", nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []string
	for _, input := range []string{"one", "two"} {
		if err := conn.Send(input); err != nil {
			t.Fatalf("Send(%q) failed: %v", input, err)
		}
		for chunk, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive error: %v", err)
			}
			chunks = append(chunks, chunk)
			break
		}
	}
	conn.Close()

	output, err := conn.Output()
	if err != nil {
		t.Fatal(err)
	}
	if output != "done" {
		t.Errorf("expected output 'done', got %q", output)
	}
	want := []string{"echo: one", "echo: two"}
	if !slices.Equal(chunks, want) {
		t.Errorf("expected chunks %v, got %v", want, chunks)
	}
}

// TestBidiJSONConnSendValidatesChunks verifies that the JSON transport path
// validates every inbound chunk against the action's input schema and that an
// invalid chunk fails the session, matching the JS runtime and the one-shot
// path (where invalid input fails the call).
func TestBidiJSONConnSendValidatesChunks(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"typed-in", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			var n int
			for {
				select {
				case _, ok := <-inCh:
					if !ok {
						return fmt.Sprintf("got %d", n), nil
					}
					n++
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
		},
	)

	t.Run("valid chunk delivered", func(t *testing.T) {
		conn, err := action.ConnectJSON(ctx, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := conn.Send(json.RawMessage(`"ok"`)); err != nil {
			t.Errorf("Send valid chunk: %v", err)
		}
		if err := conn.Close(); err != nil {
			t.Fatal(err)
		}
		out, err := conn.Output()
		if err != nil {
			t.Fatal(err)
		}
		if string(out) != `"got 1"` {
			t.Errorf("output = %s, want %q", out, `"got 1"`)
		}
	})

	t.Run("invalid chunk fails the session", func(t *testing.T) {
		conn, err := action.ConnectJSON(ctx, nil)
		if err != nil {
			t.Fatal(err)
		}
		serr := conn.Send(json.RawMessage(`123`))
		if serr == nil {
			t.Fatal("expected validation error for non-string chunk")
		}
		if gerr, ok := serr.(*GenkitError); !ok || gerr.Status != INVALID_ARGUMENT {
			t.Errorf("Send err = %v, want INVALID_ARGUMENT GenkitError", serr)
		}
		// The validation error is the session's terminal error.
		if _, oerr := conn.Output(); oerr == nil || !strings.Contains(oerr.Error(), "invalid stream chunk") {
			t.Errorf("Output err = %v, want invalid-chunk error", oerr)
		}
	})

	t.Run("null chunk validated like any payload", func(t *testing.T) {
		conn, err := action.ConnectJSON(ctx, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := conn.Send(json.RawMessage(`null`)); err == nil {
			t.Error("expected validation error for null chunk")
		}
	})
}

// TestBidiOutputSchemaValidatedOnConnection verifies that a session's final
// output is validated against the action's OutputSchema, mirroring the unary
// path.
func TestBidiOutputSchemaValidatedOnConnection(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"bad-output", api.ActionTypeCustom,
		&BidiActionOptions{OutputSchema: map[string]any{"type": "string"}},
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (int, error) {
			for range inCh {
			}
			return 42, nil
		},
	)

	conn, err := action.Connect(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	conn.Close()
	if _, err := conn.Output(); err == nil || !strings.Contains(err.Error(), "invalid output") {
		t.Errorf("Output err = %v, want invalid-output error", err)
	}
}

// TestBidiJSONConnReceiveMarshalErrorAbortsSession verifies that a stream
// chunk that cannot be marshaled aborts the session instead of leaving the
// action running with no consumer.
func TestBidiJSONConnReceiveMarshalErrorAbortsSession(t *testing.T) {
	ctx := context.Background()

	fnExited := make(chan struct{})
	action := NewBidiAction(
		"nan-stream", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- float64) (string, error) {
			defer close(fnExited)
			select {
			case outCh <- math.NaN(): // json.Marshal fails on NaN.
			case <-ctx.Done():
				return "", ctx.Err()
			}
			<-ctx.Done()
			return "", ctx.Err()
		},
	)

	conn, err := action.ConnectJSON(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	var gotErr error
	for _, rerr := range conn.Receive() {
		if rerr != nil {
			gotErr = rerr
			break
		}
	}
	if gotErr == nil {
		t.Fatal("expected marshal error from Receive")
	}
	select {
	case <-fnExited:
	case <-time.After(5 * time.Second):
		t.Fatal("session not aborted after marshal error (goroutine leak)")
	}
	// The marshal error is the session's terminal error, not a bare
	// cancellation.
	if _, oerr := conn.Output(); oerr == nil || !strings.Contains(oerr.Error(), "unsupported value") {
		t.Errorf("Output err = %v, want marshal error", oerr)
	}
}

// TestBidiInvalidChunkFailsCtxObliviousSession verifies that the poison cause
// is the session's terminal error even when the action never observes the
// cancellation and returns a nil error after its input closes.
func TestBidiInvalidChunkFailsCtxObliviousSession(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"oblivious", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "done", nil
		},
	)

	conn, err := action.ConnectJSON(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := conn.Send(json.RawMessage(`123`)); err == nil {
		t.Fatal("expected validation error for non-string chunk")
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if _, oerr := conn.Output(); oerr == nil || !strings.Contains(oerr.Error(), "invalid stream chunk") {
		t.Errorf("Output err = %v, want invalid-chunk error overriding the nil result", oerr)
	}
}

// TestBidiSendAfterCompletionFails verifies that Send fails deterministically
// once the action has completed, even when the input channel was never closed
// and still has buffer space.
func TestBidiSendAfterCompletionFails(t *testing.T) {
	action := NewBidiAction(
		"one-read", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			<-inCh
			return "done", nil
		},
	)

	conn, err := action.Connect(context.Background(), struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if err := conn.Send("only"); err != nil {
		t.Fatal(err)
	}
	<-conn.Done()

	// Without the completion pre-check, the blocking select races the free
	// buffer slot against the closed done/ctx channels and ~1/3 of sends
	// would report success for a message nothing will ever read.
	for i := range 25 {
		if err := conn.Send("late"); !errors.Is(err, ErrActionCompleted) {
			t.Fatalf("Send %d after completion = %v, want error matching ErrActionCompleted", i, err)
		}
	}
}

// TestBidiSessionWrapperPanicNotMislabeled verifies that a panic escaping the
// session wrapper (outside the user function) is reported as a panic, not
// misattributed to the action closing its output channel.
func TestBidiSessionWrapperPanicNotMislabeled(t *testing.T) {
	// An unregistered action with a $ref output schema: schema resolution
	// dereferences the nil registry after the function returns, panicking
	// inside the session wrapper.
	action := NewBidiAction(
		"ref-output", api.ActionTypeCustom,
		&BidiActionOptions{OutputSchema: SchemaRef("missing")},
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "done", nil
		},
	)

	conn, err := action.Connect(context.Background(), struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	conn.Close()
	_, oerr := conn.Output()
	if oerr == nil {
		t.Fatal("expected error from wrapper panic")
	}
	if strings.Contains(oerr.Error(), "closed its output channel") {
		t.Errorf("Output err = %v; wrapper panic mislabeled as output-channel close", oerr)
	}
	if !strings.Contains(oerr.Error(), "panic in bidi session") {
		t.Errorf("Output err = %v, want panic-in-bidi-session error", oerr)
	}
}

// TestBidiRunCallbackPanicReleasesAction verifies that a panicking stream
// callback on the unary path does not strand the action goroutine blocked on
// a stream write.
func TestBidiRunCallbackPanicReleasesAction(t *testing.T) {
	ctx := context.Background()

	fnExited := make(chan struct{})
	action := NewBidiAction(
		"cb-panic", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			defer close(fnExited)
			for i := 0; ; i++ {
				select {
				case outCh <- fmt.Sprintf("chunk %d", i):
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
		},
	)

	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected callback panic to propagate to the caller")
			}
		}()
		_, _ = action.Run(ctx, "in", func(context.Context, string) error {
			panic("callback boom")
		})
	}()

	select {
	case <-fnExited:
	case <-time.After(5 * time.Second):
		t.Fatal("bidi function did not exit after callback panic (goroutine leak)")
	}
}

// TestBidiJSONConnEmptyChunkValidated verifies that an absent chunk payload
// is validated like the unary path validates its decoded input, rather than
// silently delivering an unchecked zero value.
func TestBidiJSONConnEmptyChunkValidated(t *testing.T) {
	ctx := context.Background()

	type msg struct {
		Name string `json:"name,omitempty"`
	}
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
		"required":   []any{"name"},
	}

	action := NewBidiAction(
		"required-in", api.ActionTypeCustom,
		&BidiActionOptions{InputSchema: schema},
		func(ctx context.Context, _ struct{}, inCh <-chan msg, outCh chan<- string) (string, error) {
			for {
				select {
				case _, ok := <-inCh:
					if !ok {
						return "done", nil
					}
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
		},
	)

	conn, err := action.ConnectJSON(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := conn.Send(nil); err == nil {
		t.Error("expected validation error for empty chunk against a schema with required fields")
	}
}

// TestActionDescSchemaSentinels verifies that the struct{} sentinel type
// parameters do not leak inferred schemas into action descriptors: only
// streaming actions advertise a streamSchema and only bidi actions with a
// real Init type advertise an initSchema.
func TestActionDescSchemaSentinels(t *testing.T) {
	plain := NewAction("plain-desc", api.ActionTypeCustom, nil, nil,
		func(ctx context.Context, in string) (string, error) { return in, nil })
	if got := plain.Desc().StreamSchema; got != nil {
		t.Errorf("non-streaming action StreamSchema = %v, want nil", got)
	}
	if got := plain.Desc().InitSchema; got != nil {
		t.Errorf("non-streaming action InitSchema = %v, want nil", got)
	}

	streaming := NewStreamingAction("streaming-desc", api.ActionTypeCustom, nil, nil,
		func(ctx context.Context, in string, cb StreamCallback[string]) (string, error) { return in, nil })
	if got := streaming.Desc().StreamSchema; got == nil {
		t.Error("streaming action StreamSchema = nil, want schema")
	}

	noInit := NewBidiAction("bidi-noinit-desc", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			return "", nil
		})
	if got := noInit.Desc().InitSchema; got != nil {
		t.Errorf("bidi action without init InitSchema = %v, want nil", got)
	}
	if got := noInit.Desc().StreamSchema; got == nil {
		t.Error("bidi action StreamSchema = nil, want schema")
	}

	type Config struct{ Prefix string }
	withInit := NewBidiAction("bidi-init-desc", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			return "", nil
		})
	if got := withInit.Desc().InitSchema; got == nil {
		t.Error("bidi action with init InitSchema = nil, want schema")
	}
}

// TestBidiEchoStress exercises many concurrent sessions with many messages
// each. Run it with -race and GOMAXPROCS=1 to catch scheduling-dependent
// bugs in the connection's channel handling.
func TestBidiEchoStress(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"stress-echo", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan int, outCh chan<- int) (int, error) {
			var sum int
			for v := range inCh {
				sum += v
				select {
				case outCh <- v * 2:
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			}
			return sum, nil
		},
	)

	const sessions = 16
	const messages = 100

	var wg sync.WaitGroup
	for s := range sessions {
		wg.Add(1)
		go func() {
			defer wg.Done()
			conn, err := action.Connect(ctx, struct{}{})
			if err != nil {
				t.Error(err)
				return
			}
			go func() {
				for i := range messages {
					if err := conn.Send(i); err != nil {
						t.Error(err)
						return
					}
				}
				conn.Close()
			}()
			var count int
			for chunk, err := range conn.Receive() {
				if err != nil {
					t.Error(err)
					return
				}
				_ = chunk
				count++
			}
			if count != messages {
				t.Errorf("session %d: got %d chunks, want %d", s, count, messages)
			}
			out, err := conn.Output()
			if err != nil {
				t.Error(err)
				return
			}
			if want := messages * (messages - 1) / 2; out != want {
				t.Errorf("session %d: output %d, want %d", s, out, want)
			}
		}()
	}
	wg.Wait()
}

// TestResolveBidiActionFor verifies typed round-trip resolution of a bidi
// action from the registry.
func TestResolveBidiActionFor(t *testing.T) {
	ctx := context.Background()
	r := registry.New()

	type Config struct{ Prefix string }

	DefineBidiAction(r, "resolvable-bidi", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			var out string
			for in := range inCh {
				out = cfg.Prefix + in
			}
			return out, nil
		})

	resolved := ResolveBidiActionFor[string, string, string, Config](r, api.ActionTypeCustom, "resolvable-bidi")
	if resolved == nil {
		t.Fatal("ResolveBidiActionFor returned nil")
	}
	got, err := resolved.RunBidi(ctx, Config{Prefix: ">> "}, "hello", nil)
	if err != nil {
		t.Fatalf("RunBidi: %v", err)
	}
	if got != ">> hello" {
		t.Errorf("output = %q, want %q", got, ">> hello")
	}

	if missing := ResolveBidiActionFor[string, string, string, Config](r, api.ActionTypeCustom, "nope"); missing != nil {
		t.Errorf("expected nil for missing action, got %v", missing)
	}
}
