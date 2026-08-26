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

package a2ui

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/registry"
)

var ctx = context.Background()

func newTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	r := registry.New()
	ai.ConfigureFormats(r)
	return r
}

// echoModel returns a model that replies with the given text and records the
// messages it received.
func echoModel(t *testing.T, r *registry.Registry, name, reply string) (ai.Model, *[]*ai.Message) {
	t.Helper()
	var captured []*ai.Message
	m := ai.NewModel(name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		captured = req.Messages
		return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage(reply)}, nil
	})
	m.Register(r)
	return m, &captured
}

func TestMiddlewareInjectsInstructions(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/echo", "ok")

	cfg := &Config{}
	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hi"), ai.WithUse(cfg)); err != nil {
		t.Fatal(err)
	}

	var sys *ai.Message
	for _, msg := range *captured {
		if msg.Role == ai.RoleSystem {
			sys = msg
			break
		}
	}
	if sys == nil {
		t.Fatalf("expected a system message; messages=%v", *captured)
	}
	joined := ""
	for _, p := range sys.Content {
		joined += p.Text
	}
	if !strings.Contains(joined, "Rendering UI with A2UI") {
		t.Errorf("system prompt missing A2UI instructions: %q", joined)
	}
	if !strings.Contains(joined, BasicCatalogID) {
		t.Errorf("system prompt missing catalog id")
	}
}

func TestMiddlewareInstructionsNone(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/echo-none", "ok")

	cfg := &Config{Instructions: InstructionsNone}
	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hi"), ai.WithUse(cfg)); err != nil {
		t.Fatal(err)
	}
	for _, msg := range *captured {
		for _, p := range msg.Content {
			if strings.Contains(p.Text, "Rendering UI with A2UI") {
				t.Fatal("instructions should not be injected when InstructionsNone")
			}
		}
	}
}

func TestMiddlewareRewritesFinalMessage(t *testing.T) {
	r := newTestRegistry(t)
	catalog := BasicCatalog()
	reply := "Here you go:\n```a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"` + catalog.ID + `"}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Text","text":"hi"}]}}]` +
		"\n```"
	m, _ := echoModel(t, r, "test/rewrite", reply)

	cfg := &Config{SurfaceID: "fixed-surface", Validate: ValidateStrict}
	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("card please"), ai.WithUse(cfg))
	if err != nil {
		t.Fatal(err)
	}

	envs := EnvelopesFromParts(resp.Message.Content)
	if len(envs) != 2 {
		t.Fatalf("got %d envelopes, want 2; content=%v", len(envs), resp.Message.Content)
	}
	cs, _ := envs[0]["createSurface"].(map[string]any)
	if cs["surfaceId"] != "fixed-surface" {
		t.Errorf("surfaceId = %v, want fixed-surface", cs["surfaceId"])
	}

	// The prose before the block should remain as a text part.
	hasProse := false
	for _, p := range resp.Message.Content {
		if p.IsText() && strings.Contains(p.Text, "Here you go:") {
			hasProse = true
		}
	}
	if !hasProse {
		t.Error("expected the leading prose to be preserved as a text part")
	}
}

func TestMiddlewareTransformsStream(t *testing.T) {
	r := newTestRegistry(t)
	catalog := BasicCatalog()

	// A model that streams the reply in several text chunks, splitting the a2ui
	// fenced block across chunk boundaries.
	chunks := []string{
		"Here you go:\n``",
		"`a2ui\n[{\"createSurface\":{\"surfaceId\":\"SURFACE_ID\",\"catalogId\":\"" + catalog.ID + "\"}},",
		"{\"updateComponents\":{\"surfaceId\":\"SURFACE_ID\",\"components\":[{\"id\":\"root\",\"component\":\"Text\",\"text\":\"hi\"}]}}]\n``",
		"`\nBye.",
	}
	full := strings.Join(chunks, "")
	m := ai.NewModel("test/stream", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if cb != nil {
			for _, c := range chunks {
				if err := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(c)}}); err != nil {
					return nil, err
				}
			}
		}
		return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage(full)}, nil
	})
	m.Register(r)

	var streamedEnvelopes []Envelope
	var streamedProse strings.Builder
	cb := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		streamedEnvelopes = append(streamedEnvelopes, EnvelopesFromParts(chunk.Content)...)
		for _, p := range chunk.Content {
			if p.IsText() {
				streamedProse.WriteString(p.Text)
			}
		}
		return nil
	}

	cfg := &Config{SurfaceID: "fixed", Validate: ValidateStrict}
	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("card please"),
		ai.WithUse(cfg),
		ai.WithStreaming(cb),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(streamedEnvelopes) != 2 {
		t.Fatalf("streamed %d envelopes, want 2", len(streamedEnvelopes))
	}
	// The leading prose streams through. Trailing prose shorter than a fence is
	// legitimately held back on the streaming path (it appears in the final
	// message instead), mirroring the JS parser.
	if !strings.Contains(streamedProse.String(), "Here you go:") {
		t.Errorf("streamed prose = %q, missing leading text", streamedProse.String())
	}

	// The final message carries all prose, including the trailing run, plus the
	// same surface id as the streamed one.
	finalProse := ""
	for _, p := range resp.Message.Content {
		if p.IsText() {
			finalProse += p.Text
		}
	}
	if !strings.Contains(finalProse, "Here you go:") || !strings.Contains(finalProse, "Bye.") {
		t.Errorf("final prose = %q, missing expected text", finalProse)
	}

	finalEnvs := EnvelopesFromParts(resp.Message.Content)
	if len(finalEnvs) != 2 {
		t.Fatalf("final message has %d envelopes, want 2", len(finalEnvs))
	}
	streamCS, _ := streamedEnvelopes[0]["createSurface"].(map[string]any)
	finalCS, _ := finalEnvs[0]["createSurface"].(map[string]any)
	if streamCS["surfaceId"] != finalCS["surfaceId"] {
		t.Errorf("surface id mismatch: stream=%v final=%v", streamCS["surfaceId"], finalCS["surfaceId"])
	}
}

func TestMiddlewareSanitizesInboundA2UI(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/sanitize", "ok")

	// Simulate a client sending a surface action back as the next turn.
	actionPart := newPart([]Envelope{
		{"action": map[string]any{"name": "submit", "surfaceId": "s1", "context": map[string]any{"email": "a@b.c"}}},
	})
	userMsg := ai.NewMessage(ai.RoleUser, nil, ai.NewTextPart("submit"), actionPart)

	cfg := &Config{Instructions: InstructionsNone}
	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithMessages(userMsg), ai.WithUse(cfg)); err != nil {
		t.Fatal(err)
	}

	// The model must never see the raw a2ui data part; it should be summarized.
	for _, msg := range *captured {
		for _, p := range msg.Content {
			if IsPart(p) {
				t.Fatal("model saw a raw a2ui data part; it should have been sanitized")
			}
		}
	}
	joined := ""
	for _, msg := range *captured {
		for _, p := range msg.Content {
			joined += p.Text
		}
	}
	if !strings.Contains(joined, "UI action") || !strings.Contains(joined, "submit") {
		t.Errorf("expected sanitized action summary in messages, got %q", joined)
	}
}

// A plain text part with no a2ui block must pass through with its Metadata and
// ContentType intact (e.g. Gemini thought signatures on Metadata["signature"],
// or an application/json content type), rather than being rebuilt as a fresh
// plain text part.
func TestMiddlewarePreservesTextPartMetadata(t *testing.T) {
	r := newTestRegistry(t)
	sig := "thought-sig-123"
	m := ai.NewModel("test/meta", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		part := ai.NewTextPart("just prose, no UI")
		part.ContentType = "application/json"
		part.Metadata = map[string]any{"signature": sig}
		return &ai.ModelResponse{Request: req, Message: ai.NewMessage(ai.RoleModel, nil, part)}, nil
	})
	m.Register(r)

	cfg := &Config{Instructions: InstructionsNone, Validate: ValidateStrict}
	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hi"), ai.WithUse(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.Content) != 1 {
		t.Fatalf("got %d parts, want 1", len(resp.Message.Content))
	}
	p := resp.Message.Content[0]
	if p.Metadata == nil || p.Metadata["signature"] != sig {
		t.Errorf("thought signature dropped: metadata=%v", p.Metadata)
	}
	if p.ContentType != "application/json" {
		t.Errorf("content type = %q, want application/json (not re-typed)", p.ContentType)
	}
}

// A turn that finished abnormally (e.g. blocked) whose partial text contains a
// malformed a2ui block must not be turned into a strict-mode parse error that
// discards the response; the response passes through untouched.
func TestMiddlewareSkipsParseOnAbnormalFinish(t *testing.T) {
	r := newTestRegistry(t)
	// A malformed, unterminated a2ui block that strict mode would reject.
	reply := "partial ```a2ui\n[{\"createSurface\": bad"
	m := ai.NewModel("test/blocked", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Request:       req,
			Message:       ai.NewModelTextMessage(reply),
			FinishReason:  ai.FinishReasonBlocked,
			FinishMessage: "blocked by safety settings",
		}, nil
	})
	m.Register(r)

	cfg := &Config{Instructions: InstructionsNone, Validate: ValidateStrict}
	// Generate itself surfaces a blocked finish as an error, but the middleware
	// must not replace it with an a2ui parse error, and the response state must
	// survive. Call the model through the middleware directly to observe that.
	hooks, err := cfg.New(ctx)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := hooks.WrapModel(ctx, &ai.ModelParams{Request: &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("hi")},
	}}, func(ctx context.Context, p *ai.ModelParams) (*ai.ModelResponse, error) {
		return m.Generate(ctx, p.Request, p.Callback)
	})
	if err != nil {
		t.Fatalf("abnormal finish should not surface as a middleware error, got %v", err)
	}
	if resp == nil || resp.FinishReason != ai.FinishReasonBlocked {
		t.Fatalf("response state lost; resp=%v", resp)
	}
	if resp.FinishMessage != "blocked by safety settings" {
		t.Errorf("FinishMessage lost: %q", resp.FinishMessage)
	}
}

func TestConfigRejectsInvalidValidateMode(t *testing.T) {
	if _, err := (&Config{Validate: "strick"}).New(ctx); err == nil {
		t.Fatal("expected an error for an invalid Validate mode")
	}
}

func TestConfigRejectsUnsupportedVersion(t *testing.T) {
	if _, err := (&Config{Version: "0.9"}).New(ctx); err == nil {
		t.Fatal("expected an error for an unsupported Version")
	}
}

func TestConfigRejectsInvalidInstructions(t *testing.T) {
	if _, err := (&Config{Instructions: "prompt"}).New(ctx); err == nil {
		t.Fatal("expected an error for invalid Instructions")
	}
}
