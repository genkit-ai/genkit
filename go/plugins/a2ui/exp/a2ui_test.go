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
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/firebase/genkit/go/internal/schematest"
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

	cfg := &Surfaces{}
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

	cfg := &Surfaces{Instructions: InstructionsNone}
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

	cfg := &Surfaces{SurfaceID: "fixed-surface", Validate: ValidateStrict}
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

// The aggregated final message is not guaranteed to coalesce adjacent text: the
// Gemini plugin splits a turn into many text parts (fence, JSON body split many
// ways, close fence, then a trailing empty-text part carrying the thought
// signature). transformResponse must stitch a block spanning several parts into
// a single a2ui data part rather than flushing per part and leaking the whole
// surface back out as raw prose.
func TestMiddlewareRewritesFinalMessageSplitAcrossParts(t *testing.T) {
	r := newTestRegistry(t)
	catalog := BasicCatalog()
	sig := "thought-sig-xyz"

	// The fenced block, chopped into many adjacent text parts the way a
	// streaming provider aggregates them, followed by an empty-text part that
	// only carries a thought signature (as Gemini does).
	parts := []*ai.Part{
		ai.NewTextPart("Here you go:\n\n``"),
		ai.NewTextPart("`a2ui\n[{\"createSurface\":{\"surfaceId\":\"SURFACE_ID\","),
		ai.NewTextPart("\"catalogId\":\"" + catalog.ID + "\"}},"),
		ai.NewTextPart("{\"updateComponents\":{\"surfaceId\":\"SURFACE_ID\","),
		ai.NewTextPart("\"components\":[{\"id\":\"root\",\"component\":\"Text\","),
		ai.NewTextPart("\"text\":\"hi\"}]}}]\n``"),
		ai.NewTextPart("`"),
	}
	sigPart := ai.NewTextPart("")
	sigPart.Metadata = map[string]any{"signature": sig}
	parts = append(parts, sigPart)

	m := ai.NewModel("test/split", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Request: req, Message: ai.NewMessage(ai.RoleModel, nil, parts...)}, nil
	})
	m.Register(r)

	cfg := &Surfaces{SurfaceID: "fixed-surface", Validate: ValidateStrict}
	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("card please"), ai.WithUse(cfg))
	if err != nil {
		t.Fatal(err)
	}

	// The block spread over many parts must be stitched into exactly two
	// envelopes on a single a2ui data part.
	envs := EnvelopesFromParts(resp.Message.Content)
	if len(envs) != 2 {
		t.Fatalf("got %d envelopes, want 2; content=%v", len(envs), resp.Message.Content)
	}
	cs, _ := envs[0]["createSurface"].(map[string]any)
	if cs["surfaceId"] != "fixed-surface" {
		t.Errorf("surfaceId = %v, want fixed-surface", cs["surfaceId"])
	}

	// No text part should still contain the raw fence: the JSON must have been
	// parsed out, not leaked back as prose.
	for _, p := range resp.Message.Content {
		if p.IsText() && strings.Contains(p.Text, "a2ui") {
			t.Errorf("raw a2ui fence leaked into prose: %q", p.Text)
		}
	}

	// The leading prose survives, and the trailing empty-text signature part is
	// carried through untouched.
	hasProse := false
	hasSig := false
	for _, p := range resp.Message.Content {
		if p.IsText() && strings.Contains(p.Text, "Here you go:") {
			hasProse = true
		}
		if p.Metadata != nil && p.Metadata["signature"] == sig {
			hasSig = true
		}
	}
	if !hasProse {
		t.Error("expected the leading prose to be preserved as a text part")
	}
	if !hasSig {
		t.Error("expected the trailing thought-signature part to survive")
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

	cfg := &Surfaces{SurfaceID: "fixed", Validate: ValidateStrict}
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

	cfg := &Surfaces{Instructions: InstructionsNone}
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

	cfg := &Surfaces{Instructions: InstructionsNone, Validate: ValidateStrict}
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

	cfg := &Surfaces{Instructions: InstructionsNone, Validate: ValidateStrict}
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

// A prior assistant surface replayed as history must be reconstructed as the
// canonical fenced a2ui block the model originally emitted, NOT summarized to a
// sentinel like [rendered UI surface] (which taught the model to echo that
// literal string instead of real UI).
func TestMiddlewareReplaysPriorSurfaceAsBlock(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/replay", "ok")

	catalog := BasicCatalog()
	surfacePart := newPart([]Envelope{
		{"createSurface": map[string]any{"surfaceId": "s1", "catalogId": catalog.ID}, "version": "v0.9"},
		{"updateComponents": map[string]any{"surfaceId": "s1", "components": []any{
			map[string]any{"id": "root", "component": "Text", "text": "hi"},
		}}, "version": "v0.9"},
	})
	modelMsg := ai.NewMessage(ai.RoleModel, nil, ai.NewTextPart("Here you go:"), surfacePart)
	userMsg := ai.NewUserTextMessage("thanks")

	cfg := &Surfaces{Instructions: InstructionsNone}
	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithMessages(modelMsg, userMsg), ai.WithUse(cfg)); err != nil {
		t.Fatal(err)
	}

	var modelSeen *ai.Message
	for _, msg := range *captured {
		if msg.Role == ai.RoleModel {
			modelSeen = msg
			break
		}
	}
	if modelSeen == nil {
		t.Fatalf("expected a model message; messages=%v", *captured)
	}
	for _, p := range modelSeen.Content {
		if IsPart(p) {
			t.Fatal("model saw a raw a2ui data part; it should have been sanitized")
		}
	}
	joined := ""
	for _, p := range modelSeen.Content {
		joined += p.Text + "\n"
	}
	if strings.Contains(joined, "[rendered UI surface]") || strings.Contains(joined, "[UI surface") {
		t.Errorf("prior surface summarized to a sentinel: %q", joined)
	}
	if !strings.Contains(joined, "```a2ui") || !strings.Contains(joined, "createSurface") || !strings.Contains(joined, "updateComponents") {
		t.Errorf("prior surface not reconstructed as a fenced block: %q", joined)
	}
	if !strings.Contains(joined, "Here you go:") {
		t.Errorf("leading prose lost: %q", joined)
	}
	// The real surface id is kept verbatim so a replayed action can correlate
	// with this surface.
	if !strings.Contains(joined, `"surfaceId":"s1"`) {
		t.Errorf("real surface id not preserved verbatim: %q", joined)
	}
}

// A message whose only content is an a2ui part with an empty (or all-
// unrecognized) envelope array summarizes to nothing. The middleware must drop
// the whole message rather than send empty content downstream (which providers
// like Gemini/Vertex reject).
func TestMiddlewareDropsEmptiedMessage(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/empty", "ok")

	emptyPart := newPart(nil)
	modelMsg := ai.NewMessage(ai.RoleModel, nil, emptyPart)
	userMsg := ai.NewUserTextMessage("hi")

	cfg := &Surfaces{Instructions: InstructionsNone}
	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithMessages(modelMsg, userMsg), ai.WithUse(cfg)); err != nil {
		t.Fatal(err)
	}

	for _, msg := range *captured {
		if len(msg.Content) == 0 {
			t.Fatalf("a message was sent downstream with empty content: %v", *captured)
		}
	}
	// The still-meaningful user message survives.
	var userSeen *ai.Message
	for _, msg := range *captured {
		if msg.Role == ai.RoleUser {
			userSeen = msg
		}
	}
	if userSeen == nil || len(userSeen.Content) == 0 || userSeen.Content[0].Text != "hi" {
		t.Errorf("user message did not survive sanitizing; messages=%v", *captured)
	}
}

// Regression for the "new answer overwrites the prior surface in place" bug:
// history keeps real ids (for action correlation), so the model can copy an old
// id into a fresh createSurface. The parser must still mint a distinct id for
// that new render.
func TestMiddlewareNewRenderNeverReusesHistoryID(t *testing.T) {
	r := newTestRegistry(t)
	catalog := BasicCatalog()

	// The model copies the prior surface's real id (s1) into a brand-new
	// createSurface - exactly what it does after seeing s1 in history.
	reply := "Here you go:\n```a2ui\n" +
		`[{"createSurface":{"surfaceId":"s1","catalogId":"` + catalog.ID + `"}},` +
		`{"updateComponents":{"surfaceId":"s1","components":[{"id":"root","component":"Text","text":"new"}]}}]` +
		"\n```"
	m, captured := echoModel(t, r, "test/reuse", reply)

	priorSurface := newPart([]Envelope{
		{"createSurface": map[string]any{"surfaceId": "s1", "catalogId": catalog.ID}},
		{"updateComponents": map[string]any{"surfaceId": "s1", "components": []any{
			map[string]any{"id": "root", "component": "Text", "text": "old"},
		}}},
	})
	actionPart := newPart([]Envelope{
		{"action": map[string]any{"name": "refresh", "surfaceId": "s1"}},
	})
	modelMsg := ai.NewMessage(ai.RoleModel, nil, priorSurface)
	userMsg := ai.NewMessage(ai.RoleUser, nil, actionPart)

	cfg := &Surfaces{SurfaceID: "sfc-new"}
	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithMessages(modelMsg, userMsg), ai.WithUse(cfg))
	if err != nil {
		t.Fatal(err)
	}

	// The new render is minted onto the fixed id sfc-new, NOT the copied s1, so
	// it can't overwrite the prior surface.
	envs := EnvelopesFromParts(resp.Message.Content)
	var create, update map[string]any
	for _, e := range envs {
		if cs, ok := e["createSurface"].(map[string]any); ok {
			create = cs
		}
		if uc, ok := e["updateComponents"].(map[string]any); ok {
			update = uc
		}
	}
	if create == nil || create["surfaceId"] != "sfc-new" {
		t.Errorf("createSurface id = %v, want sfc-new", create["surfaceId"])
	}
	if update == nil || update["surfaceId"] != "sfc-new" {
		t.Errorf("updateComponents id = %v, want sfc-new", update["surfaceId"])
	}

	// Meanwhile, the sanitized history the model saw kept the real id on both
	// the reconstructed surface block and the action line (correlation).
	var modelSeen, userSeen *ai.Message
	for _, msg := range *captured {
		switch msg.Role {
		case ai.RoleModel:
			modelSeen = msg
		case ai.RoleUser:
			userSeen = msg
		}
	}
	modelText := ""
	for _, p := range modelSeen.Content {
		modelText += p.Text + "\n"
	}
	if !strings.Contains(modelText, `"surfaceId":"s1"`) {
		t.Errorf("history lost the real surface id: %q", modelText)
	}
	userText := ""
	for _, p := range userSeen.Content {
		userText += p.Text + "\n"
	}
	if !strings.Contains(userText, "on surface s1") {
		t.Errorf("action summary lost the real surface id: %q", userText)
	}
}

// Consecutive surface envelopes are grouped into one fenced block, but an action
// between them splits the output (the block precedes the action summary).
func TestMiddlewareGroupsSurfacesSplitsAroundAction(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/group", "ok")

	catalog := BasicCatalog()
	part := newPart([]Envelope{
		{"createSurface": map[string]any{"surfaceId": "s1", "catalogId": catalog.ID}},
		{"updateComponents": map[string]any{"surfaceId": "s1", "components": []any{}}},
		{"action": map[string]any{"name": "refresh", "surfaceId": "s1"}},
	})
	userMsg := ai.NewMessage(ai.RoleUser, nil, part)

	cfg := &Surfaces{Instructions: InstructionsNone}
	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithMessages(userMsg), ai.WithUse(cfg)); err != nil {
		t.Fatal(err)
	}

	var userSeen *ai.Message
	for _, msg := range *captured {
		if msg.Role == ai.RoleUser {
			userSeen = msg
		}
	}
	joined := ""
	for _, p := range userSeen.Content {
		joined += p.Text + "\n"
	}
	// Exactly one fenced block (the two surface envelopes grouped together).
	if got := strings.Count(joined, "```a2ui"); got != 1 {
		t.Errorf("got %d fenced blocks, want 1: %q", got, joined)
	}
	// Plus the action rendered as a text summary after it.
	if !strings.Contains(joined, "UI action") {
		t.Errorf("action summary missing: %q", joined)
	}
	// The block precedes the action line (source order preserved).
	if strings.Index(joined, "```a2ui") > strings.Index(joined, "UI action") {
		t.Errorf("block should precede action line: %q", joined)
	}
}

func TestSurfacesRejectsInvalidValidateMode(t *testing.T) {
	if _, err := (&Surfaces{Validate: "strick"}).New(ctx); err == nil {
		t.Fatal("expected an error for an invalid Validate mode")
	}
}

func TestSurfacesRejectsUnsupportedVersion(t *testing.T) {
	if _, err := (&Surfaces{Version: "0.9"}).New(ctx); err == nil {
		t.Fatal("expected an error for an unsupported Version")
	}
}

func TestSurfacesRejectsInvalidInstructions(t *testing.T) {
	if _, err := (&Surfaces{Instructions: "prompt"}).New(ctx); err == nil {
		t.Fatal("expected an error for invalid Instructions")
	}
}

// The plugin exists only to register the middleware so it can be resolved by
// name: from the Dev UI, from another runtime, or from a prompt file's `use:`
// list. Its descriptor must carry the name the middleware answers to, a
// description, a fully documented schema, and exactly the serializable
// options (Catalog is code-only).
func TestPluginDescribesMiddleware(t *testing.T) {
	descs, err := (&A2UI{}).Middlewares(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) != 1 {
		t.Fatalf("got %d middleware descriptors, want 1", len(descs))
	}
	d := descs[0]
	if d.Name != provider || (&A2UI{}).Name() != provider {
		t.Errorf("descriptor name = %q, plugin name = %q; both must be %q", d.Name, (&A2UI{}).Name(), provider)
	}
	if d.Description == "" {
		t.Error("middleware has no description")
	}
	schematest.AssertDescribed(t, d.Name, d.ConfigSchema)

	props, _ := d.ConfigSchema["properties"].(map[string]any)
	got := slices.Sorted(maps.Keys(props))
	want := []string{"catalogId", "instructions", "surfaceId", "validate", "version"}
	if !slices.Equal(got, want) {
		t.Errorf("config schema properties = %v, want %v", got, want)
	}
}

// TestDescriptionsUseTheDedicatedTag guards against the inline
// `jsonschema:"description=..."` form, which the schema library truncates at
// the first comma; see [schematest.AssertNoInlineDescriptions].
func TestDescriptionsUseTheDedicatedTag(t *testing.T) {
	schematest.AssertNoInlineDescriptions(t, ".")
}

// Every JSON-dispatched call (Dev UI, another runtime, a prompt file) builds
// its own config off the registered prototype, so an option one call sets must
// not leak into the next.
func TestJSONDispatchDoesNotLeakConfigBetweenCalls(t *testing.T) {
	r := newTestRegistry(t)
	m, captured := echoModel(t, r, "test/echo", "ok")
	descs, err := (&A2UI{}).Middlewares(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	descs[0].Register(r)

	injectedSystem := func(config map[string]any) bool {
		t.Helper()
		_, err := ai.GenerateWithRequest(t.Context(), r, &ai.GenerateActionOptions{
			Model:    m.Name(),
			Messages: []*ai.Message{ai.NewUserTextMessage("hi")},
			Use:      []*ai.MiddlewareRef{{Name: provider, Config: config}},
		}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		return slices.ContainsFunc(*captured, func(msg *ai.Message) bool { return msg.Role == ai.RoleSystem })
	}

	if injectedSystem(map[string]any{"instructions": InstructionsNone}) {
		t.Error("instructions=none still injected a system message")
	}
	if !injectedSystem(map[string]any{}) {
		t.Error("default dispatch injected no system message: instructions=none leaked from the previous call")
	}
}
