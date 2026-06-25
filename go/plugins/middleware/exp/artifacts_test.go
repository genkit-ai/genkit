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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

func toolResponseByName(t *testing.T, msgs []*ai.Message, name string) (any, bool) {
	t.Helper()
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolResponse() && p.ToolResponse != nil && p.ToolResponse.Name == name {
				return p.ToolResponse.Output, true
			}
		}
	}
	return nil, false
}

func decodeReadArtifact(t *testing.T, v any) readArtifactOutput {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var o readArtifactOutput
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal readArtifactOutput: %v", err)
	}
	return o
}

func TestArtifactsReadonlyOmitsWriteTool(t *testing.T) {
	hooks, err := (&Artifacts{Readonly: true}).New(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(hooks.Tools) != 1 || hooks.Tools[0].Name() != "read_artifact" {
		t.Fatalf("readonly should expose only read_artifact, got %v", toolNames(hooks.Tools))
	}

	hooks, err = (&Artifacts{}).New(ctx)
	if err != nil {
		t.Fatal(err)
	}
	names := toolNames(hooks.Tools)
	if len(names) != 2 || !contains(names, "read_artifact") || !contains(names, "write_artifact") {
		t.Fatalf("default should expose read_artifact and write_artifact, got %v", names)
	}
}

func TestArtifactsWriteThenRead(t *testing.T) {
	g := newTestGenkit(t)

	// The model writes an artifact, reads it back, then finishes.
	model := toolModel(t, g, "test/artifact-model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		var wrote, read bool
		for _, m := range req.Messages {
			for _, p := range m.Content {
				if p.IsToolResponse() && p.ToolResponse != nil {
					switch p.ToolResponse.Name {
					case "write_artifact":
						wrote = true
					case "read_artifact":
						read = true
					}
				}
			}
		}
		switch {
		case read:
			return textResp(req, "done"), nil
		case wrote:
			return toolReqResp(req, &ai.ToolRequest{Name: "read_artifact", Input: map[string]any{"name": "report.md"}}), nil
		default:
			return toolReqResp(req, &ai.ToolRequest{Name: "write_artifact", Input: map[string]any{"name": "report.md", "content": "hello world"}}), nil
		}
	})

	builder := genkitx.DefineAgent[any](g, "builder",
		aix.InlinePrompt{ai.WithModel(model), ai.WithSystem("be a builder"), ai.WithUse(&Artifacts{})},
	)

	out, err := builder.RunText(ctx, "make a report")
	if err != nil {
		t.Fatal(err)
	}

	// read_artifact returned the content written by write_artifact.
	v, ok := toolResponseByName(t, statemessages(out), "read_artifact")
	if !ok {
		t.Fatal("no read_artifact tool response found")
	}
	read := decodeReadArtifact(t, v)
	if !read.Found || read.Content != "hello world" {
		t.Errorf("read_artifact = %+v, want found with content %q", read, "hello world")
	}

	// The artifact persisted on the session.
	if !hasArtifactNamed(out.Artifacts, "report.md") {
		t.Errorf("expected artifact %q on session; got %v", "report.md", artifactNames(out.Artifacts))
	}
}

func TestArtifactsSystemPromptListing(t *testing.T) {
	g := newTestGenkit(t)

	var captured []*ai.Message
	capture := toolModel(t, g, "test/capture", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		captured = req.Messages
		return textResp(req, "ok"), nil
	})

	// A custom agent seeds an artifact, then generates with the Artifacts
	// middleware so the listing reflects the seeded artifact.
	lister := genkitx.DefineCustomAgent[any](g, "lister",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				resp.SendArtifact(&aix.Artifact{
					Name:  "notes.md",
					Parts: []*ai.Part{ai.NewTextPart("some notes here")},
				})
				if _, err := genkit.Generate(ctx, g,
					ai.WithModel(capture),
					ai.WithMessages(input.Message),
					ai.WithUse(&Artifacts{}),
				); err != nil {
					return nil, err
				}
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{Message: ai.NewModelTextMessage("listed")}, nil
		},
	)

	if _, err := lister.RunText(ctx, "what artifacts are there?"); err != nil {
		t.Fatal(err)
	}

	sys := findSystem(captured)
	if sys == nil {
		t.Fatalf("expected a system message; got %v", captured)
	}
	text := systemText(sys)
	if !strings.Contains(text, "<artifacts>") || !strings.Contains(text, "notes.md") {
		t.Errorf("system prompt missing artifact listing for notes.md; got:\n%s", text)
	}
}

func TestArtifactsNoSession(t *testing.T) {
	g := newTestGenkit(t)

	// With a plain Generate call there is no agent session.
	model := toolModel(t, g, "test/no-session", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, m := range req.Messages {
			for _, p := range m.Content {
				if p.IsToolResponse() {
					return textResp(req, "done"), nil
				}
			}
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "read_artifact", Input: map[string]any{"name": "x"}}), nil
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt("read x"), ai.WithUse(&Artifacts{}))
	if err != nil {
		t.Fatal(err)
	}
	v, ok := toolResponseByName(t, resp.History(), "read_artifact")
	if !ok {
		t.Fatal("no read_artifact tool response found")
	}
	read := decodeReadArtifact(t, v)
	if read.Found || !strings.Contains(read.Content, "no active session") {
		t.Errorf("expected a no-active-session result, got %+v", read)
	}
}

func toolNames(tools []ai.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tl := range tools {
		names = append(names, tl.Name())
	}
	return names
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
