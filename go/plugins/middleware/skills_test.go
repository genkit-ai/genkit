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

package middleware

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/registry"
)

// setupSkillsDir creates a temporary skills directory with two skills: one
// with a YAML frontmatter description and one without. Returns the absolute
// path to the skills/ directory.
func setupSkillsDir(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	skillsDir := filepath.Join(tmp, "skills")
	if err := os.Mkdir(skillsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	py := filepath.Join(skillsDir, "python")
	if err := os.Mkdir(py, 0o755); err != nil {
		t.Fatal(err)
	}
	pyMd := "---\nname: python\ndescription: A python expert skill\n---\nPython prompt content"
	if err := os.WriteFile(filepath.Join(py, "SKILL.md"), []byte(pyMd), 0o644); err != nil {
		t.Fatal(err)
	}

	js := filepath.Join(skillsDir, "javascript")
	if err := os.Mkdir(js, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(js, "SKILL.md"), []byte("Just javascript content"), 0o644); err != nil {
		t.Fatal(err)
	}
	return skillsDir
}

// captureModel returns a model that records the messages it receives and
// returns a fixed text response. The returned pointer lets the test inspect
// what the middleware produced.
func captureModel(t *testing.T, r *registry.Registry, name string) (ai.Model, *[]*ai.Message) {
	t.Helper()
	var captured []*ai.Message
	m := ai.DefineModel(r, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		captured = req.Messages
		return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("mock response")}, nil
	})
	return m, &captured
}

// toolCallingModel returns a model that issues a single tool request on its
// first call, then returns "done" once the tool response is visible in the
// messages.
func toolCallingModel(t *testing.T, r *registry.Registry, name, toolName string, input map[string]any) ai.Model {
	t.Helper()
	return ai.DefineModel(r, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
				}
			}
		}
		return &ai.ModelResponse{
			Request: req,
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: toolName, Input: input}),
				},
			},
		}, nil
	})
}

func TestSkillsInjectsSystemPrompt(t *testing.T) {
	r := newTestRegistry(t)
	skillsDir := setupSkillsDir(t)

	m, captured := captureModel(t, r, "test/capture")

	s := &Skills{SkillPaths: []string{skillsDir}}
	ai.DefineMiddleware(r, "skills", s)

	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(s)); err != nil {
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

	text := sys.Content[0].Text
	if !strings.Contains(text, "python - A python expert skill") {
		t.Errorf("system prompt missing python description: %q", text)
	}
	if !strings.Contains(text, "javascript") {
		t.Errorf("system prompt missing javascript entry: %q", text)
	}
}

func TestSkillsRegistersUseSkillTool(t *testing.T) {
	r := newTestRegistry(t)
	skillsDir := setupSkillsDir(t)

	m := toolCallingModel(t, r, "test/toolcaller", useSkillToolName, map[string]any{"skillName": "python"})

	s := &Skills{SkillPaths: []string{skillsDir}}
	ai.DefineMiddleware(r, "skills", s)

	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("use python"), ai.WithUse(s))
	if err != nil {
		t.Fatal(err)
	}

	// Find the tool response in the history.
	var got string
	for _, msg := range resp.History() {
		for _, part := range msg.Content {
			if part.IsToolResponse() && part.ToolResponse.Name == useSkillToolName {
				if out, ok := part.ToolResponse.Output.(string); ok {
					got = out
				}
			}
		}
	}
	if got == "" {
		t.Fatalf("expected a tool response with the skill content; history=%v", resp.History())
	}
	if !strings.Contains(got, "Python prompt content") {
		t.Errorf("tool response missing SKILL.md body: %q", got)
	}
}

func TestSkillsUnknownSkillReturnsError(t *testing.T) {
	r := newTestRegistry(t)
	skillsDir := setupSkillsDir(t)

	m := toolCallingModel(t, r, "test/unknown", useSkillToolName, map[string]any{"skillName": "nonexistent"})

	s := &Skills{SkillPaths: []string{skillsDir}}
	ai.DefineMiddleware(r, "skills", s)

	_, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("use skill"), ai.WithUse(s))
	if err == nil {
		t.Fatal("expected an error for unknown skill")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' in error, got %v", err)
	}
}

func TestSkillsPromptInjectionIsIdempotent(t *testing.T) {
	r := newTestRegistry(t)
	skillsDir := setupSkillsDir(t)

	m, captured := captureModel(t, r, "test/idempotent")

	s := &Skills{SkillPaths: []string{skillsDir}}
	ai.DefineMiddleware(r, "skills", s)

	// First call produces a system message with a single <skills> block.
	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(s))
	if err != nil {
		t.Fatal(err)
	}

	sys := findSystem(*captured)
	if sys == nil {
		t.Fatal("expected system message after first call")
	}
	if n := strings.Count(sys.Content[0].Text, "<skills>"); n != 1 {
		t.Errorf("first call: got %d <skills> blocks, want 1", n)
	}

	// Second call replays the prior history (which already contains the
	// injected system message). The middleware must refresh the existing
	// part in place rather than duplicating it.
	if _, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithMessages(resp.History()...),
		ai.WithUse(s),
	); err != nil {
		t.Fatal(err)
	}

	sys = findSystem(*captured)
	if sys == nil {
		t.Fatal("expected system message after second call")
	}
	var total int
	for _, p := range sys.Content {
		total += strings.Count(p.Text, "<skills>")
	}
	if total != 1 {
		t.Errorf("second call: got %d <skills> blocks across system message parts, want 1", total)
	}
}

func TestSkillsNoopWhenNoSkillsFound(t *testing.T) {
	r := newTestRegistry(t)

	m, captured := captureModel(t, r, "test/empty")

	// Point at an empty directory — no skills, so the middleware should
	// leave the request untouched.
	s := &Skills{SkillPaths: []string{t.TempDir()}}
	ai.DefineMiddleware(r, "skills", s)

	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("hello"), ai.WithUse(s)); err != nil {
		t.Fatal(err)
	}

	if findSystem(*captured) != nil {
		t.Error("did not expect a system message when no skills were found")
	}
}

func findSystem(msgs []*ai.Message) *ai.Message {
	for _, msg := range msgs {
		if msg.Role == ai.RoleSystem {
			return msg
		}
	}
	return nil
}
