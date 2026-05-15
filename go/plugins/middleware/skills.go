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
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/goccy/go-yaml"
)

// defaultSkillsPath is the directory scanned when Skills.SkillPaths is unset.
const defaultSkillsPath = "skills"

// skillsMarker marks the system prompt part injected by this middleware so it
// can be refreshed on later tool-loop iterations instead of duplicated.
const skillsMarker = "skills-instructions"

const skillsMissingDescription = "No description provided."

// useSkillToolName is the name of the tool registered by this middleware.
// The name intentionally matches the JS implementation so prompts and
// evaluations port cleanly between runtimes.
const useSkillToolName = "use_skill"

// Skills is a middleware that makes a local library of "skills" available to
// the model. A skill is a directory containing a SKILL.md file whose contents
// become specialized instructions the model can load on demand.
//
// When used, Skills:
//   - Injects a system prompt listing the available skill names and their
//     (optional) descriptions.
//   - Registers a use_skill tool that the model can call to load a skill's
//     full SKILL.md content into the conversation.
//
// SKILL.md may start with a YAML frontmatter block with name and description
// fields; if absent, only the directory name is surfaced to the model.
//
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("use the python skill to compute ..."),
//	    ai.WithUse(&middleware.Skills{SkillPaths: []string{"skills"}}),
//	)
type Skills struct {
	// SkillPaths lists directories that are scanned for skills. Each direct
	// subdirectory containing a SKILL.md file is exposed as a skill.
	// Defaults to []string{"skills"}.
	SkillPaths []string `json:"skillPaths,omitempty"`
}

// skillInfo records where a skill's SKILL.md lives and its description.
type skillInfo struct {
	Path        string
	Description string
}

// skillFrontmatter mirrors the YAML block expected at the top of a SKILL.md.
type skillFrontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

func (s *Skills) Name() string { return provider + "/skills" }

// New scans the configured skill paths and returns a [ai.Hooks] that injects
// the skills system prompt and exposes the use_skill tool. Scanning happens
// once per [ai.Generate] call; the result is captured in the returned hooks
// so WrapGenerate and the use_skill tool agree on the same skill set.
func (s *Skills) New(ctx context.Context) (*ai.Hooks, error) {
	info, err := scanSkills(s.paths())
	if err != nil {
		return nil, err
	}

	useSkill := ai.NewTool(
		useSkillToolName,
		"Use a skill by its name.",
		func(_ *ai.ToolContext, in struct {
			SkillName string `json:"skillName" jsonschema:"description=The name of the skill to use."`
		}) (string, error) {
			si, ok := info[in.SkillName]
			if !ok {
				return "", fmt.Errorf("skill %q not found", in.SkillName)
			}
			data, err := os.ReadFile(si.Path)
			if err != nil {
				return "", fmt.Errorf("failed to read skill %q: %w", in.SkillName, err)
			}
			return string(data), nil
		},
	)

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		if len(info) == 0 {
			return next(ctx, params)
		}
		params.Request = injectSkillsPrompt(params.Request, buildSkillsPrompt(info))
		return next(ctx, params)
	}

	return &ai.Hooks{
		Tools:        []ai.Tool{useSkill},
		WrapGenerate: wrapGenerate,
	}, nil
}

// paths returns the directories to scan, falling back to the default.
func (s *Skills) paths() []string {
	if len(s.SkillPaths) == 0 {
		return []string{defaultSkillsPath}
	}
	return s.SkillPaths
}

// scanSkills enumerates SKILL.md files under each path and returns a map keyed
// by the skill's directory name. Missing or unreadable paths are skipped,
// matching the JS implementation.
func scanSkills(paths []string) (map[string]skillInfo, error) {
	result := make(map[string]skillInfo)
	for _, p := range paths {
		abs, err := filepath.Abs(p)
		if err != nil {
			continue
		}
		entries, err := os.ReadDir(abs)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
				continue
			}
			skillMd := filepath.Join(abs, entry.Name(), "SKILL.md")
			data, err := os.ReadFile(skillMd)
			if err != nil {
				continue
			}
			fm := parseFrontmatter(data)
			desc := strings.TrimSpace(fm.Description)
			if desc == "" {
				desc = skillsMissingDescription
			}
			result[entry.Name()] = skillInfo{
				Path:        skillMd,
				Description: desc,
			}
		}
	}
	return result, nil
}

// parseFrontmatter extracts the YAML frontmatter (fenced by "---" lines) at
// the top of a SKILL.md. Returns the zero value if no frontmatter is present
// or it fails to parse.
func parseFrontmatter(content []byte) skillFrontmatter {
	var fm skillFrontmatter
	text := strings.TrimPrefix(string(content), "\ufeff") // strip optional BOM
	if !strings.HasPrefix(text, "---") {
		return fm
	}
	rest := text[3:]
	// The opening fence must be followed by a newline.
	if !strings.HasPrefix(rest, "\n") && !strings.HasPrefix(rest, "\r\n") {
		return fm
	}
	// Locate the closing fence ("\n---") on its own line.
	idx := strings.Index(rest, "\n---")
	if idx < 0 {
		return fm
	}
	_ = yaml.Unmarshal([]byte(rest[:idx]), &fm)
	return fm
}

// buildSkillsPrompt renders the system prompt text listing available skills.
// Skills are sorted alphabetically to produce stable output across runs.
func buildSkillsPrompt(info map[string]skillInfo) string {
	names := make([]string, 0, len(info))
	for name := range info {
		names = append(names, name)
	}
	sort.Strings(names)

	var b strings.Builder
	b.WriteString("<skills>\n")
	b.WriteString("You have access to a library of skills that serve as specialized instructions/personas.\n")
	b.WriteString("Strongly prefer to use them when working on anything related to them.\n")
	b.WriteString("Only use them once to load the context.\n")
	b.WriteString("Here are the available skills:\n")
	for _, name := range names {
		desc := info[name].Description
		if desc == "" || desc == skillsMissingDescription {
			fmt.Fprintf(&b, " - %s\n", name)
			continue
		}
		fmt.Fprintf(&b, " - %s - %s\n", name, desc)
	}
	b.WriteString("</skills>")
	return b.String()
}

// injectSkillsPrompt returns a copy of req with promptText placed in a part
// marked by skillsMarker. If such a part already exists it is replaced in
// place; otherwise the text is appended to the existing system message, or a
// new system message is prepended.
func injectSkillsPrompt(req *ai.ModelRequest, promptText string) *ai.ModelRequest {
	newReq := *req
	newReq.Messages = append([]*ai.Message(nil), req.Messages...)

	// Refresh an existing injected part in place.
	for i, msg := range newReq.Messages {
		if msg == nil {
			continue
		}
		for j, part := range msg.Content {
			if !hasSkillsMarker(part) {
				continue
			}
			if part.Text == promptText {
				return &newReq
			}
			msgCopy := msg.Clone()
			msgCopy.Content[j] = newSkillsPart(promptText)
			newReq.Messages[i] = msgCopy
			return &newReq
		}
	}

	// Append to an existing system message.
	for i, msg := range newReq.Messages {
		if msg == nil || msg.Role != ai.RoleSystem {
			continue
		}
		msgCopy := msg.Clone()
		msgCopy.Content = append(msgCopy.Content, newSkillsPart(promptText))
		newReq.Messages[i] = msgCopy
		return &newReq
	}

	// Otherwise prepend a fresh system message.
	newReq.Messages = append(
		[]*ai.Message{ai.NewSystemMessage(newSkillsPart(promptText))},
		newReq.Messages...,
	)
	return &newReq
}

// newSkillsPart builds the text part that carries the skills prompt, tagged
// with skillsMarker so later iterations can find and refresh it.
func newSkillsPart(text string) *ai.Part {
	p := ai.NewTextPart(text)
	p.Metadata = map[string]any{skillsMarker: true}
	return p
}

// hasSkillsMarker reports whether p is a text part tagged as the skills prompt.
func hasSkillsMarker(p *ai.Part) bool {
	if p == nil || !p.IsText() || p.Metadata == nil {
		return false
	}
	v, ok := p.Metadata[skillsMarker].(bool)
	return ok && v
}
