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
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
)

// artifactsMarker tags the system prompt part injected by this middleware so it
// can be refreshed each turn as the session's artifacts change.
const artifactsMarker = "artifacts-listing"

// Artifacts is a middleware that gives the model tools to interact with session
// artifacts and injects a listing of available artifacts into the system prompt.
//
// It provides:
//
//   - read_artifact: reads an artifact by name from the session and returns its
//     text content.
//   - write_artifact (unless Readonly): creates or updates a session artifact.
//     Artifacts are deduplicated by name, so writing to an existing name
//     replaces it.
//
// On every generate turn an <artifacts> block listing the names and sizes of
// the session's artifacts is injected into (or refreshed within) the system
// message, so the model knows what is available without spending context on the
// full content.
//
// This is useful standalone (e.g. a workspace-builder agent that creates files
// as artifacts) or combined with [Agents] using ArtifactStrategySession, where
// sub-agent artifacts are merged into the parent session and the model reaches
// them through these tools.
//
// Artifacts live on the active agent session, so this middleware only has an
// effect when generation runs inside an agent invocation (see
// [github.com/firebase/genkit/go/genkit/exp.DefineAgent]). With no active
// session the tools report that gracefully and the listing is empty.
//
// Usage:
//
//	builder := genkitx.DefineAgent(g, "builder",
//	    aix.InlinePrompt{
//	        ai.WithModelName("googleai/gemini-flash-latest"),
//	        ai.WithSystem("You are a code generator. Use write_artifact to create files."),
//	        ai.WithUse(&middlewarex.Artifacts{}),
//	    },
//	)
type Artifacts struct {
	// Readonly, when true, provides only the read_artifact tool; the model
	// cannot create or update artifacts. Defaults to false.
	Readonly bool `json:"readonly,omitempty"`
}

func (a *Artifacts) Name() string { return provider + "/artifacts" }

// New returns the hooks that register the artifact tools and inject the
// artifact listing into the system prompt on each turn.
func (a *Artifacts) New(ctx context.Context) (*ai.Hooks, error) {
	tools := []ai.Tool{newReadArtifactTool()}
	if !a.Readonly {
		tools = append(tools, newWriteArtifactTool())
	}

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		var arts []*aix.Artifact
		if store := aix.ArtifactStoreFromContext(ctx); store != nil {
			arts = store.Artifacts()
		}
		params.Request = injectSystemText(params.Request, artifactsMarker, buildArtifactsListing(arts))
		return next(ctx, params)
	}

	return &ai.Hooks{
		Tools:        tools,
		WrapGenerate: wrapGenerate,
	}, nil
}

type readArtifactInput struct {
	Name string `json:"name" jsonschema:"description=The name of the artifact to read."`
}

type readArtifactOutput struct {
	Name    string `json:"name"`
	Content string `json:"content"`
	Found   bool   `json:"found"`
}

func newReadArtifactTool() ai.Tool {
	return aix.NewTool("read_artifact",
		"Reads the content of a named artifact from the session. Use this to "+
			"inspect artifacts produced by sub-agents or previously created artifacts.",
		func(ctx context.Context, in readArtifactInput) (readArtifactOutput, error) {
			store := aix.ArtifactStoreFromContext(ctx)
			if store == nil {
				return readArtifactOutput{Name: in.Name, Content: "Error: no active session.", Found: false}, nil
			}
			for _, art := range store.Artifacts() {
				if art.Name == in.Name {
					return readArtifactOutput{Name: in.Name, Content: artifactText(art), Found: true}, nil
				}
			}
			return readArtifactOutput{Name: in.Name, Content: fmt.Sprintf("Artifact %q not found.", in.Name), Found: false}, nil
		})
}

type writeArtifactInput struct {
	Name    string `json:"name" jsonschema:"description=A unique name for the artifact (e.g. a filename like report.md)."`
	Content string `json:"content" jsonschema:"description=The full text content of the artifact."`
}

type writeArtifactOutput struct {
	Status string `json:"status"`
}

func newWriteArtifactTool() ai.Tool {
	return aix.NewTool("write_artifact",
		"Creates or updates a named artifact in the session. If an artifact with "+
			"the same name already exists, it is replaced. Use this to produce "+
			"files, reports, code, or other deliverables.",
		func(ctx context.Context, in writeArtifactInput) (writeArtifactOutput, error) {
			store := aix.ArtifactStoreFromContext(ctx)
			if store == nil {
				return writeArtifactOutput{Status: "Error: no active session."}, nil
			}
			store.AddArtifacts(&aix.Artifact{
				Name:  in.Name,
				Parts: []*ai.Part{ai.NewTextPart(in.Content)},
			})
			return writeArtifactOutput{Status: fmt.Sprintf("Artifact %q saved successfully.", in.Name)}, nil
		})
}

// artifactText joins an artifact's text parts with newlines, skipping
// non-text and empty parts.
func artifactText(a *aix.Artifact) string {
	var b strings.Builder
	for _, p := range a.Parts {
		if p == nil || !p.IsText() || p.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

// buildArtifactsListing renders the <artifacts> system block listing the
// session's artifacts and their sizes. Artifacts are listed in session order,
// which is stable across turns so the injected text only changes when the set
// of artifacts does.
func buildArtifactsListing(arts []*aix.Artifact) string {
	var b strings.Builder
	b.WriteString("<artifacts>\n")
	if len(arts) == 0 {
		b.WriteString("No artifacts are currently available in the session.\n")
		b.WriteString("</artifacts>")
		return b.String()
	}
	b.WriteString("The following artifacts are available in the session. ")
	b.WriteString("Use the read_artifact tool to view their content.\n")
	for _, a := range arts {
		if a == nil {
			continue
		}
		name := a.Name
		if name == "" {
			name = "(unnamed)"
		}
		fmt.Fprintf(&b, "  - %s", name)
		if text := artifactText(a); len(text) > 0 {
			fmt.Fprintf(&b, " (%d chars)", utf8.RuneCountInString(text))
		}
		if src := artifactSource(a); src != "" {
			fmt.Fprintf(&b, " [from: %s]", src)
		}
		b.WriteByte('\n')
	}
	b.WriteString("</artifacts>")
	return b.String()
}

// artifactSource returns the artifact's "source" metadata (set when an artifact
// originates from a sub-agent delegation), or "" if absent.
func artifactSource(a *aix.Artifact) string {
	if a.Metadata == nil {
		return ""
	}
	src, _ := a.Metadata["source"].(string)
	return src
}
