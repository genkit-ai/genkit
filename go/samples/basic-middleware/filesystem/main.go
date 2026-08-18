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

// This sample demonstrates the Filesystem middleware, which grants the model
// scoped file access through list_files, read_file, write_file, and
// search_and_replace tools. Everything is confined to RootDir by os.Root, which
// rejects any path resolving outside it, including via "..", absolute paths, or
// symlinks. A mock project in workspace/ gives the tools something to work on.
//
//   - exploreFlow answers a question by listing and reading files.
//   - editFlow applies a SEARCH/REPLACE edit, writing to workspace/ on disk.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to watch the tool calls in a trace of every run at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP:
//
//	curl -N -X POST 'http://localhost:8080/exploreFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"question": "Summarise what this project does and what is still pending."}}'
//
//	curl -N -X POST 'http://localhost:8080/editFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"instruction": "Mark the in-memory response cache TODO as done."}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// workspaceDir is the sandbox the model is allowed to see. All filesystem
// tool calls are rooted here; anything outside is unreachable by construction.
const workspaceDir = "./workspace"

// A struct rather than a bare string lets each field carry a description and a
// default, which the Dev UI pre-fills its form from.
type (
	// ExploreRequest asks a question about the workspace.
	ExploreRequest struct {
		Question string `json:"question" jsonschema:"default=Summarise what this project does and what is still pending." jsonschema_description:"A question about the project in workspace/"`
	}

	// EditRequest asks for a change to a file in the workspace.
	EditRequest struct {
		Instruction string `json:"instruction" jsonschema:"default=Mark the in-memory response cache TODO as done." jsonschema_description:"A change to make to a file in workspace/"`
	}
)

// model is shared by every flow below, so switching models or thinking levels
// for the whole sample is a one-line change.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// Registering the Middleware plugin exposes the built-in middleware
	// (Filesystem, Retry, Fallback, ...) to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	DefineExploreFlow(g)
	DefineEditFlow(g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineExploreFlow is read-only. AllowWriteAccess is unset, so write_file and
// search_and_replace are never registered and the model cannot modify anything.
func DefineExploreFlow(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "exploreFlow",
		func(ctx context.Context, input ExploreRequest, sendChunk core.StreamCallback[string]) (string, error) {
			text, err := genkit.GenerateText(ctx, g,
				ai.WithModel(model),
				ai.WithSystem("You are a helpful project analyst. Use the filesystem tools to explore the workspace before answering."),
				ai.WithPrompt(input.Question),
				ai.WithMaxTurns(20),
				ai.WithUse(&middleware.Filesystem{RootDir: workspaceDir}),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					// Every tool turn streams as well, and those chunks carry no
					// text, so the answer looks like it starts late rather than
					// arriving blank.
					if text := chunk.Text(); text != "" {
						return sendChunk(ctx, text)
					}
					return nil
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not explore the workspace: %w", err)
			}
			return text, nil
		})
}

// DefineEditFlow is write-enabled: AllowWriteAccess adds write_file and
// search_and_replace to the tool set.
//
// Edits land in workspace/ on disk, so a second run against an already edited
// workspace may report "search content not found".
func DefineEditFlow(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "editFlow",
		func(ctx context.Context, input EditRequest, sendChunk core.StreamCallback[string]) (string, error) {
			text, err := genkit.GenerateText(ctx, g,
				ai.WithModel(model),
				ai.WithSystem(
					"You are a careful project editor. Use the tools available to you to interact with the workspace. "+
						"Keep unrelated content unchanged.",
				),
				ai.WithPrompt("Apply the following change to the workspace and report what you did:\n\n%s", input.Instruction),
				ai.WithMaxTurns(20),
				ai.WithUse(&middleware.Filesystem{
					RootDir:          workspaceDir,
					AllowWriteAccess: true,
				}),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					if text := chunk.Text(); text != "" {
						return sendChunk(ctx, text)
					}
					return nil
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not edit the workspace: %w", err)
			}
			return text, nil
		})
}
