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
// scoped file access via list_files, read_file, write_file, and
// search_and_replace tools. All operations are confined to the configured
// RootDir — os.Root (Go 1.25+) rejects any path that resolves outside it,
// including via "..", absolute paths, or symbolic links.
//
// A ready-to-demo workspace/ directory ships alongside this sample with a
// mock project (README, docs, config, data, TODO list) so the tools have
// something interesting to read and edit.
//
// The sample defines two flows:
//
//   - exploreFlow (read-only): the model lists and reads files under
//     workspace/ to answer a question about the project.
//   - editFlow (write-enabled): the model reads a file and applies a
//     SEARCH/REPLACE edit to satisfy a change request.
//
// To run:
//
//	go run .
//
// In another terminal, ask a question about the workspace:
//
//	curl -N -X POST http://localhost:8080/exploreFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "Summarise what this project does and what is still pending."}'
//
// Apply an edit (writes are visible in workspace/todo.txt afterwards):
//
//	curl -N -X POST http://localhost:8080/editFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "Mark the in-memory response cache TODO as done."}'
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// workspaceDir is the sandbox the model is allowed to see. All filesystem
// tool calls are rooted here; anything outside is unreachable by construction.
const workspaceDir = "./workspace"

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin and the Middleware plugin.
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

// DefineExploreFlow defines a read-only flow: the model answers questions
// about the workspace by listing and reading files. AllowWriteAccess is not
// set, so write_file and search_and_replace are not registered — the model
// literally cannot modify anything.
func DefineExploreFlow(g *genkit.Genkit) {
	genkit.DefineFlow(g, "exploreFlow", func(ctx context.Context, question string) (string, error) {
		if question == "" {
			question = "Summarise what this project does based on the files available."
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			})),
			ai.WithSystem("You are a helpful project analyst. Use the filesystem tools to explore the workspace before answering."),
			ai.WithPrompt(question),
			ai.WithMaxTurns(20),
			ai.WithUse(&middleware.Filesystem{RootDir: workspaceDir}),
		)
	})
}

// DefineEditFlow defines a write-enabled flow: the model reads a file,
// identifies the right spot, and applies a SEARCH/REPLACE edit. Enabling
// AllowWriteAccess adds write_file and search_and_replace to the tool set.
//
// Changes are written to workspace/ on disk. Re-running this sample against
// a fresh checkout will re-apply the edit; re-running against an already
// edited workspace may report "search content not found" if the instruction
// has already been satisfied.
func DefineEditFlow(g *genkit.Genkit) {
	genkit.DefineFlow(g, "editFlow", func(ctx context.Context, instruction string) (string, error) {
		if instruction == "" {
			instruction = "In todo.txt, mark the in-memory response cache item as done."
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			})),
			ai.WithSystem(
				"You are a careful project editor. Use the tools available to you to interact with the workspace. "+
					"Keep unrelated content unchanged.",
			),
			ai.WithPrompt("Apply the following change to the workspace and report what you did:\n\n%s", instruction),
			ai.WithMaxTurns(20),
			ai.WithUse(&middleware.Filesystem{
				RootDir:          workspaceDir,
				AllowWriteAccess: true,
			}),
		)
	})
}
