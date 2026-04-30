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

// This sample demonstrates the Skills middleware, which makes a local library
// of "skills" — specialised instructions stored as SKILL.md files — available
// to the model on demand.
//
// When the middleware is installed, it injects a system prompt listing every
// available skill (name + description) and registers a use_skill tool. The
// model inspects the list, decides which skill fits the user's request, and
// calls use_skill("<name>") to load that skill's full SKILL.md body into the
// conversation as further instructions.
//
// Four tiny, visually distinct skills ship with this sample so the effect is
// easy to verify by eyeballing the response:
//
//   - skills/haiku/SKILL.md — reply as a strict 5-7-5 haiku.
//   - skills/pirate/SKILL.md — reply in full pirate voice ("Arr, matey!").
//   - skills/shakespeare/SKILL.md — reply in early modern English.
//   - skills/eli5/SKILL.md — reply as if explaining to a five-year-old.
//
// SKILL.md files use an optional YAML frontmatter block for name and
// description; only the description surfaces in the listing, so models can
// keep the heavier persona instructions off the hot path until actually
// loaded.
//
// To run:
//
//	go run .
//
// In another terminal, trigger the pirate skill (default prompt):
//
//	curl -X POST http://localhost:8080/askFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{}'
//
// Trigger the haiku skill:
//
//	curl -X POST http://localhost:8080/askFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "Write a haiku about debugging code."}'
//
// Trigger the eli5 skill:
//
//	curl -X POST http://localhost:8080/askFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "Explain recursion to me like I am five."}'
//
// Trigger the shakespeare skill:
//
//	curl -X POST http://localhost:8080/askFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "Describe a rainy Tuesday morning, but in the voice of Shakespeare."}'
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

// skillsDir holds the SKILL.md library the model can browse and load.
// This path is relative to the working directory at startup; `go run .`
// resolves it against this sample directory.
const skillsDir = "./skills"

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin and the Middleware plugin.
	// Registering the Middleware plugin exposes the built-in middleware
	// (Skills, Filesystem, Retry, Fallback, ...) to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	DefineAskFlow(g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineAskFlow defines a single flow that asks the model a question and lets
// it load whichever skill best matches the request. The default question
// nudges the model toward the pirate skill so a fresh run produces obviously
// skill-flavoured output ("Arr, matey!" rather than a plain paragraph).
func DefineAskFlow(g *genkit.Genkit) {
	genkit.DefineFlow(g, "askFlow", func(ctx context.Context, question string) (string, error) {
		if question == "" {
			question = "Explain how a rainbow forms, but in the voice of a pirate."
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			})),
			ai.WithSystem(
				"You have access to a use_skill tool that loads a specialised "+
					"persona or style. Before answering, decide whether any listed "+
					"skill fits the user's request, and if so, call use_skill with "+
					"that name first. Then answer in the loaded style.",
			),
			ai.WithPrompt(question),
			// Loading a skill consumes one tool-loop turn; bump the cap
			// from the default of 5 so there's headroom for the skill
			// call plus the final response.
			ai.WithMaxTurns(8),
			ai.WithUse(&middleware.Skills{SkillPaths: []string{skillsDir}}),
		)
	})
}
