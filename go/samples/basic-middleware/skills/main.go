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

// This sample demonstrates the Skills middleware, which offers the model a
// local library of specialised instructions stored as SKILL.md files.
//
// The middleware injects a system prompt listing each skill's name and
// description, and registers a use_skill tool. The model picks the skill that
// fits and calls use_skill to load its full body into the conversation, so the
// heavier instructions stay off the hot path until they are wanted.
//
// Four visually distinct skills ship here, so the effect is easy to eyeball:
// haiku, pirate, shakespeare, and eli5.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to watch the use_skill call in a trace of every run at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. The question decides which skill loads:
//
//	curl -N -X POST 'http://localhost:8080/askFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"question": "Write a haiku about debugging code."}}'
//
//	curl -N -X POST 'http://localhost:8080/askFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"question": "Explain recursion to me like I am five."}}'
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

// skillsDir holds the SKILL.md library the model can browse and load. The path
// is relative to the working directory, which `go run .` puts here.
const skillsDir = "./skills"

// AskRequest is what the flow takes. A jsonschema tag is comma-delimited, so
// the default holds no comma: one would silently truncate the value.
type AskRequest struct {
	Question string `json:"question" jsonschema:"default=Explain how a rainbow forms in the voice of a pirate." jsonschema_description:"What to ask"`
}

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
	// (Skills, Filesystem, Retry, Fallback, ...) to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	DefineAskFlow(g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineAskFlow asks the model a question and lets it load whichever skill
// matches. The default question steers it to the pirate skill, so a fresh run
// answers "Arr, matey!" rather than with a plain paragraph.
func DefineAskFlow(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "askFlow",
		func(ctx context.Context, input AskRequest, sendChunk core.StreamCallback[string]) (string, error) {
			text, err := genkit.GenerateText(ctx, g,
				ai.WithModel(model),
				ai.WithSystem(
					"You have access to a use_skill tool that loads a specialised "+
						"persona or style. Before answering, decide whether any listed "+
						"skill fits the user's request, and if so, call use_skill with "+
						"that name first. Then answer in the loaded style.",
				),
				ai.WithPrompt(input.Question),
				// Loading a skill costs one tool-loop turn, so raise the cap from
				// the default of 5 to leave room for the answer.
				ai.WithMaxTurns(8),
				ai.WithUse(&middleware.Skills{SkillPaths: []string{skillsDir}}),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					// The use_skill turn streams too, and carries no text, so the
					// answer looks like it starts late rather than arriving blank.
					if text := chunk.Text(); text != "" {
						return sendChunk(ctx, text)
					}
					return nil
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not answer the question: %w", err)
			}
			return text, nil
		})
}
