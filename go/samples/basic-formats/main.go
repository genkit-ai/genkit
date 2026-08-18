// Copyright 2025 Google LLC
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

// This sample demonstrates output formats: the setting that decides how a
// model's text is parsed back into Go values, and what a streamed chunk means
// along the way.
//
//   - characterFlow (json) streams a growing value: every chunk is the whole
//     object so far, with more fields filled in than the one before it.
//   - castFlow (jsonl) streams a list an item at a time: a finished item is
//     handed over once, not restated on every chunk.
//   - ratingFlow (enum) constrains the answer to one label, so there is
//     nothing to stream.
//
// The rest are "text" (the default, no parsing), "array" (like jsonl, but one
// JSON array on the wire instead of one object per line), and "media".
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to call the flows from a browser and read a trace of
// every run at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Streaming needs ?stream=true, otherwise only the final result
// comes back:
//
//	curl -N -X POST 'http://localhost:8080/castFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"premise": "a lighthouse keeper who befriends a whale"}}'
//
//	curl -X POST http://localhost:8080/ratingFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"premise": "a lighthouse keeper who befriends a whale"}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// StoryRequest is the input every flow here takes, so the format stays the
// only difference between them.
type StoryRequest struct {
	Premise string `json:"premise" jsonschema:"default=a lighthouse keeper who befriends a whale" jsonschema_description:"The premise of the story"`
}

// Character is one invented character. The jsonschema tags describe the fields
// to the model, which is what makes the generated values sensible.
type Character struct {
	Name       string `json:"name" jsonschema_description:"The character's name"`
	Age        int    `json:"age" jsonschema_description:"The character's age in years"`
	Hometown   string `json:"hometown" jsonschema_description:"Where the character grew up"`
	Profession string `json:"profession" jsonschema_description:"What the character does for a living"`
}

// Rating is the set of labels the rating flow may answer with. Any string
// type works; the values become the schema.
type Rating string

const (
	RatingAllAges    Rating = "all-ages"
	RatingYoungAdult Rating = "young-adult"
	RatingAdult      Rating = "adult"
)

// model is shared by every flow below so the format is the only thing that
// varies between them.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// The Google AI plugin reads the API key from GEMINI_API_KEY or
	// GOOGLE_API_KEY, which is the recommended practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	DefineCharacterWithJSON(g)
	DefineCastWithJSONL(g)
	DefineRatingWithEnum(g)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineCharacterWithJSON demonstrates the json format, which every typed
// generation gets by default.
func DefineCharacterWithJSON(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "characterFlow",
		// Asking for a value rather than a pointer means every chunk is usable
		// as it lands: a half-filled Character reads the same as one whose
		// fields have not arrived yet, so there is nothing to nil-check.
		func(ctx context.Context, input StoryRequest, sendChunk core.StreamCallback[Character]) (Character, error) {
			for val, err := range genkit.GenerateDataStream[Character](ctx, g,
				ai.WithModel(model),
				ai.WithSystem("You are a children's book author. Invent memorable, gentle characters."),
				ai.WithPrompt("Invent one character for a story about %s.", input.Premise),
			) {
				if err != nil {
					return Character{}, fmt.Errorf("could not invent a character: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				// Each chunk is the whole character so far, which is what makes
				// a UI fill in field by field.
				sendChunk(ctx, val.Chunk)
			}
			return Character{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		},
	)
}

// DefineCastWithJSONL demonstrates the jsonl format, the one to reach for when
// the output is a list. The Out type has to be a slice: the format parses one
// item per line, so it needs an array schema to work from.
func DefineCastWithJSONL(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "castFlow",
		func(ctx context.Context, input StoryRequest, sendChunk core.StreamCallback[Character]) ([]Character, error) {
			for val, err := range genkit.GenerateDataStream[[]Character](ctx, g,
				ai.WithModel(model),
				// The schema comes from the type parameter either way; this
				// changes only how the model is asked to write it out.
				ai.WithOutputFormat(ai.OutputFormatJSONL),
				ai.WithSystem("You are a children's book author. Invent memorable, gentle characters."),
				ai.WithPrompt("Invent four characters for a story about %s.", input.Premise),
			) {
				if err != nil {
					return nil, fmt.Errorf("could not invent a cast: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				// Characters land one at a time, so a chunk is work the caller
				// has not been given yet. The exception is the character still
				// being written: it arrives again next chunk, further along, so
				// a consumer wanting only finished ones must spot the repeat.
				for _, character := range val.Chunk {
					sendChunk(ctx, character)
				}
			}
			return nil, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		},
	)
}

// DefineRatingWithEnum demonstrates the enum format. WithOutputEnums sets the
// schema and the format together, so the model answers with the bare label
// rather than with JSON wrapping it.
func DefineRatingWithEnum(g *genkit.Genkit) {
	genkit.DefineFlow(g, "ratingFlow",
		func(ctx context.Context, input StoryRequest) (Rating, error) {
			rating, _, err := genkit.GenerateData[Rating](ctx, g,
				ai.WithModel(model),
				ai.WithOutputEnums(RatingAllAges, RatingYoungAdult, RatingAdult),
				ai.WithPrompt("Which audience is a story about %s written for?", input.Premise),
			)
			if err != nil {
				return "", fmt.Errorf("could not rate the story: %w", err)
			}
			// GenerateData answers with a nil value and no error when the
			// response carried no text to parse, as a blocked or truncated one
			// does, so a typed result is worth checking before dereferencing.
			if rating == nil {
				return "", status.Errorf(status.ErrInternal, "the model answered with no rating")
			}
			return *rating, nil
		},
	)
}
