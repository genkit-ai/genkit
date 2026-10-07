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

package googlegenai_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"google.golang.org/genai"
)

// genaiSuite drives one backend of the Google GenAI plugin, the Gemini API or
// Vertex AI, through the shared checklist and then the checks both backends
// owe beyond it.
type genaiSuite struct {
	livetest.Suite
	// ref builds a model reference on the backend under test.
	ref func(id string, config *genai.GenerateContentConfig) ai.ModelRef
	// flash is the Gemini model the family cases spend on.
	flash string
	// imageModel, speechModel and videoModel serve the expensive media
	// output cases. An empty one skips its case.
	imageModel, speechModel, videoModel string
	// videoRef builds references to the Veo models.
	videoRef func(id string) ai.ModelRef
}

// runGenAI walks the backend registered on g through the shared checklist and
// then the Google GenAI one, under a "googlegenai" subtest.
func runGenAI(t *testing.T, g *genkit.Genkit, s genaiSuite) {
	t.Helper()
	flash := func(config *genai.GenerateContentConfig) ai.ModelRef { return s.ref(s.flash, config) }
	gen := func(t *testing.T, opts ...ai.GenerateOption) *ai.ModelResponse {
		t.Helper()
		resp, err := genkit.Generate(t.Context(), g, opts...)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		return resp
	}
	// needModel runs a case only when the backend names a model for it.
	needModel := func(id, kind string) func() string {
		return func() string {
			if id == "" {
				return "no " + kind + " model for this backend"
			}
			return ""
		}
	}

	livetest.Run(t, g, s.Suite, livetest.Group{Name: "googlegenai", Cases: []livetest.Case{
		{Name: "google search grounding", Run: func(t *testing.T) {
			resp := gen(t,
				ai.WithModel(flash(&genai.GenerateContentConfig{
					Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
				})),
				ai.WithPrompt("Search the web: who won the most recent Nobel Prize in Physics?"))
			if resp.Text() == "" {
				t.Error("Text() is empty")
			}
			if c := firstCandidate(resp); c == nil || c.GroundingMetadata == nil {
				t.Error("the response carries no grounding metadata, want the search it was grounded on")
			}
		}},

		// Code execution answers with executable code and its result as
		// custom parts, which have to go back to the API on the next turn.
		{Name: "code execution across turns", Run: func(t *testing.T) {
			model := flash(&genai.GenerateContentConfig{
				Tools: []*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}},
			})
			first := gen(t,
				ai.WithModel(model),
				ai.WithPrompt("Run code to compute the sum of the first 50 prime numbers. Reply with just the number."))
			if !googlegenai.HasCodeExecution(first.Message) {
				t.Fatalf("the response carries no code execution parts (text %q)", first.Text())
			}
			if !strings.Contains(first.Text(), "5117") {
				t.Errorf("Text() = %q, want it to contain 5117", first.Text())
			}
			second := gen(t,
				ai.WithModel(model),
				ai.WithMessages(first.History()...),
				ai.WithPrompt("Now run code to compute the sum of the first 10 prime numbers. Reply with just the number."))
			if !strings.Contains(second.Text(), "129") {
				t.Errorf("Text() = %q, want it to contain 129", second.Text())
			}
		}},

		{Name: "video url input", Run: func(t *testing.T) {
			resp := gen(t,
				ai.WithModel(flash(nil)),
				ai.WithMessages(ai.NewUserMessage(
					ai.NewMediaPart("video/mp4", "https://www.youtube.com/watch?v=_6FYhqGgel8"),
					ai.NewTextPart("Which video game is this video about?"))))
			if !strings.Contains(strings.ToLower(resp.Text()), "mario kart") {
				t.Errorf("Text() = %q, want it to name Mario Kart", resp.Text())
			}
		}},

		// The plugin sends a data part as inline bytes, as it does a media
		// part with a data URL.
		{Name: "data part input", Run: func(t *testing.T) {
			resp := gen(t,
				ai.WithModel(flash(nil)),
				ai.WithMessages(ai.NewUserMessage(
					ai.NewDataPart(livetest.RedImage),
					ai.NewTextPart("What is the dominant color of this image? Reply with one word."))))
			if !strings.Contains(strings.ToLower(resp.Text()), "red") {
				t.Errorf("Text() = %q, want it to name the color red", resp.Text())
			}
		}},

		{
			Name: "thinking tokens reported",
			Needs: func() string {
				if s.ReasoningModel == nil {
					return "Suite.ReasoningModel is not set"
				}
				return ""
			},
			Run: func(t *testing.T) {
				resp := gen(t,
					ai.WithModel(s.ReasoningModel),
					ai.WithPrompt("Is 91 a prime number? Answer yes or no."))
				if resp.Usage == nil || resp.Usage.ThoughtsTokens == 0 {
					t.Errorf("Usage = %+v, want the thinking tokens counted", resp.Usage)
				}
			},
		},

		{Name: "context caching", Run: func(t *testing.T) {
			livetest.Expensive(t)
			first := gen(t,
				ai.WithModel(flash(nil)),
				ai.WithMessages(ai.NewUserTextMessage(cacheableDocument()).WithCacheTTL(360)),
				ai.WithPrompt("Which animal does the document mention most often? Reply with one word."))
			cache, _ := first.Message.Metadata["cache"].(map[string]any)
			if name, _ := cache["name"].(string); name == "" {
				t.Fatalf("Message.Metadata = %v, want the name of the cache it created", first.Message.Metadata)
			}
			second := gen(t,
				ai.WithModel(flash(nil)),
				ai.WithMessages(first.History()...),
				ai.WithPrompt("Which color does the document mention most often? Reply with one word."))
			if second.Usage == nil || second.Usage.CachedContentTokens == 0 {
				t.Errorf("Usage = %+v, want the second turn served from the cache", second.Usage)
			}
		}},

		{Name: "image output", Needs: needModel(s.imageModel, "image"), Run: func(t *testing.T) {
			livetest.Expensive(t)
			resp := gen(t,
				ai.WithModel(s.ref(s.imageModel, &genai.GenerateContentConfig{
					ResponseModalities: []string{"IMAGE", "TEXT"},
				})),
				ai.WithPrompt("Draw a red circle on a white background."))
			wantMedia(t, resp, "image/")
		}},

		{Name: "speech output", Needs: needModel(s.speechModel, "speech"), Run: func(t *testing.T) {
			livetest.Expensive(t)
			resp := gen(t,
				ai.WithModel(s.ref(s.speechModel, nil)),
				ai.WithPrompt("Say: the quick brown fox jumps over the lazy dog."))
			wantMedia(t, resp, "audio/")
		}},

		{Name: "veo output", Needs: needModel(s.videoModel, "Veo"), Run: func(t *testing.T) {
			livetest.Expensive(t)
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
			defer cancel()
			op, err := genkit.GenerateOperation(ctx, g,
				ai.WithModel(s.videoRef(s.videoModel)),
				ai.WithPrompt("A red ball bouncing on a white floor."))
			if err != nil {
				t.Fatalf("GenerateOperation() error = %v", err)
			}
			for !op.Done {
				select {
				case <-ctx.Done():
					t.Fatalf("the operation did not finish: %v", ctx.Err())
				case <-time.After(10 * time.Second):
				}
				if op, err = genkit.CheckModelOperation(ctx, g, op); err != nil {
					t.Fatalf("CheckModelOperation() error = %v", err)
				}
			}
			if op.Error != nil {
				t.Fatalf("operation error = %v", op.Error)
			}
			wantMedia(t, op.Output, "video/")
		}},
	}})
}

// firstCandidate returns the first raw candidate the plugin keeps on the
// response, or nil.
func firstCandidate(resp *ai.ModelResponse) *genai.Candidate {
	custom, _ := resp.Custom.(map[string]any)
	candidates, _ := custom["candidates"].([]*genai.Candidate)
	if len(candidates) == 0 {
		return nil
	}
	return candidates[0]
}

// wantMedia fails t unless resp carries a media part whose content type
// starts with prefix.
func wantMedia(t *testing.T, resp *ai.ModelResponse, prefix string) {
	t.Helper()
	if resp == nil || resp.Message == nil {
		t.Fatal("the response carries no message")
	}
	for _, p := range resp.Message.Content {
		if p.IsMedia() && strings.HasPrefix(p.ContentType, prefix) && p.Text != "" {
			return
		}
	}
	t.Errorf("the response carries no %s* media part", prefix)
}

// cacheableDocument returns a document comfortably above the minimum size
// the API caches.
func cacheableDocument() string {
	var b strings.Builder
	animals := []string{"otter", "heron", "otter", "fox", "otter"}
	colors := []string{"teal", "amber", "teal", "slate", "teal"}
	for i := range 600 {
		fmt.Fprintf(&b, "Entry %d: the %s rested beside a %s stone near the river bend. ",
			i, animals[i%len(animals)], colors[i%len(colors)])
	}
	return b.String()
}
