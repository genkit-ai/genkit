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
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
)

// The live checks run jev through OpenRouter, the gateway an ordinary key
// can reach today; TypeSafe's own API is behind a waitlist.
func TestOpenRouterLive(t *testing.T) {
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY is not set")
	}
	g := genkit.Init(t.Context(), genkit.WithPlugins(OpenRouter()))
	const model = "openrouter-decisions/~typesafe/jev-latest"

	t.Run("decision", func(t *testing.T) {
		// The documented call: the model, the state, and the type.
		out, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket":       "I was charged twice for one order and I need the duplicate refunded today.",
				"account_tier": "business",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(out))
		t.Logf("info: %s, usage: %s", base.JSONString(ResponseInfo(resp)), base.JSONString(resp.Usage))

		if _, ok := out.Department.Choice.Criteria()[out.Department.Choice]; !ok {
			t.Errorf("choice %q is outside the criteria", out.Department.Choice)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing for a double charge", out.Department.Choice)
		}
		var mass float64
		for _, p := range out.Department.Probabilities {
			mass += p
		}
		if math.Abs(mass-1) > 0.05 {
			t.Errorf("probabilities sum to %v, want about 1: %v", mass, out.Department.Probabilities)
		}
		if out.Department.Confidence <= 0 || out.Department.Confidence > 1 {
			t.Errorf("confidence = %v, want within (0, 1]", out.Department.Confidence)
		}
		if out.IsUrgent.Probability < 0.5 {
			t.Errorf("is_urgent = %v, want over 0.5 for \"today\"", out.IsUrgent.Probability)
		}
		if out.Frustration.Score < 0 || out.Frustration.Score > 2 || len(out.Frustration.Legend) != 3 {
			t.Errorf("frustration = %+v", out.Frustration)
		}
		if resp.Usage == nil || resp.Usage.InputTokens == 0 {
			t.Fatalf("usage = %+v", resp.Usage)
		}
		if ResponseInfo(resp).Model == "" {
			t.Errorf("no resolved model version on the response: %v", resp.Raw)
		}
		if resp.Usage.Custom["cost"] <= 0 {
			t.Errorf("usage custom = %v, want the gateway's cost", resp.Usage.Custom)
		}
	})

	t.Run("decision with preamble", func(t *testing.T) {
		out, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("The state is a support ticket from a business customer of an online store."),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket": "I was charged twice for one order and I need the duplicate refunded today.",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers with preamble: %s", base.JSONString(out))
		if out.Department.Choice != "billing" || out.IsUrgent.Probability < 0.5 {
			t.Errorf("with a preamble: department = %q, is_urgent = %v", out.Department.Choice, out.IsUrgent.Probability)
		}
	})

	t.Run("guidance", func(t *testing.T) {
		// Object criteria on an option, a level, and a side: the gateway
		// takes them, and the legend keeps the rubric's strings.
		out, resp, err := genkit.GenerateData[guidedTriage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket": "THIS IS THE THIRD TIME. I was charged twice and I want my money back TODAY.",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(out))
		t.Logf("info: %s, usage: %s", base.JSONString(ResponseInfo(resp)), base.JSONString(resp.Usage))
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing", out.Department.Choice)
		}
		if out.Frustration.Legend["2"] != "Very angry" {
			t.Errorf("legend = %v, want the rubric strings", out.Frustration.Legend)
		}
	})

	t.Run("enum", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("Which team should handle this ticket?"),
			ai.WithOutputEnums("billing", "technical", "sales"),
			ai.WithPrompt("The API returns 500 errors since this morning's deploy."))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("enum: %q info: %s", resp.Text(), base.JSONString(ResponseInfo(resp)))
		if resp.Text() != "technical" {
			t.Errorf("text = %q, want technical", resp.Text())
		}
	})

	t.Run("history as state", func(t *testing.T) {
		type handoff struct {
			WantsHuman Noul `json:"wants_human" jsonschema_description:"Does the user ask to talk to a human?"`
		}
		out, _, err := genkit.GenerateData[handoff](t.Context(), g,
			ai.WithModelName(model),
			ai.WithMessages(
				ai.NewUserMessage(ai.NewTextPart("Hi, I cannot log in.")),
				ai.NewModelMessage(ai.NewTextPart("Let me help. Have you tried resetting your password?")),
				ai.NewUserMessage(ai.NewTextPart("Yes, three times. Can I please just talk to a person?")),
			))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("wants_human: %v", out.WantsHuman.Probability)
		if out.WantsHuman.Probability < 0.5 {
			t.Errorf("wants_human = %v, want over 0.5", out.WantsHuman.Probability)
		}
	})

	t.Run("pinned minor version", func(t *testing.T) {
		_, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("openrouter-decisions/typesafe/jev-1.13"),
			ai.WithPrompt("I was charged twice for one order."))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("jev-1.13 resolved to %v", ResponseInfo(resp).Model)
	})

	t.Run("preview alias", func(t *testing.T) {
		// The alias is forwarded as it is, so the gateway says whether it
		// has a preview channel.
		_, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("openrouter-decisions/~typesafe/jev-preview"),
			ai.WithPrompt("I was charged twice for one order."))
		if err != nil {
			if !errors.Is(err, status.ErrInvalidArgument) || !strings.Contains(err.Error(), "jev-preview") {
				t.Errorf("error = %v, want the gateway's refusal of jev-preview", err)
			}
			t.Logf("no preview channel: %v", err)
			return
		}
		t.Logf("jev-preview resolved to %v", ResponseInfo(resp).Model)
	})

	t.Run("another vendor's model", func(t *testing.T) {
		// The same decision type, through the same plugin, to Liquid AI's d1.
		out, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("openrouter-decisions/liquid/d1"),
			ai.WithPrompt("I was charged twice for one order and I need the duplicate refunded today."))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s, info: %s", base.JSONString(out), base.JSONString(ResponseInfo(resp)))
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing for a double charge", out.Department.Choice)
		}
	})

	t.Run("listing", func(t *testing.T) {
		var names []string
		for _, desc := range genkit.LookupPlugin(g, "openrouter-decisions").(*SystemOne).ListActions(t.Context()) {
			names = append(names, desc.Name)
		}
		t.Logf("listed: %v", names)
		if !slices.Contains(names, "openrouter-decisions/liquid/d1") {
			t.Errorf("listed %v, want liquid/d1 among the decision models", names)
		}
	})

	t.Run("runtime questions", func(t *testing.T) {
		// Options from data, in the order given, with guidance on one;
		// structured instructions beside a system message.
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("The state is a request a user made to an assistant."),
			ai.WithOutputSchema(Schema(map[string]Question{
				"tool": ChoiceQuestion{
					Instructions: "Which tool serves the request?",
					Options: []ChoiceOption{
						{Name: "web_search", Criteria: "Look up facts, news, or prices on the web"},
						{Name: "calendar", Criteria: "Read or change the user's own calendar", Guidance: map[string]any{
							"examples": []string{"What meetings do I have on Friday?", "Move my 3pm to 4pm."},
						}},
						{Name: "none", Criteria: "No tool fits the request"},
					},
				},
				"effort": ScoreQuestion{
					Instructions: map[string]any{
						"question": "How much work does fulfilling the request take?",
						"field":    map[string]any{"name": "request", "description": "What the user asked for"},
					},
					Levels: []string{"A single lookup", "A few steps", "A multi-step project"},
				},
				"personal": NoulQuestion{
					Instructions: "Does the request involve the user's own data?",
					Yes:          "Names the user's files, mail, or calendar",
					No:           "Asks about the world at large",
				},
			})),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{"request": "What is on my calendar tomorrow morning?"})))
		if err != nil {
			t.Fatal(err)
		}
		var answers map[string]Answer
		if err := resp.Output(&answers); err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(answers))
		if a := answers["tool"]; a.Choice != "calendar" {
			t.Errorf("tool = %+v, want calendar", a)
		}
		if a := answers["personal"]; a.Probability < 0.5 {
			t.Errorf("personal = %+v, want over 0.5", a)
		}
		if a := answers["effort"]; a.Score < 0 || a.Score > 2 || a.Legend["0"] != "A single lookup" {
			t.Errorf("effort = %+v", a)
		}
	})

	t.Run("stateJSON with a large number", func(t *testing.T) {
		out, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithConfig(&Config{StateJSON: true}),
			ai.WithPrompt(`{"order_id": 9007199254740993, "ticket": "I was charged twice for this order."}`))
		if err != nil {
			t.Fatal(err)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing", out.Department.Choice)
		}
	})

	t.Run("document with data", func(t *testing.T) {
		type stock struct {
			InStock Noul `json:"in_stock" jsonschema_description:"Does the context show the item the user asks about as in stock?"`
		}
		out, _, err := genkit.GenerateData[stock](t.Context(), g,
			ai.WithModelName(model),
			ai.WithPrompt("Is the blue kettle available?"),
			ai.WithDocs(&ai.Document{Content: []*ai.Part{ai.NewDataPart(map[string]any{"item": "blue kettle", "stock": 12})}}))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("in_stock: %v", out.InStock.Probability)
		if out.InStock.Probability < 0.5 {
			t.Errorf("in_stock = %v, want over 0.5: the data document did not reach the state", out.InStock.Probability)
		}
	})
}

// The Workers AI checks run Cloudflare's own models on the account the
// token belongs to, with media each model can answer about only by
// reading it.
func TestWorkersAILive(t *testing.T) {
	if os.Getenv("CLOUDFLARE_API_TOKEN") == "" || os.Getenv("CLOUDFLARE_ACCOUNT_ID") == "" {
		t.Skip("CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID is not set")
	}
	g := genkit.Init(t.Context(), genkit.WithPlugins(WorkersAI("")))
	const omni = "cloudflare-decisions/@cf/cloudflare/clef-omni"

	t.Run("decision", func(t *testing.T) {
		out, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("cloudflare-decisions/@cf/cloudflare/clef-flash"),
			ai.WithPrompt("I was charged twice for one order and I need the duplicate refunded today."))
		if err != nil {
			t.Fatal(err)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing for a double charge", out.Department.Choice)
		}
		if info := ResponseInfo(resp); info.Model != "clef-flash" || resp.Usage.InputTokens == 0 {
			t.Errorf("model = %q, usage = %+v", info.Model, resp.Usage)
		}
	})

	t.Run("images", func(t *testing.T) {
		for _, model := range []string{"@cf/cloudflare/clef", "@cf/cloudflare/clef-flash", "@cf/cloudflare/clef-omni"} {
			for _, tt := range []struct {
				name  string
				color color.RGBA
				red   bool
			}{{"red", color.RGBA{R: 255, A: 255}, true}, {"blue", color.RGBA{B: 255, A: 255}, false}} {
				out, _, err := genkit.GenerateData[redCheck](t.Context(), g,
					ai.WithModelName("cloudflare-decisions/"+model),
					ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Photo from the customer."), ai.NewMediaPart("image/png", solidPNG(t, tt.color)))))
				if err != nil {
					t.Fatalf("%s: %v", model, err)
				}
				if got := out.Red.Probability > 0.5; got != tt.red {
					t.Errorf("%s on a %s square: red = %v", model, tt.name, out.Red.Probability)
				}
			}
		}
	})

	t.Run("audio", func(t *testing.T) {
		for _, tt := range []struct {
			name string
			hz   float64
			tone bool
		}{{"tone", 880, true}, {"silence", 0, false}} {
			out, _, err := genkit.GenerateData[toneCheck](t.Context(), g,
				ai.WithModelName(omni),
				ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("A recording from the device."), ai.NewMediaPart("audio/wav", sineWAV(tt.hz)))))
			if err != nil {
				t.Fatal(err)
			}
			if got := out.Tone.Probability > 0.5; got != tt.tone {
				t.Errorf("%s: tone = %v", tt.name, out.Tone.Probability)
			}
		}
	})

	t.Run("video", func(t *testing.T) {
		clip, err := os.ReadFile("testdata/red.mp4")
		if err != nil {
			t.Fatal(err)
		}
		out, _, err := genkit.GenerateData[redCheck](t.Context(), g,
			ai.WithModelName(omni),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("A clip from the camera."), ai.NewMediaPart("video/mp4", "data:video/mp4;base64,"+base64.StdEncoding.EncodeToString(clip)))))
		if err != nil {
			t.Fatal(err)
		}
		if out.Red.Probability <= 0.5 {
			t.Errorf("red = %v on a red clip", out.Red.Probability)
		}
	})
}

// redCheck asks whether the media is red, which a model can answer only
// by reading the media.
type redCheck struct {
	Red Noul `json:"red" jsonschema_description:"Is the attached image or video mostly the color red?"`
}

// toneCheck asks whether a recording holds a tone.
type toneCheck struct {
	Tone Noul `json:"tone" jsonschema_description:"Does the audio clip contain an audible tone or beep?"`
}

// solidPNG is a data URL of a small PNG of one color.
func solidPNG(t *testing.T, c color.RGBA) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = c.R, c.G, c.B, c.A
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// sineWAV is a data URL of two seconds of 16 kHz mono WAV: a sine at hz,
// or silence at 0.
func sineWAV(hz float64) string {
	const rate, seconds = 16000, 2
	samples := make([]int16, rate*seconds)
	for i := range samples {
		samples[i] = int16(12000 * math.Sin(2*math.Pi*hz*float64(i)/rate))
	}
	size := uint32(len(samples) * 2)
	header := struct {
		RIFF       [4]byte
		Size       uint32
		WAVEfmt    [8]byte
		FmtSize    uint32
		Format     uint16
		Channels   uint16
		Rate       uint32
		ByteRate   uint32
		BlockAlign uint16
		Bits       uint16
		Data       [4]byte
		DataSize   uint32
	}{[4]byte([]byte("RIFF")), 36 + size, [8]byte([]byte("WAVEfmt ")), 16, 1, 1, rate, rate * 2, 2, 16, [4]byte([]byte("data")), size}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, header)
	_ = binary.Write(&buf, binary.LittleEndian, samples)
	return "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}
