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

// This sample demonstrates tools: Go functions the model may call mid-answer,
// and what a tool hands back when its answer is more than one value.
//
// deployService simulates a rollout that takes a while, then reports how it
// went. It is a multipart tool, so it answers with two things: a *Rollout as
// its output, and a chart of the latency it recorded as an attached content
// part. Attachments reach the model and the client both, so the model can
// describe the shape of the rollout and the Dev UI can show the picture. They
// must be media or data parts; a text part is not a valid attachment.
//
// deployFlow streams the run, so the report arrives as the model writes it.
//
// basic-tools-exp is this same sample written against the in-preview tools API
// in genkit/exp. Reading the two side by side is the shortest way to see what
// that API changes and what it adds.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, which renders the attached chart and keeps a trace of the
// whole rollout at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Streaming needs ?stream=true, otherwise only the final report
// comes back:
//
//	curl -N -X POST 'http://localhost:8080/deployFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": "Ship checkout-api to production."}}'
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"net/http"
	"slices"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

type (
	// Deploy is what the model fills in to call the tool. The schema comes
	// from this type, so a field's jsonschema tag is how the model learns
	// what may go in it.
	Deploy struct {
		Service     string `json:"service" jsonschema_description:"The service to deploy"`
		Environment string `json:"environment" jsonschema:"enum=staging,enum=production" jsonschema_description:"Where to deploy it"`
	}

	// Rollout is the tool's answer, and the Output half of its multipart
	// response.
	Rollout struct {
		Service  string  `json:"service"`
		Revision string  `json:"revision"`
		Healthy  bool    `json:"healthy"`
		P95Ms    float64 `json:"p95Ms" jsonschema_description:"The p95 latency after the rollout, in milliseconds"`
	}

	// DeployRequest is what the flow takes.
	DeployRequest struct {
		Request string `json:"request" jsonschema:"default=Ship checkout-api to production." jsonschema_description:"What to deploy and where"`
	}
)

// rolloutStages are the steps the tool walks, with the p95 latency it records
// at each one. Latency falls as the new revision takes over, which is what the
// attached chart ends up showing. Fixed numbers keep the sample repeatable.
var rolloutStages = []struct {
	Name string
	P95  float64
}{
	{"building image", 154},
	{"pushing image", 151},
	{"shifting 10% of traffic", 148},
	{"shifting 50% of traffic", 121},
	{"shifting 100% of traffic", 96},
	{"checking health", 92},
}

// stageDuration stands in for work that really takes time.
const stageDuration = 250 * time.Millisecond

// model is shared by every flow below, so switching models or thinking levels
// for the whole sample is a one-line change.
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

	// The name and description are all the model knows about a tool, beside
	// the schemas inferred from the types. They are prompt, so they are worth
	// writing as carefully as one.
	//
	// A multipart tool differs from a plain one only in what it returns: the
	// value a plain tool would have answered with goes in Output, and whatever
	// is not a value goes in Content.
	deployService := genkit.DefineMultipartTool(g, "deployService",
		"Deploys a service to an environment and reports how the rollout went.",
		func(ctx *ai.ToolContext, input Deploy) (*ai.MultipartToolResponse, error) {
			latencies := make([]float64, 0, len(rolloutStages))
			for _, stage := range rolloutStages {
				time.Sleep(stageDuration)
				latencies = append(latencies, stage.P95)
			}

			revision := fmt.Sprintf("%s-00042", input.Service)
			return &ai.MultipartToolResponse{
				Output: &Rollout{
					Service:  input.Service,
					Revision: revision,
					Healthy:  true,
					P95Ms:    latencies[len(latencies)-1],
				},
				// The model receives this as a picture, so it can describe the
				// shape of the rollout rather than only its last number.
				Content: []*ai.Part{ai.NewMediaPart("image/png", barChartPNG(latencies))},
			}, nil
		})

	genkit.DefineStreamingFlow(g, "deployFlow",
		func(ctx context.Context, input DeployRequest, sendChunk core.StreamCallback[string]) (string, error) {
			for val, err := range genkit.GenerateStream(ctx, g,
				ai.WithModel(model),
				ai.WithSystem("You are a release assistant. Deploy what the user asks for, then report the outcome and what the latency chart shows, in two sentences."),
				ai.WithPrompt(input.Request),
				ai.WithTools(deployService),
			) {
				if err != nil {
					return "", fmt.Errorf("could not deploy: %w", err)
				}
				if val.Done {
					return val.Response.Text(), nil
				}
				// A tool call is several turns, so the stream carries the
				// tool's traffic as well as the model's. Only the text is
				// worth forwarding: the tool's answer goes to the model
				// rather than to the caller.
				for _, part := range val.Chunk.Content {
					if part.IsText() {
						sendChunk(ctx, part.Text) // The model writing its report.
					}
				}
			}
			return "", status.Errorf(status.ErrInternal, "the stream ended without a final result")
		})

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// barChartPNG draws the values as a bar chart and returns it as a data: URI.
// The drawing is incidental: a tool's attachments are ordinary media parts,
// whatever produced them.
func barChartPNG(values []float64) string {
	const width, height, pad, floor = 224, 96, 8, 8
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)

	low, high := slices.Min(values), slices.Max(values)
	bar := image.NewUniform(color.RGBA{R: 0x1a, G: 0x73, B: 0xe8, A: 0xff})
	slot := (width - 2*pad) / len(values)
	for i, v := range values {
		tall := floor + int((v-low)/max(high-low, 1)*float64(height-2*pad-floor))
		left := pad + i*slot
		draw.Draw(img, image.Rect(left, height-pad-tall, left+slot-3, height-pad), bar, image.Point{}, draw.Src)
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return ""
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}
