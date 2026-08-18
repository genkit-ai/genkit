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

// This sample is basic-tools, written against the in-preview tools API in
// genkit/exp. It is slated to replace the one basic-tools uses in the next
// major version, so the two are worth reading side by side: same rollout, same
// flow, same chart, and the differences are all API.
//
// The tool function takes a plain context.Context rather than *[ai.ToolContext],
// and the runtime helpers in ai/exp/tool do the rest:
//
//   - tool.AttachParts attaches the chart, so the tool returns a plain *Rollout
//     instead of an *[ai.MultipartToolResponse]. The signature stops having to
//     announce that the tool sometimes has more to say.
//   - tool.SendPartial streams structured progress while the rollout runs, so a
//     slow tool does not look like a hang.
//   - tool.SendChunk streams a chunk the tool builds itself, for an update that
//     is a line of prose rather than a value.
//
// The two streaming helpers are best-effort: with a caller that is not
// streaming they are no-ops, so the tool still works when nobody is listening,
// and the returned *Rollout is always the authoritative answer. Neither is
// written to history, since progress is for showing, not for the model to read.
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
// Or over HTTP. Streaming needs ?stream=true, otherwise the progress goes
// nowhere and only the final report comes back:
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
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
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

	// Rollout is the tool's answer, and the whole of its return type: the
	// chart it attaches does not appear here.
	Rollout struct {
		Service  string  `json:"service"`
		Revision string  `json:"revision"`
		Healthy  bool    `json:"healthy"`
		P95Ms    float64 `json:"p95Ms" jsonschema_description:"The p95 latency after the rollout, in milliseconds"`
	}

	// Progress is what tool.SendPartial sends. It is the tool's own shape, not
	// one the API dictates: any value that survives JSON works.
	Progress struct {
		Step    string `json:"step"`
		Percent int    `json:"percent"`
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

// stageDuration stands in for work that really takes time. It is what makes
// the streamed progress worth watching.
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
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithExperimental(), // This sample uses the new in-preview tools API.
	)

	// The name and description are all the model knows about a tool, beside
	// the schemas inferred from the types. They are prompt, so they are worth
	// writing as carefully as one.
	deployService := genkitx.DefineTool(g, "deployService",
		"Deploys a service to an environment and reports how the rollout went.",
		func(ctx context.Context, input Deploy) (*Rollout, error) {
			latencies := make([]float64, 0, len(rolloutStages))
			for i, stage := range rolloutStages {
				// Sent before the work, so the client sees the step it is
				// waiting on rather than the one already done.
				tool.SendPartial(ctx, Progress{
					Step:    stage.Name,
					Percent: (i + 1) * 100 / len(rolloutStages),
				})
				time.Sleep(stageDuration)
				latencies = append(latencies, stage.P95)
			}

			revision := fmt.Sprintf("%s-00042", input.Service)

			// An update with no structure worth giving it. RoleTool marks the
			// chunk as the tool's, which is how the flow below tells it from
			// the model's own text.
			tool.SendChunk(ctx, &ai.ModelResponseChunk{
				Role:    ai.RoleTool,
				Content: []*ai.Part{ai.NewTextPart(fmt.Sprintf("%s is live in %s", revision, input.Environment))},
			})

			// The model receives this as a picture, so it can describe the
			// shape of the rollout rather than only its last number.
			tool.AttachParts(ctx, ai.NewMediaPart("image/png", barChartPNG(latencies)))

			return &Rollout{
				Service:  input.Service,
				Revision: revision,
				Healthy:  true,
				P95Ms:    latencies[len(latencies)-1],
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
				// tool's traffic as well as the model's. The tool's own text
				// carries RoleTool like every other part of its message, so
				// the role is what separates the two text cases below.
				for _, part := range val.Chunk.Content {
					switch {
					case part.IsPartial():
						// From tool.SendPartial. In process the value arrives
						// as the one the tool sent; a client reading the HTTP
						// stream gets its JSON instead.
						if p, ok := part.ToolResponse.Output.(Progress); ok {
							sendChunk(ctx, fmt.Sprintf("[%3d%%] %s", p.Percent, p.Step))
						}
					case part.IsText() && val.Chunk.Role == ai.RoleTool:
						sendChunk(ctx, "deploy: "+part.Text) // From tool.SendChunk.
					case part.IsText():
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
