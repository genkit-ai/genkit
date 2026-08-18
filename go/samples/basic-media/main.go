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

// This sample demonstrates media: reading a picture, making one, and making a
// video from one.
//
//   - describeFlow sends a picture and gets words back.
//   - editFlow sends a picture and an instruction and gets a square one back.
//   - generateFlow sends only words and gets a picture back.
//   - animateFlow sends a picture and gets a video back, a while later.
//
// The first three call an image-capable text model, where only the request and
// what the response may contain change between them. animateFlow calls a
// background model instead: video generation is too slow to answer in one
// request, so it starts an operation and polls until it finishes, streaming a
// chunk on every check.
//
// A picture attaches either by reference to a file in the Files API, which pays
// off for anything large or reused, or inline as a data: URI. Both appear here.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, which renders the generated images and keeps a trace of
// every run at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Generated images come back as base64 data: URIs, so pipe
// through jq to keep a terminal readable:
//
//	curl -X POST http://localhost:8080/describeFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"question": "What time of day is this and how can you tell?"}}'
//
//	curl -X POST http://localhost:8080/generateFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"description": "a red panda reading a book in a library"}}' | jq '.result.images | length'
//
// Watch the video operation report in while it runs. The video comes back
// inline, so ask jq for the path it was also written to and open that instead:
//
//	curl -N -X POST 'http://localhost:8080/animateFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"motion": "a slow drone push toward the ridge as clouds drift past"}}' \
//	  | grep '"result"' | sed 's/^data: //' | jq -r '.result.path'
package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// The picture the reading flows work from. A real app would take the bytes
// from a request.
const (
	imagePath = "mountain.jpg"
	imageType = "image/jpeg"
	videoPath = "mountain.mp4"
	videoType = "video/mp4"
)

// model reads pictures and answers with words.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

// imageModel is the same model allowed to answer with a picture. Without IMAGE
// in ResponseModalities it describes what it would draw instead of drawing it.
var imageModel = googlegenai.ModelRef("googleai/gemini-3.1-flash-image", &genai.GenerateContentConfig{
	ResponseModalities: []string{"IMAGE", "TEXT"},
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMinimal,
	},
})

// editModel is imageModel asked for a square. ImageConfig sets the shape and
// size of a generated picture, so the 3:2 source comes back 1024x1024 while
// generateFlow, which sets none, gets the model's own default.
//
// It is a whole ref rather than a per-call ai.WithConfig because that option
// replaces a ref's config instead of merging into it, which would drop
// ResponseModalities and the pictures with it.
var editModel = googlegenai.ModelRef("googleai/gemini-3.1-flash-image", &genai.GenerateContentConfig{
	ResponseModalities: []string{"IMAGE", "TEXT"},
	ImageConfig:        &genai.ImageConfig{AspectRatio: "1:1"},
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMinimal,
	},
})

// videoModel answers over minutes rather than in one response, so it is a
// background model: the request starts an operation and the flow polls it.
// VideoModelRef is the constructor carrying Veo's config type, as ModelRef
// carries a Gemini model's and ImageModelRef carries Imagen's.
//
// Veo is on both backends under different names: Google AI serves the preview
// ids, Vertex AI the stable ones (vertexai/veo-3.1-fast-generate-001).
var videoModel = googlegenai.VideoModelRef("googleai/veo-3.1-fast-generate-preview", &genai.GenerateVideosConfig{
	NumberOfVideos:  1,
	AspectRatio:     "16:9",
	DurationSeconds: genai.Ptr(int32(4)),
})

// pollInterval is how often the video flow asks whether the operation is done.
const pollInterval = 5 * time.Second

// A struct rather than a bare string lets each field carry a jsonschema
// default, so the Dev UI pre-fills its form and the flows run without typing
// anything. The default is form fill only: a field without omitempty is
// required, so an HTTP caller still sends one.
type (
	// DescribeRequest asks something about the picture.
	DescribeRequest struct {
		Question string `json:"question" jsonschema:"default=What is in this picture?" jsonschema_description:"What to ask about the picture"`
	}

	// EditRequest says how to redraw the picture.
	EditRequest struct {
		Instruction string `json:"instruction" jsonschema:"default=Redraw this as a watercolor painting." jsonschema_description:"How the picture should be changed"`
	}

	// DrawRequest describes a picture to make from nothing.
	DrawRequest struct {
		Description string `json:"description" jsonschema:"default=a red panda reading a book in a library" jsonschema_description:"What to draw"`
	}

	// AnimateRequest says how the picture should move.
	AnimateRequest struct {
		Motion string `json:"motion" jsonschema:"default=a slow drone push toward the ridge as clouds drift past" jsonschema_description:"How the picture should move"`
	}
)

// Progress is one polling update, streamed while the video is generated so a
// caller sees the wait rather than a silent connection.
type Progress struct {
	Elapsed string `json:"elapsed"`
	Done    bool   `json:"done"`
}

// Video is what the video flow returns once the operation finishes. The video
// travels inline as a data: URI, which is what gives the Dev UI a player, and is
// written to disk as well since that copy is the usable one from a terminal.
type Video struct {
	Video string `json:"video"`
	Path  string `json:"path"`
}

// Image is what the drawing flows return: what the model said, plus the
// images it drew as renderable data: URIs. Caption is often empty, since a
// model allowed to answer with an image usually answers with only an image.
type Image struct {
	Caption string   `json:"caption,omitempty"`
	Images  []string `json:"images"`
}

func main() {
	ctx := context.Background()

	// The Google AI plugin reads the API key from GEMINI_API_KEY or
	// GOOGLE_API_KEY, which is the recommended practice.
	plugin := &googlegenai.GoogleAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	// Uploads go through the plugin's own client, so they authenticate the same
	// way the model calls do.
	client, err := plugin.Client()
	if err != nil {
		log.Fatalf("could not get the Google AI client: %v", err)
	}

	DefineDescribe(g, client)
	DefineEdit(g, client)
	DefineGenerate(g)
	DefineAnimate(g, client)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineDescribe reads a picture: the file and the question travel as two parts
// of one user message, and the answer is text.
func DefineDescribe(g *genkit.Genkit, client *genai.Client) {
	genkit.DefineStreamingFlow(g, "describeFlow",
		func(ctx context.Context, input DescribeRequest, sendChunk core.StreamCallback[string]) (string, error) {
			file, err := uploadImage(ctx, client)
			if err != nil {
				return "", err
			}
			defer deleteImage(ctx, client, file.Name)

			text, err := genkit.GenerateText(ctx, g,
				ai.WithModel(model),
				// A media part holding a Files API URI stands in for the bytes.
				// A data: URI here would work the same way and skip the upload,
				// at the cost of resending the image every time.
				ai.WithPromptParts(
					ai.NewTextPart(input.Question),
					ai.NewMediaPart(imageType, file.URI),
				),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not describe the picture: %w", err)
			}
			return text, nil
		},
	)
}

// DefineEdit redraws a picture: the same request shape as describing one,
// pointed at a model allowed to answer with one.
func DefineEdit(g *genkit.Genkit, client *genai.Client) {
	genkit.DefineFlow(g, "editFlow",
		func(ctx context.Context, input EditRequest) (*Image, error) {
			file, err := uploadImage(ctx, client)
			if err != nil {
				return nil, err
			}
			defer deleteImage(ctx, client, file.Name)

			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(editModel),
				ai.WithPromptParts(
					ai.NewTextPart(input.Instruction),
					ai.NewMediaPart(imageType, file.URI),
				),
			)
			if err != nil {
				return nil, fmt.Errorf("could not edit the picture: %w", err)
			}
			return image(resp), nil
		},
	)
}

// DefineGenerate draws from a description alone, so it is the one flow with no
// Files API round trip.
func DefineGenerate(g *genkit.Genkit) {
	genkit.DefineFlow(g, "generateFlow",
		func(ctx context.Context, input DrawRequest) (*Image, error) {
			resp, err := genkit.Generate(ctx, g, ai.WithModel(imageModel), ai.WithPrompt("Draw %s.", input.Description))
			if err != nil {
				return nil, fmt.Errorf("could not draw the picture: %w", err)
			}
			return image(resp), nil
		},
	)
}

// DefineAnimate turns the picture into a video. Video generation runs far longer
// than a request should wait, so the model is a background one: GenerateOperation
// starts the work and returns a handle, and CheckModelOperation asks whether it
// has finished.
//
// Polling in a streaming flow is what makes the wait visible: each check sends a
// chunk, so a caller watching the stream sees progress instead of a stalled
// request. The loop is the whole pattern, and any long-running model uses it.
func DefineAnimate(g *genkit.Genkit, client *genai.Client) {
	genkit.DefineStreamingFlow(g, "animateFlow",
		func(ctx context.Context, input AnimateRequest, sendChunk core.StreamCallback[Progress]) (*Video, error) {
			// Veo reads the first frame from the bytes it is given, so the
			// picture is inlined rather than uploaded: a Files API URI would
			// arrive as the URL's own characters instead of as an image.
			image, err := readImage(ctx)
			if err != nil {
				return nil, err
			}

			op, err := genkit.GenerateOperation(ctx, g,
				ai.WithModel(videoModel),
				ai.WithPromptParts(
					ai.NewTextPart(input.Motion),
					ai.NewMediaPart(imageType, image),
				),
			)
			if err != nil {
				return nil, fmt.Errorf("could not start the video: %w", err)
			}

			op, err = awaitVideo(ctx, g, op, sendChunk)
			if err != nil {
				return nil, err
			}
			// A finished operation still reports how it finished.
			if op.Error != nil {
				return nil, fmt.Errorf("could not generate the video: %w", op.Error)
			}

			uri := op.Output.Media()
			if uri == "" {
				return nil, status.Errorf(status.ErrInternal, "the operation finished with no video")
			}
			data, err := downloadVideo(ctx, client, uri)
			if err != nil {
				return nil, err
			}
			if err := os.WriteFile(videoPath, data, 0o644); err != nil {
				return nil, status.Errorf(status.ErrInternal, "could not write %s: %w", videoPath, err)
			}
			return &Video{
				Video: fmt.Sprintf("data:%s;base64,%s", videoType, base64.StdEncoding.EncodeToString(data)),
				Path:  videoPath,
			}, nil
		},
	)
}

// awaitVideo polls until the operation finishes, streaming a chunk on every
// check. The wait is one span with the checks nested inside it, so a trace shows
// how long the model took and how many times it was asked.
func awaitVideo(ctx context.Context, g *genkit.Genkit, op *ai.ModelOperation, sendChunk core.StreamCallback[Progress]) (*ai.ModelOperation, error) {
	return genkit.RunWithContext(ctx, "await-video", func(ctx context.Context) (*ai.ModelOperation, error) {
		start := time.Now()
		for !op.Done {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(pollInterval):
			}
			var err error
			if op, err = genkit.CheckModelOperation(ctx, g, op); err != nil {
				return nil, fmt.Errorf("could not check the video: %w", err)
			}
			sendChunk(ctx, Progress{
				Elapsed: time.Since(start).Round(time.Second).String(),
				Done:    op.Done,
			})
		}
		return op, nil
	})
}

// readImage reads the sample picture and inlines it as a data: URI, the other
// way to attach media alongside a Files API reference.
func readImage(ctx context.Context) (string, error) {
	return genkit.RunWithContext(ctx, "read-image", func(context.Context) (string, error) {
		data, err := os.ReadFile(imagePath)
		if err != nil {
			return "", status.Errorf(status.ErrInternal, "could not read %s: %w", imagePath, err)
		}
		return fmt.Sprintf("data:%s;base64,%s", imageType, base64.StdEncoding.EncodeToString(data)), nil
	})
}

// downloadVideo fetches the finished video. The plugin's client authenticates
// the download, so the generated URI needs no API key appended.
func downloadVideo(ctx context.Context, client *genai.Client, uri string) ([]byte, error) {
	return genkit.RunWithContext(ctx, "download-video", func(ctx context.Context) ([]byte, error) {
		data, err := client.Files.Download(ctx, genai.NewDownloadURIFromVideo(&genai.Video{URI: uri}), nil)
		if err != nil {
			return nil, status.Errorf(status.ErrInternal, "could not download the video: %w", err)
		}
		return data, nil
	})
}

// uploadImage puts the sample picture in the Files API and returns the file,
// whose URI a request can point at instead of carrying the bytes.
//
// RunWithContext makes it a step rather than an invisible side trip: it gets
// its own span, and because the callback is handed the step's context, the
// SDK's HTTP calls nest underneath it. genkit.Run would time the step correctly
// but leave those calls hanging off the flow beside it.
func uploadImage(ctx context.Context, client *genai.Client) (*genai.File, error) {
	file, err := genkit.RunWithContext(ctx, "upload-image", func(ctx context.Context) (*genai.File, error) {
		return client.Files.UploadFromPath(ctx, imagePath, &genai.UploadFileConfig{
			MIMEType:    imageType,
			DisplayName: "sample image",
		})
	})
	if err != nil {
		// A caller can neither fix nor retry this differently, so classify it
		// here, where its meaning is known. Callers up the stack add context
		// with %w, which keeps the classification reachable through errors.Is.
		return nil, status.Errorf(status.ErrInternal, "could not upload %s: %w", imagePath, err)
	}
	return file, nil
}

// deleteImage removes the uploaded file, which would otherwise persist for
// about two days. Its span is where to look when a run leaves files behind.
func deleteImage(ctx context.Context, client *genai.Client, name string) {
	_, err := genkit.RunWithContext(ctx, "delete-image", func(ctx context.Context) (any, error) {
		_, err := client.Files.Delete(ctx, name, nil)
		return nil, err
	})
	if err != nil {
		// Cleanup is best effort, and nobody is waiting on the answer, so this
		// is reported rather than returned.
		logger.Warn(ctx, "could not delete the uploaded file", "file", name, "error", err)
	}
}

// image splits a response into what the model said and what it drew. Drawn
// images arrive as raw base64, so they are wrapped as renderable data: URIs; a
// part already carrying a URI is passed through untouched.
func image(resp *ai.ModelResponse) *Image {
	out := &Image{Caption: resp.Text()}
	for _, p := range resp.MediaParts() {
		image := p.Text
		if !strings.HasPrefix(image, "data:") && !strings.Contains(image, "://") {
			image = fmt.Sprintf("data:%s;base64,%s", p.ContentType, image)
		}
		out.Images = append(out.Images, image)
	}
	return out
}
