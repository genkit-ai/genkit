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

package systemone

import (
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

func TestStateJSONKeepsNumbersVerbatim(t *testing.T) {
	// The JSON goes out as written, so an ID past a float64's integer range
	// is not rounded.
	wire, _, err := BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage(`{"id": 9007199254740993}`)}}, true, Reads{})
	if err != nil {
		t.Fatal(err)
	}
	if got := base.JSONString(wire); got != `{"id":9007199254740993}` {
		t.Errorf("state on the wire = %s, want the number unchanged", got)
	}
}

func TestMedia(t *testing.T) {
	const png = "data:image/png;base64,iVBORw0KGgo="
	const wav, mp4 = "data:audio/wav;base64,UklGRg==", "data:video/mp4;base64,AAAAIGZ0eXA="
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewUserMessage(ai.NewTextPart("Is this the product?"), ai.NewMediaPart("image/png", png), ai.NewMediaPart("audio/wav", wav)),
			ai.NewModelMessage(ai.NewTextPart("Which one?")),
			ai.NewUserMessage(ai.NewMediaPart("", "data:image/jpeg;base64,/9j/4AAQ"), ai.NewMediaPart("video/mp4", mp4)),
		},
		Docs: []*ai.Document{
			{Content: []*ai.Part{ai.NewMediaPart("image/webp", "data:image/webp;base64,UklGRg==")}},
			ai.DocumentFromText("The catalog photo.", nil),
		},
	}
	state, media, err := BuildState(req, false, Reads{Images: true, Audio: true, Video: true})
	if err != nil {
		t.Fatal(err)
	}
	// Each kind keeps its order, messages first, and leaves the state.
	want := Media{
		Images: []string{png, "data:image/jpeg;base64,/9j/4AAQ", "data:image/webp;base64,UklGRg=="},
		Audio:  []string{wav},
		Videos: []string{mp4},
	}
	if !reflect.DeepEqual(media, want) {
		t.Errorf("media = %+v, want %+v", media, want)
	}
	wantState := map[string]any{
		"messages": []map[string]any{
			{"role": "user", "content": "Is this the product?"},
			{"role": "model", "content": "Which one?"},
			{"role": "user", "content": ""},
		},
		"context": []any{"The catalog photo."},
	}
	if !reflect.DeepEqual(state, wantState) {
		t.Errorf("state = %s, want %s", base.JSONString(state), base.JSONString(wantState))
	}

	// An image alone is a request with an empty state.
	state, media, err = BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewMediaPart("image/png", png))}}, false, Reads{Images: true})
	if err != nil || state != "" || len(media.Images) != 1 {
		t.Errorf("image alone: state = %q, images = %v, err = %v", state, media.Images, err)
	}

	// A data URL that is not base64 is encoded; the part's type names it.
	_, media, err = BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewMediaPart("image/svg+xml", "data:,<svg/>"))}}, false, Reads{Images: true})
	if err != nil || len(media.Images) != 1 || media.Images[0] != "data:image/svg+xml;base64,PHN2Zy8+" {
		t.Errorf("plain data URL: images = %v, err = %v", media.Images, err)
	}

	// A payload that is not base64 is percent-decoded first.
	_, media, err = BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewMediaPart("", "data:image/png,%89PNG%0D%0A%1A%0A"))}}, false, Reads{Images: true})
	if err != nil || len(media.Images) != 1 || media.Images[0] != png {
		t.Errorf("percent-encoded data URL: images = %v, err = %v", media.Images, err)
	}

	// A message of images alone is still a turn, so another message is
	// not sent as the state with its role lost.
	state, _, err = BuildState(&ai.ModelRequest{Messages: []*ai.Message{
		ai.NewUserMessage(ai.NewMediaPart("image/png", png)),
		ai.NewModelTextMessage("Looks like water damage."),
	}}, false, Reads{Images: true})
	if want := []map[string]any{{"role": "user", "content": ""}, {"role": "model", "content": "Looks like water damage."}}; err != nil || !reflect.DeepEqual(state, want) {
		t.Errorf("image-only turn: state = %s, err = %v, want %s", base.JSONString(state), err, base.JSONString(want))
	}

	// The base64 token is read without regard to case, and the payload
	// goes out as it is.
	_, media, err = BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;BASE64,iVBORw0KGgo="))}}, false, Reads{Images: true})
	if err != nil || len(media.Images) != 1 || media.Images[0] != png {
		t.Errorf("upper-case base64 token: images = %v, err = %v", media.Images, err)
	}

	// The scheme and the media type are read without regard to case.
	_, media, err = BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewMediaPart("", "DATA:Image/PNG;base64,iVBORw0KGgo="))}}, false, Reads{Images: true})
	if err != nil || len(media.Images) != 1 || media.Images[0] != png {
		t.Errorf("upper-case data URL: images = %v, err = %v", media.Images, err)
	}
}

func TestNoStateRefused(t *testing.T) {
	// Documents are context for a state, not the state itself.
	_, _, err := BuildState(&ai.ModelRequest{
		Messages: []*ai.Message{ai.NewSystemTextMessage("Judge the ticket.")},
		Docs:     []*ai.Document{ai.DocumentFromText("The catalog photo.", nil)},
	}, false, Reads{Images: true})
	if err == nil || !strings.Contains(err.Error(), "no state") {
		t.Errorf("error = %v, want a request without messages refused", err)
	}
}

func TestMediaRefused(t *testing.T) {
	images := Reads{Images: true}
	for _, tt := range []struct {
		name  string
		part  *ai.Part
		reads Reads
		want  string
	}{
		{"text-only model", ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="), Reads{}, "reads no image/png media"},
		{"audio to a vision model", ai.NewMediaPart("audio/wav", "data:audio/wav;base64,UklGRg=="), images, "reads no audio/wav media"},
		{"video to a vision model", ai.NewMediaPart("video/mp4", "data:video/mp4;base64,AAAAIGZ0eXA="), images, "reads no video/mp4 media"},
		{"remote URL", ai.NewMediaPart("image/png", "https://example.com/cat.png"), images, "DownloadRequestMedia"},
		{"upper-case remote URL", ai.NewMediaPart("image/png", "HTTPS://example.com/cat.png"), images, "DownloadRequestMedia"},
		{"not an image, audio, or video", ai.NewMediaPart("application/pdf", "data:application/pdf;base64,JVBERi0="), Reads{Images: true, Audio: true, Video: true}, "not an image, audio, or video"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hi"), tt.part)}}, false, tt.reads)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want one containing %q", err, tt.want)
			}
		})
	}
}
