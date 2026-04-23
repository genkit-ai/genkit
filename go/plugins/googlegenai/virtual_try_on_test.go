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
//
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"encoding/base64"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// taggedPart builds a media part carrying the person/product metadata the
// virtual try-on models are addressed with.
func taggedPart(mimeType, url, typ string) *ai.Part {
	p := ai.NewMediaPart(mimeType, url)
	p.Metadata = map[string]any{"type": typ}
	return p
}

func dataURL(mimeType string, data []byte) string {
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func TestToRecontextImageSource(t *testing.T) {
	t.Parallel()

	person := taggedPart("image/png", dataURL("image/png", []byte("person-bytes")), PartMetadataTypePersonImage)
	product := taggedPart("image/jpeg", dataURL("image/jpeg", []byte("product-bytes")), PartMetadataTypeProductImage)
	product2 := taggedPart("image/jpeg", dataURL("image/jpeg", []byte("product-bytes-2")), PartMetadataTypeProductImage)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(person, product, product2)},
	}

	src, err := toRecontextImageSource(input)
	if err != nil {
		t.Fatal(err)
	}
	if src.PersonImage == nil {
		t.Fatal("person image not set")
	}
	if got, want := string(src.PersonImage.ImageBytes), "person-bytes"; got != want {
		t.Errorf("person bytes = %q, want %q", got, want)
	}
	if got, want := src.PersonImage.MIMEType, "image/png"; got != want {
		t.Errorf("person mime = %q, want %q", got, want)
	}
	if got, want := len(src.ProductImages), 2; got != want {
		t.Fatalf("product images = %d, want %d", got, want)
	}
	if got, want := string(src.ProductImages[1].ProductImage.ImageBytes), "product-bytes-2"; got != want {
		t.Errorf("second product bytes = %q, want %q", got, want)
	}
	// The prompt field is not supported by virtual try-on; the source carries
	// images only.
	if src.Prompt != "" {
		t.Errorf("prompt = %q, want empty", src.Prompt)
	}
}

func TestToRecontextImageSourceGCSURI(t *testing.T) {
	t.Parallel()

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			taggedPart("image/png", "gs://bucket/person.png", PartMetadataTypePersonImage),
			taggedPart("image/png", "gs://bucket/shirt.png", PartMetadataTypeProductImage),
		)},
	}

	src, err := toRecontextImageSource(input)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := src.PersonImage.GCSURI, "gs://bucket/person.png"; got != want {
		t.Errorf("person gcsUri = %q, want %q", got, want)
	}
	if len(src.PersonImage.ImageBytes) != 0 {
		t.Error("gs:// person image should not carry inline bytes")
	}
	if got, want := src.ProductImages[0].ProductImage.GCSURI, "gs://bucket/shirt.png"; got != want {
		t.Errorf("product gcsUri = %q, want %q", got, want)
	}
}

func TestToRecontextImageSourceErrors(t *testing.T) {
	t.Parallel()

	person := taggedPart("image/png", dataURL("image/png", []byte("person-bytes")), PartMetadataTypePersonImage)
	product := taggedPart("image/png", dataURL("image/png", []byte("product-bytes")), PartMetadataTypeProductImage)

	tests := []struct {
		name  string
		parts []*ai.Part
		want  string
	}{
		{
			name:  "no person image",
			parts: []*ai.Part{product},
			want:  PartMetadataTypePersonImage,
		},
		{
			name:  "no product image",
			parts: []*ai.Part{person},
			want:  PartMetadataTypeProductImage,
		},
		{
			name:  "untagged parts are not usable",
			parts: []*ai.Part{ai.NewMediaPart("image/png", dataURL("image/png", []byte("x")))},
			want:  PartMetadataTypePersonImage,
		},
		{
			// A tagged part that cannot be parsed is reported, not skipped:
			// skipping would claim the image is missing or silently send
			// fewer product images than asked for.
			name:  "unreadable person part",
			parts: []*ai.Part{taggedPart("image/png", "data:image/png;base64,%%%not-base64%%%", PartMetadataTypePersonImage), product},
			want:  "unreadable",
		},
		{
			// uri.Data would hand back the URL's own bytes here, which the
			// API then rejects as a corrupt image.
			name:  "http url instead of image data",
			parts: []*ai.Part{taggedPart("image/jpeg", "https://example.com/shirt.jpg", PartMetadataTypePersonImage), product},
			want:  "http(s) URLs are not fetched",
		},
		{
			name:  "gs uri without a content type",
			parts: []*ai.Part{taggedPart("", "gs://bucket/person.png", PartMetadataTypePersonImage), product},
			want:  "content type",
		},
		{
			// The API takes exactly one person, so a second is a caller error
			// rather than something to silently drop.
			name:  "two person images",
			parts: []*ai.Part{person, taggedPart("image/png", dataURL("image/png", []byte("other-person")), PartMetadataTypePersonImage), product},
			want:  "single personImage",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(tt.parts...)}}
			_, err := toRecontextImageSource(input)
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}
