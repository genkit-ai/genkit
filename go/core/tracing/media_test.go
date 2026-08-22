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

package tracing_test

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type spanExporter struct {
	spans []sdktrace.ReadOnlySpan
}

func (e *spanExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.spans = append(e.spans, spans...)
	return nil
}

func (e *spanExporter) Shutdown(context.Context) error {
	return nil
}

func TestMediaDataIsRedactedFromSpanAttributes(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	exporter := &spanExporter{}
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	otel.SetTracerProvider(provider)
	defer func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	}()

	input := &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewUserMessage(
				ai.NewMediaPart("image/png", "data:image/png;base64,secret-image-bytes"),
				ai.NewMediaPart("image/jpeg", "https://example.com/image.jpg"),
			),
		},
	}
	_, err := tracing.RunInNewSpan(
		context.Background(),
		&tracing.SpanMetadata{Name: "generate", Type: "action"},
		input,
		func(context.Context, *ai.ModelRequest) (any, error) { return nil, nil },
	)
	if err != nil {
		t.Fatalf("RunInNewSpan: %v", err)
	}
	if len(exporter.spans) != 1 {
		t.Fatalf("exported %d spans, want 1", len(exporter.spans))
	}

	var inputAttribute string
	for _, attribute := range exporter.spans[0].Attributes() {
		if string(attribute.Key) == "genkit:input" {
			inputAttribute = attribute.Value.AsString()
			break
		}
	}
	if strings.Contains(inputAttribute, "secret-image-bytes") {
		t.Fatalf("genkit:input contains inline media data: %s", inputAttribute)
	}
	if !strings.Contains(inputAttribute, `"url":"data:image/png;base64,[redacted]"`) {
		t.Errorf("genkit:input lost the redacted media URI: %s", inputAttribute)
	}
	if !strings.Contains(inputAttribute, `"url":"https://example.com/image.jpg"`) {
		t.Errorf("genkit:input lost the remote media URL: %s", inputAttribute)
	}
}

func TestMediaLikeTextIsNotTreatedAsInlineMedia(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	exporter := &spanExporter{}
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	otel.SetTracerProvider(provider)
	defer func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	}()

	input := map[string]string{
		"description": "data: is a text prefix;base64 is a suffix",
	}
	_, err := tracing.RunInNewSpan(
		context.Background(),
		&tracing.SpanMetadata{Name: "text", Type: "action"},
		input,
		func(context.Context, map[string]string) (any, error) { return nil, nil },
	)
	if err != nil {
		t.Fatalf("RunInNewSpan: %v", err)
	}
	if len(exporter.spans) != 1 {
		t.Fatalf("exported %d spans, want 1", len(exporter.spans))
	}
	for _, attr := range exporter.spans[0].Attributes() {
		if string(attr.Key) == "genkit:input" && attr.Value.AsString() != `{"description":"data: is a text prefix;base64 is a suffix"}` {
			t.Fatalf("genkit:input was changed: %s", attr.Value.AsString())
		}
	}
}
