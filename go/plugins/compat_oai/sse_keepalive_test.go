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

package compat_oai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/iotest"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go/option"
)

func sseChunk(text, finish string) string {
	f := "null"
	if finish != "" {
		f = `"` + finish + `"`
	}
	return `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"` + text + `"},"finish_reason":` + f + `}]}` + "\n\n"
}

// streamText streams one request against a server that writes body verbatim
// as text/event-stream and returns the final text.
func streamText(t *testing.T, body string, opts ...option.RequestOption) (string, error) {
	t.Helper()
	return streamTextAs(t, "text/event-stream; charset=utf-8", body, opts...)
}

// streamTextAs is streamText with the response's Content-Type set to
// contentType, or left out when it is empty.
func streamTextAs(t *testing.T, contentType, body string, opts ...option.RequestOption) (string, error) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if contentType == "" {
			// Without this, net/http sniffs and sets text/plain.
			w.Header()["Content-Type"] = nil
		} else {
			w.Header().Set("Content-Type", contentType)
		}
		for _, part := range strings.SplitAfter(body, "\n") {
			_, _ = io.WriteString(w, part)
			w.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(server.Close)

	ctx := context.Background()
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "k", BaseURL: server.URL, Opts: opts}
	o.Init(ctx)
	g := genkit.Init(ctx)
	genkit.RegisterAction(g, o.NewModel("m", ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}}))
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/m"),
		ai.WithPrompt("hi"),
		ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error { return nil }),
	)
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// TestStreamingSkipsSSEKeepAliveComments: OpenAI sends
// ": keep-alive" comments while a model is slow to start, and openai-go v1
// dispatches the blank line after a comment as an event with no data, which
// it then fails to unmarshal.
func TestStreamingSkipsSSEKeepAliveComments(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"no comment", sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"keep-alive first", ": keep-alive\n\n" + sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"keep-alive between chunks", sseChunk("hel", "") + ": keep-alive\n\n: keep-alive\n\n" + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"CRLF keep-alive", ": keep-alive\r\n\r\n" + sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"comment inside a data event", ": note\n" + sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"event with no data", "event: ping\nid: 1\n\n" + sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"stray blank lines", "\n\n" + sseChunk("hel", "") + "\n" + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"empty data values", "data:\n\ndata: \n\n" + sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"},
		{"bare CR line endings", strings.ReplaceAll(": keep-alive\n\n"+sseChunk("hel", "")+sseChunk("lo", "stop")+"data: [DONE]\n\n", "\n", "\r")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := streamText(t, tt.body)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got != "hello" {
				t.Errorf("text = %q, want %q", got, "hello")
			}
		})
	}
}

// TestCallerMiddlewareSeesRawStream pins the filter as the outermost
// middleware: one passed in Opts still sees keep-alives, e.g. to reset an
// idle timer.
func TestCallerMiddlewareSeesRawStream(t *testing.T) {
	var raw strings.Builder
	tee := option.WithMiddleware(func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		resp, err := next(req)
		if err == nil {
			resp.Body = struct {
				io.Reader
				io.Closer
			}{io.TeeReader(resp.Body, &raw), resp.Body}
		}
		return resp, err
	})
	got, err := streamText(t, ": keep-alive\n\n"+sseChunk("hello", "stop")+"data: [DONE]\n\n", tee)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got != "hello" {
		t.Errorf("text = %q, want %q", got, "hello")
	}
	if !strings.Contains(raw.String(), ": keep-alive") {
		t.Errorf("caller middleware saw %q, want the keep-alive left in", raw.String())
	}
}

// TestStreamingIgnoresErrorsWithoutData pins a deliberate change: a stream
// that ends in an error event with no data, or in a bare JSON error line, used
// to fail on the empty or invalid event. It now ends with the text so far and
// no error, as the SSE spec and openai-go v3 have it.
func TestStreamingIgnoresErrorsWithoutData(t *testing.T) {
	tests := []struct {
		name string
		tail string
	}{
		{"error event without data", "event: error\n\n"},
		{"bare JSON error line", `{"error":{"message":"boom"}}` + "\n\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := streamText(t, sseChunk("hel", "")+tt.tail)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got != "hel" {
				t.Errorf("text = %q, want %q", got, "hel")
			}
		})
	}
}

func TestSSEFilter(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{
		{"comment-only events", ": a\n\n: b\n\ndata: x\n\n", "data: x\n\n"},
		{"comment inside an event", "data: x\n: c\ndata: y\n\n", "data: x\ndata: y\n\n"},
		{"CRLF", ": a\r\n\r\ndata: x\r\n\r\n", "data: x\n\n"},
		{"bare CR", ": a\r\rdata: x\r\r", "data: x\n\n"},
		{"mixed line endings", ": a\r\ndata: x\r\rdata: y\n\n", "data: x\n\ndata: y\n\n"},
		{"fields before data are kept", "event: e\nid: 1\ndata: x\n\n", "event: e\nid: 1\ndata: x\n\n"},
		{"fields after data are kept", "data: x\nid: 1\n\n", "data: x\nid: 1\n\n"},
		{"event without data is dropped", "event: ping\nretry: 5\n\ndata: x\n\n", "data: x\n\n"},
		{"stray blank lines", "\n\ndata: x\n\n\n", "data: x\n\n"},
		{"data with no colon is dropped", "data\n\ndata: x\n\n", "data: x\n\n"},
		{"empty data value is dropped", "data:\n\ndata: \n\nevent: e\ndata:\n\ndata: x\n\n", "data: x\n\n"},
		{"empty data value before a value is kept", "data:\ndata: x\n\n", "data:\ndata: x\n\n"},
		{"empty data value after a value is kept", "data: x\ndata:\n\n", "data: x\ndata:\n\n"},
		{"unterminated final event", "data: x\n\nid: 2\ndata: y", "data: x\n\nid: 2\ndata: y"},
		{"unterminated final event without data", "data: x\n\nid: 2", "data: x\n\n"},
		{"line ending in two CRs", "data\r\r\n\ndata: x\n\n", "data: x\n\n"},
		{"held fields past the cap across lines", strings.Repeat("id: "+strings.Repeat("i", 30000)+"\n", 3) + "\n", strings.Repeat("id: "+strings.Repeat("i", 30000)+"\n", 3) + "\n"},
		{"colon in data value", "data: {\"a\":\"b\"}\n\n", "data: {\"a\":\"b\"}\n\n"},
		{"data line longer than the buffer", "data: " + strings.Repeat("x", 10000) + "\n\n", "data: " + strings.Repeat("x", 10000) + "\n\n"},
		{"comment longer than the buffer", ": " + strings.Repeat("c", 10000) + "\n\ndata: x\n\n", "data: x\n\n"},
		{"field longer than the buffer, then data", "id: " + strings.Repeat("i", 10000) + "\ndata: x\n\n", "id: " + strings.Repeat("i", 10000) + "\ndata: x\n\n"},
		{"field longer than the buffer, no data", "id: " + strings.Repeat("i", 10000) + "\n\ndata: x\n\n", "data: x\n\n"},
		{"held fields past the cap pass through", "id: " + strings.Repeat("i", maxHeldSSEFields+1) + "\n\n", "id: " + strings.Repeat("i", maxHeldSSEFields+1) + "\n\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// One byte at a time, so every line arrives split across reads.
			src := io.NopCloser(iotest.OneByteReader(strings.NewReader(tt.in)))
			got, err := io.ReadAll(iotest.OneByteReader(newSSEFilter(src)))
			if err != nil {
				t.Fatalf("ReadAll() error = %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDropDatalessSSEEventsLeavesOtherResponsesAlone(t *testing.T) {
	body := ": keep-alive\n\n"
	tests := []struct {
		name        string
		status      int
		contentType string
		want        string
	}{
		{"SSE", 200, "text/event-stream; charset=utf-8", ""},
		{"JSON", 200, "application/json", body},
		{"error status", 500, "text/event-stream", body},
		{"no content type", 200, "", body},
		{"compressed", 200, "text/event-stream", body},
		{"identity encoding", 200, "text/event-stream", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			next := func(*http.Request) (*http.Response, error) {
				h := http.Header{"Content-Length": {"15"}}
				if tt.contentType != "" {
					h.Set("Content-Type", tt.contentType)
				}
				switch tt.name {
				case "compressed":
					h.Set("Content-Encoding", "gzip")
				case "identity encoding":
					h.Set("Content-Encoding", "identity")
				}
				return &http.Response{StatusCode: tt.status, Header: h, ContentLength: 15, Body: io.NopCloser(strings.NewReader(body))}, nil
			}
			resp, err := dropDatalessSSEEvents(httptest.NewRequest("POST", "/", nil), next)
			if err != nil {
				t.Fatal(err)
			}
			got, _ := io.ReadAll(resp.Body)
			if string(got) != tt.want {
				t.Errorf("body = %q, want %q", got, tt.want)
			}
			filtered := tt.want != body
			if filtered && (resp.ContentLength != -1 || resp.Header.Get("Content-Length") != "") {
				t.Errorf("ContentLength = %d, header %q; want them cleared", resp.ContentLength, resp.Header.Get("Content-Length"))
			}
			if !filtered && resp.ContentLength != 15 {
				t.Errorf("ContentLength = %d, want it untouched", resp.ContentLength)
			}
		})
	}
}

// TestStreamingStaysIncremental pins that the filter hands each line on as it
// arrives: the server holds the second chunk back until the first has reached
// the caller, so a filter that buffered ahead would hang here.
func TestStreamingStaysIncremental(t *testing.T) {
	firstSeen := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, ": keep-alive\n\n"+sseChunk("hel", ""))
		w.(http.Flusher).Flush()
		select {
		case <-firstSeen:
		case <-time.After(5 * time.Second):
			return
		}
		_, _ = io.WriteString(w, sseChunk("lo", "stop")+"data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	ctx := context.Background()
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "k", BaseURL: server.URL}
	o.Init(ctx)
	g := genkit.Init(ctx)
	genkit.RegisterAction(g, o.NewModel("m", ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}}))
	chunks := 0
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/m"),
		ai.WithPrompt("hi"),
		ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error {
			if chunks++; chunks == 1 {
				close(firstSeen)
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "hello" {
		t.Errorf("text = %q, want %q (the first chunk did not arrive before the second was sent)", got, "hello")
	}
}

func TestSSEFilterZeroLengthRead(t *testing.T) {
	f := newSSEFilter(io.NopCloser(strings.NewReader("data: x\n\n")))
	if n, err := f.Read(nil); n != 0 || err != nil {
		t.Fatalf("Read(nil) = %d, %v; want 0, nil", n, err)
	}
}

// TestSSEFilterPassesOnSourceErrors pins that a read error from the server
// reaches the SDK after whatever was filtered before it.
func TestSSEFilterPassesOnSourceErrors(t *testing.T) {
	boom := errors.New("connection reset")
	src := io.NopCloser(io.MultiReader(strings.NewReader(": keep-alive\n\ndata: x\n\ndata: partial"), iotest.ErrReader(boom)))
	got, err := io.ReadAll(newSSEFilter(src))
	if !errors.Is(err, boom) {
		t.Fatalf("ReadAll() error = %v, want %v", err, boom)
	}
	if want := "data: x\n\ndata: partial"; string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestStreamingSkipsKeepAlivesInUnlabelledStreams covers a gateway that sends
// a stream without labelling it text/event-stream: the SDK decodes the reply
// to a streaming request as SSE regardless, so it must be filtered too.
func TestStreamingSkipsKeepAlivesInUnlabelledStreams(t *testing.T) {
	body := ": keep-alive\n\n" + sseChunk("hel", "") + sseChunk("lo", "stop") + "data: [DONE]\n\n"
	for _, contentType := range []string{"", "text/plain", "text/plain; charset=utf-8", "application/octet-stream", "application/json"} {
		t.Run(contentType, func(t *testing.T) {
			got, err := streamTextAs(t, contentType, body)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got != "hello" {
				t.Errorf("text = %q, want %q", got, "hello")
			}
		})
	}
}

func TestDropDatalessSSEEventsOnStreamingRequests(t *testing.T) {
	body := ": keep-alive\n\n"
	tests := []struct {
		name, contentType, encoding string
		status                      int
		want                        string
	}{
		{"no content type", "", "", 200, ""},
		{"text/plain", "text/plain; charset=utf-8", "", 200, ""},
		{"JSON", "application/json", "", 200, ""},
		{"problem JSON", "application/problem+json", "", 200, ""},
		{"compressed", "", "gzip", 200, body},
		{"error status", "text/plain", "", 502, body},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			next := func(*http.Request) (*http.Response, error) {
				h := http.Header{}
				if tt.contentType != "" {
					h.Set("Content-Type", tt.contentType)
				}
				if tt.encoding != "" {
					h.Set("Content-Encoding", tt.encoding)
				}
				return &http.Response{StatusCode: tt.status, Header: h, Body: io.NopCloser(strings.NewReader(body))}, nil
			}
			req := httptest.NewRequest("POST", "/", nil)
			req = req.WithContext(withStreamingRequest(req.Context()))
			resp, err := dropDatalessSSEEvents(req, next)
			if err != nil {
				t.Fatal(err)
			}
			if got, _ := io.ReadAll(resp.Body); string(got) != tt.want {
				t.Errorf("body = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestSniffedSSEBody pins that an unlabelled reply is filtered only when it
// starts like an event stream; anything else reaches the SDK as sent.
func TestSniffedSSEBody(t *testing.T) {
	html := "<!DOCTYPE html>\n<html>\n\n<body>502 Bad Gateway</body>\n\n</html>\n"
	prettyJSON := "{\n\n  \"error\": {\"message\": \"quota\"}\n\n}\n"
	tests := []struct{ name, in, want string }{
		{"HTML page", html, html},
		{"pretty-printed JSON", prettyJSON, prettyJSON},
		{"JSON after blank lines", "\n\n" + prettyJSON, "\n\n" + prettyJSON},
		{"a field-like word that is not a field", "dataset: x\n\n", "dataset: x\n\n"},
		{"keep-alive first", ": ka\n\ndata: x\n\n", "data: x\n\n"},
		{"blank lines, then a keep-alive", "\n\n: ka\n\ndata: x\n\n", "data: x\n\n"},
		{"data first", "data: x\n\n", "data: x\n\n"},
		{"data with no colon", "data\ndata: x\n\n", "data\ndata: x\n\n"},
		{"event first", "event: e\ndata: x\n\n", "event: e\ndata: x\n\n"},
		{"retry only", "retry: 5\n\n", ""},
		{"CRLF", "\r\n: ka\r\n\r\ndata: x\r\n\r\n", "data: x\n\n"},
		{"blank lines past the sniff limit", strings.Repeat("\n", maxSniffedBlankBytes+1) + "data: x\n\n", "data: x\n\n"},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := &sniffedSSEBody{src: io.NopCloser(iotest.OneByteReader(strings.NewReader(tt.in)))}
			got, err := io.ReadAll(body)
			if err != nil {
				t.Fatalf("ReadAll() error = %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}

	t.Run("error before the body shows what it is", func(t *testing.T) {
		boom := errors.New("connection reset")
		body := &sniffedSSEBody{src: io.NopCloser(io.MultiReader(strings.NewReader("\n\n"), iotest.ErrReader(boom)))}
		got, err := io.ReadAll(body)
		if !errors.Is(err, boom) || string(got) != "\n\n" {
			t.Errorf("ReadAll() = %q, %v; want %q, %v", got, err, "\n\n", boom)
		}
	})
}

// TestStreamingNonSSEReplyStillFails pins that a gateway page or JSON error
// sent with a 200 on a streaming request fails as it did without the filter,
// rather than having its lines dropped and reading as an empty completion.
func TestStreamingNonSSEReplyStillFails(t *testing.T) {
	tests := []struct{ name, contentType, body string }{
		{"HTML page", "text/html", "<!DOCTYPE html>\n<html>\n\n<body>502 Bad Gateway</body>\n\n</html>\n"},
		{"JSON error as text/plain", "text/plain", "{\n\n  \"error\": {\"message\": \"quota\"}\n\n}\n"},
		{"JSON error as application/json", "application/json", "{\n\n  \"error\": {\"message\": \"quota\"}\n\n}\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := streamTextAs(t, tt.contentType, tt.body)
			if err == nil || !strings.Contains(err.Error(), "unexpected end of JSON input") {
				t.Errorf("Generate() error = %v, want the SDK's own decode error", err)
			}
		})
	}
}

// TestStreamingUnlabelledStreamAfterRetry pins that the streaming mark
// survives the SDK's retry: the first attempt fails with a 500 and the second
// returns an unlabelled stream with keep-alives.
func TestStreamingUnlabelledStreamAfterRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts++; attempts == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"error":{"message":"try again"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(w, ": keep-alive\n\n"+sseChunk("hello", "stop")+"data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	ctx := context.Background()
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "k", BaseURL: server.URL, Opts: []option.RequestOption{option.WithMaxRetries(1)}}
	o.Init(ctx)
	g := genkit.Init(ctx)
	genkit.RegisterAction(g, o.NewModel("m", ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}}))
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/m"),
		ai.WithPrompt("hi"),
		ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error { return nil }),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if attempts != 2 {
		t.Errorf("attempts = %d, want 2", attempts)
	}
	if got := resp.Text(); got != "hello" {
		t.Errorf("text = %q, want %q", got, "hello")
	}
}

func TestLineEndingNormalizer(t *testing.T) {
	tests := []struct{ in, want string }{
		{"a\r\nb", "a\nb"},
		{"a\rb", "a\nb"},
		{"a\r\r\nb", "a\n\nb"},
		{"a\n\rb\r", "a\n\nb\n"},
		{"\r\n\r\n", "\n\n"},
	}
	for _, tt := range tests {
		// One byte at a time, so a "\r\n" always spans two reads and some
		// reads hold only a dropped "\n".
		got, err := io.ReadAll(&lineEndingNormalizer{r: iotest.OneByteReader(strings.NewReader(tt.in))})
		if err != nil {
			t.Fatalf("%q: ReadAll() error = %v", tt.in, err)
		}
		if string(got) != tt.want {
			t.Errorf("%q: got %q, want %q", tt.in, got, tt.want)
		}
		if err := iotest.TestReader(&lineEndingNormalizer{r: strings.NewReader(tt.in)}, []byte(tt.want)); err != nil {
			t.Errorf("%q: %v", tt.in, err)
		}
	}
}
