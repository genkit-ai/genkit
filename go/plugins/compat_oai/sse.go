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
	"bufio"
	"bytes"
	"context"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/openai/openai-go/option"
)

// dropDatalessSSEEvents is a middleware that removes server-sent events with
// no data (keep-alive comments, pings, empty data values) from a successful,
// uncompressed event stream. openai-go v1 dispatches those
// events and fails to unmarshal them, ending the stream with "unexpected end
// of JSON input".
func dropDatalessSSEEvents(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	resp, err := next(req)
	if err != nil || resp == nil || resp.Body == nil || resp.StatusCode < 200 || resp.StatusCode > 299 {
		return resp, err
	}
	kind := eventStreamKind(req, resp)
	if kind == notEventStream {
		return resp, err
	}
	if enc := resp.Header.Get("Content-Encoding"); enc != "" && !strings.EqualFold(enc, "identity") {
		return resp, err
	}
	if kind == labelledEventStream {
		resp.Body = newSSEFilter(resp.Body)
	} else {
		resp.Body = &sniffedSSEBody{src: resp.Body}
	}
	// The filtered body is shorter than what the server declared.
	resp.ContentLength = -1
	resp.Header.Del("Content-Length")
	return resp, err
}

// streamingRequestKey marks the context of a request the plugin streams, so
// the middleware knows the SDK will decode the reply as SSE.
type streamingRequestKey struct{}

// withStreamingRequest marks ctx as carrying a streaming request.
func withStreamingRequest(ctx context.Context) context.Context {
	return context.WithValue(ctx, streamingRequestKey{}, true)
}

type streamKind int

const (
	notEventStream      streamKind = iota
	labelledEventStream            // labelled text/event-stream
	maybeEventStream               // a streamed request's reply, labelled otherwise
)

// eventStreamKind reports whether resp is an SSE stream. One labelled
// text/event-stream always is. For a request the plugin streams, the SDK
// decodes any successful reply as SSE whatever its Content-Type, so a
// gateway that labels the stream text/plain or not at all may be one too;
// its body decides (see sniffedSSEBody). A JSON reply is left alone.
func eventStreamKind(req *http.Request, resp *http.Response) streamKind {
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaType == "text/event-stream" {
		return labelledEventStream
	}
	if streaming, _ := req.Context().Value(streamingRequestKey{}).(bool); !streaming {
		return notEventStream
	}
	if mediaType == "application/json" || strings.HasSuffix(mediaType, "+json") {
		return notEventStream
	}
	return maybeEventStream
}

// sniffedSSEBody filters a body only if it starts like an event stream: its
// first non-blank line is a comment or a data, event, id or retry field.
// Anything else, such as a gateway's HTML page or a JSON error labelled
// text/plain, passes through byte for byte, so the SDK fails on it as it
// would have without the filter rather than having its lines dropped. The
// check runs on the first Read, so it does not hold up the request.
type sniffedSSEBody struct {
	src io.ReadCloser
	r   io.Reader // set by the first Read
}

func (b *sniffedSSEBody) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if b.r == nil {
		b.r = sniffSSE(b.src)
	}
	return b.r.Read(p)
}

func (b *sniffedSSEBody) Close() error {
	return b.src.Close()
}

// sniffSSE reads src up to the first non-blank line's field name and returns
// a reader for the whole body: filtered if it is an event stream, as sent
// otherwise. It reads at most a few bytes past any leading blank lines.
func sniffSSE(src io.Reader) io.Reader {
	br := bufio.NewReader(src)
	var seen []byte
	start := -1 // index in seen of the first non-blank line
	for {
		c, err := br.ReadByte()
		if err != nil {
			// The body ended or failed before showing what it is: hand on
			// what was read and then the error, unfiltered.
			return io.MultiReader(bytes.NewReader(seen), errReader{err})
		}
		seen = append(seen, c)
		if start < 0 {
			if c == '\r' || c == '\n' {
				if len(seen) > maxHeldSSEFields {
					// Only blank lines so far; that reads as SSE, not as a
					// document.
					return newSSEFilter(io.NopCloser(io.MultiReader(bytes.NewReader(seen), br)))
				}
				continue
			}
			start = len(seen) - 1
		}
		name := seen[start : len(seen)-1]
		if c == ':' || c == '\r' || c == '\n' || len(name) > len("retry") {
			body := io.MultiReader(bytes.NewReader(seen), br)
			switch string(name) {
			case "", "data", "event", "id", "retry":
				// An empty name is a comment: the line began with ':'.
				return newSSEFilter(io.NopCloser(body))
			}
			return body
		}
	}
}

// errReader returns err from every Read.
type errReader struct{ err error }

func (r errReader) Read([]byte) (int, error) { return 0, r.err }

// maxHeldSSEFields bounds the total bytes sseFilter holds back for one event
// before its first data line. Past it the fields are passed on as they are, leaving the
// event to the SDK; real streams put a few short fields before their data.
const maxHeldSSEFields = 64 << 10

// lineAction is what sseFilter does with the rest of the line it is reading.
type lineAction int

const (
	passLine lineAction = iota // send it on
	dropLine                   // a comment
	holdLine                   // a field before the event's first data line
)

// sseFilter streams an SSE body, dropping comment lines and any event that
// ends without a data field that has a value. It reads in chunks of at most the bufio buffer
// size, deciding what to do with a line from its first chunk, so a long line
// costs no more memory than a short one. Data lines pass through as they
// arrive; the other fields of an event (event, id, retry) are held only until
// its first data line with a value, since without one the event is dropped.
type sseFilter struct {
	src     io.ReadCloser
	r       *bufio.Reader
	out     []byte // filtered bytes, returned to the caller from outIdx on
	outIdx  int    // the next byte of out to return
	held    []byte // fields of the current event before its first data value
	hasData bool   // the current event has a data field with a value
	midLine bool   // the last chunk read did not end its line
	action  lineAction
	err     error // sticky error from src, returned once out drains
}

func newSSEFilter(src io.ReadCloser) *sseFilter {
	return &sseFilter{src: src, r: bufio.NewReader(&lineEndingNormalizer{r: src})}
}

// lineEndingNormalizer rewrites the SSE line endings "\r\n" and a lone "\r"
// to "\n". The SDK's scanner splits on "\n" only, so a stream that ends lines
// with a bare "\r", which the spec allows, would otherwise reach it as one
// line. A "\r" is rewritten as soon as it is read; a "\n" right after it is
// then dropped, even when it arrives in a later read.
type lineEndingNormalizer struct {
	r       io.Reader
	afterCR bool // the last byte read was "\r"
}

func (n *lineEndingNormalizer) Read(p []byte) (int, error) {
	for {
		read, err := n.r.Read(p)
		kept := 0
		for _, b := range p[:read] {
			switch {
			case b == '\r':
				p[kept], n.afterCR = '\n', true
				kept++
			case b == '\n' && n.afterCR:
				n.afterCR = false
			default:
				p[kept], n.afterCR = b, false
				kept++
			}
		}
		// Every byte may have been a dropped "\n"; read again rather than
		// return 0, nil.
		if kept > 0 || err != nil || read == 0 {
			return kept, err
		}
	}
}

func (f *sseFilter) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for f.outIdx >= len(f.out) {
		if f.err != nil {
			return 0, f.err
		}
		// Reuse out's buffer from the start once it has been drained.
		f.out, f.outIdx = f.out[:0], 0
		chunk, err := f.r.ReadSlice('\n')
		if len(chunk) > 0 {
			f.filter(chunk, err == nil)
		}
		if err != nil && err != bufio.ErrBufferFull {
			// At EOF, fields of an event that never got data are dropped.
			f.err = err
		}
	}
	n := copy(p, f.out[f.outIdx:])
	f.outIdx += n
	return n, nil
}

// filter handles one chunk of a line; ends reports whether the chunk ends
// the line.
func (f *sseFilter) filter(chunk []byte, ends bool) {
	if !f.midLine {
		f.action = f.startLine(chunk)
	}
	f.midLine = !ends
	switch f.action {
	case passLine:
		f.out = append(f.out, chunk...)
	case holdLine:
		f.held = append(f.held, chunk...)
		if len(f.held) > maxHeldSSEFields {
			f.flushHeld()
			f.action = passLine
		}
	}
}

// startLine decides what to do with a line from its first chunk, and handles
// the blank line that ends an event.
func (f *sseFilter) startLine(chunk []byte) lineAction {
	content := bytes.TrimSuffix(chunk, []byte("\n"))
	switch {
	case len(content) == 0:
		// A blank line dispatches the event: keep it only if it has data.
		if f.hasData {
			f.out = append(f.out, chunk...)
		}
		f.held, f.hasData = f.held[:0], false
		return dropLine
	case content[0] == ':':
		return dropLine
	}
	if f.hasData {
		return passLine
	}
	name, value, _ := bytes.Cut(content, []byte(":"))
	value = bytes.TrimPrefix(value, []byte(" "))
	// A data line with an empty value is held like any other field: v1 fails
	// on an event whose data is empty, though the spec would dispatch it.
	if bytes.Equal(name, []byte("data")) && len(value) > 0 {
		f.flushHeld()
		return passLine
	}
	return holdLine
}

// flushHeld commits the current event: its held fields go out, and the rest
// of it passes through.
func (f *sseFilter) flushHeld() {
	f.out = append(f.out, f.held...)
	f.held, f.hasData = f.held[:0], true
}

func (f *sseFilter) Close() error {
	return f.src.Close()
}
