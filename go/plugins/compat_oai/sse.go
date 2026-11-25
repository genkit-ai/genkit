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
	"io"
	"mime"
	"net/http"

	"github.com/openai/openai-go/option"
)

// dropDatalessSSEEvents is a middleware that removes server-sent events
// carrying no data from a streaming response before the SDK decodes it.
//
// OpenAI sends ": keep-alive" comments while a model is slow to start, and
// gateways send similar ones. The SSE spec says an event whose data buffer is
// empty is not dispatched, but openai-go v1 dispatches one on every blank
// line and then fails to unmarshal the empty data, which ends the stream with
// "unexpected end of JSON input". openai-go v3 fixed its decoder; v1, which
// this package's public API is built on, did not.
//
// Only successful text/event-stream responses are touched, so errors and
// non-streaming replies reach the SDK as sent.
func dropDatalessSSEEvents(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	resp, err := next(req)
	if err != nil || resp == nil || resp.Body == nil || resp.StatusCode < 200 || resp.StatusCode > 299 {
		return resp, err
	}
	if mediaType, _, perr := mime.ParseMediaType(resp.Header.Get("Content-Type")); perr != nil || mediaType != "text/event-stream" {
		return resp, err
	}
	resp.Body = newSSEFilter(resp.Body)
	// The filtered body is shorter than what the server declared.
	resp.ContentLength = -1
	resp.Header.Del("Content-Length")
	return resp, err
}

// maxHeldSSEFields bounds what sseFilter holds back for one event before its
// first data line. Past it the fields are passed on as they are, leaving the
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
// ends without a data field. It reads in chunks of at most the bufio buffer
// size, deciding what to do with a line from its first chunk, so a long line
// costs no more memory than a short one. Data lines pass through as they
// arrive; the other fields of an event (event, id, retry) are held only until
// its first data line, since without one the event is dropped.
type sseFilter struct {
	src     io.ReadCloser
	r       *bufio.Reader
	out     []byte // filtered bytes not yet returned to the caller
	held    []byte // non-data fields of the current event, before any data
	hasData bool   // the current event has a data field
	midLine bool   // the last chunk read did not end its line
	action  lineAction
	err     error // sticky error from src, returned once out drains
}

func newSSEFilter(src io.ReadCloser) *sseFilter {
	return &sseFilter{src: src, r: bufio.NewReader(src)}
}

func (f *sseFilter) Read(p []byte) (int, error) {
	for len(f.out) == 0 {
		if f.err != nil {
			return 0, f.err
		}
		chunk, err := f.r.ReadSlice('\n')
		if len(chunk) > 0 {
			f.filter(chunk, err == nil)
		}
		if err != nil && err != bufio.ErrBufferFull {
			// At EOF, fields of an event that never got data are dropped.
			f.err = err
		}
	}
	n := copy(p, f.out)
	f.out = f.out[n:]
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
	content := bytes.TrimRight(chunk, "\r\n")
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
	name, _, _ := bytes.Cut(content, []byte(":"))
	if string(name) == "data" {
		f.flushHeld()
		return passLine
	}
	if f.hasData {
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
