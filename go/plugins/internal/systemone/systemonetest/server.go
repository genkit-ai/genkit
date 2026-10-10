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

// Package systemonetest provides a fake System One endpoint for the tests
// of the plugins that serve decision models.
package systemonetest

import (
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"testing"
)

// Server answers whatever questions it is sent, so a test can check the
// whole path from a Go type to the wire and back. A choice is answered with
// its first option by name, a score with 1.3, a noul with 0.93, from the
// model jev-1.13.0 with 312 input tokens and 48 output tokens.
type Server struct {
	// Models is the reply to a GET, the model listing. When empty, a GET
	// fails with a server error.
	Models string
	// Usage, when set, is the usage of every answer, in place of 312
	// input tokens and 48 output tokens.
	Usage map[string]any
	// Wrap, when set, wraps every answer in a server's envelope.
	Wrap func(reply map[string]any) any

	mu       sync.Mutex
	requests []*http.Request
	bodies   []map[string]any
	listings []string
}

// ServeHTTP implements [http.Handler].
func (s *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method == http.MethodGet {
		s.mu.Lock()
		s.listings = append(s.listings, req.URL.RequestURI())
		s.mu.Unlock()
		if s.Models == "" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(w, s.Models)
		return
	}
	raw, _ := io.ReadAll(req.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	s.mu.Lock()
	s.requests = append(s.requests, req)
	s.bodies = append(s.bodies, body)
	s.mu.Unlock()

	// A server that wraps the native body, as Workers AI does for a
	// partner's model, carries the questions under input.
	native := body
	if input, ok := body["input"].(map[string]any); ok {
		native = input
	}
	questions, _ := native["questions"].(map[string]any)
	answers := map[string]any{}
	for id, raw := range questions {
		q, _ := raw.(map[string]any)
		switch q["type"] {
		case "choice":
			criteria, _ := q["criteria"].(map[string]any)
			keys := slices.Sorted(maps.Keys(criteria))
			if len(keys) == 0 {
				http.Error(w, `{"error":{"message":"a choice needs criteria"}}`, http.StatusBadRequest)
				return
			}
			probabilities := map[string]float64{}
			for i, k := range keys {
				probabilities[k] = 0.1
				if i == 0 {
					probabilities[k] = 1 - 0.1*float64(len(keys)-1)
				}
			}
			answers[id] = map[string]any{"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": 0.6}
		case "score":
			levels, _ := q["criteria"].([]any)
			legend := map[string]any{}
			for i, level := range levels {
				legend[strconv.Itoa(i)] = level
			}
			answers[id] = map[string]any{"type": "score", "score": 1.3, "legend": legend, "probabilities": map[string]float64{"0": 0, "1": 0.7, "2": 0.3}, "confidence": 0.54}
		case "noul":
			answers[id] = map[string]any{"type": "noul", "noul": 0.93}
		}
	}
	reply := map[string]any{
		"model":   "jev-1.13.0",
		"answers": answers,
		"usage":   map[string]any{"input_tokens": 312, "output_tokens": 48},
	}
	if s.Usage != nil {
		reply["usage"] = s.Usage
	}
	var out any = reply
	if s.Wrap != nil {
		out = s.Wrap(reply)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// Last is the last request posted and its JSON body. It fails the test
// when nothing was posted.
func (s *Server) Last(t *testing.T) (*http.Request, map[string]any) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.requests) == 0 {
		t.Fatal("the endpoint was never called")
	}
	return s.requests[len(s.requests)-1], s.bodies[len(s.bodies)-1]
}

// Listings are the paths, with their queries, of the listings asked for.
func (s *Server) Listings() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.listings)
}

// Calls is how many requests were posted.
func (s *Server) Calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}
