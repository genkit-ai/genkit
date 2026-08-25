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
	"encoding/json"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestGetContentForCache_NoCacheMetadata(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{Text: "Hello"},
				},
			},
		},
	}
	gotContent, err := findCacheMarker(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotContent != nil {
		t.Errorf("expected nil content when no cache metadata, got: %#v", gotContent)
	}
}

func TestGetContentForCache_NoContentToCache(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Metadata: map[string]any{"cache": map[string]any{
					"ttlSeconds": int(160),
				}},
				// No text content
			},
		},
	}
	_, err := findCacheMarker(req)
	if err == nil {
		t.Fatalf("should fail due no text in message")
	}
}

// A long video or PDF is what caching pays off for most, and such a message
// holds no text at all, so the marker has to look at content rather than text.
func TestFindCacheMarker_MediaOnlyMessage(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewMessage(ai.RoleUser, nil, ai.NewMediaPart("video/mp4", "data:video/mp4;base64,AAAA")).WithCacheTTL(360),
			ai.NewUserTextMessage("Summarize the video."),
		},
	}

	settings, err := findCacheMarker(req)
	if err != nil {
		t.Fatalf("findCacheMarker rejected a media-only message: %v", err)
	}
	if settings == nil {
		t.Fatal("findCacheMarker returned no settings")
	}
	if settings.ttl != 360 {
		t.Errorf("ttl = %d, want 360", settings.ttl)
	}
	if settings.endIndex != 0 {
		t.Errorf("endIndex = %d, want 0", settings.endIndex)
	}
}

func TestGetContentForCache_Invalid(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleSystem,
				Content: []*ai.Part{
					{Text: "System instructions"},
				},
			},
			{
				Role:    ai.RoleUser,
				Content: []*ai.Part{{Text: "Hello user"}},
				Metadata: map[string]any{"cache": map[string]any{
					"ttlSeconds": 160,
				}},
			},
		},
	}
	err := validateContextCacheRequest(req, "gemini-2.5-fash")
	if err == nil {
		t.Fatal("expecting error, system instructions are not supported with Context Cache")
	}
}

func TestValidateContextCacheRequest_HasTools(t *testing.T) {
	req := &ai.ModelRequest{
		Tools: []*ai.ToolDefinition{{Name: "someTool"}},
	}
	err := validateContextCacheRequest(req, "gemini-2.5-flash")
	if err == nil {
		t.Fatal("expected error if Tools are present")
	}
	if !strings.Contains(err.Error(), invalidArgMessages.tools) {
		t.Errorf("expected error to contain %q, got %v", invalidArgMessages.tools, err)
	}
}

func TestValidateContextCacheRequest_Valid(t *testing.T) {
	req := &ai.ModelRequest{}
	err := validateContextCacheRequest(req, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("did not expect error, got: %v", err)
	}
}

func TestExtractCacheConfig_NoMetadata(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{{Text: "Hello"}}},
		},
	}
	cs, err := findCacheMarker(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if cs != nil {
		t.Fatalf("expecting cache settings to be nil, got %#v", cs)
	}
}

func TestExtractCacheConfig_MapTTL(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{Text: "Hello"},
				},
				Metadata: map[string]any{
					"cache": map[string]any{
						"ttlSeconds": int(123),
					},
				},
			},
		},
	}
	cs, err := findCacheMarker(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cs.endIndex != 0 {
		t.Errorf("expected endIndex=0, got %d", cs.endIndex)
	}
	if cs.ttl != 123 {
		t.Errorf("expected TTLSeconds=123, got %v", cs.ttl)
	}
}

func TestExtractCacheConfig_InvalidCacheType(t *testing.T) {
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{Text: "Hello"},
				},
				Metadata: map[string]any{
					"cache": []string{"not valid"},
				},
			},
		},
	}
	cs, err := findCacheMarker(req)
	if err == nil {
		t.Fatal("expected error for invalid cache type")
	}
	if cs != nil {
		t.Fatalf("expecting empty cache settings but got: %v", cs)
	}
}

// TestMessagesToCache pins that the cached span stops at the boundary, stays
// in chronological order (calculateCacheHash reads it in slice order, so the
// order is part of every live cache name), and is encoded exactly like the
// messages sent inline. The last part matters because the cached prefix is no
// longer sent inline as well: the copy stored on the resource is the only one
// the model ever sees.
func TestMessagesToCache(t *testing.T) {
	t.Run("chronological up to the boundary", func(t *testing.T) {
		msgs := []*ai.Message{
			ai.NewUserTextMessage("first"),
			ai.NewModelTextMessage("second"),
			ai.NewUserTextMessage("third, not cached"),
		}
		got, err := messagesToCache(msgs, 1)
		if err != nil {
			t.Fatalf("messagesToCache: %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("len = %d, want 2", len(got))
		}
		if got[0].Parts[0].Text != "first" || got[1].Parts[0].Text != "second" {
			t.Errorf("order = %q, %q; want first, second", got[0].Parts[0].Text, got[1].Parts[0].Text)
		}
	})

	t.Run("tool turns take the gemini role", func(t *testing.T) {
		msgs := []*ai.Message{
			ai.NewMessage(ai.RoleTool, nil, ai.NewToolResponsePart(&ai.ToolResponse{Name: "myTool", Output: "result"})),
			ai.NewUserTextMessage("and now?"),
		}
		got, err := messagesToCache(msgs, 0)
		if err != nil {
			t.Fatalf("messagesToCache: %v", err)
		}
		// The Gemini Content API takes only "user" and "model"; a raw "tool"
		// is rejected by Caches.Create.
		if got[0].Role != "user" {
			t.Errorf("cached role = %q, want %q", got[0].Role, "user")
		}
	})

	t.Run("contentless turns are skipped", func(t *testing.T) {
		// A blocked model turn replayed via resp.History() carries no content,
		// and the API rejects contents with zero parts.
		msgs := []*ai.Message{
			{Role: ai.RoleModel},
			ai.NewUserTextMessage("hi"),
		}
		got, err := messagesToCache(msgs, 1)
		if err != nil {
			t.Fatalf("messagesToCache: %v", err)
		}
		if len(got) != 1 {
			t.Fatalf("len = %d, want 1 (the contentless turn must be dropped)", len(got))
		}
		if got[0].Parts[0].Text != "hi" {
			t.Errorf("cached = %q, want %q", got[0].Parts[0].Text, "hi")
		}
	})
}

// TestCalculateCacheHash pins that the hash tells apart everything the cache
// lookup relies on it to tell apart. It is the only check that a named cache
// holds the request's own content, and the request no longer carries that
// content inline, so a collision silently answers from the wrong cache.
func TestCalculateCacheHash(t *testing.T) {
	hashOf := func(t *testing.T, model string, msgs ...*ai.Message) string {
		t.Helper()
		contents, err := messagesToCache(msgs, len(msgs)-1)
		if err != nil {
			t.Fatalf("messagesToCache: %v", err)
		}
		h, err := calculateCacheHash(contents, model)
		if err != nil {
			t.Fatalf("calculateCacheHash: %v", err)
		}
		return h
	}
	media := func(uri string) *ai.Message {
		return ai.NewMessage(ai.RoleUser, nil, ai.NewMediaPart("video/mp4", uri))
	}

	// URI media carries neither Text nor InlineData, the two fields the hash
	// used to fold in, so every such cache hashed alike.
	if a, b := hashOf(t, "m", media("gs://bucket/a.mp4")), hashOf(t, "m", media("gs://bucket/b.mp4")); a == b {
		t.Errorf("two different gs:// videos hash alike: %s", a)
	}
	// A cache resource belongs to the model that created it.
	if a, b := hashOf(t, "gemini-2.5-flash", ai.NewUserTextMessage("x")), hashOf(t, "gemini-2.5-pro", ai.NewUserTextMessage("x")); a == b {
		t.Errorf("same contents on two models hash alike: %s", a)
	}
	// Message boundaries have to survive: without them "ab" and "a"+"b" fold
	// into the same digest.
	if a, b := hashOf(t, "m", ai.NewUserTextMessage("ab")), hashOf(t, "m", ai.NewUserTextMessage("a"), ai.NewUserTextMessage("b")); a == b {
		t.Errorf("[ab] and [a b] hash alike: %s", a)
	}
	if a, b := hashOf(t, "m", ai.NewUserTextMessage("x")), hashOf(t, "m", ai.NewUserTextMessage("x")); a != b {
		t.Errorf("hash is not stable: %s != %s", a, b)
	}
}

func TestFindCacheMarker_NumericTTLForms(t *testing.T) {
	// ttlSeconds is an int when set from Go code but arrives as float64 or
	// json.Number after a JSON round-trip (dev UI, reflection server). All
	// numeric forms must work.
	cases := []struct {
		name string
		ttl  any
		want int
	}{
		{"int", int(160), 160},
		{"int64", int64(200), 200},
		{"float64", float64(300), 300},
		{"json.Number", json.Number("400"), 400},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role:    ai.RoleUser,
						Content: []*ai.Part{{Text: "Hello"}},
						Metadata: map[string]any{"cache": map[string]any{
							"ttlSeconds": tc.ttl,
						}},
					},
				},
			}
			cs, err := findCacheMarker(req)
			if err != nil {
				t.Fatalf("findCacheMarker: %v", err)
			}
			if cs == nil {
				t.Fatal("findCacheMarker = nil, want cache settings")
			}
			if cs.ttl != tc.want {
				t.Errorf("ttl = %d, want %d", cs.ttl, tc.want)
			}
		})
	}
}

func TestFindCacheMarker_JSONRoundTrip(t *testing.T) {
	// Simulates a request that passed through JSON serialization, the way
	// the reflection server delivers it.
	original := ai.NewUserTextMessage("cache me").WithCacheTTL(3600)
	b, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var roundTripped ai.Message
	if err := json.Unmarshal(b, &roundTripped); err != nil {
		t.Fatal(err)
	}

	cs, err := findCacheMarker(&ai.ModelRequest{Messages: []*ai.Message{&roundTripped}})
	if err != nil {
		t.Fatalf("findCacheMarker after JSON round-trip: %v", err)
	}
	if cs == nil || cs.ttl != 3600 {
		t.Fatalf("cache settings = %+v, want ttl 3600", cs)
	}
}

func TestFindCacheMarker_InvalidTTL(t *testing.T) {
	// A non-positive TTL would silently skip cache creation downstream, so
	// it must be rejected loudly, as must non-numeric values. 0.5 truncates
	// to 0 and lands in the same bucket.
	for name, ttl := range map[string]any{
		"non-numeric": true,
		"zero":        0,
		"negative":    -30,
		"fractional":  float64(0.5),
	} {
		t.Run(name, func(t *testing.T) {
			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role:    ai.RoleUser,
						Content: []*ai.Part{{Text: "Hello"}},
						Metadata: map[string]any{"cache": map[string]any{
							"ttlSeconds": ttl,
						}},
					},
				},
			}
			if _, err := findCacheMarker(req); err == nil {
				t.Fatalf("findCacheMarker = nil error, want error for ttlSeconds %v", ttl)
			}
		})
	}
}
