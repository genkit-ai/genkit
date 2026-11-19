// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

func TestIsGemmaModelName(t *testing.T) {
	cases := map[string]bool{
		"gemma-4-31b-it":                          true,
		"gemma-4-26b-a4b-it":                      true,
		"gemma-5-future-it":                       true,
		"googleai/gemma-4-31b-it":                 true,
		"models/gemma-4-31b-it":                   true,
		"publishers/google/models/gemma-4-31b-it": true,
		"gemma":                     false,
		"gemmaX":                    false,
		"gemini-2.5-flash":          false,
		"googleai/gemini-2.5-flash": false,
		"publishers/google/models/gemini-2.5-flash":     false,
		"imagen-4.0-generate-001":                       false,
		"endpoints/123":                                 false,
		"projects/p/locations/l/endpoints/gemma-tuned1": false,
	}
	for name, want := range cases {
		if got := isGemmaModelName(name); got != want {
			t.Errorf("isGemmaModelName(%q) = %v, want %v", name, got, want)
		}
	}
}

// TestClassifyModelAgreesWithIsGemmaModelName pins that every spelling
// isGemmaModelName accepts is dispatched as a Gemini-family model, and that a
// name it rejects is not classified as Gemini only because of a "gemma" prefix.
func TestClassifyModelAgreesWithIsGemmaModelName(t *testing.T) {
	for _, name := range []string{
		"gemma-4-31b-it", "googleai/gemma-4-31b-it", "models/gemma-4-31b-it",
		"publishers/google/models/gemma-4-31b-it",
	} {
		if got := ClassifyModel(name); got != ModelTypeGemini {
			t.Errorf("ClassifyModel(%q) = %v, want ModelTypeGemini", name, got)
		}
	}
	for _, name := range []string{"gemma", "gemmaX"} {
		if got := ClassifyModel(name); got != ModelTypeUnknown {
			t.Errorf("ClassifyModel(%q) = %v, want ModelTypeUnknown", name, got)
		}
	}
}

// unregistered Gemma IDs resolve through the same generic Gemini defaults the
// JS plugin's gemma-generic uses, so no Gemma-specific claim is made for them.
func TestUnregisteredGemmaModelUsesGenericGeminiOptions(t *testing.T) {
	const name = "gemma-5-future-it"
	opts := GetModelOptions(name, googleAIProvider)
	if !hitUnknownModelFallback(opts, name) {
		t.Errorf("label = %q, want the raw ID for an unregistered model", opts.Label)
	}
	if opts.Supports != defaultGeminiOpts.Supports {
		t.Errorf("Supports = %#v, want the generic Gemini defaults", opts.Supports)
	}
}

var (
	sigA = []byte("sig-a")
	sigB = []byte("sig-b")
	sigC = []byte("sig-c")
)

// gemmaHistory builds a conversation whose model turns carry what Gemini
// returns: reasoning parts, and a thought signature on the first part emitted
// (here a tool request and a text part). It is built afresh each call so a
// test can compare the caller's copy against an untouched twin.
func gemmaHistory() []*ai.Message {
	signedCall := ai.NewToolRequestPart(&ai.ToolRequest{Name: "weather", Ref: "1", Input: map[string]any{"city": "Paris"}})
	signedCall.Metadata = map[string]any{"signature": append([]byte(nil), sigB...), "keep": "me"}
	signedText := ai.NewTextPart("It is sunny.")
	signedText.Metadata = map[string]any{"signature": append([]byte(nil), sigC...)}
	// A stream ends a text reply with an empty part that is only a signature.
	sigOnly := ai.NewTextPart("")
	sigOnly.Metadata = map[string]any{"signature": append([]byte(nil), sigA...)}
	return []*ai.Message{
		ai.NewUserTextMessage("What is the weather in Paris?"),
		ai.NewModelMessage(ai.NewReasoningPart("SECRET-THOUGHT-1", sigA), signedCall),
		ai.NewMessage(ai.RoleTool, nil, ai.NewToolResponsePart(&ai.ToolResponse{Name: "weather", Ref: "1", Output: map[string]any{"temp": 21}})),
		ai.NewModelMessage(ai.NewReasoningPart("SECRET-THOUGHT-ONLY-TURN", sigA)),
		ai.NewModelMessage(signedText, sigOnly),
		ai.NewUserTextMessage("And in Rome?"),
	}
}

// wireServer is a fake Gemini backend that records the bodies sent to
// generateContent / streamGenerateContent and to the cachedContents create.
type wireServer struct {
	*httptest.Server
	mu       sync.Mutex
	generate map[string]any
	cacheNew map[string]any
}

func newWireServer(t *testing.T) *wireServer {
	t.Helper()
	ws := &wireServer{}
	ws.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decode := func() map[string]any {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode %s: %v", r.URL.Path, err)
			}
			return body
		}
		ws.mu.Lock()
		defer ws.mu.Unlock()
		switch {
		case strings.Contains(r.URL.Path, "cachedContents") && r.Method == http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"cachedContents":[]}`)
		case strings.Contains(r.URL.Path, "cachedContents"):
			ws.cacheNew = decode()
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"name":"cachedContents/c1","displayName":%q}`, ws.cacheNew["displayName"])
		case strings.HasSuffix(r.URL.Path, ":streamGenerateContent"):
			ws.generate = decode()
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n")
		default:
			ws.generate = decode()
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
		}
	}))
	t.Cleanup(ws.Close)
	return ws
}

// flatParts returns every part of every content in a decoded request body,
// paired with the content's index.
func flatParts(body map[string]any) (parts []map[string]any, contentIdx []int) {
	contents, _ := body["contents"].([]any)
	for i, c := range contents {
		ps, _ := c.(map[string]any)["parts"].([]any)
		for _, p := range ps {
			parts = append(parts, p.(map[string]any))
			contentIdx = append(contentIdx, i)
		}
	}
	return
}

func countContents(body map[string]any) int {
	c, _ := body["contents"].([]any)
	return len(c)
}

func hasThoughts(body map[string]any) bool {
	parts, _ := flatParts(body)
	for _, p := range parts {
		if p["thought"] == true || p["thoughtSignature"] != nil || strings.Contains(fmt.Sprint(p["text"]), "SECRET-THOUGHT") {
			return true
		}
	}
	return false
}

func runWire(t *testing.T, client *genai.Client, model string, stream bool, input *ai.ModelRequest) *ai.ModelResponse {
	t.Helper()
	var cb func(context.Context, *ai.ModelResponseChunk) error
	if stream {
		cb = func(context.Context, *ai.ModelResponseChunk) error { return nil }
	}
	r, err := generate(context.Background(), client, model, input, &genai.GenerateContentConfig{}, cb)
	if err != nil {
		t.Fatalf("generate(%s, stream=%v): %v", model, stream, err)
	}
	return r
}

func TestGenerateStripsThoughtsForGemma(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			ws := newWireServer(t)
			client := newTestClient(t, ws.URL)
			input := &ai.ModelRequest{Messages: gemmaHistory()}
			orig := input.Messages
			r := runWire(t, client, "gemma-4-31b-it", stream, input)

			if hasThoughts(ws.generate) {
				t.Errorf("Gemma request carries thoughts or signatures: %v", ws.generate["contents"])
			}
			// The thoughts-only turn is skipped: 6 messages -> 5 contents.
			if got := countContents(ws.generate); got != 5 {
				t.Errorf("contents = %d, want 5 (the all-thoughts turn skipped)", got)
			}
			// The signed tool request survives with its call intact, so the
			// tool response that follows still has its functionCall.
			parts, idx := flatParts(ws.generate)
			var callIdx, respIdx = -1, -1
			for i, p := range parts {
				if fc, ok := p["functionCall"].(map[string]any); ok {
					if fc["name"] != "weather" || fmt.Sprint(fc["args"]) != "map[city:Paris]" {
						t.Errorf("functionCall = %v, want weather(city=Paris)", fc)
					}
					callIdx = idx[i]
				}
				if _, ok := p["functionResponse"]; ok {
					respIdx = idx[i]
				}
			}
			if callIdx < 0 || respIdx != callIdx+1 {
				t.Errorf("functionCall in content %d, functionResponse in content %d; want the response right after its call", callIdx, respIdx)
			}
			// The signed text survives as text.
			var texts []string
			for _, p := range parts {
				if s, ok := p["text"].(string); ok {
					texts = append(texts, s)
				}
			}
			if !contains(texts, "It is sunny.") {
				t.Errorf("signed text part was dropped: %q", texts)
			}

			// The caller's history is untouched, and the response still
			// carries the request as sent.
			if !reflect.DeepEqual(input.Messages, gemmaHistory()) {
				t.Error("generate mutated the caller's messages")
			}
			if &input.Messages[0] != &orig[0] || r.Request != input {
				t.Error("r.Request is not the caller's request")
			}
		})
	}
}

func TestGenerateKeepsThoughtsForGemini(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			ws := newWireServer(t)
			runWire(t, newTestClient(t, ws.URL), "gemini-2.5-flash", stream, &ai.ModelRequest{Messages: gemmaHistory()})
			parts, _ := flatParts(ws.generate)
			var thoughts, sigs int
			for _, p := range parts {
				if p["thought"] == true {
					thoughts++
				}
				if p["thoughtSignature"] != nil {
					sigs++
				}
			}
			if thoughts != 2 || sigs < 3 {
				t.Errorf("Gemini request kept %d thought parts and %d signatures, want 2 and at least 3", thoughts, sigs)
			}
		})
	}
}

// Gemma is served by the Gemini API only, so the stripping is scoped to that
// backend: the same model name on Vertex is left alone.
func TestGenerateKeepsThoughtsForGemmaOnVertex(t *testing.T) {
	ws := newWireServer(t)
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend:     genai.BackendVertexAI,
		Project:     "p",
		Location:    "us-central1",
		HTTPClient:  ws.Client(),
		HTTPOptions: genai.HTTPOptions{BaseURL: ws.URL},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	runWire(t, client, "gemma-4-31b-it", false, &ai.ModelRequest{Messages: gemmaHistory()})
	if !hasThoughts(ws.generate) {
		t.Error("Vertex request had its thoughts stripped; the Gemma handling is Gemini API only")
	}
}

func TestGenerateCachePathDoesNotSendThoughtsToGemma(t *testing.T) {
	// The cache marker is on the first model turn, which carries a thought.
	history := func() []*ai.Message {
		signed := ai.NewTextPart("cached answer")
		signed.Metadata = map[string]any{"signature": append([]byte(nil), sigC...)}
		return []*ai.Message{
			ai.NewUserTextMessage("a long document"),
			ai.NewModelMessage(ai.NewReasoningPart("SECRET-THOUGHT-CACHED", sigA), signed).WithCacheTTL(3600),
			ai.NewUserTextMessage("a follow-up question"),
		}
	}
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			ws := newWireServer(t)
			input := &ai.ModelRequest{Messages: history()}
			runWire(t, newTestClient(t, ws.URL), "gemma-4-31b-it", stream, input)

			if ws.cacheNew == nil {
				t.Fatal("no cache was created")
			}
			if hasThoughts(ws.cacheNew) {
				t.Errorf("the cached prefix carries thoughts: %v", ws.cacheNew["contents"])
			}
			if hasThoughts(ws.generate) {
				t.Errorf("the inline tail carries thoughts: %v", ws.generate["contents"])
			}
			var cached []string
			parts, _ := flatParts(ws.cacheNew)
			for _, p := range parts {
				if s, ok := p["text"].(string); ok {
					cached = append(cached, s)
				}
			}
			if !contains(cached, "cached answer") {
				t.Errorf("cached prefix = %q, want the signed text kept", cached)
			}
			if !reflect.DeepEqual(input.Messages, history()) {
				t.Error("generate mutated the caller's messages")
			}
		})
	}

	// The hash is over the stripped content: two histories that differ only in
	// their thoughts reuse one cache.
	t.Run("hash ignores thoughts", func(t *testing.T) {
		hashOf := func(thought string) any {
			ws := newWireServer(t)
			h := history()
			h[1].Content[0] = ai.NewReasoningPart(thought, sigA)
			runWire(t, newTestClient(t, ws.URL), "gemma-4-31b-it", false, &ai.ModelRequest{Messages: h})
			return ws.cacheNew["displayName"]
		}
		if a, b := hashOf("one"), hashOf("two"); a != b {
			t.Errorf("cache hash differs with the thoughts: %v vs %v", a, b)
		}
	})
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func TestStripGemmaThoughts(t *testing.T) {
	signed := func(p *ai.Part) *ai.Part {
		p.Metadata = map[string]any{"signature": []byte("sig"), "other": 1}
		return p
	}
	call := signed(ai.NewToolRequestPart(&ai.ToolRequest{Name: "t", Input: map[string]any{"a": 1}}))
	text := signed(ai.NewTextPart("hello"))
	emptySigned := signed(ai.NewTextPart(""))
	plain := ai.NewTextPart("plain")

	in := []*ai.Part{ai.NewReasoningPart("r", []byte("sig")), call, text, emptySigned, plain}
	got, changed := stripGemmaThoughts(in)
	if !changed {
		t.Fatal("changed = false, want true")
	}
	if len(got) != 3 || got[0].ToolRequest == nil || got[1].Text != "hello" || got[2] != plain {
		t.Fatalf("stripped parts = %+v, want [tool request, hello, plain]", got)
	}
	for _, p := range got[:2] {
		if _, ok := p.Metadata["signature"]; ok {
			t.Errorf("%+v still carries its signature", p)
		}
		if p.Metadata["other"] != 1 {
			t.Errorf("%+v lost its other metadata", p)
		}
	}
	// The originals keep their signature.
	if call.Metadata["signature"] == nil || text.Metadata["signature"] == nil {
		t.Error("stripGemmaThoughts mutated the caller's parts")
	}

	// A signature-only metadata map is dropped rather than left empty.
	only := ai.NewTextPart("x")
	only.Metadata = map[string]any{"signature": []byte("s")}
	if g, _ := stripGemmaThoughts([]*ai.Part{only}); g[0].Metadata != nil {
		t.Errorf("Metadata = %v, want nil", g[0].Metadata)
	}
}

func TestStripGemmaThoughtsNoOp(t *testing.T) {
	parts := []*ai.Part{ai.NewTextPart("a"), ai.NewTextPart("b")}
	got, changed := stripGemmaThoughts(parts)
	if changed || &got[0] != &parts[0] {
		t.Error("nothing to strip, but the slice was copied")
	}

	msgs := []*ai.Message{ai.NewUserTextMessage("a"), ai.NewModelMessage(ai.NewTextPart("b"))}
	out, changed := stripGemmaThoughtsFromMessages(msgs)
	if changed || &out[0] != &msgs[0] {
		t.Error("nothing to strip, but the message slice was copied")
	}

	// Only the changed message is copied.
	msgs = append(msgs, ai.NewModelMessage(ai.NewReasoningPart("r", []byte("s")), ai.NewTextPart("c")))
	out, changed = stripGemmaThoughtsFromMessages(msgs)
	if !changed || out[0] != msgs[0] || out[1] != msgs[1] || out[2] == msgs[2] {
		t.Error("want unchanged messages shared and the changed one copied")
	}
	if len(msgs[2].Content) != 2 {
		t.Error("the original message was modified")
	}
	if len(out[2].Content) != 1 || out[2].Content[0].Text != "c" {
		t.Errorf("stripped message content = %+v", out[2].Content)
	}
}
