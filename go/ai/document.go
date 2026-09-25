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

package ai

import (
	"encoding/json"
	"maps"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// A Document is a piece of data that can be embedded, indexed, or retrieved.
// It includes metadata. It can contain multiple parts.
type Document struct {
	// The data that is part of this document.
	Content []*Part `json:"content,omitempty"`
	// The metadata for this document.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// A Part is one part of a [Document]. This may be plain text or it
// may be a URL (possibly a "data:" URL with embedded data).
type Part struct {
	Kind         PartKind       `json:"kind,omitempty"`
	ContentType  string         `json:"contentType,omitempty"`  // valid for kind==blob
	Text         string         `json:"text,omitempty"`         // valid for kind∈{text,blob}
	Data         any            `json:"data,omitempty"`         // valid for kind==data
	ToolRequest  *ToolRequest   `json:"toolRequest,omitempty"`  // valid for kind==partToolRequest
	ToolResponse *ToolResponse  `json:"toolResponse,omitempty"` // valid for kind==partToolResponse
	Resource     *ResourcePart  `json:"resource,omitempty"`     // valid for kind==partResource
	Custom       map[string]any `json:"custom,omitempty"`       // valid for plugin-specific custom parts
	Interrupt    *ToolInterrupt `json:"-"`                      // valid for kind==partToolRequest
	Restart      *ToolRestart   `json:"-"`                      // valid for kind==partToolRequest
	Metadata     map[string]any `json:"metadata,omitempty"`     // valid for all kinds
}

// ToolInterrupt is the interrupt state of a tool request [Part]. A non-nil
// Interrupt on a tool request part means the tool paused execution and returned
// control to the caller; the caller resolves it with [ToolAction.RestartWith]
// or [ToolAction.RespondWith].
//
// On the wire it is carried in the part's metadata map (under "interrupt", or
// "resolvedInterrupt" once resolved, plus "interruptedBy" when a hook raised
// it) for compatibility with the JS runtime;
// marshaling folds it in and unmarshaling lifts it back out. In process the
// state lives on [Part.Interrupt] alone: the generate loop and unmarshaling
// set the field and leave the metadata map to user and plugin metadata, so
// the key is not on Metadata in process. A part assembled with the key
// instead of the field reads as the same state through [Part.IsInterrupt]
// and [InterruptAs], while its field stays nil, so read the state through
// those rather than through the field.
type ToolInterrupt struct {
	// Data is the payload the tool interrupted with, e.g. the question it
	// needs answered. It must serialize to a JSON object (a struct or a map);
	// nil means the tool interrupted without data.
	Data any
	// Resolved reports whether the interrupt has been resolved (by a restart
	// that re-executed the tool or by a caller-provided response). A resolved
	// interrupt is kept for history; the part no longer awaits resolution.
	Resolved bool
	// RaisedBy names the WrapTool hook that raised the interrupt, as the
	// middleware's name (suffixed "#n" when the name repeats in the chain), and
	// is empty when the tool itself did. A restart answers the stage named
	// here: the hook reads the payload with
	// [github.com/firebase/genkit/go/ai/tool.ResumeData] and the tool then runs
	// as a fresh call. Carried on the wire as "interruptedBy".
	RaisedBy string
	// ReleasedBy names, in chain order, the WrapTool hooks that let the call
	// through before it interrupted. A restart that keeps the call's input
	// reports [github.com/firebase/genkit/go/ai/tool.Released] to exactly
	// these hooks, so a hook that approved the call is not asked again, while
	// a hook added to the chain since, or one facing a replaced input, decides
	// afresh. Carried on the wire as "releasedBy".
	ReleasedBy []string
}

// ToolRestart marks a tool request [Part] as a restart of an interrupted call,
// carrying the data the caller sends back to the tool when it re-executes.
//
// On the wire it is carried in the part's metadata map (under "resumed" and
// "replacedInput") for compatibility with the JS runtime; marshaling folds it
// in and unmarshaling lifts it back out. As with [ToolInterrupt], a part
// assembled with the keys reads as the same state through [Part.IsRestart]
// and the generate loop while its [Part.Restart] field stays nil.
type ToolRestart struct {
	// Resume is the payload the restarted tool reads from [ToolContext.Resumed],
	// e.g. the user's answer to the question the tool interrupted with. It must
	// serialize to a JSON object (a struct or a map); nil means a bare restart,
	// i.e. restarting is itself the approval.
	Resume any
	// OriginalInput preserves the tool's original input when the caller
	// provided a new one for re-execution (via [WithNewInput]). On the wire it
	// is carried under the metadata key "replacedInput" for compatibility with
	// the JS runtime.
	OriginalInput any
}

// Clone returns a shallow copy of the Part with its own Metadata and Custom
// maps and its own Interrupt and Restart state. Callers can add or remove map
// keys or flip interrupt state without mutating the original. When Data, or a
// payload of the interrupt or restart state, holds a map[string]any or []any
// (the common cases for data parts, e.g. A2UI envelopes, and for every payload
// that crossed the wire), the top-level container is cloned too so callers can
// mutate its keys/elements without disturbing the original; nested values are
// still shared by reference. Cloning both shapes (rather than only maps) keeps
// the isolation guarantee independent of whether a payload is an object or an
// array.
func (p *Part) Clone() *Part {
	if p == nil {
		return nil
	}
	cp := *p
	cp.Custom = maps.Clone(p.Custom)
	cp.Metadata = maps.Clone(p.Metadata)
	cp.Data = cloneContainer(p.Data)
	if p.Interrupt != nil {
		i := *p.Interrupt
		i.Data = cloneContainer(i.Data)
		i.ReleasedBy = slices.Clone(i.ReleasedBy)
		cp.Interrupt = &i
	}
	if p.Restart != nil {
		r := *p.Restart
		r.Resume = cloneContainer(r.Resume)
		r.OriginalInput = cloneContainer(r.OriginalInput)
		cp.Restart = &r
	}
	return &cp
}

// cloneContainer returns a copy of v's top-level container when v is a
// map[string]any or a []any, and v itself otherwise (a struct, a scalar, or
// nil). It is what [Part.Clone] applies to every payload it owns.
func cloneContainer(v any) any {
	switch d := v.(type) {
	case map[string]any:
		return maps.Clone(d)
	case []any:
		return slices.Clone(d)
	}
	return v
}

// Clone returns a shallow copy of the Message with its own Content slice
// and Metadata map. Callers can replace parts or add metadata keys without
// mutating the original.
func (m *Message) Clone() *Message {
	if m == nil {
		return nil
	}
	cp := *m
	cp.Content = slices.Clone(m.Content)
	cp.Metadata = maps.Clone(m.Metadata)
	return &cp
}

// PartKind is what a [Part] carries: text, media, a tool request, and so on.
type PartKind int8

const (
	PartText PartKind = iota
	PartMedia
	PartData
	PartToolRequest
	PartToolResponse
	PartCustom
	PartReasoning
	PartResource
)

// partKindNames maps each valid PartKind to its wire name, mirroring the field
// names of the JS part union. Shared by [PartKind.String] and [Part.Validate]
// so the two cannot drift.
var partKindNames = map[PartKind]string{
	PartText:         "text",
	PartMedia:        "media",
	PartData:         "data",
	PartToolRequest:  "toolRequest",
	PartToolResponse: "toolResponse",
	PartCustom:       "custom",
	PartReasoning:    "reasoning",
	PartResource:     "resource",
}

// String returns the wire name of the part kind (e.g. "toolRequest"), or
// "unknown" for a value outside the defined kinds.
func (k PartKind) String() string {
	if name, ok := partKindNames[k]; ok {
		return name
	}
	return "unknown"
}

// NewTextPart returns a Part containing text.
func NewTextPart(text string) *Part {
	return &Part{Kind: PartText, ContentType: "plain/text", Text: text}
}

// NewJSONPart returns a Part containing JSON.
func NewJSONPart(text string) *Part {
	return &Part{Kind: PartText, ContentType: "application/json", Text: text}
}

// NewMediaPart returns a Part containing structured data described
// by the given mimeType.
func NewMediaPart(mimeType, contents string) *Part {
	return &Part{Kind: PartMedia, ContentType: mimeType, Text: contents}
}

// NewDataPart returns a Part containing arbitrary structured data. The value is
// serialized to the part's `data` field as-is, so it may be a string, a map, a
// slice, or any other JSON-serializable value.
func NewDataPart(data any) *Part {
	return &Part{Kind: PartData, Data: data}
}

// NewToolRequestPart returns a Part containing a request from
// the model to the client to run a Tool.
// (Only genkit plugins should need to use this function.)
func NewToolRequestPart(r *ToolRequest) *Part {
	return &Part{Kind: PartToolRequest, ToolRequest: r}
}

// NewToolResponsePart returns a Part containing the results
// of applying a Tool that the model requested.
func NewToolResponsePart(r *ToolResponse) *Part {
	return &Part{Kind: PartToolResponse, ToolResponse: r}
}

// NewResponseForToolRequest returns a Part containing the results
// of executing the tool request part.
func NewResponseForToolRequest(p *Part, output any) *Part {
	if !p.IsToolRequest() {
		return nil
	}
	return &Part{Kind: PartToolResponse, ToolResponse: &ToolResponse{
		Name:   p.ToolRequest.Name,
		Ref:    p.ToolRequest.Ref,
		Output: output,
	}}
}

// NewCustomPart returns a Part containing custom plugin-specific data.
func NewCustomPart(customData map[string]any) *Part {
	return &Part{Kind: PartCustom, Custom: customData}
}

// NewReasoningPart returns a Part containing reasoning text
func NewReasoningPart(text string, signature []byte) *Part {
	p := &Part{
		Kind:        PartReasoning,
		ContentType: "plain/text",
		Text:        text,
	}
	if len(signature) > 0 {
		// Cloned because the part outlives the call: [Part.Clone] copies the
		// metadata map but not the bytes under it, so a caller reusing a
		// buffer across chunks would mutate signatures already handed off.
		p.Metadata = map[string]any{"signature": slices.Clone(signature)}
	}
	return p
}

// NewResourcePart returns a Part containing a resource reference.
func NewResourcePart(uri string) *Part {
	return &Part{Kind: PartResource, Resource: &ResourcePart{Uri: uri}}
}

// IsText reports whether the [Part] contains plain text.
func (p *Part) IsText() bool {
	return p != nil && p.Kind == PartText
}

// IsMedia reports whether the [Part] contains structured media data.
func (p *Part) IsMedia() bool {
	return p != nil && p.Kind == PartMedia
}

// IsData reports whether the [Part] contains unstructured data.
func (p *Part) IsData() bool {
	return p != nil && p.Kind == PartData
}

// DataString returns a data part's payload as a string: the string as-is if
// Data already holds one (e.g. a "data:" URI), otherwise its JSON encoding.
// This is the single place provider converters and [github.com/firebase/genkit/go/plugins/internal/uri.Data]
// should read a data part's payload from, so a structured Data value (a
// map/slice, as [NewDataPart] and the A2UI middleware now produce) yields the
// same answer everywhere instead of failing one converter and silently
// dropping in another. Returns "" for a nil Part or nil Data.
func (p *Part) DataString() string {
	if p == nil || p.Data == nil {
		return ""
	}
	if s, ok := p.Data.(string); ok {
		return s
	}
	b, err := json.Marshal(p.Data)
	if err != nil {
		return ""
	}
	return string(b)
}

// IsToolRequest reports whether the [Part] contains a request to run a tool.
func (p *Part) IsToolRequest() bool {
	return p != nil && p.Kind == PartToolRequest
}

// IsToolResponse reports whether the [Part] contains the result of running a tool.
func (p *Part) IsToolResponse() bool {
	return p != nil && p.Kind == PartToolResponse
}

// IsInterrupt reports whether the [Part] contains a tool request whose
// interrupt is awaiting resolution. Resolved interrupts are kept on the part
// (see [ToolInterrupt.Resolved]) but no longer count.
func (p *Part) IsInterrupt() bool {
	it := p.interruptState()
	return it != nil && !it.Resolved
}

// IsRestart reports whether the [Part] contains a tool request that restarts an
// interrupted call.
func (p *Part) IsRestart() bool {
	return p.restartState() != nil
}

// IsPartial reports whether the [Part] contains a partial tool response
// streamed during tool execution (e.g., a progress update).
func (p *Part) IsPartial() bool {
	return p != nil && p.IsToolResponse() && p.Metadata != nil && p.Metadata["partial"] == true
}

// NewPartialToolResponsePart returns a [Part] containing a partial tool response.
// Partial tool responses are streamed during tool execution for client-side
// display (e.g., progress indicators) and are not included in conversation history.
func NewPartialToolResponsePart(r *ToolResponse) *Part {
	p := NewToolResponsePart(r)
	p.Metadata = map[string]any{"partial": true}
	return p
}

// IsCustom reports whether the [Part] contains custom plugin-specific data.
func (p *Part) IsCustom() bool {
	return p != nil && p.Kind == PartCustom
}

// IsReasoning reports whether the [Part] contains a reasoning text
func (p *Part) IsReasoning() bool {
	return p != nil && p.Kind == PartReasoning
}

// IsImage reports whether the [Part] contains an image.
func (p *Part) IsImage() bool {
	if p == nil || !p.IsMedia() {
		return false
	}
	return IsImageContentType(p.ContentType) || strings.HasPrefix(p.Text, "data:image/")
}

// IsVideo reports whether the [Part] contains a video.
func (p *Part) IsVideo() bool {
	if p == nil || !p.IsMedia() {
		return false
	}
	return IsVideoContentType(p.ContentType) || strings.HasPrefix(p.Text, "data:video/")
}

// IsAudio reports whether the [Part] contains an audio file.
func (p *Part) IsAudio() bool {
	if p == nil || !p.IsMedia() {
		return false
	}
	return IsAudioContentType(p.ContentType) || strings.HasPrefix(p.Text, "data:audio/")
}

// IsResource reports whether the [Part] contains a resource reference.
func (p *Part) IsResource() bool {
	return p != nil && p.Kind == PartResource
}

// MarshalJSON is called by the JSON marshaler to write out a Part.
func (p *Part) MarshalJSON() ([]byte, error) {
	if p == nil {
		return nil, status.Errorf(ErrInvalidPart, "part is nil")
	}

	// This is not handled by the schema generator because
	// Part is defined in TypeScript as a union.
	meta := p.wireMetadata()
	switch p.Kind {
	case PartText:
		v := textPart{
			Text:     p.Text,
			Metadata: meta,
		}
		return json.Marshal(v)
	case PartMedia:
		v := mediaPart{
			Media: &Media{
				ContentType: p.ContentType,
				Url:         p.Text,
			},
			Metadata: meta,
		}
		return json.Marshal(v)
	case PartData:
		v := dataPart{
			Data:     p.Data,
			Metadata: meta,
		}
		return json.Marshal(v)
	case PartToolRequest:
		v := toolRequestPart{
			ToolRequest: p.ToolRequest,
			Metadata:    meta,
		}
		return json.Marshal(v)
	case PartToolResponse:
		v := toolResponsePart{
			ToolResponse: p.ToolResponse,
			Metadata:     meta,
		}
		return json.Marshal(v)
	case PartResource:
		v := resourcePart{
			Resource: p.Resource,
			Metadata: meta,
		}
		return json.Marshal(v)
	case PartCustom:
		v := customPart{
			Custom:   p.Custom,
			Metadata: meta,
		}
		return json.Marshal(v)
	case PartReasoning:
		v := reasoningPart{
			Reasoning: p.Text,
			Metadata:  meta,
		}
		return json.Marshal(v)
	default:
		return nil, status.Errorf(ErrInvalidPart, "invalid part kind %v", p.Kind)
	}
}

// Metadata keys of the interrupt and resume wire contract, mirroring the JS
// runtime so that messages cross runtimes intact. In memory this state lives
// on the typed [Part] fields Interrupt and Restart: marshaling folds it into
// these keys and unmarshaling lifts it back out, and a tool request part
// hand-assembled with the keys instead of the fields reads as the same state
// through [Part.interruptState] and [Part.restartState].
const (
	// metaInterrupt marks a tool request part as interrupted. Holds the
	// interrupt data object, or true for a bare interrupt.
	metaInterrupt = "interrupt"
	// metaResolvedInterrupt preserves the original interrupt data on a tool
	// request part once its interrupt has been resolved.
	metaResolvedInterrupt = "resolvedInterrupt"
	// metaResumed marks a tool request part as a restart of an interrupted
	// call. Holds the resume data object, or true for a bare restart. Also
	// used on tool message metadata to carry resume metadata.
	metaResumed = "resumed"
	// metaReplacedInput preserves the original input on a restart part when
	// the caller replaced it.
	metaReplacedInput = "replacedInput"
	// metaInterruptedBy names, on an interrupted tool request part, the
	// WrapTool hook that raised the interrupt (ToolInterrupt.RaisedBy);
	// absent when the tool itself did. A Go-only key: the JS runtime has no
	// tool hooks and carries it through untouched.
	metaInterruptedBy = "interruptedBy"
	// metaReleasedBy lists, on an interrupted tool request part, the WrapTool
	// hooks that let the call through before it interrupted
	// (ToolInterrupt.ReleasedBy). A Go-only key, like metaInterruptedBy.
	metaReleasedBy = "releasedBy"
	// metaInterruptResponse marks a caller-provided tool response part that
	// resolves an interrupt in place of re-executing the tool.
	metaInterruptResponse = "interruptResponse"
)

// wireMetadata returns the metadata map to serialize for the Part, folding the
// typed Interrupt and Restart fields into the wire keys. The Part's own
// Metadata map is never mutated.
func (p *Part) wireMetadata() map[string]any {
	if p.Interrupt == nil && p.Restart == nil {
		return p.Metadata
	}
	m := maps.Clone(p.Metadata)
	if m == nil {
		m = make(map[string]any, 2)
	}
	if it := p.Interrupt; it != nil {
		// The typed state is authoritative: a stale key of the other
		// resolution state, left by a part assembled with the keys and then
		// resolved on the field, would otherwise read back as the state.
		key, stale := metaInterrupt, metaResolvedInterrupt
		if it.Resolved {
			key, stale = stale, key
		}
		m[key] = orTrue(it.Data)
		delete(m, stale)
		if it.RaisedBy != "" {
			m[metaInterruptedBy] = it.RaisedBy
		} else {
			delete(m, metaInterruptedBy)
		}
		if len(it.ReleasedBy) > 0 {
			m[metaReleasedBy] = it.ReleasedBy
		} else {
			delete(m, metaReleasedBy)
		}
	}
	if rs := p.Restart; rs != nil {
		m[metaResumed] = orTrue(rs.Resume)
		if rs.OriginalInput != nil {
			m[metaReplacedInput] = rs.OriginalInput
		} else {
			delete(m, metaReplacedInput)
		}
	}
	return m
}

// orTrue encodes an optional payload for the wire: a nil payload, a nil map
// or pointer included, is carried as the JSON literal true (a "bare"
// interrupt or restart), matching the JS runtime, rather than as null, which
// reads back as no state.
func orTrue(v any) any {
	if base.IsNil(v) {
		return true
	}
	return v
}

// wirePayload decodes a wire marker written by orTrue, the way the JS runtime
// reads these keys, by truthiness: an absent key, null, false, "" and 0 mean
// no state, true means state with no payload, and any other value is the
// payload itself. A part that carries "interrupt": null therefore reads as a plain
// tool request, not as a bare interrupt.
func wirePayload(v any) (payload any, set bool) {
	switch b := v.(type) {
	case nil:
		return nil, false
	case bool:
		return nil, b
	case string:
		return v, b != ""
	case float64:
		return v, b != 0
	}
	return v, true
}

// wireStrings reads a list of strings a wire key carries: a []string set in
// process or the []any a JSON decode yields. Entries that are not strings are
// dropped, and an empty or absent list is nil.
func wireStrings(v any) []string {
	switch l := v.(type) {
	case []string:
		if len(l) == 0 {
			return nil
		}
		return slices.Clone(l)
	case []any:
		var out []string
		for _, e := range l {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// interruptState returns the interrupt state of a tool request part: the
// typed field, or the state a part hand-assembled with the wire keys
// describes. Every reader of interrupt state goes through it, so such a part
// behaves like one built by the loop, without being copied or mutated. Nil
// for any other part, and for a tool request part with no [ToolRequest]: it
// names no tool, so no verb could act on it, and [Part.Validate] reports it.
func (p *Part) interruptState() *ToolInterrupt {
	if !p.IsToolRequest() || p.ToolRequest == nil {
		return nil
	}
	if p.Interrupt != nil {
		return p.Interrupt
	}
	raisedBy, _ := p.Metadata[metaInterruptedBy].(string)
	releasedBy := wireStrings(p.Metadata[metaReleasedBy])
	if v, ok := wirePayload(p.Metadata[metaInterrupt]); ok {
		return &ToolInterrupt{Data: v, RaisedBy: raisedBy, ReleasedBy: releasedBy}
	}
	if v, ok := wirePayload(p.Metadata[metaResolvedInterrupt]); ok {
		return &ToolInterrupt{Data: v, Resolved: true, RaisedBy: raisedBy, ReleasedBy: releasedBy}
	}
	return nil
}

// restartState is [Part.interruptState] for the restart state. A part marked
// "resumed": false is not a restart: the tool re-executes without a resume
// payload, as it would for a request the model made afresh.
func (p *Part) restartState() *ToolRestart {
	if !p.IsToolRequest() || p.ToolRequest == nil {
		return nil
	}
	if p.Restart != nil {
		return p.Restart
	}
	resume, resumed := wirePayload(p.Metadata[metaResumed])
	original := p.Metadata[metaReplacedInput]
	if !resumed && original == nil {
		return nil
	}
	return &ToolRestart{Resume: resume, OriginalInput: original}
}

// liftWireMetadata moves the interrupt and restart state a tool request part
// carries in its wire keys onto the typed fields, leaving the map to user and
// plugin metadata. Typed state already present is kept.
func (p *Part) liftWireMetadata() {
	if !p.IsToolRequest() || p.Metadata == nil {
		return
	}
	p.Interrupt = p.interruptState()
	p.Restart = p.restartState()
	p.Metadata = stripWireKeys(p.Metadata)
}

// stripWireKeys deletes the wire keys from m in place and returns m, or nil
// when nothing is left.
func stripWireKeys(m map[string]any) map[string]any {
	for _, key := range [...]string{metaInterrupt, metaResolvedInterrupt, metaInterruptedBy, metaReleasedBy, metaResumed, metaReplacedInput} {
		delete(m, key)
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// typedClone returns a copy of p with its wire-key state lifted onto the typed
// fields, for the loop to mark interrupted or resolved without mutating the
// caller's part.
func (p *Part) typedClone() *Part {
	cp := p.Clone()
	cp.liftWireMetadata()
	return cp
}

// Validate checks that the Part's fields are consistent with its Kind: that the
// kind's own payload field is set and that no field belonging to another kind
// is set (e.g. no ToolResponse on a tool request part, no Interrupt on a text
// part). It reports the first inconsistency found.
//
// A restart part may still carry the interrupt it resolves: that is the shape
// the JS runtime's restartTool builds, and the restart supersedes it.
func (p *Part) Validate() error {
	if p == nil {
		return status.Errorf(ErrInvalidPart, "part is nil")
	}
	if _, ok := partKindNames[p.Kind]; !ok {
		return status.Errorf(ErrInvalidPart, "invalid part kind %d", int8(p.Kind))
	}
	fields := []struct {
		name    string
		set     bool
		validOn bool
	}{
		// Text parts carry a content type too ("plain/text", "application/json"),
		// so ContentType is valid on every kind that has text or media.
		{"Text", p.Text != "", p.Kind == PartText || p.Kind == PartMedia || p.Kind == PartReasoning},
		{"ContentType", p.ContentType != "", p.Kind == PartText || p.Kind == PartMedia || p.Kind == PartReasoning},
		{"Data", p.Data != nil, p.Kind == PartData},
		{"ToolRequest", p.ToolRequest != nil, p.Kind == PartToolRequest},
		{"ToolResponse", p.ToolResponse != nil, p.Kind == PartToolResponse},
		{"Resource", p.Resource != nil, p.Kind == PartResource},
		{"Custom", p.Custom != nil, p.Kind == PartCustom},
		{"Interrupt", p.Interrupt != nil, p.Kind == PartToolRequest},
		{"Restart", p.Restart != nil, p.Kind == PartToolRequest},
	}
	for _, f := range fields {
		if f.set && !f.validOn {
			return status.Errorf(ErrInvalidPart, "field %s is not valid on a %s part", f.name, p.Kind)
		}
	}
	required := map[PartKind]struct {
		name string
		set  bool
	}{
		PartToolRequest:  {"ToolRequest", p.ToolRequest != nil},
		PartToolResponse: {"ToolResponse", p.ToolResponse != nil},
		PartResource:     {"Resource", p.Resource != nil},
		PartCustom:       {"Custom", p.Custom != nil},
	}
	if r, ok := required[p.Kind]; ok && !r.set {
		return status.Errorf(ErrInvalidPart, "field %s is required on a %s part", r.name, p.Kind)
	}
	return nil
}

type partSchema struct {
	Text         string         `json:"text,omitempty" yaml:"text,omitempty"`
	Media        *Media         `json:"media,omitempty" yaml:"media,omitempty"`
	Data         any            `json:"data,omitempty" yaml:"data,omitempty"`
	ToolRequest  *ToolRequest   `json:"toolRequest,omitempty" yaml:"toolRequest,omitempty"`
	ToolResponse *ToolResponse  `json:"toolResponse,omitempty" yaml:"toolResponse,omitempty"`
	Resource     *ResourcePart  `json:"resource,omitempty" yaml:"resource,omitempty"`
	Custom       map[string]any `json:"custom,omitempty" yaml:"custom,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
	// Reasoning is a pointer so that a reasoning part with empty text stays a
	// reasoning part: what marks the kind is the key being present, not the
	// text being non-empty.
	Reasoning *string `json:"reasoning,omitempty" yaml:"reasoning,omitempty"`
}

// unmarshalPartFromSchema updates Part p based on the schema s.
func (p *Part) unmarshalPartFromSchema(s partSchema) {
	switch {
	case s.Media != nil:
		p.Kind = PartMedia
		p.Text = s.Media.Url
		p.ContentType = s.Media.ContentType
	case s.ToolRequest != nil:
		p.Kind = PartToolRequest
		p.ToolRequest = s.ToolRequest
	case s.ToolResponse != nil:
		p.Kind = PartToolResponse
		p.ToolResponse = s.ToolResponse
	case s.Resource != nil:
		p.Kind = PartResource
		p.Resource = s.Resource
	case s.Custom != nil:
		p.Kind = PartCustom
		p.Custom = s.Custom
	case s.Reasoning != nil:
		p.Kind = PartReasoning
		p.Text = *s.Reasoning
		p.ContentType = "plain/text"
	case s.Data != nil:
		p.Kind = PartData
		p.Data = s.Data
	default:
		// Note: if part is completely empty, we use text by default.
		p.Kind = PartText
		p.Text = s.Text
		p.ContentType = ""
	}
	p.Metadata = s.Metadata
	// Lifting keeps typed state already on the part, so state left by an
	// earlier decode into the same Part is cleared first.
	p.Interrupt, p.Restart = nil, nil
	p.liftWireMetadata()
}

// UnmarshalJSON is called by the JSON unmarshaler to read a Part.
func (p *Part) UnmarshalJSON(b []byte) error {
	var s partSchema
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	p.unmarshalPartFromSchema(s)
	return nil
}

// UnmarshalYAML implements goccy/go-yaml library's InterfaceUnmarshaler interface.
func (p *Part) UnmarshalYAML(unmarshal func(any) error) error {
	var s partSchema
	if err := unmarshal(&s); err != nil {
		return err
	}
	p.unmarshalPartFromSchema(s)
	return nil
}

// JSONSchemaAlias tells the JSON schema reflection code to use a different
// type for the schema for this type. This is needed because the JSON
// marshaling of Part uses a schema that matches the TypeScript code,
// rather than the natural JSON marshaling. This matters because the
// current JSON validation code works by marshaling the JSON.
func (Part) JSONSchemaAlias() any {
	return partSchema{}
}

// DocumentFromText returns a [Document] containing a single plain text part.
// This takes ownership of the metadata map.
func DocumentFromText(text string, metadata map[string]any) *Document {
	return &Document{
		Content: []*Part{
			{
				Kind: PartText,
				Text: text,
			},
		},
		Metadata: metadata,
	}
}

// IsImageContentType checks if the content type represents an image.
func IsImageContentType(contentType string) bool {
	return strings.HasPrefix(contentType, "image/") || strings.HasPrefix(contentType, "data:image/")
}

// IsVideoContentType checks if the content type represents a video.
func IsVideoContentType(contentType string) bool {
	return strings.HasPrefix(contentType, "video/") || strings.HasPrefix(contentType, "data:video/")
}

// IsAudioContentType checks if the content type represents an audio file.
func IsAudioContentType(contentType string) bool {
	return strings.HasPrefix(contentType, "audio/") || strings.HasPrefix(contentType, "data:audio/")
}
