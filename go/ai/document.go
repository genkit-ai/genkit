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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core"
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
	ContentType  string         `json:"contentType,omitempty"`  // valid for kind∈{media,data}
	Text         string         `json:"text,omitempty"`         // valid for kind∈{text,media,data,reasoning}
	ToolRequest  *ToolRequest   `json:"toolRequest,omitempty"`  // valid for kind==toolRequest
	ToolResponse *ToolResponse  `json:"toolResponse,omitempty"` // valid for kind==toolResponse
	Resource     *ResourcePart  `json:"resource,omitempty"`     // valid for kind==resource
	Custom       map[string]any `json:"custom,omitempty"`       // valid for plugin-specific custom parts
	Interrupt    *ToolInterrupt `json:"-"`                      // valid for kind==toolRequest
	Restart      *ToolRestart   `json:"-"`                      // valid for kind==toolRequest
	Metadata     map[string]any `json:"metadata,omitempty"`     // valid for all kinds
}

// ToolInterrupt is the interrupt state of a tool request [Part]. A non-nil
// Interrupt on a tool request part means the tool paused execution and
// returned control to the caller; the caller resolves it with [Part]s built by
// the tool package's Restart or Respond verbs.
//
// On the wire it is carried in the part's metadata map (under "interrupt", or
// "resolvedInterrupt" once resolved) for compatibility with the JS runtime;
// marshaling folds it in and unmarshaling lifts it back out.
type ToolInterrupt struct {
	// Data is the payload the tool interrupted with, e.g. the question it
	// needs answered. It must serialize to a JSON object (a struct or a map);
	// nil means the tool interrupted without data.
	Data any
	// Resolved reports whether the interrupt has been resolved (by a restart
	// that re-executed the tool or by a caller-provided response). A resolved
	// interrupt is kept for history; the part no longer awaits resolution.
	Resolved bool
}

// ToolRestart marks a tool request [Part] as a restart of an interrupted
// call, carrying the data the caller sends back to the tool when it
// re-executes.
//
// On the wire it is carried in the part's metadata map (under "resumed" and
// "replacedInput") for compatibility with the JS runtime; marshaling folds it
// in and unmarshaling lifts it back out.
type ToolRestart struct {
	// Resume is the payload delivered to the tool function's resume
	// parameter, e.g. the user's answer to the question the tool interrupted
	// with. It must serialize to a JSON object (a struct or a map); nil means
	// a bare restart, i.e. restarting is itself the approval.
	Resume any
	// OriginalInput preserves the tool's original input when the caller
	// provided a new one for re-execution (via [WithNewInput]). On the wire it
	// is carried under the metadata key "replacedInput" for compatibility with
	// the JS runtime.
	OriginalInput any
}

// Clone returns a shallow copy of the Part with its own Metadata and Custom
// maps and its own Interrupt and Restart state. Callers can add or remove map
// keys or flip interrupt state without mutating the original.
func (p *Part) Clone() *Part {
	if p == nil {
		return nil
	}
	cp := *p
	cp.Custom = maps.Clone(p.Custom)
	cp.Metadata = maps.Clone(p.Metadata)
	if p.Interrupt != nil {
		i := *p.Interrupt
		cp.Interrupt = &i
	}
	if p.Restart != nil {
		r := *p.Restart
		cp.Restart = &r
	}
	return &cp
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

// Part metadata keys and values shared with the JS runtime. Unlike the lifted
// interrupt state, parts carrying these keep them in the Metadata map, since
// plugins, middleware, and user code match on them directly.
const (
	// PartMetaPurpose marks why a part was injected into the request.
	PartMetaPurpose = "purpose"
	// PartPurposeOutput marks an injected output-format instruction part.
	PartPurposeOutput = "output"
	// PartPurposeContext marks an injected grounding-context part. Place a
	// part with this purpose and [PartMetaPending] to control where retrieved
	// documents are inserted.
	PartPurposeContext = "context"
	// PartMetaPending marks a placeholder part whose content has not been
	// filled in yet.
	PartMetaPending = "pending"
	// PartMetaThoughtSignature carries a model thought signature that
	// round-trips to the model on subsequent requests. Access it with
	// [Part.ThoughtSignature] and [Part.SetThoughtSignature], which handle
	// the raw-bytes vs base64-string encoding.
	PartMetaThoughtSignature = "thoughtSignature"
)

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

// partKindNames maps each valid PartKind to its wire name, mirroring the
// field names of the JS part union. Shared by [PartKind.String] and
// [Part.Validate] so the two cannot drift.
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
	return &Part{Kind: PartText, Text: text}
}

// NewJSONPart returns a Part containing JSON.
func NewJSONPart(text string) *Part {
	return &Part{Kind: PartText, Text: text}
}

// NewMediaPart returns a Part containing structured data described
// by the given mimeType.
func NewMediaPart(mimeType, contents string) *Part {
	return &Part{Kind: PartMedia, ContentType: mimeType, Text: contents}
}

// NewDataPart returns a Part containing raw string data.
func NewDataPart(contents string) *Part {
	return &Part{Kind: PartData, Text: contents}
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
// of executing the tool request part. Returns an error if p is not a
// tool request part.
func NewResponseForToolRequest(p *Part, output any) (*Part, error) {
	if !p.IsToolRequest() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.NewResponseForToolRequest: part is not a tool request")
	}
	return &Part{Kind: PartToolResponse, ToolResponse: &ToolResponse{
		Name:   p.ToolRequest.Name,
		Ref:    p.ToolRequest.Ref,
		Output: output,
	}}, nil
}

// NewCustomPart returns a Part containing custom plugin-specific data.
func NewCustomPart(customData map[string]any) *Part {
	return &Part{Kind: PartCustom, Custom: customData}
}

// NewReasoningPart returns a Part containing reasoning text and the model's
// thought signature, which round-trips to the model on subsequent requests.
func NewReasoningPart(text string, signature []byte) *Part {
	p := &Part{Kind: PartReasoning, Text: text}
	p.SetThoughtSignature(signature)
	return p
}

// ThoughtSignature returns the model thought signature attached to the Part,
// or nil if there is none. Signatures live in the part's metadata under
// [PartMetaThoughtSignature] (shared with the JS runtime); this accessor
// decodes both the in-memory raw bytes and the base64 string a JSON round
// trip produces.
func (p *Part) ThoughtSignature() []byte {
	if p == nil || p.Metadata == nil {
		return nil
	}
	switch sig := p.Metadata[PartMetaThoughtSignature].(type) {
	case []byte:
		return sig
	case string:
		if decoded, err := base64.StdEncoding.DecodeString(sig); err == nil {
			return decoded
		}
	}
	return nil
}

// SetThoughtSignature attaches a model thought signature to the Part,
// storing it in the part's metadata under [PartMetaThoughtSignature]. An
// empty signature removes the entry.
func (p *Part) SetThoughtSignature(signature []byte) {
	if len(signature) == 0 {
		delete(p.Metadata, PartMetaThoughtSignature)
		return
	}
	if p.Metadata == nil {
		p.Metadata = make(map[string]any, 1)
	}
	p.Metadata[PartMetaThoughtSignature] = signature
}

// NewResourcePart returns a Part containing a resource reference.
func NewResourcePart(uri string) *Part {
	return &Part{Kind: PartResource, Resource: &ResourcePart{URI: uri}}
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
	return p != nil && p.IsToolRequest() && p.Interrupt != nil && !p.Interrupt.Resolved
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
		return nil, fmt.Errorf("part is nil")
	}

	// This is not handled by the schema generator because
	// Part is defined in TypeScript as a union.
	switch p.Kind {
	case PartText:
		v := textPart{
			Text:     p.Text,
			Metadata: p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartMedia:
		v := mediaPart{
			Media: &Media{
				ContentType: p.ContentType,
				URL:         p.Text,
			},
			Metadata: p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartData:
		v := dataPart{
			Data:     p.Text,
			Metadata: p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartToolRequest:
		v := toolRequestPart{
			ToolRequest: p.ToolRequest,
			Metadata:    p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartToolResponse:
		v := toolResponsePart{
			ToolResponse: p.ToolResponse,
			Metadata:     p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartResource:
		v := resourcePart{
			Resource: p.Resource,
			Metadata: p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartCustom:
		v := customPart{
			Custom:   p.Custom,
			Metadata: p.wireMetadata(),
		}
		return json.Marshal(v)
	case PartReasoning:
		v := reasoningPart{
			Reasoning: p.Text,
			Metadata:  p.wireMetadata(),
		}
		return json.Marshal(v)
	default:
		return nil, fmt.Errorf("invalid part kind %d", int8(p.Kind))
	}
}

// wireMetadata returns the metadata map to serialize for the Part, folding
// the typed Interrupt and Restart fields into the keys the wire protocol
// expects (mirroring the JS runtime). The Part's own Metadata map is never
// mutated.
func (p *Part) wireMetadata() map[string]any {
	if p.Interrupt == nil && p.Restart == nil {
		return p.Metadata
	}
	m := maps.Clone(p.Metadata)
	if m == nil {
		m = make(map[string]any, 2)
	}
	if it := p.Interrupt; it != nil {
		key := base.ToolMetaInterrupt
		if it.Resolved {
			key = base.ToolMetaResolvedInterrupt
		}
		m[key] = orTrue(it.Data)
	}
	if rs := p.Restart; rs != nil {
		m[base.ToolMetaResumed] = orTrue(rs.Resume)
		if rs.OriginalInput != nil {
			m[base.ToolMetaReplacedInput] = rs.OriginalInput
		}
	}
	return m
}

// orTrue encodes an optional payload for the wire: a nil payload is carried
// as the JSON literal true (a "bare" interrupt or restart), matching the JS
// runtime.
func orTrue(v any) any {
	if v == nil {
		return true
	}
	return v
}

// payloadOf decodes the wire encoding written by orTrue: true means no
// payload.
func payloadOf(v any) any {
	if b, ok := v.(bool); ok && b {
		return nil
	}
	return v
}

// liftWireMetadata populates the typed Interrupt and Restart fields from
// their wire keys in the metadata map, removing the lifted keys so the map
// holds only user and plugin metadata.
func (p *Part) liftWireMetadata() {
	m := p.Metadata
	if m == nil {
		return
	}
	if v, ok := m[base.ToolMetaInterrupt]; ok {
		p.Interrupt = &ToolInterrupt{Data: payloadOf(v)}
		delete(m, base.ToolMetaInterrupt)
	} else if v, ok := m[base.ToolMetaResolvedInterrupt]; ok {
		p.Interrupt = &ToolInterrupt{Data: payloadOf(v), Resolved: true}
		delete(m, base.ToolMetaResolvedInterrupt)
	}
	if v, ok := m[base.ToolMetaResumed]; ok {
		p.Restart = &ToolRestart{Resume: payloadOf(v)}
		delete(m, base.ToolMetaResumed)
	}
	if v, ok := m[base.ToolMetaReplacedInput]; ok {
		if p.Restart == nil {
			p.Restart = &ToolRestart{}
		}
		p.Restart.OriginalInput = v
		delete(m, base.ToolMetaReplacedInput)
	}
	if len(m) == 0 {
		p.Metadata = nil
	}
}

// Validate checks that the Part's fields are consistent with its Kind: that
// the kind's own payload field is set and that no field belonging to another
// kind is set (e.g. no ToolResponse on a tool request part, no Interrupt on a
// text part). It reports the first inconsistency found.
func (p *Part) Validate() error {
	if p == nil {
		return core.NewError(core.INVALID_ARGUMENT, "part is nil")
	}
	if _, ok := partKindNames[p.Kind]; !ok {
		return core.NewError(core.INVALID_ARGUMENT, "invalid part kind %d", int8(p.Kind))
	}
	fields := []struct {
		name    string
		set     bool
		validOn bool
	}{
		{"Text", p.Text != "", p.Kind == PartText || p.Kind == PartMedia || p.Kind == PartData || p.Kind == PartReasoning},
		{"ContentType", p.ContentType != "", p.Kind == PartMedia || p.Kind == PartData},
		{"ToolRequest", p.ToolRequest != nil, p.Kind == PartToolRequest},
		{"ToolResponse", p.ToolResponse != nil, p.Kind == PartToolResponse},
		{"Resource", p.Resource != nil, p.Kind == PartResource},
		{"Custom", p.Custom != nil, p.Kind == PartCustom},
		{"Interrupt", p.Interrupt != nil, p.Kind == PartToolRequest},
		{"Restart", p.Restart != nil, p.Kind == PartToolRequest},
	}
	for _, f := range fields {
		if f.set && !f.validOn {
			return core.NewError(core.INVALID_ARGUMENT, "field %s is not valid on a %s part", f.name, p.Kind)
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
		return core.NewError(core.INVALID_ARGUMENT, "field %s is required on a %s part", r.name, p.Kind)
	}
	if p.Interrupt != nil && !p.Interrupt.Resolved && p.Restart != nil {
		return core.NewError(core.INVALID_ARGUMENT, "part cannot both await an interrupt and be a restart; resolve the interrupt first")
	}
	return nil
}

type partSchema struct {
	Text         string         `json:"text,omitempty" yaml:"text,omitempty"`
	Media        *Media         `json:"media,omitempty" yaml:"media,omitempty"`
	Data         string         `json:"data,omitempty" yaml:"data,omitempty"`
	ToolRequest  *ToolRequest   `json:"toolRequest,omitempty" yaml:"toolRequest,omitempty"`
	ToolResponse *ToolResponse  `json:"toolResponse,omitempty" yaml:"toolResponse,omitempty"`
	Resource     *ResourcePart  `json:"resource,omitempty" yaml:"resource,omitempty"`
	Custom       map[string]any `json:"custom,omitempty" yaml:"custom,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
	Reasoning    string         `json:"reasoning,omitempty" yaml:"reasoning,omitempty"`
}

// unmarshalPartFromSchema updates Part p based on the schema s.
func (p *Part) unmarshalPartFromSchema(s partSchema) {
	switch {
	case s.Media != nil:
		p.Kind = PartMedia
		p.Text = s.Media.URL
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
	case s.Reasoning != "":
		p.Kind = PartReasoning
		p.Text = s.Reasoning
	default:
		p.Kind = PartText
		p.Text = s.Text
		p.ContentType = ""
		if s.Data != "" {
			// Note: if part is completely empty, we use text by default.
			p.Kind = PartData
			p.Text = s.Data
		}
	}
	p.Metadata = s.Metadata
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
