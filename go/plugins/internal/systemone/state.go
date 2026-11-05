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

package systemone

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"net/url"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// SystemPreamble gathers the system messages into the text that goes in
// front of every question's instructions. The model has no system role: the
// state is what it evaluates and the questions are what it is told, so a
// system message belongs with the questions, never in the state. Only text
// can be instructions; loop plumbing left in a history is skipped.
func SystemPreamble(messages []*ai.Message) (string, error) {
	var texts []string
	for _, msg := range messages {
		if msg.Role != ai.RoleSystem {
			continue
		}
		var sb strings.Builder
		for _, part := range msg.Content {
			if isFormatInstructions(part) {
				continue
			}
			if !part.IsText() {
				return "", status.Errorf(status.ErrInvalidArgument, "systemone: a %s part cannot be instructions; a system message is text only", partKindName(part))
			}
			sb.WriteString(part.Text)
		}
		if text := strings.TrimSpace(sb.String()); text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n\n"), nil
}

// Reads says which kinds of media a model reads besides text and JSON.
type Reads struct {
	Images, Audio, Video bool
}

// Media is the media a request carries, as base64 data URLs by kind, each
// in the order its parts appear, messages first and documents after.
// Media is an extension some servers make to the protocol, in the images,
// audio, and videos fields.
type Media struct {
	Images, Audio, Videos []string
}

// len is how many media parts there are, of every kind.
func (m Media) len() int { return len(m.Images) + len(m.Audio) + len(m.Videos) }

// BuildState turns the request into the state and the media: one message
// as its value, several as {role, content} records, and with documents
// attached an object of both, so a question can name the context it is
// about. System messages are instructions, not state, and are left out.
//
// Media parts of a kind the model reads become the request's media and
// contribute nothing to the state; a media part of any other kind is
// refused. A message of media alone is still a turn, with the empty string
// as its content, so the turns around it keep their roles and a request of
// that one message sends the empty string as its state.
func BuildState(req *ai.ModelRequest, stateJSON bool, reads Reads) (any, Media, error) {
	b := &stateBuilder{reads: reads}
	records := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		if msg.Role == ai.RoleSystem {
			continue
		}
		collected := b.media.len()
		value, err := b.messageValue(msg, stateJSON)
		if err != nil {
			return nil, Media{}, err
		}
		if value == nil {
			if b.media.len() == collected {
				continue
			}
			value = ""
		}
		records = append(records, map[string]any{"role": string(msg.Role), "content": value})
	}
	if len(records) == 0 {
		return nil, Media{}, status.Errorf(status.ErrInvalidArgument, "systemone: the request carries no state; a system message is instructions, so pass a prompt or messages too")
	}
	docs := make([]any, 0, len(req.Docs))
	for _, doc := range req.Docs {
		value, err := b.documentValue(doc)
		if err != nil {
			return nil, Media{}, err
		}
		if value != nil {
			docs = append(docs, value)
		}
	}
	switch {
	case len(docs) > 0:
		return map[string]any{"messages": records, "context": docs}, b.media, nil
	case len(records) == 1:
		return records[0]["content"], b.media, nil
	default:
		return records, b.media, nil
	}
}

// stateBuilder reads the state out of messages and documents, and collects
// the media it meets on the way.
type stateBuilder struct {
	reads Reads
	media Media
}

// messageValue is a message's contribution to the state: the value of its
// one part, or the values of all of them. A message with nothing to
// contribute yields nil.
func (b *stateBuilder) messageValue(msg *ai.Message, stateJSON bool) (any, error) {
	values := make([]any, 0, len(msg.Content))
	for _, part := range msg.Content {
		value, err := b.partValue(part, stateJSON)
		if err != nil {
			return nil, err
		}
		if value != nil {
			values = append(values, value)
		}
	}
	switch len(values) {
	case 0:
		return nil, nil
	case 1:
		return values[0], nil
	default:
		return values, nil
	}
}

// partValue is a part's contribution to the state. Text is sent as is, or
// as the JSON it holds when the config asks, kept verbatim so a number too
// large for a float64 reaches the model unchanged; a data part is sent as
// its value; media is collected and contributes nothing. Loop plumbing
// left in a history is skipped rather than sent as state, and any other
// kind of part is refused rather than stringified.
func (b *stateBuilder) partValue(part *ai.Part, stateJSON bool) (any, error) {
	if isFormatInstructions(part) {
		return nil, nil
	}
	switch {
	case part.IsText():
		if stateJSON {
			var value json.RawMessage
			if err := json.Unmarshal([]byte(part.Text), &value); err != nil {
				return nil, status.Errorf(status.ErrInvalidArgument, "systemone: stateJSON is set but the text is not JSON: %w", err)
			}
			return value, nil
		}
		return part.Text, nil
	case part.IsData():
		return part.Data, nil
	case part.IsMedia():
		return nil, b.collect(part)
	default:
		return nil, status.Errorf(status.ErrInvalidArgument, "systemone: a %s part cannot be state; the model takes text, JSON, and media only", partKindName(part))
	}
}

// collect adds a media part to the request's media, or refuses it when the
// model does not read its kind.
func (b *stateBuilder) collect(part *ai.Part) error {
	dataURL, contentType, err := mediaURL(part)
	if err != nil {
		return err
	}
	var reads bool
	var media *[]string
	switch {
	case ai.IsImageContentType(contentType):
		reads, media = b.reads.Images, &b.media.Images
	case ai.IsAudioContentType(contentType):
		reads, media = b.reads.Audio, &b.media.Audio
	case ai.IsVideoContentType(contentType):
		reads, media = b.reads.Video, &b.media.Videos
	default:
		return status.Errorf(status.ErrInvalidArgument, "systemone: a %s part is not an image, audio, or video; the models take no other media", cmp.Or(contentType, "media"))
	}
	if !reads {
		return status.Errorf(status.ErrInvalidArgument, "systemone: the model reads no %s media", contentType)
	}
	*media = append(*media, dataURL)
	return nil
}

// documentValue is a document's contribution to the context: its content,
// read as a message's is, or its content with its metadata when it has
// any, so a passage keeps its title and source for a question to refer
// to. Text stays text whatever the config says, since a retrieved passage
// is prose, not a rendered template. A document of media alone with no
// metadata contributes nothing beside its media.
func (b *stateBuilder) documentValue(doc *ai.Document) (any, error) {
	content, err := b.messageValue(&ai.Message{Content: doc.Content}, false)
	if err != nil {
		return nil, err
	}
	if len(doc.Metadata) == 0 {
		return content, nil
	}
	return map[string]any{"content": content, "metadata": doc.Metadata}, nil
}

// mediaURL is a media part as the media fields take it: a base64 data URL,
// and its media type. A base64 payload goes out as it is, under the part's
// media type, since the server decodes it anyway; any other payload is
// percent-decoded (RFC 2397) and encoded. The scheme, the base64 token,
// and the media type are read without regard to case. A remote URL is
// refused: the servers take inline media only, and fetching it here would
// hide a network call, so the caller downloads it, as
// [ai.DownloadRequestMedia] does.
func mediaURL(part *ai.Part) (dataURL, contentType string, err error) {
	scheme, rest, _ := strings.Cut(part.Text, ":")
	switch strings.ToLower(scheme) {
	case "http", "https", "gs":
		return "", "", status.Errorf(status.ErrInvalidArgument, "systemone: the media %q is a URL, and the models take inline media only; download it first, for example with the ai.DownloadRequestMedia middleware", part.Text)
	case "data":
	default:
		return "", "", status.Errorf(status.ErrInvalidArgument, "systemone: a media part must carry its content as a data URL")
	}
	header, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return "", "", status.Errorf(status.ErrInvalidArgument, "systemone: reading the media: the data URL has no comma")
	}
	mediaType, _, _ := strings.Cut(header, ";")
	contentType = strings.ToLower(cmp.Or(part.ContentType, mediaType))
	prefix := "data:" + contentType + ";base64,"
	if strings.HasSuffix(strings.ToLower(header), ";base64") {
		return prefix + payload, contentType, nil
	}
	data, err := url.PathUnescape(payload)
	if err != nil {
		return "", "", status.Errorf(status.ErrInvalidArgument, "systemone: reading the media: %w", err)
	}
	return prefix + base64.StdEncoding.EncodeToString([]byte(data)), contentType, nil
}

// isFormatInstructions reports whether the part is the output instructions
// the generate loop injects for a model without constrained output. The
// loop never injects one for this model, but a history recorded from
// another model's turn keeps it in that turn's messages, and when such a
// history is the state the part is plumbing, not something anyone said.
func isFormatInstructions(part *ai.Part) bool {
	purpose, _ := part.Metadata["purpose"].(string)
	return purpose == "output"
}

// partKindName names a part's kind for an error message.
func partKindName(part *ai.Part) string {
	switch {
	case part.IsMedia():
		return "media"
	case part.IsToolRequest():
		return "tool request"
	case part.IsToolResponse():
		return "tool response"
	case part.IsReasoning():
		return "reasoning"
	case part.IsResource():
		return "resource"
	default:
		return "custom"
	}
}
