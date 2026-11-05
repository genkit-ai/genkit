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

// Package uri extracts the content-type and data from a media part.
// This is used by the googleai and vertexai plugins.
package uri

import (
	"encoding/base64"
	"errors"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// Data returns the content type and bytes of the media part.
func Data(p *ai.Part) (string, []byte, error) {
	if !p.IsMedia() && !p.IsData() {
		return "", nil, errors.New("not a media part")
	}

	uri := p.Text
	if uri == "" {
		// Data parts carry their payload on Data, not Text. Read it through the
		// shared accessor so a string payload (e.g. a "data:image/png;base64,..."
		// URI) comes through as-is and a structured payload (map/slice) comes
		// through as its JSON encoding, matching what the provider converters do.
		uri = p.DataString()
	}

	if strings.HasPrefix(uri, "gs://") || strings.HasPrefix(uri, "http") {
		if p.ContentType == "" {
			return "", nil, errors.New("must supply contentType when using media from gs:// or http(s):// URLs")
		}
		return p.ContentType, []byte(uri), nil
	}

	if contents, isData := strings.CutPrefix(uri, "data:"); isData {
		prefix, data, found := strings.Cut(contents, ",")
		if !found {
			return "", nil, errors.New("failed to parse data URI: missing comma")
		}

		var dataBytes []byte
		if p, isBase64 := strings.CutSuffix(prefix, ";base64"); isBase64 {
			prefix = p
			var err error
			dataBytes, err = base64.StdEncoding.DecodeString(data)
			if err != nil {
				return "", nil, err
			}
		} else {
			dataBytes = []byte(data)
		}

		contentType := p.ContentType
		if contentType == "" {
			contentType = prefix
		}

		return contentType, dataBytes, nil
	}

	return "", nil, errors.New("could not convert media part to genai part: missing file data")
}
