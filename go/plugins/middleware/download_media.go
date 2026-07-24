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

package middleware

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// DownloadRequestMedia is a middleware that downloads media referenced by an
// HTTP(S) URL in a request and replaces the media part with an inline base64
// data URL. It is useful for models whose APIs cannot fetch remote media
// themselves.
//
// It hooks the Model stage, so each individual model API call has its media
// downloaded (not the whole generate loop).
//
// Usage as call-scoped middleware:
//
//	resp, err := ai.Generate(ctx, r,
//	    ai.WithModel(m),
//	    ai.WithMessages(msgWithRemoteMediaURL),
//	    ai.WithUse(&middleware.DownloadRequestMedia{MaxBytes: 10 << 20}),
//	)
//
// Provider plugins that always need the behavior can instead bake it into a
// model at definition time with [DownloadRequestMedia.WrapModelFunc].
type DownloadRequestMedia struct {
	// MaxBytes caps the number of bytes read per media URL. Zero means no cap.
	MaxBytes int64 `json:"maxBytes,omitempty"`
	// Filter, when set, decides whether a media part is downloaded. Parts for
	// which it returns false are left untouched. It is a Go-only field and is
	// not populated from JSON config.
	Filter func(part *ai.Part) bool `json:"-"`
}

func (d *DownloadRequestMedia) Name() string { return provider + "/download-request-media" }

func (d *DownloadRequestMedia) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{
		WrapModel: d.wrapModel,
	}, nil
}

func (d *DownloadRequestMedia) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	if err := d.download(params.Request); err != nil {
		return nil, err
	}
	return next(ctx, params)
}

// WrapModelFunc returns fn wrapped so that HTTP(S) media URLs in each request
// are downloaded and inlined as base64 data URLs before fn runs. Provider
// plugins use this to bake the behavior into a model at definition time; end
// users can instead attach the middleware to a single call via [ai.WithUse].
func (d *DownloadRequestMedia) WrapModelFunc[Config any](fn ai.ModelFunc[Config]) ai.ModelFunc[Config] {
	return func(ctx context.Context, req *ai.ModelRequest, cfg Config, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if err := d.download(req); err != nil {
			return nil, err
		}
		return fn(ctx, req, cfg, cb)
	}
}

// download replaces each HTTP(S) media part in req with an inline base64 data
// URL, mutating req in place.
func (d *DownloadRequestMedia) download(req *ai.ModelRequest) error {
	client := &http.Client{}
	for _, message := range req.Messages {
		for j, part := range message.Content {
			if !part.IsMedia() || !strings.HasPrefix(part.Text, "http") || (d.Filter != nil && !d.Filter(part)) {
				continue
			}

			mediaURL := part.Text

			resp, err := client.Get(mediaURL)
			if err != nil {
				return status.Errorf(status.ErrInvalidArgument, "downloading media %q: %w", mediaURL, err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				return status.Errorf(status.ErrUnknown, "HTTP error downloading media %q: %s", mediaURL, string(body))
			}

			contentType := part.ContentType
			if contentType == "" {
				contentType = resp.Header.Get("Content-Type")
			}

			var data []byte
			if d.MaxBytes > 0 {
				data, err = io.ReadAll(io.LimitReader(resp.Body, d.MaxBytes))
			} else {
				data, err = io.ReadAll(resp.Body)
			}
			if err != nil {
				return status.Errorf(status.ErrUnknown, "reading media %q: %w", mediaURL, err)
			}

			message.Content[j] = ai.NewMediaPart(contentType, fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(data)))
		}
	}
	return nil
}
