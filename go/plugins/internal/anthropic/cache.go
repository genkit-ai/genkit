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

package anthropic

import (
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
)

// cacheControlFromMetadata reads part.metadata.cache_control (JS Anthropic parity).
// Expected shape: {"type":"ephemeral","ttl":"5m"|"1h"}.
func cacheControlFromMetadata(md map[string]any) (anthropic.CacheControlEphemeralParam, bool, error) {
	if md == nil {
		return anthropic.CacheControlEphemeralParam{}, false, nil
	}
	raw, ok := md["cache_control"]
	if !ok || raw == nil {
		return anthropic.CacheControlEphemeralParam{}, false, nil
	}
	cc, ok := raw.(map[string]any)
	if !ok {
		return anthropic.CacheControlEphemeralParam{}, false, fmt.Errorf("cache_control must be an object")
	}

	typ, _ := cc["type"].(string)
	if typ == "" {
		typ = "ephemeral"
	}
	if typ != "ephemeral" {
		return anthropic.CacheControlEphemeralParam{}, false, fmt.Errorf("unsupported cache_control type %q", typ)
	}

	param := anthropic.NewCacheControlEphemeralParam()
	if ttlRaw, exists := cc["ttl"]; exists && ttlRaw != nil {
		ttl, ok := ttlRaw.(string)
		if !ok {
			return anthropic.CacheControlEphemeralParam{}, false, fmt.Errorf("cache_control.ttl must be a string")
		}
		switch anthropic.CacheControlEphemeralTTL(ttl) {
		case anthropic.CacheControlEphemeralTTLTTL5m, anthropic.CacheControlEphemeralTTLTTL1h:
			param.TTL = anthropic.CacheControlEphemeralTTL(ttl)
		case "":
			// omit → SDK/API default (5m)
		default:
			return anthropic.CacheControlEphemeralParam{}, false, fmt.Errorf("cache_control.ttl must be %q or %q", anthropic.CacheControlEphemeralTTLTTL5m, anthropic.CacheControlEphemeralTTLTTL1h)
		}
	}
	return param, true, nil
}

// applyCacheControl attaches cache_control to supported content-block variants.
// Thinking / redacted_thinking blocks are left unchanged (API rejects cache on them).
func applyCacheControl(block *anthropic.ContentBlockParamUnion, md map[string]any) error {
	cc, ok, err := cacheControlFromMetadata(md)
	if err != nil || !ok {
		return err
	}
	switch {
	case block.OfText != nil:
		block.OfText.CacheControl = cc
	case block.OfImage != nil:
		block.OfImage.CacheControl = cc
	case block.OfDocument != nil:
		block.OfDocument.CacheControl = cc
	case block.OfToolUse != nil:
		block.OfToolUse.CacheControl = cc
	case block.OfToolResult != nil:
		block.OfToolResult.CacheControl = cc
	}
	return nil
}
