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

// CacheControlOptions configures an Anthropic ephemeral cache breakpoint.
// Mirrors JS cacheControl() / CacheControlEphemeral.
type CacheControlOptions struct {
	// TTL is the cache lifetime: "5m" (default) or "1h".
	TTL string
}

// CacheControl returns part metadata that attaches Anthropic prompt caching
// (cache_control) to a content block, matching the JS helper:
//
//	metadata: { ...cacheControl({ ttl: '5m' }) }
//
// Merge into an existing map or assign directly to ai.Part.Metadata / system
// text parts:
//
//	p := ai.NewTextPart(longSystem)
//	p.Metadata = anthropic.CacheControl(&anthropic.CacheControlOptions{TTL: "5m"})
func CacheControl(opts *CacheControlOptions) map[string]any {
	cc := map[string]any{"type": "ephemeral"}
	if opts != nil && opts.TTL != "" {
		cc["ttl"] = opts.TTL
	}
	return map[string]any{"cache_control": cc}
}
