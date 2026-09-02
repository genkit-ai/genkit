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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"google.golang.org/genai"
)

var invalidArgMessages = struct {
	modelVersion string
	tools        string
	systemPrompt string
}{
	tools:        "tools are not supported with context caching",
	systemPrompt: "system prompts are not supported with context caching",
}

// handleCache checks if caching should be used and attempts to find or create
// the cache. It returns the cached content along with the messages that still
// have to be sent inline: the cached prefix is not among them, since it reaches
// the model through the CachedContent resource and sending it inline as well
// would bill it twice (#6137).
func handleCache(
	ctx context.Context,
	client *genai.Client,
	request *ai.ModelRequest,
	model string,
) (*genai.CachedContent, []*ai.Message, error) {
	cs, err := findCacheMarker(request)
	if err != nil {
		return nil, nil, err
	}
	// no cache mark found
	if cs == nil {
		return nil, request.Messages, nil
	}
	// index out of bounds
	if cs.endIndex < 0 || cs.endIndex >= len(request.Messages) {
		return nil, nil, status.Errorf(status.ErrInvalidArgument, "end of cached contents, index %d is invalid", cs.endIndex)
	}

	// since context caching is only available for specific model versions, we
	// must make sure the configuration has the right version
	err = validateContextCacheRequest(request, model)
	if err != nil {
		return nil, nil, err
	}

	messages, err := messagesToCache(request.Messages, cs.endIndex)
	if err != nil {
		return nil, nil, err
	}
	hash, err := calculateCacheHash(messages, model)
	if err != nil {
		return nil, nil, err
	}

	// A name only records that an earlier turn built a cache. The resource
	// lives no longer than its TTL, and the request it was built from may have
	// moved on since, so a name that no longer resolves falls back to a fresh
	// cache instead of failing. The caller cannot act on that failure anyway:
	// the name comes from Genkit's own response metadata, not from them.
	var cache *genai.CachedContent
	if cs.name != "" {
		cache, err = lookupCache(ctx, client, cs.name)
		switch {
		case err != nil:
			logger.Warn(ctx, "context cache lookup failed, looking for another cache with the same contents",
				"name", cs.name, "error", wrapAPIError(err))
			cache = nil
		case cache.DisplayName != hash:
			logger.Warn(ctx, "context cache no longer matches the request messages, looking for another cache with the same contents",
				"name", cs.name)
			cache = nil
		}
	}

	// The name only travels back to the caller through the response metadata,
	// so a caller who rebuilds the same prefix instead of replaying
	// resp.History() arrives here without one. Look for a cache that already
	// holds this content before paying to store a second copy of it.
	if cache == nil {
		cache, err = findCacheByHash(ctx, client, hash)
		if err != nil {
			logger.Warn(ctx, "context cache scan failed, creating a fresh cache", "error", err)
		}
	}

	if cache == nil {
		if cs.ttl <= 0 {
			// A name-only marker that resolved to nothing. There is no ttl to
			// build a replacement with, so send the whole request: the answer
			// is still right, it just is not cached.
			logger.Warn(ctx, "no usable context cache and no ttlSeconds to create one, sending the full request",
				"name", cs.name)
			return nil, request.Messages, nil
		}
		cache, err = client.Caches.Create(ctx, model, &genai.CreateCachedContentConfig{
			DisplayName: hash,
			TTL:         time.Duration(cs.ttl) * time.Second,
			Contents:    messages,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("cache creation error: %w", wrapAPIError(err))
		}
	}

	return cache, request.Messages[cs.endIndex+1:], nil
}

// cacheScanLimit bounds the search for a reusable cache. The scan runs before
// every request that asks for explicit caching, and a project can hold far
// more cache resources than are worth paging through; giving up only costs a
// fresh cache.
const cacheScanLimit = 200

// findCacheByHash looks for a cache resource whose DisplayName is hash, which
// is how [calculateCacheHash] records what a cache holds. A scan failure is
// not fatal: the caller falls back to creating a cache.
func findCacheByHash(ctx context.Context, client *genai.Client, hash string) (*genai.CachedContent, error) {
	scanned := 0
	for c, err := range client.Caches.All(ctx) {
		if err != nil {
			return nil, wrapAPIError(err)
		}
		if c.DisplayName == hash {
			return c, nil
		}
		if scanned++; scanned >= cacheScanLimit {
			logger.Warn(ctx, "gave up looking for a reusable context cache", "scanned", scanned)
			return nil, nil
		}
	}
	return nil, nil
}

// messagesToCache converts the messages through cacheEndIdx (inclusive) into
// the contents stored on the CachedContent resource. It shares toGeminiContents
// with the inline path so that the cached prefix is encoded exactly like the
// messages sent alongside it: same role mapping, same handling of contentless
// turns. The order is chronological and feeds calculateCacheHash, so changing
// it invalidates every live cache name.
func messagesToCache(m []*ai.Message, cacheEndIdx int) ([]*genai.Content, error) {
	return toGeminiContents(m[:cacheEndIdx+1])
}

// validateContextCacheRequest checks for supported models and checks if Tools
// are being provided in the request
func validateContextCacheRequest(request *ai.ModelRequest, modelVersion string) error {
	if len(request.Tools) > 0 {
		return status.Errorf(status.ErrInvalidArgument, "%s", invalidArgMessages.tools)
	}
	for _, m := range request.Messages {
		if m.Role == ai.RoleSystem {
			return status.Errorf(status.ErrInvalidArgument, "%s", invalidArgMessages.systemPrompt)
		}
	}

	return nil
}

type cacheSettings struct {
	ttl      int
	name     string
	endIndex int
}

// findCacheMarker finds the cache mark in the list of request messages. The
// marked message and everything before it is cached: endIndex is inclusive.
// The scan runs newest to oldest, so the newest cache name in the history wins
// and only the newest ttl marker is honoured.
func findCacheMarker(request *ai.ModelRequest) (*cacheSettings, error) {
	cacheName, cacheNameIdx := "", -1

	for i := len(request.Messages) - 1; i >= 0; i-- {
		m := request.Messages[i]
		if m.Metadata == nil {
			continue
		}

		cacheVal, exists := m.Metadata["cache"]
		if !exists || cacheVal == nil {
			continue
		}

		c, ok := cacheVal.(map[string]any)
		if !ok {
			return nil, status.Errorf(status.ErrInvalidArgument, "cache metadata should be map but got: %T", cacheVal)
		}

		// A message can carry the name of a cache an earlier turn built, the
		// ttl that builds one, or both. Several turns of a replayed history
		// carry a name, so keep the first seen: the scan runs newest to oldest.
		name, hasName := c["name"].(string)
		if hasName && cacheName == "" {
			cacheName, cacheNameIdx = name, i
		}

		// ttlSeconds arrives as an int when set in Go code and as a float64 or
		// json.Number after a JSON round-trip (dev UI, reflection server), so
		// accept any numeric form. A non-positive TTL (including a fractional
		// value that truncates to zero) would silently skip cache creation in
		// handleCache, so reject it here instead.
		if tv, ok := c["ttlSeconds"]; ok {
			t, isNum := castToInt64(tv)
			if !isNum {
				return nil, status.Errorf(status.ErrInvalidArgument, "invalid type for cache ttlSeconds, expected a number, got %T", tv)
			}
			if t <= 0 {
				return nil, status.Errorf(status.ErrInvalidArgument, "invalid cache ttlSeconds, expected a positive number of whole seconds, got %v", tv)
			}
			// Any content is cacheable, not just text: a long video or PDF is
			// what caching pays off for most, and [ai.Message.Text] reports a
			// message holding only media as empty.
			if len(m.Content) == 0 {
				return nil, status.Errorf(status.ErrInvalidArgument, "no content to cache, message is empty")
			}
			return &cacheSettings{
				ttl:      int(t),
				name:     cacheName,
				endIndex: i,
			}, nil
		}

		if hasName {
			continue
		}

		return nil, status.Errorf(status.ErrInvalidArgument, "invalid cache metadata, expected ttlSeconds or name, got: %v", c)
	}

	// A name with no ttl anywhere names a cache an earlier turn built. The
	// marked message closes the cached span, the same way a ttl marker does;
	// handleCache checks that against the cache's own contents before trusting
	// it, and falls back to sending the whole request if it does not hold up.
	if cacheName != "" {
		return &cacheSettings{name: cacheName, endIndex: cacheNameIdx}, nil
	}
	return nil, nil
}

// lookupCache retrieves a *genai.CachedContent from a given cache name
func lookupCache(ctx context.Context, client *genai.Client, name string) (*genai.CachedContent, error) {
	if name == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "empty cache name detected")
	}

	return client.Caches.Get(ctx, name, nil)
}

// calculateCacheHash generates a sha256 key over the contents of a cache. It
// is stored as the resource's DisplayName and re-checked on every reuse, so it
// has to cover everything that tells one cached prefix apart from another.
// Hashing the JSON encoding does that: it reaches every part kind and keeps
// field and message boundaries, where folding selected fields into a bare
// digest silently collided. Two different gs:// videos both hashed to the
// digest of no bytes at all, because URI media carries neither Text nor
// InlineData. The model is part of the key because a cache resource belongs to
// the model that created it, matching the JS and Python plugins.
func calculateCacheHash(content []*genai.Content, model string) (string, error) {
	b, err := json.Marshal(struct {
		Model    string           `json:"model"`
		Contents []*genai.Content `json:"contents"`
	}{Model: model, Contents: content})
	if err != nil {
		// Reachable through tool inputs and outputs, which carry caller-owned
		// map[string]any into FunctionCall.Args and FunctionResponse.Response.
		return "", status.Errorf(status.ErrInvalidArgument, "cannot hash cache contents: %w", err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// cacheMetadata writes in the metadata map the cache name used in the
// request
func cacheMetadata(m map[string]any, cc *genai.CachedContent) map[string]any {
	// keep the original metadata if no cache was used in the request
	if cc == nil {
		return m
	}

	cache, ok := m["cache"].(map[string]any)
	if !ok {
		m = map[string]any{
			"cache": map[string]any{
				"name": cc.Name,
			},
		}
		return m
	}

	cache["name"] = cc.Name
	return m
}
