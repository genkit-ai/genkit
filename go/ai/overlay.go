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

package ai

import "slices"

// A plugin's model catalog answers two questions that must agree: what
// ListActions advertises, and what ResolveAction builds to serve a request. A
// caller who knows better than the catalog supplies an override through plugin
// config, and both paths lay it over what the plugin already knows.
//
// Overlaying rather than replacing is what lets a caller pin one capability
// without restating the label, the config schema, and the rest a model needs
// set to work at all. A zero-value field means "not specified": every field of
// both options structs distinguishes its zero value from a meaningful one,
// which [TestOverlayCoversEveryField] holds them to.

// Overlay returns o with every field set in override replacing o's. A field
// left at its zero value in override keeps o's.
func (o ModelOptions) Overlay(override ModelOptions) ModelOptions {
	if override.ConfigSchema != nil {
		o.ConfigSchema = override.ConfigSchema
	}
	if override.Label != "" {
		o.Label = override.Label
	}
	if override.Stage != "" {
		o.Stage = override.Stage
	}
	if override.Supports != nil {
		o.Supports = override.Supports
	}
	if override.Versions != nil {
		o.Versions = override.Versions
	}
	if override.Metadata != nil {
		o.Metadata = override.Metadata
	}
	return o
}

// Overlay returns o with every field set in override replacing o's. A field
// left at its zero value in override keeps o's.
func (o EmbedderOptions) Overlay(override EmbedderOptions) EmbedderOptions {
	if override.ConfigSchema != nil {
		o.ConfigSchema = override.ConfigSchema
	}
	if override.Label != "" {
		o.Label = override.Label
	}
	if override.Supports != nil {
		o.Supports = override.Supports
	}
	if override.Dimensions != 0 {
		o.Dimensions = override.Dimensions
	}
	if override.Metadata != nil {
		o.Metadata = override.Metadata
	}
	return o
}

// The Supports field of every options struct is a pointer, so that a zero value
// can mean "not specified" for [ModelOptions.Overlay]. That makes it easy for a
// plugin to point a whole table of models at one shared capability struct, so
// the constructors copy it rather than keeping the caller's pointer, and take a
// nil one to mean no capabilities.

func cloneModelSupports(s *ModelSupports) *ModelSupports {
	if s == nil {
		return &ModelSupports{}
	}
	c := *s
	c.ContentType = slices.Clone(s.ContentType)
	c.Output = slices.Clone(s.Output)
	return &c
}

func cloneEmbedderSupports(s *EmbedderSupports) *EmbedderSupports {
	if s == nil {
		return &EmbedderSupports{}
	}
	c := *s
	c.Input = slices.Clone(s.Input)
	return &c
}

func cloneRetrieverSupports(s *RetrieverSupports) *RetrieverSupports {
	if s == nil {
		return &RetrieverSupports{}
	}
	c := *s
	return &c
}
