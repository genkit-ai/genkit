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
	"context"
	"fmt"
	"maps"
	"net/url"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/yosida95/uritemplate/v3"
)

// normalizeURI normalizes a URI for template matching by removing query parameters,
// fragments, and trailing slashes from the path.
func normalizeURI(rawURI string) string {
	// Parse the URI
	u, err := url.Parse(rawURI)
	if err != nil {
		// If parsing fails, return the original URI
		return rawURI
	}

	// Remove query parameters and fragment
	u.RawQuery = ""
	u.Fragment = ""

	// Remove trailing slash from path (but not from the root path)
	if len(u.Path) > 1 && strings.HasSuffix(u.Path, "/") {
		u.Path = strings.TrimSuffix(u.Path, "/")
	}

	return u.String()
}

// matches checks if a URI matches the given URI template.
func matches(templateStr, uri string) (bool, error) {
	template, err := uritemplate.New(templateStr)
	if err != nil {
		return false, fmt.Errorf("invalid URI template %q: %w", templateStr, err)
	}

	normalizedURI := normalizeURI(uri)
	values := template.Match(normalizedURI)
	return len(values) > 0, nil
}

// extractVariables extracts variables from a URI using the given URI template.
func extractVariables(templateStr, uri string) (map[string]string, error) {
	template, err := uritemplate.New(templateStr)
	if err != nil {
		return nil, fmt.Errorf("invalid URI template %q: %w", templateStr, err)
	}

	normalizedURI := normalizeURI(uri)
	values := template.Match(normalizedURI)
	if len(values) == 0 {
		return nil, fmt.Errorf("URI %q does not match template", uri)
	}

	// Convert uritemplate.Values to string map
	result := make(map[string]string)
	for name, value := range values {
		result[name] = value.String()
	}
	return result, nil
}

// ResourceInput represents the input to a resource function.
type ResourceInput struct {
	URI       string            `json:"uri"`       // The resource URI
	Variables map[string]string `json:"variables"` // Extracted variables from URI template matching
}

// ResourceOutput represents the output from a resource function.
type ResourceOutput struct {
	Content []*Part `json:"content"` // The content parts returned by the resource
}

// ResourceOptions configures a resource definition.
type ResourceOptions struct {
	URI         string         // Static URI (mutually exclusive with Template)
	Template    string         // URI template (mutually exclusive with URI)
	Description string         // Optional description
	Metadata    map[string]any // Optional metadata
}

// ResourceFunc is a function that loads content for a resource.
type ResourceFunc = func(context.Context, *ResourceInput) (*ResourceOutput, error)

// resource is the internal implementation of the Resource interface.
// It holds the underlying core action and allows looking up resources
// by name without knowing their specific input/output api.
type resource struct {
	core.ActionDef[*ResourceInput, *ResourceOutput, struct{}]
}

// Resource represents an instance of a resource.
type Resource interface {
	// Name returns the name of the resource.
	Name() string
	// Matches reports whether this resource matches the given URI.
	Matches(uri string) bool
	// ExtractVariables extracts variables from a URI using this resource's template.
	ExtractVariables(uri string) (map[string]string, error)
	// Execute runs the resource with the given input.
	Execute(ctx context.Context, input *ResourceInput) (*ResourceOutput, error)
	// Register registers the resource with the given registry.
	Register(r api.Registry)
}

// DefineResource creates a resource and registers it with the given Registry.
func DefineResource(r api.Registry, name string, opts *ResourceOptions, fn ResourceFunc) Resource {
	metadata := resourceMetadata(name, opts)
	return &resource{ActionDef: *core.DefineAction(r, name, api.ActionTypeResource, metadata, nil, fn)}
}

// NewResource creates a resource but does not register it in the registry.
// It can be registered later via the Register method.
func NewResource(name string, opts *ResourceOptions, fn ResourceFunc) Resource {
	metadata := resourceMetadata(name, opts)
	metadata["dynamic"] = true
	return &resource{ActionDef: *core.NewAction(name, api.ActionTypeResource, metadata, nil, fn)}
}

// resourceMetadata creates the metadata common to both DefineResource and NewResource.
func resourceMetadata(name string, opts *ResourceOptions) map[string]any {
	// Validate options - panic like other Define* functions
	if name == "" {
		panic("resource name is required")
	}
	if opts.URI != "" && opts.Template != "" {
		panic("cannot specify both URI and Template")
	}
	if opts.URI == "" && opts.Template == "" {
		panic("must specify either URI or Template")
	}

	// Create metadata with resource-specific information
	metadata := map[string]any{
		"type":        api.ActionTypeResource,
		"name":        name,
		"description": opts.Description,
		"resource": map[string]any{
			"uri":      opts.URI,
			"template": opts.Template,
		},
	}

	// Add user metadata
	if opts.Metadata != nil {
		maps.Copy(metadata, opts.Metadata)
	}

	return metadata
}

// Matches reports whether this resource matches the given URI.
func (r *resource) Matches(uri string) bool {
	resourceMeta, ok := r.Desc().Metadata["resource"].(map[string]any)
	if !ok {
		return false
	}

	// Check static URI
	if staticURI, ok := resourceMeta["uri"].(string); ok && staticURI != "" {
		return staticURI == uri
	}

	// Check template
	if template, ok := resourceMeta["template"].(string); ok && template != "" {
		matches, err := matches(template, uri)
		if err != nil {
			return false
		}
		return matches
	}

	return false
}

// ExtractVariables extracts variables from a URI using this resource's template.
func (r *resource) ExtractVariables(uri string) (map[string]string, error) {
	resourceMeta, ok := r.Desc().Metadata["resource"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("no resource metadata found")
	}

	// Static URI has no variables
	if staticURI, ok := resourceMeta["uri"].(string); ok && staticURI != "" {
		if staticURI == uri {
			return map[string]string{}, nil
		}
		return nil, fmt.Errorf("URI %q does not match static URI %q", uri, staticURI)
	}

	// Extract from template
	if template, ok := resourceMeta["template"].(string); ok && template != "" {
		return extractVariables(template, uri)
	}

	return nil, fmt.Errorf("no URI or template found in resource metadata")
}

// Execute runs the resource with the given input.
func (r *resource) Execute(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
	return r.Run(ctx, input, nil)
}

// FindMatchingResource finds a resource that matches the given URI.
func FindMatchingResource(r api.Registry, uri string) (Resource, *ResourceInput, error) {
	for _, a := range r.ListActions() {
		if action, ok := a.(*core.ActionDef[*ResourceInput, *ResourceOutput, struct{}]); ok {
			res := &resource{ActionDef: *action}
			if res.Matches(uri) {
				variables, err := res.ExtractVariables(uri)
				if err != nil {
					return nil, nil, err
				}
				return res, &ResourceInput{URI: uri, Variables: variables}, nil
			}
		}
	}

	return nil, nil, fmt.Errorf("no resource found for URI %q", uri)
}

// LookupResource looks up the resource in the registry by provided name and returns it.
func LookupResource(r api.Registry, name string) Resource {
	action := core.ResolveActionFor[*ResourceInput, *ResourceOutput, struct{}](r, api.ActionTypeResource, name)
	if action == nil {
		return nil
	}
	return &resource{ActionDef: *action}
}
