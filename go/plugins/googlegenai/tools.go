// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"google.golang.org/genai"
)

const (
	// toolNameRegex validates tool names.
	toolNameRegex = "^[a-zA-Z_][a-zA-Z0-9_.-]{0,63}$"
)

// toGeminiTools translates a slice of [ai.ToolDefinition] to a slice of [genai.Tool].
func toGeminiTools(inTools []*ai.ToolDefinition) ([]*genai.Tool, error) {
	var outTools []*genai.Tool
	functions := []*genai.FunctionDeclaration{}

	for _, t := range inTools {
		if !validToolName(t.Name) {
			return nil, fmt.Errorf(`invalid tool name: %q, must start with a letter or an underscore, must be alphanumeric, underscores, dots or dashes with a max length of 64 chars`, t.Name)
		}
		inputSchema, err := toGeminiSchema(t.InputSchema, t.InputSchema)
		if err != nil {
			return nil, err
		}
		fd := &genai.FunctionDeclaration{
			Name:        t.Name,
			Parameters:  inputSchema,
			Description: t.Description,
		}
		functions = append(functions, fd)
	}

	if len(functions) > 0 {
		outTools = append(outTools, &genai.Tool{
			FunctionDeclarations: functions,
		})
	}

	return outTools, nil
}

// toGeminiFunctionResponsePart translates a slice of [ai.Part] to a slice of [genai.FunctionResponsePart]
func toGeminiFunctionResponsePart(parts []*ai.Part) ([]*genai.FunctionResponsePart, error) {
	frp := []*genai.FunctionResponsePart{}
	for _, p := range parts {
		switch {
		case p.IsData():
			contentType, data, err := uri.Data(p)
			if err != nil {
				return nil, err
			}
			frp = append(frp, genai.NewFunctionResponsePartFromBytes(data, contentType))
		case p.IsMedia():
			if strings.HasPrefix(p.Text, "data:") {
				contentType, data, err := uri.Data(p)
				if err != nil {
					return nil, err
				}
				frp = append(frp, genai.NewFunctionResponsePartFromBytes(data, contentType))
				continue
			}
			frp = append(frp, genai.NewFunctionResponsePartFromURI(p.Text, p.ContentType))
		default:
			return nil, fmt.Errorf("unsupported function response part type: %d", p.Kind)
		}
	}
	return frp, nil
}

// mergeTools consolidates all FunctionDeclarations into a single Tool
// while preserving non-function tools (Retrieval, GoogleSearch, CodeExecution, etc.)
func mergeTools(ts []*genai.Tool) []*genai.Tool {
	var decls []*genai.FunctionDeclaration
	var out []*genai.Tool

	for _, t := range ts {
		if t == nil {
			continue
		}
		if len(t.FunctionDeclarations) == 0 {
			out = append(out, t)
			continue
		}
		decls = append(decls, t.FunctionDeclarations...)
		if cpy := cloneToolWithoutFunctions(t); cpy != nil && !reflect.ValueOf(*cpy).IsZero() {
			out = append(out, cpy)
		}
	}

	if len(decls) > 0 {
		out = append([]*genai.Tool{{FunctionDeclarations: decls}}, out...)
	}
	return out
}

func cloneToolWithoutFunctions(t *genai.Tool) *genai.Tool {
	if t == nil {
		return nil
	}
	clone := *t
	clone.FunctionDeclarations = nil
	return &clone
}

// toGeminiToolChoice translates tool choice settings to Gemini tool config.
func toGeminiToolChoice(toolConfig *genai.ToolConfig, toolChoice ai.ToolChoice, tools []*ai.ToolDefinition) (*genai.ToolConfig, error) {
	var mode genai.FunctionCallingConfigMode
	switch toolChoice {
	case "":
		// No ToolChoice was set; leave whatever ToolConfig the user supplied
		// via config alone so RetrievalConfig and IncludeServerSideToolInvocations
		// survive instead of being clobbered to nil.
		return toolConfig, nil
	case ai.ToolChoiceAuto:
		mode = genai.FunctionCallingConfigModeAuto
	case ai.ToolChoiceRequired:
		mode = genai.FunctionCallingConfigModeAny
	case ai.ToolChoiceNone:
		mode = genai.FunctionCallingConfigModeNone
	default:
		return nil, fmt.Errorf("tool choice mode %q not supported", toolChoice)
	}

	var toolNames []string
	// Per docs, only set AllowedToolNames with mode set to ANY.
	if mode == genai.FunctionCallingConfigModeAny {
		for _, t := range tools {
			toolNames = append(toolNames, t.Name)
		}
	}

	// If a config already exists, just add as a parameter
	if toolConfig != nil {
		toolConfig.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode:                 mode,
			AllowedFunctionNames: toolNames,
		}
		return toolConfig, nil
	}

	return &genai.ToolConfig{
		FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 mode,
			AllowedFunctionNames: toolNames,
		},
	}, nil
}

var toolNameRegexCompiled = regexp.MustCompile(toolNameRegex)

// validToolName checks whether the provided tool name matches the
// following criteria:
// - Start with a letter or an underscore
// - Must be alphanumeric and can include underscores, dots or dashes
// - Maximum length of 64 chars
func validToolName(n string) bool {
	return toolNameRegexCompiled.MatchString(n)
}
