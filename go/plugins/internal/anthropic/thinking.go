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

package anthropic

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/anthropics/anthropic-sdk-go"
)

// ThinkingDisplay is the Anthropic thinking display mode for adaptive thinking.
type ThinkingDisplay string

const (
	ThinkingDisplaySummarized ThinkingDisplay = "summarized"
	ThinkingDisplayOmitted    ThinkingDisplay = "omitted"
	minThinkingBudgetTokens                   = 1024
)

// ThinkingConfig is the Genkit-shaped extended-thinking config (JS ThinkingConfigSchema parity).
type ThinkingConfig struct {
	Enabled      *bool           `json:"enabled,omitempty"`
	BudgetTokens *int64          `json:"budgetTokens,omitempty"`
	Adaptive     *bool           `json:"adaptive,omitempty"`
	Display      ThinkingDisplay `json:"display,omitempty"`
}

// thinkingConfigSchema is the JSON Schema advertised for map-style model config.
func thinkingConfigSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"enabled": map[string]any{
				"type": "boolean",
			},
			"budgetTokens": map[string]any{
				"type":    "integer",
				"minimum": minThinkingBudgetTokens,
			},
			"adaptive": map[string]any{
				"type": "boolean",
			},
			"display": map[string]any{
				"type": "string",
				"enum": []any{string(ThinkingDisplaySummarized), string(ThinkingDisplayOmitted)},
			},
		},
		"additionalProperties": true,
		"description":          "The thinking configuration to use for the request. Thinking allows the model to reason about the request and provide a better response.",
	}
}

func overlayThinkingConfigSchema(schema map[string]any) {
	if schema == nil {
		return
	}
	props, _ := schema["properties"].(map[string]any)
	if props == nil {
		props = map[string]any{}
		schema["properties"] = props
	}
	props["thinking"] = thinkingConfigSchema()
}

func looksLikeGenkitThinkingConfig(v any) bool {
	switch v.(type) {
	case ThinkingConfig, *ThinkingConfig:
		return true
	}
	m, ok := asStringAnyMap(v)
	if !ok {
		return false
	}
	_, hasEnabled := m["enabled"]
	_, hasBudget := m["budgetTokens"]
	_, hasAdaptive := m["adaptive"]
	_, hasDisplay := m["display"]
	return hasEnabled || hasBudget || hasAdaptive || hasDisplay
}

func asStringAnyMap(v any) (map[string]any, bool) {
	m, ok := v.(map[string]any)
	return m, ok
}

func parseThinkingConfig(v any) (ThinkingConfig, error) {
	switch c := v.(type) {
	case ThinkingConfig:
		return c, nil
	case *ThinkingConfig:
		if c == nil {
			return ThinkingConfig{}, nil
		}
		return *c, nil
	default:
		// Prefer the raw map so non-integer budgetTokens (e.g. 1024.5) can be
		// rejected with a clear error instead of a JSON unmarshal failure.
		if m, ok := asStringAnyMap(v); ok {
			return thinkingConfigFromMap(m)
		}
		data, err := json.Marshal(v)
		if err != nil {
			return ThinkingConfig{}, fmt.Errorf("invalid thinking config: %w", err)
		}
		var cfg ThinkingConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			return ThinkingConfig{}, fmt.Errorf("invalid thinking config: %w", err)
		}
		return cfg, nil
	}
}

func thinkingConfigFromMap(m map[string]any) (ThinkingConfig, error) {
	var cfg ThinkingConfig
	if raw, exists := m["enabled"]; exists && raw != nil {
		b, ok := raw.(bool)
		if !ok {
			return ThinkingConfig{}, fmt.Errorf("enabled must be a boolean")
		}
		cfg.Enabled = &b
	}
	if raw, exists := m["adaptive"]; exists && raw != nil {
		b, ok := raw.(bool)
		if !ok {
			return ThinkingConfig{}, fmt.Errorf("adaptive must be a boolean")
		}
		cfg.Adaptive = &b
	}
	if raw, exists := m["display"]; exists && raw != nil {
		s, ok := raw.(string)
		if !ok {
			return ThinkingConfig{}, fmt.Errorf("display must be a string")
		}
		cfg.Display = ThinkingDisplay(s)
	}
	if raw, exists := m["budgetTokens"]; exists && raw != nil {
		n, err := asInt64(raw)
		if err != nil {
			return ThinkingConfig{}, fmt.Errorf("budgetTokens must be an integer")
		}
		cfg.BudgetTokens = &n
	}
	return cfg, nil
}

func asInt64(v any) (int64, error) {
	switch n := v.(type) {
	case int:
		return int64(n), nil
	case int32:
		return int64(n), nil
	case int64:
		return n, nil
	case float32:
		if n < float32(math.MinInt64) || n > float32(math.MaxInt64) || float32(int64(n)) != n {
			return 0, fmt.Errorf("not an integer")
		}
		return int64(n), nil
	case float64:
		if n < float64(math.MinInt64) || n > float64(math.MaxInt64) || math.Trunc(n) != n {
			return 0, fmt.Errorf("not an integer")
		}
		return int64(n), nil
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, err
		}
		return i, nil
	default:
		return 0, fmt.Errorf("unsupported number type %T", v)
	}
}

func validateThinkingConfig(c ThinkingConfig) error {
	enabled := c.Enabled != nil && *c.Enabled
	adaptive := c.Adaptive != nil && *c.Adaptive

	if enabled && adaptive {
		return fmt.Errorf("cannot use both enabled and adaptive thinking modes simultaneously")
	}
	if c.Display != "" && c.Display != ThinkingDisplaySummarized && c.Display != ThinkingDisplayOmitted {
		return fmt.Errorf("display must be %q or %q", ThinkingDisplaySummarized, ThinkingDisplayOmitted)
	}
	if c.Display != "" && !adaptive {
		return fmt.Errorf("display can only be set when adaptive thinking is enabled")
	}

	// budgetTokens is only meaningful for enabled thinking (explicit or implied).
	implicitEnabled := c.Enabled == nil && c.BudgetTokens != nil && !adaptive
	if enabled || implicitEnabled {
		if c.BudgetTokens == nil {
			return fmt.Errorf("budgetTokens is required when thinking is enabled")
		}
		if *c.BudgetTokens < minThinkingBudgetTokens {
			return fmt.Errorf("budgetTokens must be >= %d", minThinkingBudgetTokens)
		}
	}
	return nil
}

// toAnthropicThinkingConfig maps Genkit ThinkingConfig onto Anthropic SDK thinking params.
// Returns ok=false when the config is empty / a no-op (thinking omitted).
func toAnthropicThinkingConfig(v any) (anthropic.ThinkingConfigParamUnion, bool, error) {
	cfg, err := parseThinkingConfig(v)
	if err != nil {
		return anthropic.ThinkingConfigParamUnion{}, false, err
	}
	if err := validateThinkingConfig(cfg); err != nil {
		return anthropic.ThinkingConfigParamUnion{}, false, err
	}

	if cfg.Adaptive != nil && *cfg.Adaptive {
		adaptive := anthropic.NewThinkingConfigAdaptiveParam()
		if cfg.Display != "" {
			adaptive.SetExtraFields(map[string]any{"display": string(cfg.Display)})
		}
		return anthropic.ThinkingConfigParamUnion{OfAdaptive: &adaptive}, true, nil
	}

	if cfg.Enabled != nil && *cfg.Enabled {
		return anthropic.ThinkingConfigParamOfEnabled(*cfg.BudgetTokens), true, nil
	}

	if cfg.Enabled != nil && !*cfg.Enabled {
		disabled := anthropic.NewThinkingConfigDisabledParam()
		return anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}, true, nil
	}

	if cfg.BudgetTokens != nil {
		return anthropic.ThinkingConfigParamOfEnabled(*cfg.BudgetTokens), true, nil
	}

	return anthropic.ThinkingConfigParamUnion{}, false, nil
}
