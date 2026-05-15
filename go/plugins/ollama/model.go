// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package ollama

import (
	"errors"
	"fmt"
	"maps"

	"github.com/firebase/genkit/go/ai"
)

var topLevelOpts = map[string]struct{}{
	"think":      {},
	"keep_alive": {},
}

// Ollama has two API endpoints, one with a chat interface and another with a generate response interface.
// That's why have multiple request interfaces for the Ollama API below.

/*
TODO: Support optional, advanced parameters:
format: the format to return a response in. Currently the only accepted value is json
options: additional model parameters listed in the documentation for the Modelfile such as temperature
system: system message to (overrides what is defined in the Modelfile)
template: the prompt template to use (overrides what is defined in the Modelfile)
context: the context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory
stream: if false the response will be returned as a single response object, rather than a stream of objects
raw: if true no formatting will be applied to the prompt. You may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API
*/
type ollamaChatRequest struct {
	Messages  []*ollamaMessage `json:"messages"`
	Images    []string         `json:"images,omitempty"`
	Model     string           `json:"model"`
	Stream    bool             `json:"stream"`
	Format    string           `json:"format,omitempty"`
	Tools     []ollamaTool     `json:"tools,omitempty"`
	Think     *ThinkOption     `json:"think,omitempty"`
	Options   map[string]any   `json:"options,omitempty"`
	KeepAlive string           `json:"keep_alive,omitempty"`
}

func (o *ollamaChatRequest) ApplyOptions(cfg any) error {
	if cfg == nil {
		return nil
	}

	switch config := cfg.(type) {
	case GenerateContentConfig:
		return o.applyGenerateContentConfig(&config)
	case *GenerateContentConfig:
		return o.applyGenerateContentConfig(config)
	case map[string]any:
		return o.applyMapAny(config)
	case *ai.GenerationCommonConfig:
		return o.applyGenerationCommonConfig(config)
	case ai.GenerationCommonConfig:
		return o.applyGenerationCommonConfig(&config)
	default:
		return fmt.Errorf("unexpected config type: %T", cfg)
	}
}

func (o *ollamaChatRequest) applyGenerateContentConfig(cfg *GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}

	if cfg.Think != nil {
		o.Think = cfg.Think
	}

	if cfg.KeepAlive != "" {
		o.KeepAlive = cfg.KeepAlive
	}

	opts := make(map[string]any)
	if cfg.Seed != nil {
		opts["seed"] = *cfg.Seed
	}
	if cfg.Temperature != nil {
		opts["temperature"] = *cfg.Temperature
	}
	if cfg.TopK != nil {
		opts["top_k"] = *cfg.TopK
	}
	if cfg.TopP != nil {
		opts["top_p"] = *cfg.TopP
	}
	if cfg.MinP != nil {
		opts["min_p"] = *cfg.MinP
	}
	if len(cfg.Stop) > 0 {
		opts["stop"] = cfg.Stop
	}
	if cfg.NumCtx != nil {
		opts["num_ctx"] = *cfg.NumCtx
	}
	if cfg.NumPredict != nil {
		opts["num_predict"] = *cfg.NumPredict
	}

	if len(opts) > 0 {
		if o.Options == nil {
			o.Options = make(map[string]any)
		}
		maps.Copy(o.Options, opts)
	}

	return nil
}

func (o *ollamaChatRequest) applyGenerationCommonConfig(cfg *ai.GenerationCommonConfig) error {
	if cfg == nil {
		return nil
	}

	m := make(map[string]any)

	if cfg.MaxOutputTokens > 0 {
		m["num_predict"] = cfg.MaxOutputTokens
	}
	if len(cfg.StopSequences) > 0 {
		m["stop"] = cfg.StopSequences
	}
	if cfg.Temperature != 0 {
		m["temperature"] = cfg.Temperature
	}
	if cfg.TopK > 0 {
		m["top_k"] = cfg.TopK
	}
	if cfg.TopP > 0 {
		m["top_p"] = cfg.TopP
	}

	return o.applyMapAny(m)
}

func (o *ollamaChatRequest) applyMapAny(m map[string]any) error {
	if len(m) == 0 {
		return nil
	}
	opts := o.Options
	if opts == nil {
		opts = make(map[string]any)
	}
	for k, v := range m {
		if _, isTopLevel := topLevelOpts[k]; isTopLevel {
			switch k {
			case "think":
				switch tv := v.(type) {
				case bool:
					o.Think = ThinkEnabled(tv)
				case string:
					o.Think = ThinkEffort(tv)
				case *ThinkOption:
					o.Think = tv
				default:
					return fmt.Errorf("think must be a boolean or string, got: %T", v)
				}
			case "keep_alive":
				if s, ok := v.(string); ok {
					o.KeepAlive = s
				} else {
					return errors.New("keep_alive must be string")
				}
			}
			continue
		}
		opts[k] = v
	}

	if len(opts) > 0 {
		o.Options = opts
	}

	return nil
}
