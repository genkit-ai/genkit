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

// Package bedrock provides a Genkit plugin for Amazon Bedrock — text
// generation (Claude, Nova, Llama, Mistral, AI21, Cohere Command, DeepSeek,
// Writer Palmyra) via the Converse API, embedders via InvokeModel (Titan text +
// image, Cohere text + image, Nova multimodal), reranking via [Rerank] (Cohere
// Rerank), and image generation via InvokeModel (Titan Image, Nova Canvas,
// Stable Diffusion).
//
// The plugin uses the standard AWS credential chain via
// [config.LoadDefaultConfig]. To use it:
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&bedrock.Bedrock{Region: "us-east-1"}))
//	model := bedrock.DefineModel(g, "anthropic.claude-3-5-sonnet-20241022-v2:0", nil)
//
// Cross-region inference profile IDs (e.g. "us.anthropic.claude-...") are
// passed through verbatim as Bedrock model IDs.
//
// Each Claude / Nova / Llama / Mistral foundation model requires a one-time
// "Request model access" approval per region in the AWS Bedrock console
// before it becomes callable. The plugin surfaces the underlying AWS error
// when access has not been granted.
//
// Request timeouts are not exposed as a plugin field — callers use
// [context.WithTimeout] on the call site.
package bedrock

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "bedrock"

// Bedrock is the Amazon Bedrock plugin. Register one instance per region via
// [genkit.WithPlugins] during [genkit.Init].
//
// Auth resolution order:
//  1. AWSConfig (if set) is used verbatim — Region is ignored.
//  2. Otherwise [config.LoadDefaultConfig] is called with [config.WithRegion]
//     only if Region is non-empty, allowing the standard chain to determine
//     credentials (env, profile, EC2/ECS role, SSO, web identity).
type Bedrock struct {
	// Region targets a specific AWS region (e.g. "us-east-1"). If empty,
	// the region resolves through the AWS SDK's default chain (AWS_REGION
	// env var, ~/.aws/config, EC2/ECS metadata).
	Region string

	// AWSConfig, if non-nil, is used verbatim and Region is ignored.
	// Intended for advanced wiring (custom credential providers, fakes,
	// shared config across plugins).
	AWSConfig *aws.Config

	client  *bedrockruntime.Client
	mu      sync.Mutex
	initted bool
}

// Name returns the plugin's provider identifier ("bedrock").
func (b *Bedrock) Name() string { return provider }

// Init constructs the bedrockruntime client. It panics on credential or
// config errors — the standard Genkit plugin lifecycle does not allow Init to
// return an error.
func (b *Bedrock) Init(ctx context.Context) []api.Action {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.initted {
		panic("bedrock.Init already called")
	}

	var cfg aws.Config
	if b.AWSConfig != nil {
		cfg = *b.AWSConfig
	} else {
		opts := []func(*config.LoadOptions) error{}
		if b.Region != "" {
			opts = append(opts, config.WithRegion(b.Region))
		}
		c, err := config.LoadDefaultConfig(ctx, opts...)
		if err != nil {
			panic(fmt.Errorf("bedrock.Init: load AWS config: %w", err))
		}
		cfg = c
	}
	if cfg.Region == "" {
		panic("bedrock.Init: no AWS region resolved; set Bedrock.Region, AWS_REGION, or a region in ~/.aws/config")
	}

	b.client = bedrockruntime.NewFromConfig(cfg)
	b.initted = true
	return []api.Action{}
}

// Client returns the underlying bedrockruntime client. Returns nil before
// Init. Provided as an escape hatch for advanced use cases not covered by the
// plugin's higher-level API.
func (b *Bedrock) Client() *bedrockruntime.Client { return b.client }

// DefineModel registers a Converse-backed Bedrock model. The name argument is
// the full Bedrock model ID, including any cross-region inference profile
// prefix (e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0" or
// "us.anthropic.claude-3-5-sonnet-20241022-v2:0").
//
// If opts is nil, default capability options are looked up from the built-in
// model registry; pass an explicit opts to override or to register a model
// not in the registry.
func DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	if name == "" {
		return nil, errors.New("bedrock.DefineModel: model name required")
	}
	if opts == nil {
		opts = defaultModelOptions(name)
	}
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = configSchema()
	}
	if opts.Label == "" {
		opts.Label = "Bedrock - " + name
	}
	m := genkit.DefineModel(g, api.NewName(provider, name), opts, func(ctx context.Context, req *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
		p, _ := genkit.LookupPlugin(g, provider).(*Bedrock)
		if p == nil {
			return nil, errors.New("bedrock plugin not registered; pass &bedrock.Bedrock{...} to genkit.WithPlugins")
		}
		if !p.initted {
			return nil, errors.New("bedrock.DefineModel: plugin not initialized")
		}
		return generate(ctx, p.client, name, req, cb)
	})
	return m, nil
}

// DefineEmbedder registers a Bedrock embedder by model ID. Supported models
// are Titan ("amazon.titan-embed-text-v1", "v2:0", "amazon.titan-embed-image-v1"),
// Cohere ("cohere.embed-english-v3", "cohere.embed-multilingual-v3", with image
// input via the Cohere multimodal embedders), and Amazon Nova multimodal
// ("amazon.nova-2-multimodal-embeddings-v1:0"). The JSON wire shape is inferred
// from the model-ID prefix.
//
// Documents carrying image media are embedded as images; Nova and per-document
// Cohere image requests issue one InvokeModel call per document, while
// text-only Cohere requests are batched into a single call.
//
// If opts is nil, capability metadata is derived from the model ID
// (multimodal model IDs advertise image input; everything else is text-only).
func DefineEmbedder(g *genkit.Genkit, name string, opts *ai.EmbedderOptions) (ai.Embedder, error) {
	if name == "" {
		return nil, errors.New("bedrock.DefineEmbedder: name required")
	}
	if opts == nil {
		opts = defaultEmbedderOptions(name)
	}
	if opts.Label == "" {
		opts.Label = "Bedrock - " + name
	}
	emb := genkit.DefineEmbedder(g, api.NewName(provider, name), opts, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		p, _ := genkit.LookupPlugin(g, provider).(*Bedrock)
		if p == nil {
			return nil, errors.New("bedrock plugin not registered")
		}
		if !p.initted {
			return nil, errors.New("bedrock.DefineEmbedder: plugin not initialized")
		}
		return embedderFunc(p.client, name)(ctx, req)
	})
	return emb, nil
}

// DefineImager registers a Bedrock image-generation model. Supported model-ID
// shapes are Titan ("amazon.titan-image-generator-v1"), Nova Canvas
// ("amazon.nova-canvas-v1:0"), and Stable Diffusion
// ("stable-diffusion-xl-v1", "stability.sd3-large-*", etc.). The JSON wire
// shape is inferred from the model-ID prefix.
//
// Images are returned as base64 PNG embedded in [ai.Media] parts.
func DefineImager(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	if name == "" {
		return nil, errors.New("bedrock.DefineImager: name required")
	}
	if opts == nil {
		opts = &ai.ModelOptions{
			Supports: &ai.ModelSupports{Media: true},
			Stage:    ai.ModelStageStable,
		}
	}
	if opts.Label == "" {
		opts.Label = "Bedrock - " + name
	}
	m := genkit.DefineModel(g, api.NewName(provider, name), opts, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		p, _ := genkit.LookupPlugin(g, provider).(*Bedrock)
		if p == nil {
			return nil, errors.New("bedrock plugin not registered")
		}
		if !p.initted {
			return nil, errors.New("bedrock.DefineImager: plugin not initialized")
		}
		return imagerFunc(p.client, name)(ctx, req, cb)
	})
	return m, nil
}

// Model returns a previously registered model, or nil.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// Embedder returns a previously registered embedder, or nil.
func Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(provider, name))
}
