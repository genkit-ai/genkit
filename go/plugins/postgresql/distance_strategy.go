// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package postgresql

import "fmt"

type DistanceStrategy interface {
	String() string
	operator() string
	searchFunction() string
	similaritySearchFunction() string
}

type Euclidean struct{}

func (e Euclidean) String() string {
	return "euclidean"
}

func (e Euclidean) operator() string {
	return "<->"
}

func (e Euclidean) searchFunction() string {
	return "vector_l2_ops"
}

func (e Euclidean) similaritySearchFunction() string {
	return "l2_distance"
}

type CosineDistance struct{}

func (c CosineDistance) String() string {
	return "cosineDistance"
}

func (c CosineDistance) operator() string {
	return "<=>"
}

func (c CosineDistance) searchFunction() string {
	return "vector_cosine_ops"
}

func (c CosineDistance) similaritySearchFunction() string {
	return "cosine_distance"
}

type InnerProduct struct{}

func (i InnerProduct) String() string {
	return "innerProduct"
}

func (i InnerProduct) operator() string {
	return "<#>"
}

func (i InnerProduct) searchFunction() string {
	return "vector_ip_ops"
}

func (i InnerProduct) similaritySearchFunction() string {
	return "inner_product"
}

type Index interface {
	Options() string
}

type BaseIndex struct {
	name             string
	indexType        string
	options          Index
	distanceStrategy DistanceStrategy
	partialIndexes   []string
}

func NewBaseIndex(indexName, indexType string, strategy DistanceStrategy, partialIndexes []string, opts Index) BaseIndex {
	return BaseIndex{
		name:             indexName,
		indexType:        indexType,
		distanceStrategy: strategy,
		partialIndexes:   partialIndexes,
		options:          opts,
	}
}

// HNSWOptions holds the configuration for the hnsw index.
type HNSWOptions struct {
	M              int
	EfConstruction int
}

func (h HNSWOptions) Options() string {
	return fmt.Sprintf("(m = %d, ef_construction = %d)", h.M, h.EfConstruction)
}

// IVFFlatOptions holds the configuration for the ivfflat index.
type IVFFlatOptions struct {
	Lists int
}

func (i IVFFlatOptions) Options() string {
	return fmt.Sprintf("(lists = %d)", i.Lists)
}

// indexOptions returns the specific options for the index based on the index api.
func (index *BaseIndex) indexOptions() string {
	return index.options.Options()
}
