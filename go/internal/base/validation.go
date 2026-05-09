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

package base

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/xeipuuv/gojsonschema"
)

// ValidateValue will validate any value against the expected schema.
// It will return an error if it doesn't match the schema, otherwise it will return nil.
func ValidateValue(data any, schema map[string]any) error {
	if schema == nil {
		return nil
	}
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("data is not a valid JSON type: %w", err)
	}
	return ValidateJSON(dataBytes, schema)
}

// ValidateJSON will validate JSON against the expected schema.
// It will return an error if it doesn't match the schema, otherwise it will return nil.
func ValidateJSON(dataBytes json.RawMessage, schema map[string]any) error {
	if schema == nil {
		return nil
	}
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("expected schema is not valid: %w", err)
	}
	return ValidateRaw(dataBytes, schemaBytes)
}

// ValidateRaw will validate JSON data against the JSON schema.
// It will return an error if it doesn't match the schema, otherwise it will return nil.
func ValidateRaw(dataBytes json.RawMessage, schemaBytes json.RawMessage) error {
	var data any
	// Do this check separately from below to get a better error message.
	if err := json.Unmarshal(dataBytes, &data); err != nil {
		return fmt.Errorf("data is not valid JSON: %w", err)
	}

	schemaLoader := gojsonschema.NewBytesLoader(schemaBytes)
	documentLoader := gojsonschema.NewBytesLoader(dataBytes)

	result, err := gojsonschema.Validate(schemaLoader, documentLoader)
	if err != nil {
		return fmt.Errorf("failed to validate data against expected schema: %w", err)
	}

	if !result.Valid() {
		var errors []string
		for _, err := range result.Errors() {
			errors = append(errors, fmt.Sprintf("- %s", err))
		}
		return fmt.Errorf("data did not match expected schema:\n%s", strings.Join(errors, "\n"))
	}

	return nil
}

// ValidateIsJSONArray will validate if the schema represents a JSON array.
func ValidateIsJSONArray(schema map[string]any) bool {
	if sType, ok := schema["type"]; !ok || sType != "array" {
		return false
	}

	if _, ok := schema["items"]; !ok {
		return false
	}

	return true
}

// Validates if the given string is a valid JSON string
func ValidJSON(s string) bool {
	var js json.RawMessage
	return json.Unmarshal([]byte(s), &js) == nil
}
