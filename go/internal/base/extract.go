// Copyright 2024 Google LLC
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

package base

import (
	"encoding/json"
	"strings"
)

// ExtractJSON extracts JSON from string with lenient parsing rules.
// It handles both complete and partial JSON structures.
func ExtractJSON(text string) (any, error) {
	var openingChar, closingChar rune
	var startPos int = -1
	nestingCount := 0
	inString := false
	escapeNext := false

	for i, char := range text {
		// Replace non-breaking space with regular space
		if char == '\u00A0' {
			char = ' '
		}

		if escapeNext {
			escapeNext = false
			continue
		}

		if char == '\\' {
			escapeNext = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if openingChar == 0 && (char == '{' || char == '[') {
			// Look for opening character
			openingChar = char
			if char == '{' {
				closingChar = '}'
			} else {
				closingChar = ']'
			}
			startPos = i
			nestingCount++
		} else if char == openingChar {
			// Increment nesting for matching opening character
			nestingCount++
		} else if char == closingChar {
			// Decrement nesting for matching closing character
			nestingCount--
			if nestingCount == 0 {
				// Reached end of target element
				jsonStr := text[startPos : i+1]
				var result any
				err := json.Unmarshal([]byte(jsonStr), &result)
				if err != nil {
					return nil, err
				}
				return result, nil
			}
		}
	}

	if startPos != -1 && nestingCount > 0 {
		// If an incomplete JSON structure is detected, try to parse it partially
		jsonStr := text[startPos:]
		result, err := ParsePartialJSON(jsonStr)
		if err != nil {
			return nil, err
		}
		return result, nil
	}

	return nil, nil
}

// ParsePartialJSON attempts to parse incomplete JSON by completing it.
func ParsePartialJSON(jsonStr string) (any, error) {
	// Try to parse as-is first
	var result any
	err := json.Unmarshal([]byte(jsonStr), &result)
	if err == nil {
		return result, nil
	}

	// If it fails, try to complete the JSON structure
	completed := CompleteJSON(jsonStr)
	err = json.Unmarshal([]byte(completed), &result)
	return result, err
}

// CompleteJSON attempts to complete an incomplete JSON string.
func CompleteJSON(jsonStr string) string {
	jsonStr = strings.TrimSpace(jsonStr)
	if jsonStr == "" {
		return "{}"
	}

	// Count unclosed structures
	var openBraces, openBrackets int
	inString := false
	escapeNext := false

	for _, char := range jsonStr {
		if escapeNext {
			escapeNext = false
			continue
		}

		if char == '\\' {
			escapeNext = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		switch char {
		case '{':
			openBraces++
		case '}':
			openBraces--
		case '[':
			openBrackets++
		case ']':
			openBrackets--
		}
	}

	// Close any unclosed string
	if inString {
		jsonStr += "\""
	}

	// Remove trailing comma if present (before closing)
	jsonStr = strings.TrimRight(jsonStr, " \t\n\r")
	jsonStr = strings.TrimSuffix(jsonStr, ",")

	// Close open structures
	for i := 0; i < openBrackets; i++ {
		jsonStr += "]"
	}
	for i := 0; i < openBraces; i++ {
		jsonStr += "}"
	}

	return jsonStr
}

// ExtractItemsResult contains the result of extracting items from an array.
type ExtractItemsResult struct {
	Items  []any
	Cursor int
}

// ExtractItems extracts complete objects from the first array found in the text.
// Processes text from the cursor position and returns both complete items
// and the new cursor position.
func ExtractItems(text string, cursor int) ExtractItemsResult {
	items := []any{}
	currentCursor := cursor

	// Find the first array start if we haven't already processed any text
	if cursor == 0 {
		arrayStart := strings.Index(text, "[")
		if arrayStart == -1 {
			return ExtractItemsResult{Items: items, Cursor: len(text)}
		}
		currentCursor = arrayStart + 1
	}

	objectStart := -1
	braceCount := 0
	inString := false
	escapeNext := false

	// Process the text from the cursor position
	for i := currentCursor; i < len(text); i++ {
		char := rune(text[i])

		if escapeNext {
			escapeNext = false
			continue
		}

		if char == '\\' {
			escapeNext = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if char == '{' {
			if braceCount == 0 {
				objectStart = i
			}
			braceCount++
		} else if char == '}' {
			braceCount--
			if braceCount == 0 && objectStart != -1 {
				var obj any
				err := json.Unmarshal([]byte(text[objectStart:i+1]), &obj)
				if err == nil {
					items = append(items, obj)
					currentCursor = i + 1
					objectStart = -1
				}
			}
		} else if char == ']' && braceCount == 0 {
			// End of array
			break
		}
	}

	return ExtractItemsResult{
		Items:  items,
		Cursor: currentCursor,
	}
}
