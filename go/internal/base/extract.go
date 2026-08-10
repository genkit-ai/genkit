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
	"unicode/utf8"
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

// What the grammar allows at the current position. A token that is not allowed
// where it appears means the document is malformed rather than truncated, and
// completion rewinds instead of trying to make sense of it.
const (
	allowValue = 1 << iota // a value may start here
	allowKey               // an object key may start here
	allowColon             // the ':' between a key and its value
	allowComma             // the ',' between members
	allowClose             // this container's own closer
)

// jsonFrame is a container the scan is currently inside. The outermost frame is
// the document itself, which has no closer and holds exactly one value.
type jsonFrame struct {
	closer byte // '}' or ']', or 0 for the document
	allow  uint8
}

// CompleteJSON attempts to complete an incomplete JSON string. The result is
// always valid JSON.
//
// Containers are closed innermost first, so an object nested in an array is
// closed before the array. A string value cut off mid-stream is kept and
// closed, minus any dangling escape sequence, and a truncated keyword is
// finished because only one keyword can start with a given letter. A tail that
// cannot be finished without inventing a value is dropped instead: a
// half-written key, a key whose value has not arrived yet, or a truncated
// number such as "1.".
//
// Input that is malformed rather than truncated is cut back to the last
// position that was still well formed, which is why the grammar is tracked at
// all: a trailing comma before a closer, a container opened where a key
// belongs, or a raw control character inside a string all end the scan and take
// everything after them with it.
func CompleteJSON(jsonStr string) string {
	s := strings.TrimSpace(jsonStr)
	if s == "" {
		return "{}"
	}

	stack := []jsonFrame{{allow: allowValue}}
	// safe is the length of the prefix that becomes valid JSON once the closers
	// for stack are appended. It only advances past a completed value, so a
	// truncated or malformed tail is discarded by rewinding to it.
	safe := 0

	// valueDone records that a complete value ended at end, leaving its
	// container ready for a separator and the document ready for nothing.
	valueDone := func(end int) {
		top := &stack[len(stack)-1]
		if top.closer == 0 {
			top.allow = 0
		} else {
			top.allow = allowComma | allowClose
		}
		safe = end
	}

	for i := 0; i < len(s); {
		top := &stack[len(stack)-1]

		switch c := s[i]; c {
		case ' ', '\t', '\n', '\r':
			i++

		case '{', '[':
			if top.allow&allowValue == 0 {
				return closeJSON(s[:safe], stack)
			}
			frame := jsonFrame{closer: '}', allow: allowKey | allowClose}
			if c == '[' {
				frame = jsonFrame{closer: ']', allow: allowValue | allowClose}
			}
			stack = append(stack, frame)
			i++
			safe = i

		case '}', ']':
			if top.allow&allowClose == 0 || top.closer != c {
				return closeJSON(s[:safe], stack)
			}
			stack = stack[:len(stack)-1]
			i++
			valueDone(i)

		case ',':
			if top.allow&allowComma == 0 {
				return closeJSON(s[:safe], stack)
			}
			if top.closer == '}' {
				top.allow = allowKey
			} else {
				top.allow = allowValue
			}
			i++

		case ':':
			if top.allow&allowColon == 0 {
				return closeJSON(s[:safe], stack)
			}
			top.allow = allowValue
			i++

		case '"':
			isKey := top.allow&allowKey != 0
			if !isKey && top.allow&allowValue == 0 {
				return closeJSON(s[:safe], stack)
			}
			end, cut := scanJSONString(s, i)
			if end < 0 {
				// The string is still streaming. A partial value is worth
				// keeping; a partial key has no value to attach to.
				if isKey {
					return closeJSON(s[:safe], stack)
				}
				return closeJSON(trimPartialRune(s[:cut])+`"`, stack)
			}
			i = end
			if isKey {
				top.allow = allowColon
			} else {
				valueDone(i)
			}

		default:
			if top.allow&allowValue == 0 {
				return closeJSON(s[:safe], stack)
			}
			end := scanJSONAtom(s, i)
			if !json.Valid([]byte(s[i:end])) {
				if kw := completeKeyword(s[i:end]); kw != "" && end == len(s) {
					return closeJSON(s[:i]+kw, stack)
				}
				// A truncated number would have to be guessed at, so the whole
				// member goes instead.
				return closeJSON(s[:safe], stack)
			}
			i = end
			valueDone(i)
		}
	}

	return closeJSON(s[:safe], stack)
}

// closeJSON appends the closers for the containers left open in stack,
// innermost first. The document frame has no closer to append.
func closeJSON(prefix string, stack []jsonFrame) string {
	var b strings.Builder
	b.Grow(len(prefix) + len(stack))
	b.WriteString(prefix)
	for i := len(stack) - 1; i >= 0; i-- {
		if stack[i].closer != 0 {
			b.WriteByte(stack[i].closer)
		}
	}
	if b.Len() == 0 {
		return "{}"
	}
	return b.String()
}

// scanJSONString scans the string literal starting at s[start], which is the
// opening quote. It returns the index just past the closing quote, or -1 if the
// literal does not end there along with cut, the length of the prefix that
// stays valid once a closing quote is appended.
//
// Content JSON does not permit inside a string ends the literal early, at the
// offending byte, rather than being carried into the result. A closing quote
// later in the input cannot rescue it: the bytes before that quote would still
// be illegal.
func scanJSONString(s string, start int) (end, cut int) {
	for i := start + 1; i < len(s); i++ {
		switch c := s[i]; {
		case c == '"':
			return i + 1, 0

		case c < 0x20:
			// A raw control character. Models emit these for newlines inside
			// string values, where JSON requires \n.
			return -1, i

		case c == '\\':
			// The escape and everything it consumes must both have arrived, or
			// the quote we append would be swallowed by the escape.
			if i+1 >= len(s) {
				return -1, i
			}
			switch s[i+1] {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			case 'u':
				if i+5 >= len(s) || !isHex(s[i+2]) || !isHex(s[i+3]) || !isHex(s[i+4]) || !isHex(s[i+5]) {
					return -1, i
				}
				i += 4 // Skip \uXXXX; the loop's i++ accounts for the 'u'.
			default:
				return -1, i // Not an escape JSON recognizes.
			}
			i++
		}
	}
	return -1, len(s)
}

// isHex reports whether c is a hexadecimal digit.
func isHex(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}

// completeKeyword returns the JSON keyword that tok is an unfinished prefix of.
// No two keywords share a first letter, so the keyword a truncated one was
// going to become is the only one it could have become.
func completeKeyword(tok string) string {
	if tok == "" {
		return ""
	}
	for _, kw := range []string{"true", "false", "null"} {
		if len(tok) < len(kw) && strings.HasPrefix(kw, tok) {
			return kw
		}
	}
	return ""
}

// trimPartialRune drops a trailing UTF-8 sequence that a chunk boundary cut in
// half, so the closed string does not decode to a replacement character.
func trimPartialRune(s string) string {
	for i := 0; i < utf8.UTFMax && s != ""; i++ {
		r, size := utf8.DecodeLastRuneInString(s)
		if size == 0 || r != utf8.RuneError || size > 1 {
			// size > 1 means the input really does encode U+FFFD.
			break
		}
		s = s[:len(s)-size]
	}
	return s
}

// scanJSONAtom returns the end of the number or keyword starting at s[start].
func scanJSONAtom(s string, start int) int {
	for i := start; i < len(s); i++ {
		switch s[i] {
		case ',', ':', '{', '}', '[', ']', '"', ' ', '\t', '\n', '\r':
			return i
		}
	}
	return len(s)
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
