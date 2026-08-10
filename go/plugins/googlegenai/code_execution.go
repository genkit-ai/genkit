// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"github.com/firebase/genkit/go/ai"
)

// CodeExecutionResult represents the result of a code execution.
type CodeExecutionResult struct {
	Outcome string `json:"outcome"`
	Output  string `json:"output"`
}

// ExecutableCode represents executable code.
type ExecutableCode struct {
	Language string `json:"language"`
	Code     string `json:"code"`
}

// newCodeExecutionResultPart returns a Part containing the result of code execution.
// This is internal and used by translateCandidate.
func newCodeExecutionResultPart(outcome string, output string) *ai.Part {
	return ai.NewCustomPart(map[string]any{
		"codeExecutionResult": map[string]any{
			"outcome": outcome,
			"output":  output,
		},
	})
}

// newExecutableCodePart returns a Part containing executable code.
// This is internal and used by translateCandidate.
func newExecutableCodePart(language string, code string) *ai.Part {
	return ai.NewCustomPart(map[string]any{
		"executableCode": map[string]any{
			"language": language,
			"code":     code,
		},
	})
}

// ToCodeExecutionResult tries to convert an ai.Part to a CodeExecutionResult.
// Returns nil if the part doesn't contain code execution results.
func ToCodeExecutionResult(part *ai.Part) *CodeExecutionResult {
	if !part.IsCustom() {
		return nil
	}

	codeExec, ok := part.Custom["codeExecutionResult"]
	if !ok {
		return nil
	}

	result, ok := codeExec.(map[string]any)
	if !ok {
		return nil
	}

	outcome, _ := result["outcome"].(string)
	output, _ := result["output"].(string)

	return &CodeExecutionResult{
		Outcome: outcome,
		Output:  output,
	}
}

// ToExecutableCode tries to convert an ai.Part to an ExecutableCode.
// Returns nil if the part doesn't contain executable code.
func ToExecutableCode(part *ai.Part) *ExecutableCode {
	if !part.IsCustom() {
		return nil
	}

	execCode, ok := part.Custom["executableCode"]
	if !ok {
		return nil
	}

	code, ok := execCode.(map[string]any)
	if !ok {
		return nil
	}

	language, _ := code["language"].(string)
	codeStr, _ := code["code"].(string)

	return &ExecutableCode{
		Language: language,
		Code:     codeStr,
	}
}

// HasCodeExecution checks if a message contains code execution results or executable code.
func HasCodeExecution(msg *ai.Message) bool {
	return GetCodeExecutionResult(msg) != nil || GetExecutableCode(msg) != nil
}

// GetExecutableCode returns the first executable code from a message.
// Returns nil if the message doesn't contain executable code.
func GetExecutableCode(msg *ai.Message) *ExecutableCode {
	for _, part := range msg.Content {
		if code := ToExecutableCode(part); code != nil {
			return code
		}
	}
	return nil
}

// GetCodeExecutionResult returns the first code execution result from a message.
// Returns nil if the message doesn't contain a code execution result.
func GetCodeExecutionResult(msg *ai.Message) *CodeExecutionResult {
	for _, part := range msg.Content {
		if result := ToCodeExecutionResult(part); result != nil {
			return result
		}
	}
	return nil
}
