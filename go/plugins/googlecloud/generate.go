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

package googlecloud

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// GenerateTelemetry implements telemetry collection for model generate actions
type GenerateTelemetry struct {
	actionCounter    metric.Int64Counter   // genkit/ai/generate/requests
	latencies        metric.Int64Histogram // genkit/ai/generate/latency
	inputCharacters  metric.Int64Counter   // genkit/ai/generate/input/characters
	inputTokens      metric.Int64Counter   // genkit/ai/generate/input/tokens
	inputImages      metric.Int64Counter   // genkit/ai/generate/input/images
	inputVideos      metric.Int64Counter   // genkit/ai/generate/input/videos
	inputAudio       metric.Int64Counter   // genkit/ai/generate/input/audio
	outputCharacters metric.Int64Counter   // genkit/ai/generate/output/characters
	outputTokens     metric.Int64Counter   // genkit/ai/generate/output/tokens
	thinkingTokens   metric.Int64Counter   // genkit/ai/generate/thinking/tokens
	outputImages     metric.Int64Counter   // genkit/ai/generate/output/images
	outputVideos     metric.Int64Counter   // genkit/ai/generate/output/videos
	outputAudio      metric.Int64Counter   // genkit/ai/generate/output/audio
}

// NewGenerateTelemetry creates a new generate telemetry module with all required metrics
func NewGenerateTelemetry() *GenerateTelemetry {
	meter := otel.Meter("genkit")

	actionCounter, _ := meter.Int64Counter("genkit/ai/generate/requests", metric.WithDescription("Counts calls to genkit generate actions."), metric.WithUnit("1"))
	latencies, _ := meter.Int64Histogram("genkit/ai/generate/latency", metric.WithDescription("Latencies when interacting with a Genkit model."), metric.WithUnit("ms"))
	inputCharacters, _ := meter.Int64Counter("genkit/ai/generate/input/characters", metric.WithDescription("Counts input characters to any Genkit model."), metric.WithUnit("1"))
	inputTokens, _ := meter.Int64Counter("genkit/ai/generate/input/tokens", metric.WithDescription("Counts input tokens to a Genkit model."), metric.WithUnit("1"))
	inputImages, _ := meter.Int64Counter("genkit/ai/generate/input/images", metric.WithDescription("Counts input images to a Genkit model."), metric.WithUnit("1"))
	inputVideos, _ := meter.Int64Counter("genkit/ai/generate/input/videos", metric.WithDescription("Counts input videos to a Genkit model."), metric.WithUnit("1"))
	inputAudio, _ := meter.Int64Counter("genkit/ai/generate/input/audio", metric.WithDescription("Counts input audio files to a Genkit model."), metric.WithUnit("1"))
	outputCharacters, _ := meter.Int64Counter("genkit/ai/generate/output/characters", metric.WithDescription("Counts output characters from a Genkit model."), metric.WithUnit("1"))
	outputTokens, _ := meter.Int64Counter("genkit/ai/generate/output/tokens", metric.WithDescription("Counts output tokens from a Genkit model."), metric.WithUnit("1"))
	thinkingTokens, _ := meter.Int64Counter("genkit/ai/generate/thinking/tokens", metric.WithDescription("Counts thinking tokens from a Genkit model."), metric.WithUnit("1"))
	outputImages, _ := meter.Int64Counter("genkit/ai/generate/output/images", metric.WithDescription("Count output images from a Genkit model."), metric.WithUnit("1"))
	outputVideos, _ := meter.Int64Counter("genkit/ai/generate/output/videos", metric.WithDescription("Count output videos from a Genkit model."), metric.WithUnit("1"))
	outputAudio, _ := meter.Int64Counter("genkit/ai/generate/output/audio", metric.WithDescription("Count output audio files from a Genkit model."), metric.WithUnit("1"))

	return &GenerateTelemetry{
		actionCounter:    actionCounter,
		latencies:        latencies,
		inputCharacters:  inputCharacters,
		inputTokens:      inputTokens,
		inputImages:      inputImages,
		inputVideos:      inputVideos,
		inputAudio:       inputAudio,
		outputCharacters: outputCharacters,
		outputTokens:     outputTokens,
		thinkingTokens:   thinkingTokens,
		outputImages:     outputImages,
		outputVideos:     outputVideos,
		outputAudio:      outputAudio,
	}
}

// Tick processes a span for generate telemetry
func (g *GenerateTelemetry) Tick(span sdktrace.ReadOnlySpan, logInputOutput bool, projectID string) {
	attributes := span.Attributes()

	modelName := truncate(extractStringAttribute(attributes, "genkit:name"), 1024)
	path := extractStringAttribute(attributes, "genkit:path")
	inputStr := extractStringAttribute(attributes, "genkit:input")
	outputStr := extractStringAttribute(attributes, "genkit:output")

	var input ai.GenerateActionOptions
	var output ai.ModelResponse

	if inputStr != "" {
		json.Unmarshal([]byte(inputStr), &input)
	}

	if outputStr != "" {
		json.Unmarshal([]byte(outputStr), &output)
	}

	errName := g.extractErrorName(span)

	featureName := truncate(g.extractFeatureName(attributes, path))

	sessionId := extractStringAttribute(attributes, "genkit:sessionId")
	threadName := extractStringAttribute(attributes, "genkit:threadName")

	if input.Config != nil {
		g.recordGenerateActionConfigLogs(span, modelName, featureName, path, &input, projectID, sessionId, threadName)
	}

	if inputStr != "" && logInputOutput {
		g.recordGenerateActionInputLogs(span, modelName, featureName, path, &input, projectID, sessionId, threadName)
	}

	if outputStr != "" && logInputOutput {
		g.recordGenerateActionOutputLogs(span, modelName, featureName, path, &output, projectID, sessionId, threadName)
	}
	if featureName == "" || featureName == "<unknown>" {
		featureName = "generate"
	}

	if inputStr != "" {
		g.recordGenerateActionMetrics(modelName, featureName, path, &output, errName)
	}
}

// recordGenerateActionMetrics records all metrics for a generate action
func (g *GenerateTelemetry) recordGenerateActionMetrics(modelName, featureName, path string, output *ai.ModelResponse, errName string) {
	status := "success"
	if errName != "" {
		status = "failure"
	}

	attrs := []attribute.KeyValue{
		attribute.String("modelName", modelName),
		attribute.String("featureName", featureName),
		attribute.String("path", path),
		attribute.String("status", status),
		attribute.String("source", "go"),
		attribute.String("sourceVersion", internal.Version),
	}

	errorAttrs := attrs
	if errName != "" {
		errorAttrs = append(attrs, attribute.String("error", errName))
	}
	g.actionCounter.Add(context.Background(), 1, metric.WithAttributes(errorAttrs...))

	if output != nil && output.LatencyMs > 0 {
		g.latencies.Record(context.Background(), int64(output.LatencyMs), metric.WithAttributes(attrs...))
	}

	if usage := output.Usage; usage != nil {
		opt := metric.WithAttributes(attrs...)
		g.inputTokens.Add(context.Background(), int64(usage.InputTokens), opt)
		g.inputCharacters.Add(context.Background(), int64(usage.InputCharacters), opt)
		g.inputImages.Add(context.Background(), int64(usage.InputImages), opt)
		g.inputVideos.Add(context.Background(), int64(usage.InputVideos), opt)
		g.inputAudio.Add(context.Background(), int64(usage.InputAudioFiles), opt)
		g.outputTokens.Add(context.Background(), int64(usage.OutputTokens), opt)
		g.outputCharacters.Add(context.Background(), int64(usage.OutputCharacters), opt)
		g.thinkingTokens.Add(context.Background(), int64(usage.ThoughtsTokens), opt)
		g.outputImages.Add(context.Background(), int64(usage.OutputImages), opt)
		g.outputVideos.Add(context.Background(), int64(usage.OutputVideos), opt)
		g.outputAudio.Add(context.Background(), int64(usage.OutputAudioFiles), opt)
	} else {
		slog.Warn("GenerateTelemetry.Tick: No usage data available", "output_is_nil", output == nil)
	}
}

// recordGenerateActionConfigLogs logs configuration information
func (g *GenerateTelemetry) recordGenerateActionConfigLogs(span sdktrace.ReadOnlySpan, model, featureName, qualifiedPath string, input *ai.GenerateActionOptions, projectID, sessionID, threadName string) {
	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	path := truncatePath(toDisplayPath(qualifiedPath))
	sharedMetadata := createCommonLogAttributes(span, projectID)

	logData := map[string]interface{}{
		"model":         model,
		"path":          path,
		"qualifiedPath": qualifiedPath,
		"featureName":   featureName,
		"source":        "go",
		"sourceVersion": internal.Version,
	}

	if sessionID != "" {
		logData["sessionId"] = sessionID
	}
	if threadName != "" {
		logData["threadName"] = threadName
	}

	for k, v := range sharedMetadata {
		logData[k] = v
	}

	if input.Config != nil {
		if configMap, ok := input.Config.(map[string]interface{}); ok {
			if maxTokens, exists := configMap["maxOutputTokens"]; exists {
				logData["maxOutputTokens"] = maxTokens
			}
			if stopSeqs, exists := configMap["stopSequences"]; exists {
				logData["stopSequences"] = stopSeqs
			}
		}
	}

	logData["source"] = "go"
	logData["sourceVersion"] = internal.Version

	message := fmt.Sprintf("[genkit] Config[%s, %s]", path, model)
	slog.InfoContext(ctx, message, "data", logData)
}

// recordGenerateActionInputLogs logs input information
func (g *GenerateTelemetry) recordGenerateActionInputLogs(span sdktrace.ReadOnlySpan, model, featureName, qualifiedPath string, input *ai.GenerateActionOptions, projectID, sessionID, threadName string) {
	if input.Messages == nil {
		return
	}

	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	path := truncatePath(toDisplayPath(qualifiedPath))
	sharedMetadata := createCommonLogAttributes(span, projectID)

	baseLogData := map[string]interface{}{
		"model":         model,
		"path":          path,
		"qualifiedPath": qualifiedPath,
		"featureName":   featureName,
	}

	if sessionID != "" {
		baseLogData["sessionId"] = sessionID
	}
	if threadName != "" {
		baseLogData["threadName"] = threadName
	}

	for k, v := range sharedMetadata {
		baseLogData[k] = v
	}

	messages := len(input.Messages)
	for msgIdx, msg := range input.Messages {
		parts := len(msg.Content)
		for partIdx, part := range msg.Content {
			logData := make(map[string]interface{})
			for k, v := range baseLogData {
				logData[k] = v
			}

			partCounts := g.toPartCounts(partIdx, parts, msgIdx, messages)
			logData["content"] = g.toPartLogContent(part)
			logData["role"] = msg.Role
			logData["partIndex"] = partIdx
			logData["totalParts"] = parts
			logData["messageIndex"] = msgIdx
			logData["totalMessages"] = messages

			message := fmt.Sprintf("[genkit] Input[%s, %s] %s", path, model, partCounts)
			slog.InfoContext(ctx, message, MetadataKey, logData)
		}
	}
}

// recordGenerateActionOutputLogs logs output information
func (g *GenerateTelemetry) recordGenerateActionOutputLogs(span sdktrace.ReadOnlySpan, model, featureName, qualifiedPath string, output *ai.ModelResponse, projectID, sessionID, threadName string) {
	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	path := truncatePath(toDisplayPath(qualifiedPath))
	sharedMetadata := createCommonLogAttributes(span, projectID)

	baseLogData := map[string]interface{}{
		"model":         model,
		"path":          path,
		"qualifiedPath": qualifiedPath,
		"featureName":   featureName,
	}

	if sessionID != "" {
		baseLogData["sessionId"] = sessionID
	}
	if threadName != "" {
		baseLogData["threadName"] = threadName
	}

	for k, v := range sharedMetadata {
		baseLogData[k] = v
	}

	var message *ai.Message
	if output.Message != nil {
		message = output.Message
	}

	if message != nil && message.Content != nil {
		parts := len(message.Content)
		for partIdx, part := range message.Content {
			logData := make(map[string]interface{})
			for k, v := range baseLogData {
				logData[k] = v
			}

			partCounts := g.toPartCounts(partIdx, parts, 0, 1)

			if output.FinishMessage != "" {
				logData["finishMessage"] = truncate(output.FinishMessage)
			}

			logData["content"] = g.toPartLogContent(part)
			logData["role"] = message.Role
			logData["partIndex"] = partIdx
			logData["totalParts"] = parts
			logData["candidateIndex"] = 0
			logData["totalCandidates"] = 1
			logData["messageIndex"] = 0
			logData["finishReason"] = output.FinishReason

			message := fmt.Sprintf("[genkit] Output[%s, %s] %s", path, model, partCounts)
			slog.InfoContext(ctx, message, MetadataKey, logData)
		}
	}
}

// Helper functions

func (g *GenerateTelemetry) extractErrorName(span sdktrace.ReadOnlySpan) string {
	if span.Status().Code == codes.Error {
		return span.Status().Description
	}
	for _, event := range span.Events() {
		if event.Name == "exception" {
			for _, attr := range event.Attributes {
				if string(attr.Key) == "exception.type" {
					return attr.Value.AsString()
				}
			}
		}
	}
	return ""
}

func (g *GenerateTelemetry) extractFeatureName(attributes []attribute.KeyValue, path string) string {
	flowName := extractStringAttribute(attributes, "genkit:metadata:flow:name")
	if flowName != "" {
		return flowName
	}
	pathFeature := extractOuterFeatureNameFromPath(path)
	return pathFeature
}

func (g *GenerateTelemetry) toPartCounts(partOrdinal, parts, msgOrdinal, messages int) string {
	if parts > 1 && messages > 1 {
		return fmt.Sprintf("(part %s in message %s)", g.xOfY(partOrdinal, parts), g.xOfY(msgOrdinal, messages))
	}
	if parts > 1 {
		return fmt.Sprintf("(part %s)", g.xOfY(partOrdinal, parts))
	}
	if messages > 1 {
		return fmt.Sprintf("(message %s)", g.xOfY(msgOrdinal, messages))
	}
	return ""
}

func (g *GenerateTelemetry) xOfY(x, y int) string {
	return fmt.Sprintf("%d of %d", x+1, y)
}

// toPartLogContent processes different part types correctly based on Part.Kind
func (g *GenerateTelemetry) toPartLogContent(part *ai.Part) string {
	switch part.Kind {
	case ai.PartText:
		return truncate(part.Text)
	case ai.PartData:
		return truncate(part.Text)
	case ai.PartMedia:
		return g.toPartLogMedia(part)
	case ai.PartCustom:
		if part.Custom != nil {
			data, _ := json.Marshal(part.Custom)
			return truncate(string(data))
		}
	case ai.PartToolRequest:
		if part.ToolRequest != nil {
			return g.toPartLogToolRequest(part.ToolRequest)
		}
	case ai.PartToolResponse:
		if part.ToolResponse != nil {
			return g.toPartLogToolResponse(part.ToolResponse)
		}
	}
	return "<unknown format>"
}

func (g *GenerateTelemetry) toPartLogMedia(part *ai.Part) string {
	// For media parts, the content is stored in the Text field
	// and ContentType indicates the media type
	if strings.HasPrefix(part.Text, "data:") {
		splitIdx := strings.Index(part.Text, "base64,")
		if splitIdx < 0 {
			return "<unknown media format>"
		}
		prefix := part.Text[:splitIdx+7]
		hasher := sha256.New()
		hasher.Write([]byte(part.Text[splitIdx+7:]))
		hashedContent := hex.EncodeToString(hasher.Sum(nil))
		return fmt.Sprintf("%s<sha256(%s)>", prefix, hashedContent)
	}
	return truncate(part.Text)
}

func (g *GenerateTelemetry) toPartLogToolRequest(tool *ai.ToolRequest) string {
	var inputText string
	if str, ok := tool.Input.(string); ok {
		inputText = str
	} else {
		data, _ := json.Marshal(tool.Input)
		inputText = string(data)
	}
	return truncate(fmt.Sprintf("Tool request: %s, ref: %s, input: %s", tool.Name, tool.Ref, inputText))
}

func (g *GenerateTelemetry) toPartLogToolResponse(tool *ai.ToolResponse) string {
	var outputText string
	if str, ok := tool.Output.(string); ok {
		outputText = str
	} else {
		data, _ := json.Marshal(tool.Output)
		outputText = string(data)
	}
	return truncate(fmt.Sprintf("Tool response: %s, ref: %s, output: %s", tool.Name, tool.Ref, outputText))
}

// Utility functions

// toDisplayPath converts qualified paths to display paths
// Converts /{name1,t:type}/{name2,t:type} to "name1 > name2"
func toDisplayPath(qualifiedPath string) string {
	if qualifiedPath == "" {
		return "<unknown>"
	}

	re := regexp.MustCompile(`\{([^,}]+),[^}]+\}`)
	matches := re.FindAllStringSubmatch(qualifiedPath, -1)

	if len(matches) == 0 {
		return qualifiedPath
	}

	var names []string
	for _, match := range matches {
		if len(match) > 1 {
			names = append(names, match[1])
		}
	}

	return strings.Join(names, " > ")
}
