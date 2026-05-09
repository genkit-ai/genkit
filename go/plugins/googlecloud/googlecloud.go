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

// The googlecloud package supports telemetry (tracing, metrics and logging) using
// Google Cloud services.
package googlecloud

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"cloud.google.com/go/logging"
	mexporter "github.com/GoogleCloudPlatform/opentelemetry-operations-go/exporter/metric"
	texporter "github.com/GoogleCloudPlatform/opentelemetry-operations-go/exporter/trace"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"

	"go.opentelemetry.io/contrib/detectors/gcp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

// Global sync.Once for showing logging setup instructions only once across all recovery attempts
var showLoggingInstructionsOnce sync.Once

// stderrLogger is a stable logger that always writes to stderr and is never
// replaced by calls to slog.SetDefault. Use this for error paths to ensure
// messages are visible even if the default logger is misconfigured.
var stderrLogger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
	Level: slog.LevelInfo,
}))

// EnableGoogleCloudTelemetry enables comprehensive telemetry export to Google Cloud Observability suite.
// This directly initializes telemetry without requiring plugin registration.
//
// Example usage:
//
//	// Zero-config (auto-detects project ID)
//	googlecloud.EnableGoogleCloudTelemetry(nil)
//	g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
//
//	// With configuration
//	googlecloud.EnableGoogleCloudTelemetry(&googlecloud.GoogleCloudTelemetryOptions{
//		ProjectID:      "my-project",
//		ForceDevExport: true,
//	})
//	g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
func EnableGoogleCloudTelemetry(options *GoogleCloudTelemetryOptions) {
	if options == nil {
		options = &GoogleCloudTelemetryOptions{}
	}

	initializeTelemetry(options)
}

// initializeTelemetry is the internal function that sets up Google Cloud telemetry directly.
func initializeTelemetry(opts *GoogleCloudTelemetryOptions) {
	// Auto-discover credentials and project ID from environment
	authConfig, authErr := CredentialsFromEnvironment()
	if authErr != nil {
		slog.Debug("Could not auto-discover credentials from environment", "error", authErr)
	}

	// Use provided credentials or fall back to auto-discovered ones
	credentials := opts.Credentials
	if credentials == nil && authErr == nil {
		credentials = authConfig.Credentials
	}

	// Use provided project ID or fall back to auto-discovered one
	projectID := opts.ProjectID
	if projectID == "" {
		if envProjectID := os.Getenv("GOOGLE_CLOUD_PROJECT"); envProjectID != "" {
			projectID = envProjectID
		} else if envProjectID := os.Getenv("GCLOUD_PROJECT"); envProjectID != "" {
			projectID = envProjectID
		} else if authErr == nil && credentials != nil && credentials.ProjectID != "" {
			projectID = credentials.ProjectID
		}
	}

	metricInterval := 5 * time.Second
	if os.Getenv("GENKIT_ENV") != "dev" && !opts.ForceDevExport {
		metricInterval = 300 * time.Second
	}
	if opts.MetricExportIntervalMillis != nil {
		metricInterval = time.Duration(*opts.MetricExportIntervalMillis) * time.Millisecond
	}

	logLevel := slog.LevelInfo

	finalResource := resource.NewWithAttributes(
		semconv.SchemaURL,
	)

	if gcpResource, err := gcp.NewDetector().Detect(context.Background()); err == nil {
		finalResource, _ = resource.Merge(finalResource, gcpResource)
	}

	var spanProcessors []sdktrace.SpanProcessor

	shouldExport := opts.ForceDevExport || os.Getenv("GENKIT_ENV") != "dev"
	if shouldExport && !opts.DisableTraces {
		var traceOpts []texporter.Option
		traceOpts = append(traceOpts, texporter.WithProjectID(projectID))

		if credentials != nil {
			traceOpts = append(traceOpts, texporter.WithTraceClientOptions([]option.ClientOption{option.WithCredentials(credentials)}))
		}

		baseExporter, err := texporter.New(traceOpts...)
		if err != nil {
			slog.Error("Failed to create Google Cloud trace exporter", "error", err, "error_type", fmt.Sprintf("%T", err))
			return
		}

		modules := []Telemetry{
			NewPathTelemetry(),
			NewGenerateTelemetry(),
			NewFeatureTelemetry(),
			NewActionTelemetry(),
			NewEngagementTelemetry(),
		}

		adjustingExporter := &AdjustingTraceExporter{
			exporter:          baseExporter,
			modules:           modules,
			logInputAndOutput: !opts.DisableLoggingInputAndOutput, // Default true, disable if flag set
			projectId:         projectID,
		}

		// Note: Ideally we should use SimpleSpanProcessor in dev mode for immediate export
		// However I noticed there's a strange interaction effect with Dev UI that causes listActions
		// to hang forever. So we use BatchSpanProcessor in all cases for now.
		batchProcessor := sdktrace.NewBatchSpanProcessor(adjustingExporter)
		spanProcessors = append(spanProcessors, batchProcessor)

		if !opts.DisableMetrics {
			slog.Debug("Setting up metrics provider...")
			if err := setMeterProvider(projectID, metricInterval, credentials, finalResource); err != nil {
				slog.Error("Failed to set up Google Cloud metrics", "error", err)
			}
			slog.Debug("Metrics provider setup complete")
		}
		slog.Debug("Setting up log handler...")
		if err := setLogHandler(projectID, logLevel, credentials); err != nil {
			slog.Error("Failed to set up Google Cloud logging", "error", err)
		}
		slog.Debug("Log handler setup complete")
		slog.Info("Google Cloud telemetry fully initialized", "project_id", projectID, "modules", len(modules))
	} else {
		slog.Info("Google Cloud telemetry resource configured, export disabled in dev environment", "project_id", projectID)
	}

	var tpOptions []sdktrace.TracerProviderOption
	tpOptions = append(tpOptions, sdktrace.WithResource(finalResource))

	if opts.Sampler != nil {
		tpOptions = append(tpOptions, sdktrace.WithSampler(opts.Sampler))
	}

	for _, processor := range spanProcessors {
		tpOptions = append(tpOptions, sdktrace.WithSpanProcessor(processor))
	}

	tp := sdktrace.NewTracerProvider(tpOptions...)
	otel.SetTracerProvider(tp) // Set as global TracerProvider

	slog.Info("Google Cloud telemetry TracerProvider configured", "processors", len(spanProcessors))
	setupGracefulShutdown(tp)
}

func setMeterProvider(projectID string, interval time.Duration, credentials *google.Credentials, res *resource.Resource) error {
	var metricOpts []mexporter.Option
	metricOpts = append(metricOpts, mexporter.WithProjectID(projectID))

	if credentials != nil {
		clientOpts := []option.ClientOption{option.WithCredentials(credentials)}
		for _, opt := range clientOpts {
			metricOpts = append(metricOpts, mexporter.WithMonitoringClientOptions(opt))
		}
	}

	mexp, err := mexporter.New(metricOpts...)
	if err != nil {
		return fmt.Errorf("failed to create metrics exporter: %w", err)
	}
	r := sdkmetric.NewPeriodicReader(mexp, sdkmetric.WithInterval(interval))
	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(r),
		sdkmetric.WithResource(res),
	)
	otel.SetMeterProvider(mp)
	return nil
}

// FlushMetrics forces an immediate flush of all pending metrics to Google Cloud.
// This is useful for short-lived processes or when you want to ensure metrics
// are exported before continuing execution.
func FlushMetrics(ctx context.Context) error {
	if mp := otel.GetMeterProvider(); mp != nil {
		if flusher, ok := mp.(interface{ ForceFlush(context.Context) error }); ok {
			return flusher.ForceFlush(ctx)
		}
	}
	return nil
}

func setLogHandler(projectID string, level slog.Leveler, credentials *google.Credentials) error {
	return setupGCPLogger(projectID, level, credentials)
}

func setupGCPLogger(projectID string, level slog.Leveler, credentials *google.Credentials) error {
	var clientOpts []option.ClientOption
	if credentials != nil {
		clientOpts = append(clientOpts, option.WithCredentials(credentials))
	}

	c, err := logging.NewClient(context.Background(), "projects/"+projectID, clientOpts...)
	if err != nil {
		return fmt.Errorf("failed to create logging client: %w", err)
	}
	// Set up error handling for async logging failures with recursive recovery
	c.OnError = func(err error) {
		slog.SetDefault(stderrLogger)
		stderrLogger.Warn("Unable to send logs to Google Cloud", "error", err)
		if loggingDenied(err) {
			showLoggingInstructionsOnce.Do(func() {
				fmt.Fprint(os.Stderr, loggingDeniedHelpText(projectID))
			})
		}
		// TODO: Reinitialize logger if possible
		// For now, we left the code below commented out because re-initialization was causing the
		// logger to swallow errors for some reason.
		/*
			stderrLogger.Error("Unable to send logs to Google Cloud. Re-initializing logger.")
			// Assume the logger is compromised, and we need a new one
			// Reinitialize the logger with a new instance with the same config

			if setupErr := setupGCPLogger(projectID, level, credentials); setupErr != nil {
				stderrLogger.Error("Failed to reinitialize GCP logger", "error", setupErr)
			}
		*/
	}
	logger := c.Logger("genkit_log")
	slog.SetDefault(slog.New(newHandler(level, logger.Log, projectID)))
	return nil
}

// AdjustingTraceExporter combines PII filtering and telemetry processing
type AdjustingTraceExporter struct {
	exporter          sdktrace.SpanExporter
	modules           []Telemetry
	logInputAndOutput bool
	projectId         string
	spansProcessed    int64
}

func (e *AdjustingTraceExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	adjustedSpans := e.adjust(spans)
	err := e.exporter.ExportSpans(ctx, adjustedSpans)
	if err != nil {
		slog.Error("Unable to send telemetry to Google Cloud",
			"error", err,
			"span_count", len(adjustedSpans),
			"project_id", e.projectId,
			"error_type", fmt.Sprintf("%T", err))
	}
	return err
}

func (e *AdjustingTraceExporter) Shutdown(ctx context.Context) error {
	slog.Info("Shutting down adjusting trace exporter", "spans_processed", e.spansProcessed)
	return e.exporter.Shutdown(ctx)
}

func (e *AdjustingTraceExporter) ForceFlush(ctx context.Context) error {
	if flusher, ok := e.exporter.(interface{ ForceFlush(context.Context) error }); ok {
		return flusher.ForceFlush(ctx)
	}
	return nil
}

func (e *AdjustingTraceExporter) adjust(spans []sdktrace.ReadOnlySpan) []sdktrace.ReadOnlySpan {
	var adjustedSpans []sdktrace.ReadOnlySpan

	for _, span := range spans {
		// Filter out Google Cloud SDK internal operations, but keep user spans like HTTP POST
		// Only exclude spans that are clearly internal Google Cloud telemetry operations
		// Note: These service names are stable Google Cloud APIs, but this list may need
		// updates if new internal telemetry services are added in the future
		spanName := span.Name()
		isInternalGoogleCloudSpan := strings.Contains(spanName, "google.monitoring.v3.MetricService") ||
			strings.Contains(spanName, "google.devtools.cloudtrace.v2.TraceService") ||
			strings.Contains(spanName, "google.logging.v2.LoggingServiceV2")

		if isInternalGoogleCloudSpan {
			continue
		}

		e.tickTelemetry(span)
		e.spansProcessed++

		adjustedSpan := span
		adjustedSpan = e.redactInputOutput(adjustedSpan)
		adjustedSpan = e.markErrorSpanAsError(adjustedSpan)
		adjustedSpan = e.markFailedSpan(adjustedSpan)
		adjustedSpan = e.markGenkitFeature(adjustedSpan)
		adjustedSpan = e.markGenkitModel(adjustedSpan)
		adjustedSpan = e.setRootState(adjustedSpan)
		adjustedSpan = e.normalizeLabels(adjustedSpan)

		adjustedSpans = append(adjustedSpans, adjustedSpan)
	}

	return adjustedSpans
}

// tickTelemetry processes telemetry for each span using all configured modules
func (e *AdjustingTraceExporter) tickTelemetry(span sdktrace.ReadOnlySpan) {
	if len(e.modules) == 0 {
		return
	}

	// Only process Genkit spans (skip internal Google Cloud SDK spans)
	attrs := span.Attributes()
	hasGenkitType := false
	for _, attr := range attrs {
		if string(attr.Key) == "genkit:type" {
			hasGenkitType = true
			break
		}
	}
	if !hasGenkitType {
		return
	}

	attributes := span.Attributes()
	spanType := extractStringAttribute(attributes, "genkit:type")
	subtype := extractStringAttribute(attributes, "genkit:metadata:subtype")
	isRoot := extractBoolAttribute(attributes, "genkit:isRoot")

	pathTelemetry := NewPathTelemetry()
	generateTelemetry := NewGenerateTelemetry()
	featureTelemetry := NewFeatureTelemetry()
	actionTelemetry := NewActionTelemetry()
	engagementTelemetry := NewEngagementTelemetry()

	// Always process path telemetry
	pathTelemetry.Tick(span, e.logInputAndOutput, e.projectId)

	if isRoot {
		// Report top level feature request and latency only for root spans
		featureTelemetry.Tick(span, e.logInputAndOutput, e.projectId)
	} else {
		if spanType == "action" && subtype == "model" {
			// Report generate metrics for all model actions
			generateTelemetry.Tick(span, e.logInputAndOutput, e.projectId)
		}
		if spanType == "action" && subtype == "tool" {
			// TODO: Report input and output for tool actions
		}
		if spanType == "action" || spanType == "flow" || spanType == "flowStep" || spanType == "util" {
			// Report request and latency metrics for all actions (including tools)
			actionTelemetry.Tick(span, e.logInputAndOutput, e.projectId)
		}
	}

	if spanType == "userEngagement" {
		// Report user acceptance and feedback metrics
		engagementTelemetry.Tick(span, e.logInputAndOutput, e.projectId)
	}
}

// redactInputOutput applies PII filtering
func (e *AdjustingTraceExporter) redactInputOutput(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	hasInput := false
	hasOutput := false

	for _, attr := range span.Attributes() {
		if attr.Key == "genkit:input" {
			hasInput = true
		}
		if attr.Key == "genkit:output" {
			hasOutput = true
		}
	}

	if !hasInput && !hasOutput {
		return span
	}

	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			var newAttrs []attribute.KeyValue
			for _, attr := range attrs {
				if attr.Key == "genkit:input" {
					newAttrs = append(newAttrs, attribute.String("genkit:input", "<redacted>"))
				} else if attr.Key == "genkit:output" {
					newAttrs = append(newAttrs, attribute.String("genkit:output", "<redacted>"))
				} else {
					newAttrs = append(newAttrs, attr)
				}
			}
			return newAttrs
		},
	}
}

// markErrorSpanAsError adds error marking for GCP Trace
func (e *AdjustingTraceExporter) markErrorSpanAsError(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	if span.Status().Code != codes.Error {
		return span
	}

	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			newAttrs := make([]attribute.KeyValue, len(attrs))
			copy(newAttrs, attrs)
			newAttrs = append(newAttrs, attribute.String("/http/status_code", "599"))
			return newAttrs
		},
	}
}

// markFailedSpan marks spans that are failure sources
func (e *AdjustingTraceExporter) markFailedSpan(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	var isFailureSource bool
	var name, path string

	for _, attr := range span.Attributes() {
		if attr.Key == "genkit:isFailureSource" {
			isFailureSource = attr.Value.AsBool()
		}
		if attr.Key == "genkit:name" {
			name = attr.Value.AsString()
		}
		if attr.Key == "genkit:path" {
			path = attr.Value.AsString()
		}
	}

	if !isFailureSource {
		return span
	}

	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			newAttrs := make([]attribute.KeyValue, len(attrs))
			copy(newAttrs, attrs)
			newAttrs = append(newAttrs, attribute.String("genkit:failedSpan", name))
			newAttrs = append(newAttrs, attribute.String("genkit:failedPath", path))
			return newAttrs
		},
	}
}

// markGenkitFeature marks root spans with feature name
func (e *AdjustingTraceExporter) markGenkitFeature(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	var isRoot bool
	var name string

	for _, attr := range span.Attributes() {
		if attr.Key == "genkit:isRoot" {
			isRoot = attr.Value.AsBool()
		}
		if attr.Key == "genkit:name" {
			name = attr.Value.AsString()
		}
	}

	if !isRoot || name == "" {
		return span
	}

	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			newAttrs := make([]attribute.KeyValue, len(attrs))
			copy(newAttrs, attrs)
			newAttrs = append(newAttrs, attribute.String("genkit:feature", name))
			return newAttrs
		},
	}
}

// markGenkitModel marks model spans with model name
func (e *AdjustingTraceExporter) markGenkitModel(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	var subtype, name string

	for _, attr := range span.Attributes() {
		if attr.Key == "genkit:metadata:subtype" {
			subtype = attr.Value.AsString()
		}
		if attr.Key == "genkit:name" {
			name = attr.Value.AsString()
		}
	}

	if subtype != "model" || name == "" {
		return span
	}

	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			newAttrs := make([]attribute.KeyValue, len(attrs))
			copy(newAttrs, attrs)
			newAttrs = append(newAttrs, attribute.String("genkit:model", name))
			return newAttrs
		},
	}
}

// normalizeLabels converts attribute keys from : to /
func (e *AdjustingTraceExporter) normalizeLabels(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			newAttrs := make([]attribute.KeyValue, len(attrs))
			for i, attr := range attrs {
				newKey := strings.ReplaceAll(string(attr.Key), ":", "/")
				newAttrs[i] = attribute.KeyValue{
					Key:   attribute.Key(newKey),
					Value: attr.Value,
				}
			}
			return newAttrs
		},
	}
}

// setRootState copies genkit:state to genkit:rootState for root spans
func (e *AdjustingTraceExporter) setRootState(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	var isRoot bool
	var state string

	for _, attr := range span.Attributes() {
		if attr.Key == "genkit:isRoot" {
			isRoot = attr.Value.AsBool()
		}
		if attr.Key == "genkit:state" {
			state = attr.Value.AsString()
		}
	}

	if !isRoot || state == "" {
		return span
	}

	return &spanWithModifiedAttributes{
		ReadOnlySpan: span,
		modifyFunc: func(attrs []attribute.KeyValue) []attribute.KeyValue {
			newAttrs := make([]attribute.KeyValue, len(attrs))
			copy(newAttrs, attrs)
			newAttrs = append(newAttrs, attribute.String("genkit:rootState", state))
			return newAttrs
		},
	}
}

// spanWithModifiedAttributes wraps a span and modifies its attributes
type spanWithModifiedAttributes struct {
	sdktrace.ReadOnlySpan
	modifyFunc func([]attribute.KeyValue) []attribute.KeyValue
}

func (s *spanWithModifiedAttributes) Attributes() []attribute.KeyValue {
	return s.modifyFunc(s.ReadOnlySpan.Attributes())
}

// setupGracefulShutdown sets up signal handlers to flush telemetry on process exit
func setupGracefulShutdown(tp *sdktrace.TracerProvider) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		slog.Info("Received shutdown signal, flushing telemetry...")

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		var hasErrors bool

		if err := tp.ForceFlush(ctx); err != nil {
			slog.Error("Failed to flush spans during shutdown", "error", err)
			hasErrors = true
		}

		if mp := otel.GetMeterProvider(); mp != nil {
			hasErrors = shutdownMetricsProvider(ctx, mp) || hasErrors
		}

		if err := tp.Shutdown(ctx); err != nil {
			slog.Error("Failed to shutdown TracerProvider", "error", err)
			hasErrors = true
		}

		if hasErrors {
			slog.Warn("Telemetry shutdown completed with errors")
		} else {
			slog.Info("Telemetry shutdown completed successfully")
		}
		os.Exit(0)
	}()
}

// shutdownMetricsProvider handles metrics provider flush and shutdown operations
func shutdownMetricsProvider(ctx context.Context, mp metric.MeterProvider) bool {
	hasErrors := false
	if flusher, ok := mp.(interface{ ForceFlush(context.Context) error }); ok {
		if err := flusher.ForceFlush(ctx); err != nil {
			slog.Error("Failed to flush metrics during shutdown", "error", err)
			hasErrors = true
		}
	}
	if shutdowner, ok := mp.(interface{ Shutdown(context.Context) error }); ok {
		if err := shutdowner.Shutdown(ctx); err != nil {
			slog.Error("Failed to shutdown MeterProvider", "error", err)
			hasErrors = true
		}
	}
	return hasErrors
}
