#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Real-time Tracing Demo
# ======================
#
# Demonstrates live telemetry and trace visualization.
#
# Prerequisites:
#   - GEMINI_API_KEY environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Real-time Tracing Demo" "📡"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY    Required. Your Gemini API key"
    echo ""
    echo "This demo shows:"
    echo "  - Live span streaming"
    echo "  - Trace visualization"
    echo "  - Performance monitoring"
    echo ""
    echo "Get an API key from: https://makersuite.google.com/app/apikey"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Real-time Tracing Demo" "📡"

check_env_var "GEMINI_API_KEY" "https://makersuite.google.com/app/apikey" || true

install_deps

# Enable realtime telemetry for this demo
export GENKIT_ENABLE_REALTIME_TELEMETRY=true

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
