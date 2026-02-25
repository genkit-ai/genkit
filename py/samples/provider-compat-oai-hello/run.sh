#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# OpenAI Compatibility Demo
# =========================
#
# Demonstrates using the OpenAI compatibility plugin with Genkit.
#
# Prerequisites:
#   - OPENAI_API_KEY environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "OpenAI Compatibility Demo" "🔌"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY    Required. Your OpenAI API key"
    echo ""
    echo "Get an API key from: https://platform.openai.com/api-keys"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "OpenAI Compatibility Demo" "🔌"

check_env_var "OPENAI_API_KEY" "https://platform.openai.com/api-keys" || true

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
