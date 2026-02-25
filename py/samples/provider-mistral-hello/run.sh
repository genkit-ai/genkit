#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Mistral AI Hello World Demo
# ===========================
#
# Demonstrates usage of Mistral AI models with Genkit.
#
# Prerequisites:
#   - MISTRAL_API_KEY environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Mistral AI Hello World" "🇫🇷"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MISTRAL_API_KEY    Required. Your Mistral AI API key"
    echo ""
    echo "Get an API key from: https://console.mistral.ai/"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Mistral AI Hello World" "🇫🇷"

check_env_var "MISTRAL_API_KEY" "https://console.mistral.ai/" || true

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
