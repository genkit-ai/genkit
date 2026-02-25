#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Cohere AI Hello World Demo
# ===========================
#
# Demonstrates usage of Cohere AI models with Genkit.
#
# Prerequisites:
#   - COHERE_API_KEY environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Cohere AI Hello World" "🔷"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  COHERE_API_KEY    Required. Your Cohere API key"
    echo "  CO_API_KEY        Alternative. Your Cohere API key"
    echo ""
    echo "Get an API key from: https://dashboard.cohere.com/api-keys"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Cohere AI Hello World" "🔷"

check_env_var "COHERE_API_KEY" "https://dashboard.cohere.com/api-keys" || true

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
