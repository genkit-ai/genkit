#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# DeepSeek Hello World Demo
# =========================
#
# Demonstrates usage of DeepSeek models with Genkit.
#
# Prerequisites:
#   - DEEPSEEK_API_KEY environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "DeepSeek Hello World" "🧠"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DEEPSEEK_API_KEY    Required. Your DeepSeek API key"
    echo ""
    echo "Get an API key from: https://platform.deepseek.com/"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "DeepSeek Hello World" "🧠"

check_env_var "DEEPSEEK_API_KEY" "https://platform.deepseek.com/" || true

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
