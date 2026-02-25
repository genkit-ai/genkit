#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Hugging Face Hello World Demo
# =============================
#
# Demonstrates usage of Hugging Face models with Genkit.
#
# Prerequisites:
#   - HF_TOKEN environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Hugging Face Hello World" "🤗"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HF_TOKEN    Required. Your Hugging Face API token"
    echo ""
    echo "Get a token from: https://huggingface.co/settings/tokens"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Hugging Face Hello World" "🤗"

check_env_var "HF_TOKEN" "https://huggingface.co/settings/tokens" || true

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
