#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Microsoft Foundry Hello World Demo
# ====================================
#
# Demonstrates usage of Microsoft Foundry (Azure OpenAI) models with Genkit.
#
# Prerequisites:
#   - AZURE_OPENAI_API_KEY environment variable set
#   - AZURE_OPENAI_ENDPOINT environment variable set
#   - Optional: AZURE_OPENAI_API_VERSION (defaults to plugin's DEFAULT_API_VERSION)
#   - Optional: AZURE_OPENAI_DEPLOYMENT (defaults to gpt-4o)
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Microsoft Foundry Hello World" "🔷"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AZURE_OPENAI_API_KEY       Required. Your Azure OpenAI API key"
    echo "  AZURE_OPENAI_ENDPOINT      Required. Your Azure OpenAI endpoint URL"
    echo "  AZURE_OPENAI_API_VERSION   API version (default: plugin default)"
    echo "  AZURE_OPENAI_DEPLOYMENT    Deployment name (default: gpt-4o)"
    echo ""
    echo "Finding Your Credentials:"
    echo "  1. Go to Microsoft Foundry Portal: https://ai.azure.com/"
    echo "  2. Select your Project > Models > Deployments > [Deployment]"
    echo "  3. Open the Details pane to find:"
    echo "     - Target URI → contains endpoint URL and API version"
    echo "     - Key → your API key"
    echo "     - Name → your deployment name"
    echo ""
    echo "Portal: https://ai.azure.com/"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Microsoft Foundry Hello World" "🔷"

# Required credentials
check_env_var "AZURE_OPENAI_API_KEY" "https://ai.azure.com/" || true
check_env_var "AZURE_OPENAI_ENDPOINT" "https://ai.azure.com/" || true

# API version: only export if user explicitly set it; otherwise let the plugin default apply.
if [[ -n "${AZURE_OPENAI_API_VERSION:-}" ]]; then
    export AZURE_OPENAI_API_VERSION
fi
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4o}"

if [[ -n "${AZURE_OPENAI_API_VERSION:-}" ]]; then
    check_env_var "AZURE_OPENAI_API_VERSION" "" || true
fi
check_env_var "AZURE_OPENAI_DEPLOYMENT" "" || true

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
