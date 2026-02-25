#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# FastAPI Hello - BugBot Demo
# ============================
#
# Demonstrates Genkit FastAPI plugin integration with streaming responses.
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
    print_banner "FastAPI BugBot Demo" "🤖"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY    Required. Your Gemini API key"
    echo ""
    echo "Get an API key from: https://aistudio.google.com/apikey"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "FastAPI BugBot Demo" "🤖"

check_env_var "GEMINI_API_KEY" "https://aistudio.google.com/apikey" || true

install_deps

echo -e "${BLUE}Starting FastAPI BugBot...${NC}"
echo -e "  API:     ${GREEN}http://localhost:8080${NC}"
echo -e "  Docs:    ${GREEN}http://localhost:8080/docs${NC}"
echo -e "  Dev UI:  ${GREEN}http://localhost:4000${NC}"
echo ""

genkit_start_with_browser -- uv run python src/main.py
