#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Evaluator Demo
# ==============
#
# Demonstrates using Genkit evaluators to assess model outputs.
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
    print_banner "Evaluator Demo" "📊"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY    Required. Your Gemini API key"
    echo ""
    echo "This demo shows:"
    echo "  - Output quality evaluation"
    echo "  - Custom evaluators"
    echo "  - Scoring metrics"
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

print_banner "Evaluator Demo" "📊"

check_env_var "GEMINI_API_KEY" "https://makersuite.google.com/app/apikey" || true

install_deps

# shellcheck disable=SC2016  # Single quotes intentional for sh -c.
genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d evaluator_demo \
        -d docs \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json;*.pdf' \
        -R \
        -- sh -c '
          # Initialize database if missing or if PDFs are newer
          if [ -f __db_pdf_qa.json ]; then
            for f in docs/*.pdf; do
              if [ "$f" -nt __db_pdf_qa.json ]; then
                rm -f __db_pdf_qa.json
                break
              fi
            done
          fi
          if [ ! -f __db_pdf_qa.json ]; then
            uv run -m evaluator_demo.main --setup
          fi
          uv run -m evaluator_demo.main "$@"' "$0" "$@"
