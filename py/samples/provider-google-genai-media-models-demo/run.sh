#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Media Models Demo
# =================
#
# Demonstrates multimodal capabilities with images, audio, and video.
#
# Prerequisites:
#   - GEMINI_API_KEY environment variable set
#   - Generative Language API enabled on the project tied to your API key
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

# APIs required by this sample.
REQUIRED_APIS=(
    "generativelanguage.googleapis.com"
)

print_help() {
    print_banner "Media Models Demo" "📸"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY           Required. Your Gemini API key"
    echo "  GOOGLE_CLOUD_PROJECT     Optional. Used to auto-enable APIs via gcloud"
    echo ""
    echo "This demo shows:"
    echo "  - Image understanding"
    echo "  - Audio transcription"
    echo "  - Video analysis"
    echo "  - Multimodal prompting"
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

print_banner "Media Models Demo" "📸"

check_env_var "GEMINI_API_KEY" "https://makersuite.google.com/app/apikey" || true

# Enable required Google Cloud APIs.
# The Generative Language API must be enabled on the GCP project associated
# with your API key, otherwise the SDK returns 403 PERMISSION_DENIED.
if command -v gcloud &> /dev/null; then
    # Try to discover the project: explicit env var > gcloud default.
    _project="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
    if [[ -n "$_project" ]]; then
        GOOGLE_CLOUD_PROJECT="$_project" enable_required_apis "${REQUIRED_APIS[@]}" || true
    else
        echo -e "${YELLOW}No GCP project detected — skipping API enablement.${NC}"
        echo -e "If you hit a 403, enable the API manually:"
        echo -e "  ${GREEN}gcloud services enable generativelanguage.googleapis.com --project=YOUR_PROJECT${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}gcloud not found — skipping API enablement.${NC}"
    echo -e "If you hit a 403 PERMISSION_DENIED, enable the Generative Language API:"
    echo -e "  ${GREEN}https://console.developers.google.com/apis/api/generativelanguage.googleapis.com${NC}"
    echo ""
fi

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d data \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
