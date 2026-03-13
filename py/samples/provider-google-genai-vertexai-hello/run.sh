#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Vertex AI Hello World Demo
# ==========================
#
# Demonstrates using Google Cloud Vertex AI with Genkit.
#
# This script automates most of the setup:
#   - Detects/prompts for GOOGLE_CLOUD_PROJECT
#   - Checks gcloud authentication
#   - Enables required APIs
#   - Installs dependencies
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --setup  # Run setup only (check auth, enable APIs)
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

# Required APIs for this demo
REQUIRED_APIS=(
    "aiplatform.googleapis.com"  # Vertex AI API
)

print_help() {
    print_banner "Vertex AI Hello World" "☁️"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo "  --setup    Run setup only (auth check, enable APIs)"
    echo ""
    echo "The script will automatically:"
    echo "  1. Prompt for GOOGLE_CLOUD_PROJECT if not set"
    echo "  2. Check gcloud authentication"
    echo "  3. Enable required APIs (with your permission)"
    echo "  4. Install dependencies"
    echo "  5. Start the demo and open the browser"
    echo ""
    echo "Required APIs (enabled automatically):"
    echo "  - Vertex AI API (aiplatform.googleapis.com)"
    print_help_footer
}

# Main
case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
    --setup)
        print_banner "Setup" "⚙️"
        run_gcp_setup "${REQUIRED_APIS[@]}" || exit 1
        echo -e "${GREEN}Setup complete!${NC}"
        echo ""
        exit 0
        ;;
esac

print_banner "Vertex AI Hello World" "☁️"

# Run GCP setup (checks gcloud, auth, enables APIs)
run_gcp_setup "${REQUIRED_APIS[@]}" || exit 1

# Install dependencies
install_deps

# Start the demo
genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
