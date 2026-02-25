#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Firestore Retriever Demo
# ========================
#
# Demonstrates using Firestore as a vector store.
#
# This script automates most of the setup:
#   - Detects/prompts for GOOGLE_CLOUD_PROJECT
#   - Checks gcloud authentication
#   - Enables required APIs
#   - Installs dependencies
#
# Note: You still need to create a Firestore vector index manually.
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
    "firestore.googleapis.com"   # Firestore API
    "aiplatform.googleapis.com"  # Vertex AI API (for embeddings)
)

print_help() {
    print_banner "Firestore Retriever Demo" "🔥"
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
    echo "  - Firestore API (firestore.googleapis.com)"
    echo "  - Vertex AI API (aiplatform.googleapis.com)"
    echo ""
    echo -e "${YELLOW}Note:${NC} You must create a Firestore vector index manually:"
    echo ""
    echo "  gcloud firestore indexes composite create \\"
    echo "    --project=\$GOOGLE_CLOUD_PROJECT \\"
    echo "    --collection-group=films \\"
    echo "    --query-scope=COLLECTION \\"
    echo "    --field-config=vector-config='{\"dimension\":\"768\",\"flat\": {}}',field-path=embedding"
    echo ""
    print_help_footer
}

# Print reminder about Firestore index
print_firestore_index_reminder() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}REMINDER: Create Firestore vector index if not already done:${NC}"
    echo ""
    echo "  gcloud firestore indexes composite create \\"
    echo "    --project=${GOOGLE_CLOUD_PROJECT:-YOUR_PROJECT} \\"
    echo "    --collection-group=films \\"
    echo "    --query-scope=COLLECTION \\"
    echo "    --field-config=vector-config='{\"dimension\":\"768\",\"flat\": {}}',field-path=embedding"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
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
        print_firestore_index_reminder
        echo -e "${GREEN}Setup complete!${NC}"
        echo ""
        exit 0
        ;;
esac

print_banner "Firestore Retriever Demo" "🔥"

# Run GCP setup (checks gcloud, auth, enables APIs)
run_gcp_setup "${REQUIRED_APIS[@]}" || exit 1

# Remind about Firestore index
print_firestore_index_reminder

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
