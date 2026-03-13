#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Vertex AI Vector Search with BigQuery Demo
# ==========================================
#
# Demonstrates enterprise-grade vector search using GCP services.
#
# Prerequisites:
#   - GOOGLE_CLOUD_PROJECT environment variable set
#   - gcloud CLI authenticated
#   - BigQuery dataset and Vertex AI index configured
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Vertex AI Vector Search (BigQuery)" "📊"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GOOGLE_CLOUD_PROJECT    Required. Your GCP project ID"
    echo ""
    echo "Getting Started:"
    echo "  1. Authenticate: gcloud auth application-default login"
    echo "  2. Set project: export GOOGLE_CLOUD_PROJECT=your-project"
    echo "  3. Configure BigQuery dataset and Vertex AI index"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Vertex AI Vector Search (BigQuery)" "📊"

check_env_var "PROJECT_ID" "" || exit 1
check_env_var "LOCATION" "" || exit 1
check_env_var "BIGQUERY_DATASET_NAME" "" || exit 1
check_env_var "BIGQUERY_TABLE_NAME" "" || exit 1
check_env_var "VECTOR_SEARCH_DEPLOYED_INDEX_ID" "" || exit 1

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
