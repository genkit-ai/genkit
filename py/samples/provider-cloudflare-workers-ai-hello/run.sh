#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Cloudflare Workers AI Hello World Demo
# =======================================
#
# Demonstrates usage of Cloudflare Workers AI models with Genkit.
#
# Prerequisites:
#   - CLOUDFLARE_ACCOUNT_ID environment variable set
#   - CLOUDFLARE_API_TOKEN environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "Cloudflare Workers AI Hello World" "☁️"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CLOUDFLARE_ACCOUNT_ID    Required. Your Cloudflare account ID"
    echo "  CLOUDFLARE_API_TOKEN     Required. API token with Workers AI permissions"
    echo ""
    echo "Get credentials from: https://dash.cloudflare.com/"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Cloudflare Workers AI Hello World" "☁️"

check_env_var "CLOUDFLARE_ACCOUNT_ID" "https://dash.cloudflare.com/" || true
check_env_var "CLOUDFLARE_API_TOKEN" "https://dash.cloudflare.com/profile/api-tokens" || true

echo ""

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
