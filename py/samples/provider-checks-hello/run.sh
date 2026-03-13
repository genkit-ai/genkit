#!/usr/bin/env bash
# Copyright 2025 Google LLC
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

# Checks Plugin Hello World Demo
# ================================
#
# Demonstrates using the Checks API with Genkit for AI safety evaluation.
#
# Prerequisites:
#   - GEMINI_API_KEY environment variable set
#   - GCLOUD_PROJECT environment variable set
#   - Checks API enabled on your GCP project
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Source shared helper utilities.
source "$(dirname "${SCRIPT_DIR}")/_common.sh"

print_help() {
    print_banner "Checks Plugin Hello World" "🛡️"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY    Required. Your Gemini API key"
    echo "  GCLOUD_PROJECT    Required. Your GCP project ID"
    echo ""
    echo "Get a Gemini API key from: https://aistudio.google.com/app/apikey"
    print_help_footer
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

# Main execution
print_banner "Checks Plugin Hello World" "🛡️"

# Prompt for required env vars if not set.
check_env_var "GCLOUD_PROJECT" "" || true
check_env_var "GEMINI_API_KEY" "https://aistudio.google.com/app/apikey" || true

# The Checks API requires specific OAuth scopes. Standard ADC credentials
# (from `gcloud auth application-default login`) don't include the `checks`
# scope, resulting in a 403 ACCESS_TOKEN_SCOPE_INSUFFICIENT error.
CHECKS_SCOPES="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/checks"
CHECKS_API="checks.googleapis.com"

# Verify the Checks API is enabled on the configured GCP project.
ensure_checks_api_enabled() {
    local project="${GCLOUD_PROJECT:-}"
    if [[ -z "$project" ]]; then
        echo -e "${YELLOW}⚠ GCLOUD_PROJECT not set — skipping API enablement check.${NC}"
        return
    fi

    echo -e "${BLUE}Checking if ${CHECKS_API} is enabled on project '${project}'...${NC}"

    if gcloud services list --project="${project}" --filter="config.name:${CHECKS_API}" --format="value(config.name)" 2>/dev/null | grep -q "${CHECKS_API}"; then
        echo -e "${GREEN}✓ ${CHECKS_API} is enabled${NC}"
    else
        echo -e "${YELLOW}✗ ${CHECKS_API} is NOT enabled on project '${project}'.${NC}"
        echo ""
        if [[ -t 0 ]] && [ -c /dev/tty ]; then
            echo -en "Run ${GREEN}gcloud services enable ${CHECKS_API} --project=${project}${NC} now? [Y/n]: "
            local response
            read -r response < /dev/tty
            if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
                echo ""
                gcloud services enable "${CHECKS_API}" --project="${project}"
                echo -e "${GREEN}✓ ${CHECKS_API} enabled${NC}"
            else
                echo -e "${YELLOW}Skipping. The Checks API calls will fail.${NC}"
            fi
        else
            echo "Run: gcloud services enable ${CHECKS_API} --project=${project}"
        fi
    fi
    echo ""
}

# Ensure ADC credentials include the Checks OAuth scope.
# Always re-authenticates because there is no reliable way to inspect
# which scopes an existing ADC token was minted with.
ensure_checks_adc() {
    echo -e "${BLUE}Authenticating with Checks API scopes...${NC}"
    echo -e "Running: ${GREEN}gcloud auth application-default login --scopes=${CHECKS_SCOPES}${NC}"
    echo ""
    gcloud auth application-default login --scopes="${CHECKS_SCOPES}"
    echo ""
}

ensure_checks_api_enabled
ensure_checks_adc

install_deps

genkit_start_with_browser -- \
  uv tool run --from watchdog watchmedo auto-restart \
    -d src \
    -d ../../packages \
    -d ../../plugins \
    -p '*.py;*.prompt;*.json' \
    -R \
    -- uv run src/main.py "$@"
