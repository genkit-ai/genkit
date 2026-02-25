#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# AWS Bedrock Hello World Demo
# ============================
#
# Demonstrates usage of AWS Bedrock models with Genkit.
#
# Prerequisites:
#   - AWS credentials configured (env vars, credentials file, or IAM role)
#   - AWS_REGION environment variable set
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

print_help() {
    print_banner "AWS Bedrock Hello World" "☁️"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_REGION              Required. AWS region (e.g., us-east-1)"
    echo "  AWS_ACCESS_KEY_ID       AWS access key ID (or use credentials file)"
    echo "  AWS_SECRET_ACCESS_KEY   AWS secret access key (or use credentials file)"
    echo "  AWS_PROFILE             AWS profile name from credentials file"
    echo ""
    echo "Setup Guide: https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html"
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "AWS Bedrock Hello World" "☁️"

# Check AWS CLI is installed (offers to install if missing)
check_aws_installed || true

# Set default region if not provided
export AWS_REGION="${AWS_REGION:-us-east-1}"

check_env_var "AWS_REGION" "https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html"

# AWS authentication: support three methods
#   1. API key (AWS_BEARER_TOKEN_BEDROCK) — simplest, requires inference profiles
#   2. IAM credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY) — explicit keys
#   3. AWS CLI profile / credentials file — uses `aws configure`
#
# We check for explicit env vars first, then fall back to AWS CLI auth check.
if [[ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    echo -e "${YELLOW}No AWS credentials found in environment.${NC}"
    echo ""
    echo "Choose an authentication method:"
    echo "  1. AWS CLI credentials (aws configure)"
    echo "  2. Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY"
    echo "  3. Set AWS_BEARER_TOKEN_BEDROCK (API key)"
    echo ""

    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Enter choice [${YELLOW}1${NC}]: "
        choice=""
        read -r choice < /dev/tty
        choice="${choice:-1}"

        case "$choice" in
            2)
                check_env_var "AWS_ACCESS_KEY_ID" "https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html"
                check_env_var "AWS_SECRET_ACCESS_KEY" "https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html"
                ;;
            3)
                check_env_var "AWS_BEARER_TOKEN_BEDROCK" "https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html"
                ;;
            *)
                # Default: use AWS CLI auth (prompts `aws configure` if needed)
                check_aws_auth || true
                ;;
        esac
    else
        # Non-interactive: try AWS CLI auth silently
        check_aws_auth || true
    fi
else
    if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
        echo -e "${GREEN}✓ Using API key authentication (AWS_BEARER_TOKEN_BEDROCK)${NC}"
    else
        echo -e "${GREEN}✓ Using IAM credentials (AWS_ACCESS_KEY_ID)${NC}"
    fi
fi
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
