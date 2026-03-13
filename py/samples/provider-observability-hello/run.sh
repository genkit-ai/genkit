#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Observability Hello World Demo
# ===============================
#
# Demonstrates exporting Genkit telemetry to third-party observability
# platforms (Sentry, Honeycomb, Datadog, Grafana Cloud, Axiom).
#
# Prerequisites:
#   - GEMINI_API_KEY environment variable set
#   - At least one observability backend API key set (or select interactively)
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

# Backend definitions: name, env var, description, signup URL
BACKENDS=(
    "honeycomb|HONEYCOMB_API_KEY|Honeycomb (best query experience)|https://ui.honeycomb.io/signup"
    "datadog|DD_API_KEY|Datadog (full APM suite)|https://app.datadoghq.com/signup"
    "sentry|SENTRY_DSN|Sentry (error tracking + tracing)|https://sentry.io/signup/"
    "grafana|GRAFANA_API_KEY|Grafana Cloud (open-source stack)|https://grafana.com/auth/sign-up"
    "axiom|AXIOM_TOKEN|Axiom (fast SQL queries)|https://app.axiom.co/register"
)

print_help() {
    print_banner "Observability Hello World" "📡"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY        Required. Your Gemini API key"
    echo ""
    echo "Optional (set one to enable a backend, or select interactively):"
    for entry in "${BACKENDS[@]}"; do
        IFS='|' read -r _ env_var desc _ <<< "$entry"
        printf "  %-24s%s\n" "$env_var" "$desc"
    done
    echo ""
    echo "Override auto-detection:"
    echo "  OBSERVABILITY_BACKEND Backend name (honeycomb, datadog, sentry, grafana, axiom)"
    echo ""
    echo "Get a Gemini API key from: https://makersuite.google.com/app/apikey"
    print_help_footer
}

# Detect which backend has credentials already set.
detect_backend() {
    if [[ -n "${OBSERVABILITY_BACKEND:-}" ]]; then
        echo "$OBSERVABILITY_BACKEND"
        return
    fi
    for entry in "${BACKENDS[@]}"; do
        IFS='|' read -r name env_var _ _ <<< "$entry"
        if [[ -n "${!env_var:-}" ]]; then
            echo "$name"
            return
        fi
    done
    echo ""
}

# Prompt the user to choose a backend interactively.
choose_backend() {
    echo ""
    echo -e "${BLUE}Choose an observability backend:${NC}"
    echo ""
    local i=1
    for entry in "${BACKENDS[@]}"; do
        IFS='|' read -r _ _ desc _ <<< "$entry"
        echo "  ${i}) ${desc}"
        ((i++))
    done
    echo ""
    echo "  0) Skip — no backend (traces still visible in Genkit DevUI)"
    echo ""

    local choice
    while true; do
        echo -en "${GREEN}Enter choice [0-${#BACKENDS[@]}]: ${NC}"
        read -r choice < /dev/tty
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 0 && choice <= ${#BACKENDS[@]} )); then
            break
        fi
        echo -e "${YELLOW}Invalid choice. Try again.${NC}"
    done

    if (( choice == 0 )); then
        echo ""
        echo -e "${YELLOW}Skipping backend setup. Traces are still visible in the Genkit DevUI.${NC}"
        return
    fi

    local selected="${BACKENDS[$((choice - 1))]}"
    IFS='|' read -r name env_var desc signup_url <<< "$selected"

    echo ""
    echo -e "${BLUE}Selected: ${desc}${NC}"
    echo -e "Sign up at: ${GREEN}${signup_url}${NC}"
    echo ""

    # Prompt for the API key / DSN
    echo -en "Enter ${env_var}: "
    local value
    read -r value < /dev/tty

    if [[ -z "$value" ]]; then
        echo -e "${YELLOW}No value entered. Skipping backend setup.${NC}"
        return
    fi

    export "$env_var"="$value"
    export OBSERVABILITY_BACKEND="$name"
    echo -e "${GREEN}✓ ${env_var} set, using ${name} backend${NC}"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

# Main execution
print_banner "Observability Hello World" "📡"

check_env_var "GEMINI_API_KEY" "https://makersuite.google.com/app/apikey" || true

# Auto-detect or interactively choose backend
detected=$(detect_backend)
if [[ -n "$detected" ]]; then
    echo -e "${GREEN}✓ Using ${detected} backend (auto-detected)${NC}"
elif [[ -t 0 ]] && [ -c /dev/tty ]; then
    # Interactive terminal — let the user choose
    choose_backend
else
    echo -e "${YELLOW}⚠ No observability backend detected. Traces are still visible in the Genkit DevUI.${NC}"
    echo -e "  Set one of: HONEYCOMB_API_KEY, DD_API_KEY, SENTRY_DSN, GRAFANA_API_KEY, AXIOM_TOKEN"
fi

install_deps

# Start with hot reloading and auto-open browser
genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
