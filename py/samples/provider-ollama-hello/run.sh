#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Ollama Hello World Demo
# =======================
#
# Demonstrates local LLM inference, tools, vision, reasoning, and embeddings
# with Genkit.
#
# Prerequisites:
#   - Ollama (auto-installed if missing)
#
# Usage:
#   ./run.sh          # Start the demo with Dev UI
#   ./run.sh --help   # Show this help message

set -euo pipefail

cd "$(dirname "$0")"
source "../_common.sh"

# Models used by this sample
MODELS=(
    "gemma3:latest"       # General generation & structured output
    "mistral-nemo:latest" # Tool calling (gablorken, currency, weather)
    "llava:latest"        # Vision / image description
    "moondream:v2"        # Vision / object detection (detect_objects flow)
    "nomic-embed-text"    # Embeddings for RAG
)

# HuggingFace models (pulled via hf.co/ prefix, requires Ollama 0.4+)
HF_MODELS=(
    "hf.co/Mungert/Fathom-R1-14B-GGUF"  # Reasoning (math, chain-of-thought)
)

pull_models() {
    echo ""
    echo "Pulling required models..."
    for model in "${MODELS[@]}"; do
        echo -e "  Pulling ${CYAN}${model}${NC}..."
        ollama pull "$model" 2>/dev/null || echo -e "  ${YELLOW}⚠${NC} Could not pull ${model} (will retry on first use)"
    done
    for model in "${HF_MODELS[@]}"; do
        echo -e "  Pulling ${CYAN}${model}${NC} (from HuggingFace)..."
        ollama pull "$model" 2>/dev/null || echo -e "  ${YELLOW}⚠${NC} Could not pull ${model} (will retry on first use)"
    done
    echo ""
}

print_help() {
    print_banner "Ollama Hello World" "🦙"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  - Ollama (auto-installed if missing)"
    echo ""
    echo "Models (auto-pulled on first run):"
    for model in "${MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "HuggingFace models:"
    for model in "${HF_MODELS[@]}"; do
        echo "  - $model"
    done
    print_help_footer
}

case "${1:-}" in
    --help|-h)
        print_help
        exit 0
        ;;
esac

print_banner "Ollama Hello World" "🦙"

check_ollama_installed || exit 1
pull_models

install_deps

genkit_start_with_browser -- \
    uv tool run --from watchdog watchmedo auto-restart \
        -d src \
        -d ../../packages \
        -d ../../plugins \
        -p '*.py;*.prompt;*.json' \
        -R \
        -- uv run src/main.py "$@"
