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

# Genkit Conformance Tests — Developer Environment Setup
# =======================================================
#
# Interactive script that:
#   1. Installs required tools (uv, Node.js, pnpm, genkit CLI)
#   2. Optionally installs Ollama (for local model testing)
#   3. Reads conform.toml to discover all plugins and their env vars
#   4. Interactively configures API keys for each provider
#   5. Persists environment variables to ~/.environment
#   6. Auto-sources ~/.environment in your shell (zsh, bash, fish)
#
# Supported platforms:
#   - macOS (Homebrew)
#   - Ubuntu / Debian (apt)
#   - Fedora (dnf)
#
# Usage:
#   ./setup.sh              # Full interactive setup
#   ./setup.sh --check      # Check what's installed (no changes)
#   ./setup.sh --keys-only  # Skip tools, configure API keys only
#   ./setup.sh --help       # Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFORM_TOML="${SCRIPT_DIR}/conform.toml"

# ── Colors ────────────────────────────────────────────────────────────

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    DIM='\033[2m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' DIM='' BOLD='' NC=''
fi

# ── Globals ───────────────────────────────────────────────────────────

ENV_FILE="$HOME/.environment"
CHECK_ONLY=false
KEYS_ONLY=false
TOOLS_CHANGED=0
KEYS_CONFIGURED=0
KEYS_SKIPPED=0

# ── Argument parsing ─────────────────────────────────────────────────

show_help() {
    cat <<'EOF'

  Genkit Conformance Tests — Developer Environment Setup

  Usage:
    ./setup.sh              Full interactive setup (tools + keys)
    ./setup.sh --check      Check what's installed (no changes)
    ./setup.sh --keys-only  Skip tools, configure API keys only
    ./setup.sh --help       Show this help

  What it does:
    1. Installs development tools (uv, Node.js, pnpm, genkit CLI)
    2. Optionally installs Ollama (for local model testing)
    3. Reads conform.toml to discover all plugins and their env vars
    4. Interactively configures API keys for each provider
    5. Saves environment variables to ~/.environment
    6. Ensures ~/.environment is sourced by your shell

  Environment file:
    All API keys are saved to ~/.environment so they persist across
    terminal sessions. The script adds a source line to your shell's
    RC file (e.g., ~/.zshrc, ~/.bashrc) if not already present.

  Re-running:
    Safe to re-run at any time. Existing values are shown as defaults
    and tools already installed are skipped.

EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --check)     CHECK_ONLY=true ;;
        --keys-only) KEYS_ONLY=true ;;
        --help|-h)   show_help ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Run ./setup.sh --help for usage"
            exit 1
            ;;
    esac
done

# ── Platform detection ────────────────────────────────────────────────

OS="$(uname -s)"
DISTRO="unknown"
PKG_MGR="none"

_detect_platform() {
    if [[ "$OS" == "Darwin" ]]; then
        DISTRO="macos"
        if command -v brew &>/dev/null; then
            PKG_MGR="brew"
        fi
    elif [[ "$OS" == "Linux" ]]; then
        if [[ -f /etc/os-release ]]; then
            # shellcheck disable=SC1091
            . /etc/os-release
            DISTRO="${ID:-unknown}"
        fi
        if command -v apt-get &>/dev/null; then
            PKG_MGR="apt"
        elif command -v dnf &>/dev/null; then
            PKG_MGR="dnf"
        elif command -v brew &>/dev/null; then
            PKG_MGR="brew"
        fi
    fi
}

_detect_platform

# ── Helper functions ──────────────────────────────────────────────────

_is_installed() {
    command -v "$1" &>/dev/null
}

_section() {
    echo ""
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${DIM}$(printf '─%.0s' $(seq 1 ${#1}))${NC}"
}

_confirm() {
    local prompt="$1"
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "${prompt} [Y/n]: "
        local response
        read -r response < /dev/tty
        [[ -z "$response" || "$response" =~ ^[Yy] ]]
    else
        return 0
    fi
}

_install_pkg() {
    local cmd="$1"
    local brew_pkg="${2:--}"
    local apt_pkg="${3:--}"
    local dnf_pkg="${4:--}"

    if _is_installed "$cmd"; then
        echo -e "  ${GREEN}✓${NC} $cmd ${DIM}($(command -v "$cmd"))${NC}"
        return 0
    fi

    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} $cmd — not installed"
        return 1
    fi

    case "$PKG_MGR" in
        brew)
            if [[ "$brew_pkg" != "-" ]]; then
                echo -e "  ${BLUE}→${NC} Installing $cmd via brew..."
                brew install "$brew_pkg"
                echo -e "  ${GREEN}✓${NC} $cmd installed"
                TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
                return 0
            fi
            ;;
        apt)
            if [[ "$apt_pkg" != "-" ]]; then
                echo -e "  ${BLUE}→${NC} Installing $cmd via apt..."
                sudo apt-get update -qq
                sudo apt-get install -y -qq "$apt_pkg"
                echo -e "  ${GREEN}✓${NC} $cmd installed"
                TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
                return 0
            fi
            ;;
        dnf)
            if [[ "$dnf_pkg" != "-" ]]; then
                echo -e "  ${BLUE}→${NC} Installing $cmd via dnf..."
                sudo dnf install -y -q "$dnf_pkg"
                echo -e "  ${GREEN}✓${NC} $cmd installed"
                TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
                return 0
            fi
            ;;
    esac

    echo -e "  ${RED}✗${NC} $cmd — could not install (no supported package manager)"
    return 1
}

# ── Tool installers ───────────────────────────────────────────────────

_install_uv() {
    if _is_installed uv; then
        echo -e "  ${GREEN}✓${NC} uv ${DIM}($(uv --version 2>/dev/null || echo 'installed'))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} uv — not installed"
        return 1
    fi
    echo -e "  ${BLUE}→${NC} Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env" || true
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "  ${GREEN}✓${NC} uv installed"
    TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
}

_install_node() {
    if _is_installed node; then
        echo -e "  ${GREEN}✓${NC} node ${DIM}($(node --version 2>/dev/null))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} node — not installed"
        return 1
    fi
    _install_pkg node node nodejs nodejs || {
        echo -e "  ${YELLOW}!${NC} Node.js is required for the genkit CLI"
        echo "       Install from: https://nodejs.org/"
        return 1
    }
}

_install_pnpm() {
    if _is_installed pnpm; then
        echo -e "  ${GREEN}✓${NC} pnpm ${DIM}($(pnpm --version 2>/dev/null))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} pnpm — not installed"
        return 1
    fi
    if _is_installed corepack; then
        echo -e "  ${BLUE}→${NC} Installing pnpm via corepack..."
        corepack enable
        corepack prepare pnpm@latest --activate
        echo -e "  ${GREEN}✓${NC} pnpm installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    elif _is_installed npm; then
        echo -e "  ${BLUE}→${NC} Installing pnpm via npm..."
        npm install -g pnpm
        echo -e "  ${GREEN}✓${NC} pnpm installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    else
        echo -e "  ${BLUE}→${NC} Installing pnpm via standalone installer..."
        curl -fsSL https://get.pnpm.io/install.sh | sh -
        export PNPM_HOME="$HOME/.local/share/pnpm"
        export PATH="$PNPM_HOME:$PATH"
        if _is_installed pnpm; then
            echo -e "  ${GREEN}✓${NC} pnpm installed"
            TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
        else
            echo -e "  ${YELLOW}!${NC} pnpm install may need a shell restart"
            return 1
        fi
    fi
}

_install_genkit() {
    if _is_installed genkit; then
        echo -e "  ${GREEN}✓${NC} genkit CLI ${DIM}($(command -v genkit))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} genkit CLI — not installed"
        return 1
    fi
    if _is_installed pnpm; then
        echo -e "  ${BLUE}→${NC} Installing genkit CLI via pnpm..."
        pnpm install -g genkit-cli
        echo -e "  ${GREEN}✓${NC} genkit CLI installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    elif _is_installed npm; then
        echo -e "  ${BLUE}→${NC} Installing genkit CLI via npm..."
        npm install -g genkit-cli
        echo -e "  ${GREEN}✓${NC} genkit CLI installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    else
        echo -e "  ${YELLOW}!${NC} Neither pnpm nor npm found — install genkit CLI manually:"
        echo "       pnpm install -g genkit-cli"
        return 1
    fi
}

_install_ollama() {
    if _is_installed ollama; then
        echo -e "  ${GREEN}✓${NC} ollama ${DIM}($(command -v ollama))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} ollama — not installed ${DIM}(optional: local model testing)${NC}"
        return 1
    fi
    if ! _confirm "  Install Ollama? (for local model conformance testing)"; then
        echo -e "  ${DIM}Skipped ollama${NC}"
        return 0
    fi

    case "$OS" in
        Darwin)
            if [[ "$PKG_MGR" == "brew" ]]; then
                echo -e "  ${BLUE}→${NC} Installing Ollama via brew..."
                brew install ollama
            else
                echo -e "  ${BLUE}→${NC} Installing Ollama..."
                curl -fsSL https://ollama.com/install.sh | sh
            fi
            ;;
        Linux)
            echo -e "  ${BLUE}→${NC} Installing Ollama via official installer..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        *)
            echo "  Visit: https://ollama.com/download"
            return 1
            ;;
    esac
    if _is_installed ollama; then
        echo -e "  ${GREEN}✓${NC} ollama installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    fi
}

_is_ollama_reachable() {
    curl -sf http://localhost:11434/api/version &>/dev/null
}

_ensure_ollama_running() {
    # Enable and start the ollama service so that 'ollama pull' and
    # conformance tests can reach the local server.
    if ! _is_installed ollama; then
        return 0
    fi
    if $CHECK_ONLY; then
        # Report service status without changing anything.
        if _is_ollama_reachable; then
            echo -e "  ${GREEN}✓${NC} ollama service ${DIM}(running)${NC}"
        else
            echo -e "  ${YELLOW}✗${NC} ollama service — not running"
        fi
        return 0
    fi

    # Already reachable — nothing to do.
    if _is_ollama_reachable; then
        echo -e "  ${GREEN}✓${NC} ollama service ${DIM}(already running)${NC}"
        return 0
    fi

    echo -e "  ${BLUE}→${NC} Starting ollama service..."

    case "$OS" in
        Linux)
            if command -v systemctl &>/dev/null && systemctl list-unit-files ollama.service &>/dev/null; then
                # The official Linux installer creates a systemd unit.
                sudo systemctl enable --now ollama 2>/dev/null \
                    && echo -e "  ${GREEN}✓${NC} ollama enabled via systemd" \
                    || echo -e "  ${YELLOW}!${NC} systemctl failed — falling back to ollama serve"
            fi
            ;;
        Darwin)
            if [[ "$PKG_MGR" == "brew" ]]; then
                brew services start ollama 2>/dev/null \
                    && echo -e "  ${GREEN}✓${NC} ollama started via brew services" \
                    || echo -e "  ${YELLOW}!${NC} brew services failed — falling back to ollama serve"
            fi
            ;;
    esac

    # Fallback: if the service still isn't reachable, start in background.
    if ! _is_ollama_reachable; then
        nohup ollama serve &>/dev/null &
        # Wait briefly for the server to come up.
        local retries=0
        while [[ $retries -lt 10 ]]; do
            if _is_ollama_reachable; then
                break
            fi
            sleep 1
            retries=$((retries + 1))
        done
        if _is_ollama_reachable; then
            echo -e "  ${GREEN}✓${NC} ollama serve started (background)"
        else
            echo -e "  ${RED}✗${NC} Could not start ollama service in background — model pulls may fail"
        fi
    fi
}

_pull_ollama_models() {
    # Pull all models referenced in ollama/model-conformance.yaml.
    # Data-driven: adding a new model entry to the YAML automatically
    # includes it here — no hardcoded model names.
    local spec_file="${SCRIPT_DIR}/ollama/model-conformance.yaml"

    if [[ ! -f "$spec_file" ]]; then
        echo -e "  ${DIM}No ollama model-conformance.yaml found — skipping${NC}"
        return 0
    fi
    if ! _is_installed ollama; then
        return 0
    fi

    echo ""
    echo -e "${DIM}Pulling Ollama models referenced in model-conformance.yaml...${NC}"
    echo ""

    # Extract model names: lines matching "- model: ollama/<name>".
    local models
    models=$(sed -nE 's~^- model: ollama/([^[:space:]]+)[[:space:]]*$~\1~p' "$spec_file")

    if [[ -z "$models" ]]; then
        echo -e "  ${DIM}No ollama models found in spec${NC}"
        return 0
    fi

    # Build an associative array of already-pulled models for O(1) lookups.
    local pulled_raw
    pulled_raw=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || true)
    declare -A pulled_set
    while IFS= read -r entry; do
        [[ -z "$entry" ]] && continue
        # Strip ":latest" suffix so both "model" and "model:latest" match.
        pulled_set["${entry%%:*}"]=1
    done <<< "$pulled_raw"

    local pull_count=0
    while IFS= read -r model; do
        [[ -z "$model" ]] && continue

        # Check if already pulled (O(1) associative array lookup).
        local base="${model%%:*}"
        if [[ -n "${pulled_set[$base]+x}" ]]; then
            echo -e "  ${GREEN}✓${NC} ${model} ${DIM}(already pulled)${NC}"
            continue
        fi

        if $CHECK_ONLY; then
            echo -e "  ${YELLOW}✗${NC} ${model} — not pulled"
            continue
        fi

        echo -e "  ${BLUE}→${NC} Pulling ${model}..."
        if ollama pull "$model"; then
            echo -e "  ${GREEN}✓${NC} ${model} pulled"
            pull_count=$((pull_count + 1))
        else
            echo -e "  ${RED}✗${NC} Failed to pull ${model}. Check Ollama service and network connectivity."
        fi
    done <<< "$models"

    if [[ $pull_count -gt 0 ]]; then
        TOOLS_CHANGED=$((TOOLS_CHANGED + pull_count))
    fi
}

# ── Environment file management ──────────────────────────────────────

_env_file_get() {
    local var_name="$1"
    if [[ -f "$ENV_FILE" ]]; then
        grep -E "^export ${var_name}=" "$ENV_FILE" 2>/dev/null \
            | tail -1 \
            | sed -E "s/^export ${var_name}=['\"]?([^'\"]*)['\"]?$/\1/" \
            || true
    fi
}

_env_file_set() {
    local var_name="$1"
    local value="$2"

    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" <<'HEADER'
# ~/.environment — Shared environment variables
# Auto-generated by Genkit setup scripts.
# Source this file in your shell RC (done automatically by setup.sh).
#
# Manual edits are preserved — the setup script only updates lines
# that start with "export VARIABLE_NAME=".

HEADER
        chmod 600 "$ENV_FILE"
    fi

    # Remove any existing line for this variable.
    if grep -qE "^export ${var_name}=" "$ENV_FILE" 2>/dev/null; then
        local tmp
        tmp=$(mktemp)
        grep -vE "^export ${var_name}=" "$ENV_FILE" > "$tmp"
        mv "$tmp" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
    fi

    echo "export ${var_name}=\"${value}\"" >> "$ENV_FILE"
    export "$var_name"="$value"
}

_ensure_shell_sources_env() {
    _section "Shell Integration"

    local configured=0

    for rc_file in "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if [[ -f "$rc_file" ]]; then
            if ! grep -qF "$ENV_FILE" "$rc_file" 2>/dev/null; then
                if ! $CHECK_ONLY; then
                    {
                        echo ""
                        echo "# Genkit environment variables"
                        echo "[ -f \"$ENV_FILE\" ] && source \"$ENV_FILE\""
                    } >> "$rc_file"
                    echo -e "  ${GREEN}✓${NC} Added source line to ${DIM}$rc_file${NC}"
                else
                    echo -e "  ${YELLOW}✗${NC} Not sourced in ${DIM}$rc_file${NC}"
                fi
                configured=$((configured + 1))
            else
                echo -e "  ${GREEN}✓${NC} Already sourced in ${DIM}$rc_file${NC}"
            fi
        fi
    done

    local zshrc="$HOME/.zshrc"
    if [[ -f "$zshrc" ]]; then
        if ! grep -qF "$ENV_FILE" "$zshrc" 2>/dev/null; then
            if ! $CHECK_ONLY; then
                {
                    echo ""
                    echo "# Genkit environment variables"
                    echo "[ -f \"$ENV_FILE\" ] && source \"$ENV_FILE\""
                } >> "$zshrc"
                echo -e "  ${GREEN}✓${NC} Added source line to ${DIM}$zshrc${NC}"
            else
                echo -e "  ${YELLOW}✗${NC} Not sourced in ${DIM}$zshrc${NC}"
            fi
            configured=$((configured + 1))
        else
            echo -e "  ${GREEN}✓${NC} Already sourced in ${DIM}$zshrc${NC}"
        fi
    elif [[ "$(basename "${SHELL:-}")" == "zsh" ]] && ! $CHECK_ONLY; then
        echo "# Genkit environment variables" > "$zshrc"
        echo "[ -f \"$ENV_FILE\" ] && source \"$ENV_FILE\"" >> "$zshrc"
        echo -e "  ${GREEN}✓${NC} Created ${DIM}$zshrc${NC} with source line"
        configured=$((configured + 1))
    fi

    local fish_config="$HOME/.config/fish/config.fish"
    if [[ -d "$HOME/.config/fish" ]] && [[ -f "$fish_config" ]]; then
        if ! grep -qF "$ENV_FILE" "$fish_config" 2>/dev/null; then
            if ! $CHECK_ONLY; then
                {
                    echo ""
                    echo "# Genkit environment variables"
                    echo "# Requires bass plugin: fisher install edc/bass"
                    echo "if test -f $ENV_FILE"
                    echo "    bass source $ENV_FILE"
                    echo "end"
                } >> "$fish_config"
                echo -e "  ${GREEN}✓${NC} Added source lines to ${DIM}$fish_config${NC}"
            else
                echo -e "  ${YELLOW}✗${NC} Not sourced in ${DIM}$fish_config${NC}"
            fi
            configured=$((configured + 1))
        else
            echo -e "  ${GREEN}✓${NC} Already sourced in ${DIM}$fish_config${NC}"
        fi
    fi

    if [[ $configured -eq 0 ]]; then
        echo -e "  ${DIM}No shell RC changes needed${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}Environment file:${NC} $ENV_FILE"
    echo -e "  ${DIM}Run \`source $ENV_FILE\` to load in current session${NC}"
}

# ── API key prompt ────────────────────────────────────────────────────

_prompt_key() {
    local var_name="$1"
    local description="$2"
    local doc_url="${3:-}"
    local is_secret="${4:-true}"

    local current_val="${!var_name:-}"
    if [[ -z "$current_val" ]]; then
        current_val="$(_env_file_get "$var_name")"
        if [[ -n "$current_val" ]]; then
            export "$var_name"="$current_val"
        fi
    fi

    echo ""
    echo -e "  ${BOLD}${var_name}${NC} — ${description}"
    if [[ -n "$doc_url" ]]; then
        echo -e "  ${DIM}Get credentials: ${CYAN}${doc_url}${NC}"
    fi

    local display_val=""
    if [[ -n "$current_val" ]]; then
        if [[ "$is_secret" == "true" ]]; then
            if [[ ${#current_val} -gt 8 ]]; then
                local num_asterisks=$((${#current_val} - 4))
                local asterisks
                asterisks=$(printf '*%.0s' $(seq 1 "$num_asterisks"))
                display_val="${current_val:0:4}${asterisks}"
            else
                display_val="******"
            fi
        else
            display_val="$current_val"
        fi
    fi

    if ! [[ -t 0 ]] || ! [ -c /dev/tty ]; then
        if [[ -n "$current_val" ]]; then
            echo -e "  ${GREEN}✓${NC} Using existing value"
        else
            echo -e "  ${DIM}Skipped (non-interactive)${NC}"
            KEYS_SKIPPED=$((KEYS_SKIPPED + 1))
        fi
        return 0
    fi

    if [[ -n "$display_val" ]]; then
        echo -e "  ${GREEN}✓${NC} Current: ${YELLOW}${display_val}${NC}"
        echo -en "  New value (Enter=keep current, 'skip'=clear): "
    else
        echo -en "  Enter value (or 'skip' to skip): "
    fi

    local input_val
    if read -r input_val < /dev/tty; then
        if [[ "$input_val" == "skip" || "$input_val" == "SKIP" || "$input_val" == "s" ]]; then
            echo -e "  ${DIM}Skipped${NC}"
            KEYS_SKIPPED=$((KEYS_SKIPPED + 1))
            return 0
        fi

        if [[ -z "$input_val" ]]; then
            if [[ -n "$current_val" ]]; then
                _env_file_set "$var_name" "$current_val"
                echo -e "  ${GREEN}✓${NC} Kept existing value"
                KEYS_CONFIGURED=$((KEYS_CONFIGURED + 1))
            else
                echo -e "  ${DIM}No value set${NC}"
                KEYS_SKIPPED=$((KEYS_SKIPPED + 1))
            fi
            return 0
        fi

        _env_file_set "$var_name" "$input_val"
        echo -e "  ${GREEN}✓${NC} Saved to $ENV_FILE"
        KEYS_CONFIGURED=$((KEYS_CONFIGURED + 1))
    else
        echo ""
        echo -e "  ${DIM}Skipped${NC}"
        KEYS_SKIPPED=$((KEYS_SKIPPED + 1))
    fi
}

# ── Parse conform.toml for plugin env vars ────────────────────────────
# Reads [conform.env] section to discover plugins and their required
# environment variables. This is the single source of truth — adding a
# new plugin to conform.toml automatically includes it here.

# Documentation URLs for each env var (for interactive prompts).
declare -A DOC_URLS=(
    [GEMINI_API_KEY]="https://aistudio.google.com/apikey"
    [ANTHROPIC_API_KEY]="https://console.anthropic.com/settings/keys"
    [OPENAI_API_KEY]="https://platform.openai.com/api-keys"
    [DEEPSEEK_API_KEY]="https://platform.deepseek.com/api_keys"
    [XAI_API_KEY]="https://console.x.ai/"
    [MISTRAL_API_KEY]="https://console.mistral.ai/api-keys/"
    [COHERE_API_KEY]="https://dashboard.cohere.com/api-keys"
    [HF_TOKEN]="https://huggingface.co/settings/tokens"
    [CLOUDFLARE_ACCOUNT_ID]="https://dash.cloudflare.com/"
    [CLOUDFLARE_API_TOKEN]="https://developers.cloudflare.com/fundamentals/api/get-started/create-token/"
    [AZURE_OPENAI_API_KEY]="https://ai.azure.com/"
    [AZURE_OPENAI_ENDPOINT]="https://ai.azure.com/"
    [AWS_REGION]="https://docs.aws.amazon.com/general/latest/gr/bedrock.html"
    [GOOGLE_CLOUD_PROJECT]="https://console.cloud.google.com/"
)

# Human-readable descriptions for env vars.
declare -A VAR_DESCRIPTIONS=(
    [GEMINI_API_KEY]="Gemini API key (Google AI)"
    [ANTHROPIC_API_KEY]="Anthropic API key (Claude)"
    [OPENAI_API_KEY]="OpenAI API key"
    [DEEPSEEK_API_KEY]="DeepSeek API key"
    [XAI_API_KEY]="xAI API key (Grok)"
    [MISTRAL_API_KEY]="Mistral AI API key"
    [COHERE_API_KEY]="Cohere API key"
    [HF_TOKEN]="Hugging Face API token"
    [CLOUDFLARE_ACCOUNT_ID]="Cloudflare account ID"
    [CLOUDFLARE_API_TOKEN]="Cloudflare API token"
    [AZURE_OPENAI_API_KEY]="Azure OpenAI API key"
    [AZURE_OPENAI_ENDPOINT]="Azure OpenAI endpoint URL"
    [AWS_REGION]="AWS region (e.g., us-east-1)"
    [GOOGLE_CLOUD_PROJECT]="GCP project ID"
)

# Non-secret vars (shown in plain text, not masked).
declare -A NON_SECRET_VARS=(
    [CLOUDFLARE_ACCOUNT_ID]=1
    [AZURE_OPENAI_ENDPOINT]=1
    [AWS_REGION]=1
    [GOOGLE_CLOUD_PROJECT]=1
)

# Parse [conform.env] from conform.toml.
# Output: lines of "plugin_name VAR1 VAR2 ..."
_parse_conform_env() {
    if [[ ! -f "$CONFORM_TOML" ]]; then
        echo -e "${RED}ERROR${NC}: conform.toml not found at ${CONFORM_TOML}" >&2
        exit 1
    fi

    local in_env_section=false
    while IFS= read -r line; do
        # Strip comments and trailing whitespace.
        line="${line%%#*}"
        line="${line%"${line##*[![:space:]]}"}"

        # Detect section headers.
        if [[ "$line" =~ ^\[conform\.env\]$ ]]; then
            in_env_section=true
            continue
        elif [[ "$line" =~ ^\[.+\]$ ]]; then
            in_env_section=false
            continue
        fi

        if $in_env_section && [[ -n "$line" ]]; then
            # Parse: plugin-name = ["VAR1", "VAR2"]
            local plugin vars_str
            plugin=$(echo "$line" | sed -E 's/^([a-zA-Z0-9_-]+)[[:space:]]*=.*/\1/')
            vars_str=$(echo "$line" | sed -E 's/^[^=]+=[[:space:]]*//' )
            # Extract var names from the array syntax.
            local vars
            vars=$(echo "$vars_str" | tr -d '[]"' | tr ',' ' ' | tr -s ' ')
            vars="${vars## }"
            vars="${vars%% }"
            echo "${plugin} ${vars}"
        fi
    done < "$CONFORM_TOML"
}

# ── Main ──────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║     Genkit Conformance Tests — Environment Setup             ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${DIM}Platform: $OS / $DISTRO / pkg: $PKG_MGR${NC}"
echo -e "${DIM}Config:   $CONFORM_TOML${NC}"
echo -e "${DIM}Env file: $ENV_FILE${NC}"

# ── Phase 1: Install tools ────────────────────────────────────────────

if ! $KEYS_ONLY; then

    _section "Required Tools"

    if $CHECK_ONLY; then
        echo "Checking installed tools..."
    else
        echo "Installing required development tools..."
    fi
    echo ""

    _install_uv || true
    _install_node || true
    _install_pnpm || true
    _install_genkit || true

    if _is_installed python3; then
        echo -e "  ${GREEN}✓${NC} python3 ${DIM}($(python3 --version 2>/dev/null))${NC}"
    else
        _install_pkg python3 python3 python3 python3 || true
    fi

    if _is_installed git; then
        echo -e "  ${GREEN}✓${NC} git ${DIM}($(git --version 2>/dev/null | head -c 20))${NC}"
    else
        _install_pkg git git git git || true
    fi

    _section "Optional Tools"

    echo -e "${DIM}Ollama is needed for local model conformance testing (ollama plugin).${NC}"
    echo ""

    _install_ollama || true
    _ensure_ollama_running
    _pull_ollama_models

fi  # end !KEYS_ONLY

# ── Phase 2: Configure API keys from conform.toml ────────────────────

_section "API Key Configuration"

echo -e "Configuring API keys for conformance test plugins."
echo -e "Keys are read from ${CYAN}conform.toml${NC} [conform.env] section."
echo -e "Type ${BOLD}skip${NC} or press ${BOLD}Enter${NC} to skip any key you don't need."
echo -e "All values are saved to ${CYAN}$ENV_FILE${NC}."

if ! [[ -t 0 ]] || ! [ -c /dev/tty ]; then
    echo ""
    echo -e "${YELLOW}Non-interactive mode detected. Skipping API key prompts.${NC}"
    echo "Run this script in an interactive terminal to configure keys."
else
    # Collect unique env vars across all plugins (preserving order).
    declare -A seen_set=()

    while IFS= read -r line; do
        plugin="${line%% *}"
        vars="${line#* }"

        if [[ -z "$vars" || "$vars" == "$plugin" ]]; then
            continue  # No env vars (e.g., ollama).
        fi

        echo ""
        echo -e "  ${BOLD}── ${plugin} ──${NC}"

        for var in $vars; do
            if [[ -n "${seen_set[$var]:-}" ]]; then
                # Already prompted for this var (shared across plugins).
                echo -e "  ${DIM}${var} — already configured above${NC}"
                continue
            fi
            seen_set[$var]=1

            desc="${VAR_DESCRIPTIONS[$var]:-$var}"
            url="${DOC_URLS[$var]:-}"
            secret="true"
            if [[ -n "${NON_SECRET_VARS[$var]:-}" ]]; then
                secret="false"
            fi

            _prompt_key "$var" "$desc" "$url" "$secret"
        done
    done < <(_parse_conform_env)
fi

# ── Phase 3: Shell integration ────────────────────────────────────────

_ensure_shell_sources_env

# ── Phase 4: Install Python dependencies ──────────────────────────────

if ! $KEYS_ONLY && ! $CHECK_ONLY; then
    _section "Python Dependencies"

    if _is_installed uv; then
        py_root="${REPO_ROOT}/py"
        if [[ -f "${py_root}/pyproject.toml" ]]; then
            echo "Syncing workspace dependencies..."
            echo ""
            (cd "$py_root" && uv sync)
            echo -e "  ${GREEN}✓${NC} Workspace dependencies installed"
        else
            echo -e "  ${DIM}Not in monorepo — skipping workspace sync${NC}"
        fi
    fi
fi

# ── Phase 5: Show readiness summary ──────────────────────────────────

_section "Plugin Readiness"

echo -e "Checking which plugins have all required env vars set..."
echo ""

ready_count=0
not_ready_count=0

while IFS= read -r line; do
    plugin="${line%% *}"
    vars="${line#* }"

    if [[ -z "$vars" || "$vars" == "$plugin" ]]; then
        echo -e "  ${GREEN}●${NC} ${BOLD}${plugin}${NC}  ${DIM}(no credentials needed)${NC}"
        ready_count=$((ready_count + 1))
        continue
    fi

    all_set=true
    colored_vars=""
    for var in $vars; do
        val="${!var:-}"
        if [[ -n "$val" ]]; then
            colored_vars="${colored_vars}${BLUE}${var}${NC} "
        else
            colored_vars="${colored_vars}${RED}${var}${NC} "
            all_set=false
        fi
    done

    if $all_set; then
        echo -e "  ${GREEN}●${NC} ${BOLD}${plugin}${NC}  ${DIM}(${NC}${colored_vars}${DIM})${NC}"
        ready_count=$((ready_count + 1))
    else
        echo -e "  ${RED}○${NC} ${BOLD}${plugin}${NC}  ${DIM}(${NC}${colored_vars}${DIM})${NC}"
        not_ready_count=$((not_ready_count + 1))
    fi
done < <(_parse_conform_env)

echo ""
echo -e "  ${GREEN}●${NC} Ready    ${RED}○${NC} Missing env vars"
echo -e "  ${BLUE}VAR${NC} Set     ${RED}VAR${NC} Not set"

# ── Summary ───────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}Setup Complete${NC}"
echo ""

if $CHECK_ONLY; then
    echo -e "Run ${GREEN}./setup.sh${NC} (without --check) to install missing tools."
else
    if [[ $TOOLS_CHANGED -gt 0 ]]; then
        echo -e "  ${GREEN}✓${NC} ${TOOLS_CHANGED} tool(s) installed or updated"
    fi
    if [[ $KEYS_CONFIGURED -gt 0 ]]; then
        echo -e "  ${GREEN}✓${NC} ${KEYS_CONFIGURED} API key(s) configured"
    fi
    if [[ $KEYS_SKIPPED -gt 0 ]]; then
        echo -e "  ${DIM}  ${KEYS_SKIPPED} key(s) skipped${NC}"
    fi

    echo ""
    echo -e "  ${BOLD}Next steps:${NC}"
    echo ""
    echo -e "  1. Load your environment:    ${GREEN}source ~/.environment${NC}"
    echo -e "  2. Run a single plugin:      ${GREEN}py/bin/conform check-model anthropic${NC}"
    echo -e "  3. Run all plugins:          ${GREEN}py/bin/conform check-model${NC}"
    echo -e "  4. Check readiness:          ${GREEN}py/bin/conform list${NC}"
    echo ""
    echo -e "  ${DIM}Tip: New terminal sessions auto-load ~/.environment${NC}"
    echo -e "  ${DIM}Tip: Re-run ./setup.sh any time to add more keys${NC}"
fi

echo ""
