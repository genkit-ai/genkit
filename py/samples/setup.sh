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

# Genkit Python Samples — Developer Environment Setup
# =====================================================
#
# Interactive script that:
#   1. Installs required tools (uv, Node.js, pnpm, genkit CLI, watchdog)
#   2. Installs optional tools (Ollama, gcloud, aws, az)
#   3. Configures cloud auth via CLI (gcloud auth, aws configure, az login)
#   4. Walks you through acquiring API keys for all sample providers
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
#
# The script is safe to re-run — it skips tools already installed and
# preserves existing environment variable values as defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Colors ────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

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

  Genkit Python Samples — Developer Environment Setup

  Usage:
    ./setup.sh              Full interactive setup (tools + keys)
    ./setup.sh --check      Check what's installed (no changes)
    ./setup.sh --keys-only  Skip tools, configure API keys only
    ./setup.sh --help       Show this help

  What it does:
    1. Installs development tools (uv, Node.js, pnpm, genkit CLI, watchdog)
    2. Optionally installs Ollama, gcloud, aws, az CLIs
    3. Configures cloud authentication via CLI tools
    4. Interactively configures API keys for all sample providers
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

# Print a section header.
_section() {
    echo ""
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${DIM}$(printf '─%.0s' $(seq 1 ${#1}))${NC}"
}

# Prompt the user with a yes/no question. Default is yes.
# Usage: _confirm "Install uv?" && do_thing
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

# Install a package using the system package manager.
# Usage: _install_pkg <cmd> <brew-pkg> <apt-pkg> <dnf-pkg>
# Pass "-" to skip a package manager.
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

# ── Public registry configuration ─────────────────────────────────────
# Corporate environments often configure npm/pnpm and pip/uv to use
# private registries (via .npmrc, pip.conf, or env vars). This silently
# poisons lock files — packages resolve to different URLs/hashes, causing
# "uv lock" or "pnpm install" to modify the lockfile on every run.
#
# This function detects private registry overrides and resets them to the
# canonical public registries for the current session.

_configure_public_registries() {
    _section "Package Registries"

    echo -e "${DIM}Ensuring public registries are used (avoids lock file corruption).${NC}"
    echo ""

    local warnings=0

    # ── npm / pnpm ────────────────────────────────────────────────────
    local npm_public="https://registry.npmjs.org/"

    # Check environment variable override.
    if [[ -n "${NPM_CONFIG_REGISTRY:-}" ]] && [[ "$NPM_CONFIG_REGISTRY" != "$npm_public" ]]; then
        echo -e "  ${YELLOW}!${NC} NPM_CONFIG_REGISTRY is set to a private registry:"
        echo -e "     ${DIM}${NPM_CONFIG_REGISTRY}${NC}"
        if ! $CHECK_ONLY; then
            unset NPM_CONFIG_REGISTRY
            echo -e "  ${GREEN}✓${NC} Unset NPM_CONFIG_REGISTRY for this session"
        fi
        warnings=$((warnings + 1))
    fi

    # Check user-level .npmrc.
    if [[ -f "$HOME/.npmrc" ]]; then
        local user_registry
        user_registry=$(grep -E '^registry\s*=' "$HOME/.npmrc" 2>/dev/null \
            | sed 's/^registry\s*=\s*//' | tr -d ' ' || true)
        if [[ -n "$user_registry" ]] && [[ "$user_registry" != "$npm_public" ]]; then
            echo -e "  ${YELLOW}!${NC} ~/.npmrc has a private registry:"
            echo -e "     ${DIM}${user_registry}${NC}"
            echo -e "     ${DIM}This may cause pnpm lock file changes.${NC}"
            warnings=$((warnings + 1))
        fi
    fi

    # Ensure the project-level .npmrc (if any) uses public registry.
    local project_npmrc="${SCRIPT_DIR}/../../.npmrc"
    if [[ -f "$project_npmrc" ]]; then
        local proj_registry
        proj_registry=$(grep -E '^registry\s*=' "$project_npmrc" 2>/dev/null \
            | sed 's/^registry\s*=\s*//' | tr -d ' ' || true)
        if [[ -n "$proj_registry" ]]; then
            echo -e "  ${GREEN}✓${NC} Project .npmrc registry: ${DIM}${proj_registry}${NC}"
        fi
    fi

    if _is_installed pnpm; then
        local pnpm_registry
        pnpm_registry=$(pnpm config get registry 2>/dev/null || echo "")
        if [[ -n "$pnpm_registry" ]] \
            && [[ "$pnpm_registry" != "$npm_public" ]] \
            && [[ "$pnpm_registry" != "undefined" ]]; then
            echo -e "  ${YELLOW}!${NC} pnpm registry override detected:"
            echo -e "     ${DIM}${pnpm_registry}${NC}"
            warnings=$((warnings + 1))
        else
            echo -e "  ${GREEN}✓${NC} pnpm registry: ${DIM}${npm_public}${NC}"
        fi
    fi

    # ── pip / uv ──────────────────────────────────────────────────────
    local pypi_public="https://pypi.org/simple/"

    # Check PIP_INDEX_URL / UV_INDEX_URL.
    for var_name in PIP_INDEX_URL UV_INDEX_URL; do
        local val="${!var_name:-}"
        if [[ -n "$val" ]] && [[ "$val" != "$pypi_public" ]] && [[ "$val" != "https://pypi.org/simple" ]]; then
            echo -e "  ${YELLOW}!${NC} ${var_name} is set to a private index:"
            echo -e "     ${DIM}${val}${NC}"
            if ! $CHECK_ONLY; then
                unset "$var_name"
                echo -e "  ${GREEN}✓${NC} Unset ${var_name} for this session"
            fi
            warnings=$((warnings + 1))
        fi
    done

    # Check PIP_EXTRA_INDEX_URL / UV_EXTRA_INDEX_URL (these add extra
    # sources, which can cause resolution differences).
    for var_name in PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL; do
        local val="${!var_name:-}"
        if [[ -n "$val" ]]; then
            echo -e "  ${YELLOW}!${NC} ${var_name} is set (extra index):"
            echo -e "     ${DIM}${val}${NC}"
            if ! $CHECK_ONLY; then
                unset "$var_name"
                echo -e "  ${GREEN}✓${NC} Unset ${var_name} for this session"
            fi
            warnings=$((warnings + 1))
        fi
    done

    # Check pip.conf / uv.toml for index overrides.
    local pip_conf=""
    if [[ -f "$HOME/.config/pip/pip.conf" ]]; then
        pip_conf="$HOME/.config/pip/pip.conf"
    elif [[ -f "$HOME/.pip/pip.conf" ]]; then
        pip_conf="$HOME/.pip/pip.conf"
    elif [[ "$OS" == "Darwin" ]] && [[ -f "$HOME/Library/Application Support/pip/pip.conf" ]]; then
        pip_conf="$HOME/Library/Application Support/pip/pip.conf"
    fi

    if [[ -n "$pip_conf" ]]; then
        local pip_idx
        pip_idx=$(grep -iE '^index-url\s*=' "$pip_conf" 2>/dev/null \
            | sed 's/^index-url\s*=\s*//' | tr -d ' ' || true)
        if [[ -n "$pip_idx" ]] && [[ "$pip_idx" != "$pypi_public" ]] && [[ "$pip_idx" != "https://pypi.org/simple" ]]; then
            echo -e "  ${YELLOW}!${NC} pip.conf has a private index:"
            echo -e "     ${DIM}${pip_conf}: ${pip_idx}${NC}"
            echo -e "     ${DIM}This may cause uv lock file changes.${NC}"
            warnings=$((warnings + 1))
        fi
    fi

    if _is_installed uv; then
        echo -e "  ${GREEN}✓${NC} uv index: ${DIM}${pypi_public}${NC}"
    fi

    if [[ $warnings -eq 0 ]]; then
        echo -e "  ${GREEN}✓${NC} All registries point to public sources"
    else
        echo ""
        echo -e "  ${YELLOW}⚠${NC}  ${warnings} private registry override(s) detected."
        if $CHECK_ONLY; then
            echo -e "     Run ${GREEN}./setup.sh${NC} to fix for the current session."
        else
            echo -e "     Overrides have been unset for this session."
            echo -e "     ${DIM}To permanently fix, remove the registry lines from${NC}"
            echo -e "     ${DIM}~/.npmrc, ~/.config/pip/pip.conf, or your shell RC file.${NC}"
        fi
    fi
}

# ── Tool installers ───────────────────────────────────────────────────
# These match the tools required by the sample run.sh scripts:
#   - uv          (every run.sh: uv sync, uv run)
#   - Node.js     (genkit CLI dependency)
#   - pnpm        (JS monorepo package manager)
#   - genkit CLI  (every run.sh: genkit start)
#   - watchdog    (every run.sh: watchmedo auto-restart for hot-reload)
#   - python3     (runtime)
#   - git         (monorepo development)

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
        echo -e "  ${YELLOW}!${NC} Node.js is required for pnpm and the genkit CLI"
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
    # corepack (bundled with Node.js 16.10+) is the recommended way.
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
        # Standalone installer as last resort.
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
        echo "       Or: curl -sL cli.genkit.dev | bash"
        return 1
    fi
}

_install_watchdog() {
    # watchdog provides watchmedo, used by every run.sh for hot-reload.
    if _is_installed watchmedo; then
        echo -e "  ${GREEN}✓${NC} watchmedo ${DIM}(watchdog — $(command -v watchmedo))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} watchmedo — not installed (watchdog package)"
        return 1
    fi
    if _is_installed uv; then
        echo -e "  ${BLUE}→${NC} Installing watchdog (provides watchmedo for hot-reload)..."
        uv tool install watchdog
        echo -e "  ${GREEN}✓${NC} watchdog installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    elif _is_installed pip; then
        echo -e "  ${BLUE}→${NC} Installing watchdog via pip..."
        pip install watchdog
        echo -e "  ${GREEN}✓${NC} watchdog installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    else
        echo -e "  ${YELLOW}!${NC} Could not install watchdog (need uv or pip)"
        return 1
    fi
}

# ── Optional tool installers ─────────────────────────────────────────
# These match tools used by specific samples.

_install_ollama() {
    # Used by: provider-ollama-hello
    if _is_installed ollama; then
        echo -e "  ${GREEN}✓${NC} ollama ${DIM}($(command -v ollama))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} ollama — not installed ${DIM}(optional: provider-ollama-hello)${NC}"
        return 1
    fi
    if ! _confirm "  Install Ollama? (for local model inference)"; then
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
            # Prefer system package manager when the package is available,
            # then fall back to the official curl installer.
            local installed_via_pkg=false
            if [[ "$PKG_MGR" == "apt" ]]; then
                if apt-cache show ollama &>/dev/null 2>&1; then
                    echo -e "  ${BLUE}→${NC} Installing Ollama via apt..."
                    sudo apt-get update -qq
                    sudo apt-get install -y -qq ollama
                    installed_via_pkg=true
                fi
            elif [[ "$PKG_MGR" == "dnf" ]]; then
                if dnf info ollama &>/dev/null 2>&1; then
                    echo -e "  ${BLUE}→${NC} Installing Ollama via dnf..."
                    sudo dnf install -y -q ollama
                    installed_via_pkg=true
                fi
            fi

            if ! $installed_via_pkg; then
                echo -e "  ${BLUE}→${NC} Installing Ollama via official installer..."
                curl -fsSL https://ollama.com/install.sh | sh
            fi
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

_install_gcloud() {
    if _is_installed gcloud; then
        echo -e "  ${GREEN}✓${NC} gcloud ${DIM}($(command -v gcloud))${NC}"
        # If gcloud was NOT installed via a package manager, offer to update components.
        if ! $CHECK_ONLY; then
            local gcloud_path
            gcloud_path="$(command -v gcloud)"
            local is_pkg_managed=false
            case "$PKG_MGR" in
                brew)
                    # Homebrew installs to /opt/homebrew or /usr/local/Cellar
                    if [[ "$gcloud_path" == */Cellar/* || "$gcloud_path" == */homebrew/* || "$gcloud_path" == */Caskroom/* ]]; then
                        is_pkg_managed=true
                    fi
                    ;;
                apt|dnf)
                    # System package installs to /usr/bin or /usr/lib
                    if [[ "$gcloud_path" == /usr/bin/* || "$gcloud_path" == /usr/lib/* || "$gcloud_path" == /snap/* ]]; then
                        is_pkg_managed=true
                    fi
                    ;;
            esac
            if ! $is_pkg_managed; then
                if _confirm "  Update gcloud components?"; then
                    echo -e "  ${BLUE}→${NC} Running gcloud components update..."
                    gcloud components update --quiet 2>/dev/null || true
                    echo -e "  ${GREEN}✓${NC} gcloud components updated"
                fi
            fi
        fi
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} gcloud — not installed ${DIM}(optional: GCP/Vertex AI samples)${NC}"
        return 1
    fi
    if ! _confirm "  Install Google Cloud SDK? (for GCP/Vertex AI samples)"; then
        echo -e "  ${DIM}Skipped gcloud${NC}"
        return 0
    fi

    case "$OS" in
        Darwin)
            if [[ "$PKG_MGR" == "brew" ]]; then
                echo -e "  ${BLUE}→${NC} Installing via brew..."
                brew install --cask google-cloud-sdk
            else
                echo -e "  ${BLUE}→${NC} Installing via curl..."
                curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
                # shellcheck disable=SC1091
                source "$HOME/google-cloud-sdk/path.bash.inc" 2>/dev/null || true
            fi
            ;;
        Linux)
            echo -e "  ${BLUE}→${NC} Installing via curl..."
            curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
            # shellcheck disable=SC1091
            source "$HOME/google-cloud-sdk/path.bash.inc" 2>/dev/null || true
            ;;
        *)
            echo "  Visit: https://cloud.google.com/sdk/docs/install"
            return 1
            ;;
    esac
    if _is_installed gcloud; then
        echo -e "  ${GREEN}✓${NC} gcloud installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    fi
}

_install_aws_cli() {
    if _is_installed aws; then
        echo -e "  ${GREEN}✓${NC} aws ${DIM}($(command -v aws))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} aws — not installed ${DIM}(optional: Bedrock samples)${NC}"
        return 1
    fi
    if ! _confirm "  Install AWS CLI? (for Amazon Bedrock samples)"; then
        echo -e "  ${DIM}Skipped aws${NC}"
        return 0
    fi

    case "$OS" in
        Darwin)
            if [[ "$PKG_MGR" == "brew" ]]; then
                echo -e "  ${BLUE}→${NC} Installing via brew..."
                brew install awscli
            else
                curl -fsSL "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o /tmp/AWSCLIV2.pkg
                sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
                rm -f /tmp/AWSCLIV2.pkg
            fi
            ;;
        Linux)
            if ! _is_installed unzip; then
                _install_pkg unzip unzip unzip unzip || return 1
            fi
            curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
            unzip -qo /tmp/awscliv2.zip -d /tmp
            sudo /tmp/aws/install || /tmp/aws/install --install-dir "$HOME/.local/aws-cli" --bin-dir "$HOME/.local/bin"
            rm -rf /tmp/awscliv2.zip /tmp/aws
            ;;
        *)
            echo "  Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
            return 1
            ;;
    esac
    if _is_installed aws; then
        echo -e "  ${GREEN}✓${NC} aws CLI installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    fi
}

_install_az_cli() {
    if _is_installed az; then
        echo -e "  ${GREEN}✓${NC} az ${DIM}($(command -v az))${NC}"
        return 0
    fi
    if $CHECK_ONLY; then
        echo -e "  ${YELLOW}✗${NC} az — not installed ${DIM}(optional: Azure AI samples)${NC}"
        return 1
    fi
    if ! _confirm "  Install Azure CLI? (for Azure AI Foundry samples)"; then
        echo -e "  ${DIM}Skipped az${NC}"
        return 0
    fi

    case "$OS" in
        Darwin)
            if [[ "$PKG_MGR" == "brew" ]]; then
                echo -e "  ${BLUE}→${NC} Installing via brew..."
                brew install azure-cli
            else
                curl -fsSL https://aka.ms/InstallAzureCLIDeb | bash
            fi
            ;;
        Linux)
            case "${DISTRO}" in
                debian|ubuntu|linuxmint|pop)
                    curl -fsSL https://aka.ms/InstallAzureCLIDeb | sudo bash
                    ;;
                fedora|rhel|centos|rocky|alma)
                    sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
                    sudo dnf install -y https://packages.microsoft.com/config/rhel/9.0/packages-microsoft-prod.rpm 2>/dev/null \
                        || sudo yum install -y https://packages.microsoft.com/config/rhel/9.0/packages-microsoft-prod.rpm
                    sudo dnf install -y azure-cli || sudo yum install -y azure-cli
                    ;;
                *)
                    echo "  Visit: https://learn.microsoft.com/cli/azure/install-azure-cli"
                    return 1
                    ;;
            esac
            ;;
        *)
            echo "  Visit: https://learn.microsoft.com/cli/azure/install-azure-cli"
            return 1
            ;;
    esac
    if _is_installed az; then
        echo -e "  ${GREEN}✓${NC} az CLI installed"
        TOOLS_CHANGED=$((TOOLS_CHANGED + 1))
    fi
}

# ── Cloud authentication via CLI tools ────────────────────────────────

_configure_gcloud_auth() {
    if ! _is_installed gcloud; then
        return 0
    fi

    echo ""
    echo -e "  ${BOLD}── Google Cloud Authentication ──${NC}"
    echo -e "  ${DIM}Used by: provider-google-genai-vertexai-*, provider-vertex-ai-*,${NC}"
    echo -e "  ${DIM}         provider-firestore-retriever, framework-realtime-tracing-demo${NC}"

    # Check if already authenticated.
    if gcloud auth application-default print-access-token &>/dev/null 2>&1; then
        local account
        account=$(gcloud config get-value account 2>/dev/null || echo "unknown")
        echo -e "  ${GREEN}✓${NC} Application Default Credentials found ${DIM}($account)${NC}"

        # Check project.
        local project
        project=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -n "$project" ]]; then
            echo -e "  ${GREEN}✓${NC} Current project: ${DIM}$project${NC}"
            _env_file_set "GOOGLE_CLOUD_PROJECT" "$project"
        fi
        return 0
    fi

    if ! [[ -t 0 ]] || ! [ -c /dev/tty ]; then
        echo -e "  ${DIM}Skipped (non-interactive)${NC}"
        return 0
    fi

    if _confirm "  Run 'gcloud auth application-default login'?"; then
        echo ""
        gcloud auth application-default login
        echo ""

        # Set project if not configured.
        local project
        project=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -z "$project" ]]; then
            echo -en "  Enter your GCP project ID: "
            local proj_input
            read -r proj_input < /dev/tty
            if [[ -n "$proj_input" ]]; then
                gcloud config set project "$proj_input"
                _env_file_set "GOOGLE_CLOUD_PROJECT" "$proj_input"
                echo -e "  ${GREEN}✓${NC} Project set to $proj_input"
            fi
        else
            _env_file_set "GOOGLE_CLOUD_PROJECT" "$project"
            echo -e "  ${GREEN}✓${NC} Using project: $project"
        fi

        # Enable commonly needed APIs.
        project="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}"
        if [[ -n "$project" ]]; then
            if _confirm "  Enable Vertex AI + Firestore APIs for project '$project'?"; then
                local apis=(
                    "aiplatform.googleapis.com"
                    "firestore.googleapis.com"
                    "generativelanguage.googleapis.com"
                )
                for api in "${apis[@]}"; do
                    echo -e "  ${BLUE}→${NC} Enabling $api..."
                    if gcloud services enable "$api" --project="$project" 2>/dev/null; then
                        echo -e "  ${GREEN}✓${NC} $api enabled"
                    else
                        echo -e "  ${YELLOW}!${NC} Could not enable $api (check permissions)"
                    fi
                done
            fi
        fi
    else
        echo -e "  ${DIM}Skipped GCP auth${NC}"
    fi
}

_configure_aws_auth() {
    if ! _is_installed aws; then
        return 0
    fi

    echo ""
    echo -e "  ${BOLD}── AWS Authentication ──${NC}"
    echo -e "  ${DIM}Used by: provider-amazon-bedrock-hello${NC}"

    # Check if already authenticated.
    if aws sts get-caller-identity &>/dev/null 2>&1; then
        local identity
        identity=$(aws sts get-caller-identity --query 'Arn' --output text 2>/dev/null || echo "authenticated")
        echo -e "  ${GREEN}✓${NC} AWS credentials found ${DIM}($identity)${NC}"

        # Save region to env file if set.
        local region
        region=$(aws configure get region 2>/dev/null || echo "")
        if [[ -n "$region" ]]; then
            _env_file_set "AWS_REGION" "$region"
            echo -e "  ${GREEN}✓${NC} Region: $region"
        fi
        return 0
    fi

    if ! [[ -t 0 ]] || ! [ -c /dev/tty ]; then
        echo -e "  ${DIM}Skipped (non-interactive)${NC}"
        return 0
    fi

    if _confirm "  Run 'aws configure'?"; then
        echo ""
        aws configure
        echo ""

        # Save region to env file.
        local region
        region=$(aws configure get region 2>/dev/null || echo "")
        if [[ -n "$region" ]]; then
            _env_file_set "AWS_REGION" "$region"
        fi
        echo -e "  ${GREEN}✓${NC} AWS credentials configured"
    else
        echo -e "  ${DIM}Skipped AWS auth${NC}"
    fi
}

_configure_az_auth() {
    if ! _is_installed az; then
        return 0
    fi

    echo ""
    echo -e "  ${BOLD}── Azure Authentication ──${NC}"
    echo -e "  ${DIM}Used by: provider-microsoft-foundry-hello${NC}"

    # Check if already authenticated.
    if az account show &>/dev/null 2>&1; then
        local account
        account=$(az account show --query 'user.name' --output tsv 2>/dev/null || echo "authenticated")
        echo -e "  ${GREEN}✓${NC} Azure credentials found ${DIM}($account)${NC}"
        return 0
    fi

    if ! [[ -t 0 ]] || ! [ -c /dev/tty ]; then
        echo -e "  ${DIM}Skipped (non-interactive)${NC}"
        return 0
    fi

    if _confirm "  Run 'az login'?"; then
        echo ""
        az login
        echo ""
        echo -e "  ${GREEN}✓${NC} Azure credentials configured"
    else
        echo -e "  ${DIM}Skipped Azure auth${NC}"
    fi
}

# ── Environment file management ──────────────────────────────────────

# Read a variable's current value from ~/.environment.
_env_file_get() {
    local var_name="$1"
    if [[ -f "$ENV_FILE" ]]; then
        grep -E "^export ${var_name}=" "$ENV_FILE" 2>/dev/null \
            | tail -1 \
            | sed -E "s/^export ${var_name}=['\"]?([^'\"]*)['\"]?$/\1/" \
            || true
    fi
}

# Set a variable in ~/.environment. Creates the file if needed.
_env_file_set() {
    local var_name="$1"
    local value="$2"

    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" <<'HEADER'
# ~/.environment — Shared environment variables
# Auto-generated by Genkit samples setup.sh
# Source this file in your shell RC (done automatically by setup.sh).
#
# Manual edits are preserved — the setup script only updates lines
# that start with "export VARIABLE_NAME=".

HEADER
        chmod 600 "$ENV_FILE"
    fi

    # Remove any existing line for this variable (portable sed -i).
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

# Ensure ~/.environment is sourced by the user's shell RC file.
_ensure_shell_sources_env() {
    _section "Shell Integration"

    local configured=0

    # ── Bash ──
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

    # ── Zsh ──
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

    # ── Fish ──
    local fish_config="$HOME/.config/fish/config.fish"
    if [[ -d "$HOME/.config/fish" ]]; then
        if [[ -f "$fish_config" ]]; then
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
                    echo -e "  ${DIM}  Note: Fish requires the 'bass' plugin (fisher install edc/bass)${NC}"
                else
                    echo -e "  ${YELLOW}✗${NC} Not sourced in ${DIM}$fish_config${NC}"
                fi
                configured=$((configured + 1))
            else
                echo -e "  ${GREEN}✓${NC} Already sourced in ${DIM}$fish_config${NC}"
            fi
        fi
    fi

    if [[ $configured -eq 0 ]]; then
        echo -e "  ${DIM}No shell RC changes needed${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}Environment file:${NC} $ENV_FILE"
    echo -e "  ${DIM}Run \`source $ENV_FILE\` to load in current session${NC}"
}

# ── API key prompts ───────────────────────────────────────────────────

# Prompt for a single API key/variable.
# Usage: _prompt_key "VAR_NAME" "description" "doc_url" [secret]
#
# - Shows the documentation URL before prompting.
# - Shows the current value (masked if secret) as default.
# - Empty input keeps the current value.
# - Entering "skip" skips the variable.
_prompt_key() {
    local var_name="$1"
    local description="$2"
    local doc_url="$3"
    local is_secret="${4:-true}"

    # Check current value: environment first, then file.
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

    # Build display value (masked for secrets).
    local display_val=""
    if [[ -n "$current_val" ]]; then
        if [[ "$is_secret" == "true" ]]; then
            if [[ ${#current_val} -gt 8 ]]; then
                display_val="${current_val:0:4}$(printf '*%.0s' $(seq 1 $((${#current_val} - 4))))"
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

# ── Provider key sections ─────────────────────────────────────────────
# Each section matches a sample's run.sh check_env_var calls.

_configure_google_ai_keys() {
    echo ""
    echo -e "  ${BOLD}── Google AI / Gemini ──${NC}"
    echo -e "  ${DIM}Used by: provider-google-genai-hello, all framework-* samples,${NC}"
    echo -e "  ${DIM}         dev-local-vectorstore-hello, web-flask-hello, web-endpoints-hello${NC}"

    _prompt_key "GEMINI_API_KEY" \
        "Gemini API key (used by most samples)" \
        "https://aistudio.google.com/apikey"
}

_configure_anthropic_keys() {
    echo ""
    echo -e "  ${BOLD}── Anthropic (Claude) ──${NC}"
    echo -e "  ${DIM}Used by: provider-anthropic-hello${NC}"

    _prompt_key "ANTHROPIC_API_KEY" \
        "Anthropic API key" \
        "https://console.anthropic.com/settings/keys"
}

_configure_openai_keys() {
    echo ""
    echo -e "  ${BOLD}── OpenAI ──${NC}"
    echo -e "  ${DIM}Used by: provider-compat-oai-hello${NC}"

    _prompt_key "OPENAI_API_KEY" \
        "OpenAI API key" \
        "https://platform.openai.com/api-keys"
}

_configure_deepseek_keys() {
    echo ""
    echo -e "  ${BOLD}── DeepSeek ──${NC}"
    echo -e "  ${DIM}Used by: provider-deepseek-hello${NC}"

    _prompt_key "DEEPSEEK_API_KEY" \
        "DeepSeek API key" \
        "https://platform.deepseek.com/api_keys"
}

_configure_xai_keys() {
    echo ""
    echo -e "  ${BOLD}── xAI (Grok) ──${NC}"
    echo -e "  ${DIM}Used by: provider-xai-hello${NC}"

    _prompt_key "XAI_API_KEY" \
        "xAI API key" \
        "https://console.x.ai/"
}

_configure_mistral_keys() {
    echo ""
    echo -e "  ${BOLD}── Mistral AI ──${NC}"
    echo -e "  ${DIM}Used by: provider-mistral-hello${NC}"

    _prompt_key "MISTRAL_API_KEY" \
        "Mistral AI API key" \
        "https://console.mistral.ai/api-keys/"
}

_configure_huggingface_keys() {
    echo ""
    echo -e "  ${BOLD}── Hugging Face ──${NC}"
    echo -e "  ${DIM}Used by: provider-huggingface-hello${NC}"

    _prompt_key "HF_TOKEN" \
        "Hugging Face API token" \
        "https://huggingface.co/settings/tokens"
}

_configure_cohere_keys() {
    echo ""
    echo -e "  ${BOLD}── Cohere ──${NC}"
    echo -e "  ${DIM}Used by: provider-cohere-hello${NC}"

    _prompt_key "COHERE_API_KEY" \
        "Cohere API key" \
        "https://dashboard.cohere.com/api-keys"
}

_configure_cloudflare_keys() {
    echo ""
    echo -e "  ${BOLD}── Cloudflare Workers AI ──${NC}"
    echo -e "  ${DIM}Used by: provider-cloudflare-workers-ai-hello${NC}"

    _prompt_key "CLOUDFLARE_ACCOUNT_ID" \
        "Cloudflare account ID" \
        "https://dash.cloudflare.com/" \
        "false"

    _prompt_key "CLOUDFLARE_API_TOKEN" \
        "Cloudflare API token" \
        "https://developers.cloudflare.com/fundamentals/api/get-started/create-token/"
}

_configure_azure_keys() {
    echo ""
    echo -e "  ${BOLD}── Azure AI Foundry ──${NC}"
    echo -e "  ${DIM}Used by: provider-microsoft-foundry-hello${NC}"
    echo -e "  ${DIM}Tip: If using managed identity, you can skip the API key.${NC}"

    _prompt_key "AZURE_AI_FOUNDRY_ENDPOINT" \
        "Azure AI Foundry endpoint URL" \
        "https://ai.azure.com/" \
        "false"

    _prompt_key "AZURE_AI_FOUNDRY_API_KEY" \
        "Azure AI Foundry API key" \
        "https://ai.azure.com/"
}

_configure_aws_keys() {
    echo ""
    echo -e "  ${BOLD}── AWS Bedrock ──${NC}"
    echo -e "  ${DIM}Used by: provider-amazon-bedrock-hello${NC}"
    echo -e "  ${DIM}Tip: If you ran 'aws configure' above, these are already set.${NC}"

    _prompt_key "AWS_REGION" \
        "AWS region (e.g., us-east-1)" \
        "https://docs.aws.amazon.com/general/latest/gr/bedrock.html" \
        "false"

    _prompt_key "AWS_ACCESS_KEY_ID" \
        "AWS access key ID (skip if using IAM roles/SSO)" \
        "https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html"

    _prompt_key "AWS_SECRET_ACCESS_KEY" \
        "AWS secret access key (skip if using IAM roles/SSO)" \
        "https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html"
}

_configure_gcp_keys() {
    echo ""
    echo -e "  ${BOLD}── Google Cloud Platform ──${NC}"
    echo -e "  ${DIM}Used by: provider-google-genai-vertexai-*, provider-vertex-ai-*,${NC}"
    echo -e "  ${DIM}         provider-firestore-retriever, framework-realtime-tracing-demo${NC}"
    echo -e "  ${DIM}Tip: If you configured gcloud auth above, the project is already set.${NC}"

    _prompt_key "GOOGLE_CLOUD_PROJECT" \
        "GCP project ID" \
        "https://console.cloud.google.com/" \
        "false"
}

_configure_observability_keys() {
    echo ""
    echo -e "  ${BOLD}── Observability Backends ──${NC}"
    echo -e "  ${DIM}Used by: provider-observability-hello${NC}"
    echo -e "  ${DIM}Configure only the backends you use. Skip the rest.${NC}"

    _prompt_key "SENTRY_DSN" \
        "Sentry DSN" \
        "https://docs.sentry.io/concepts/otlp/"

    _prompt_key "HONEYCOMB_API_KEY" \
        "Honeycomb API key" \
        "https://docs.honeycomb.io/configure/environments/manage-api-keys/"

    _prompt_key "DD_API_KEY" \
        "Datadog API key" \
        "https://app.datadoghq.com/organization-settings/api-keys"

    _prompt_key "GRAFANA_OTLP_ENDPOINT" \
        "Grafana Cloud OTLP endpoint" \
        "https://grafana.com/ > My Account > Stack > OpenTelemetry" \
        "false"

    _prompt_key "GRAFANA_USER_ID" \
        "Grafana Cloud instance ID (numeric)" \
        "https://grafana.com/ > My Account > Stack > OpenTelemetry" \
        "false"

    _prompt_key "GRAFANA_API_KEY" \
        "Grafana Cloud API key" \
        "https://grafana.com/ > My Account > Stack > OpenTelemetry"

    _prompt_key "AXIOM_TOKEN" \
        "Axiom API token" \
        "https://app.axiom.co/settings/tokens"
}

# ── Main ──────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║       Genkit Python Samples — Environment Setup              ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${DIM}Platform: $OS / $DISTRO / pkg: $PKG_MGR${NC}"
echo -e "${DIM}Environment file: $ENV_FILE${NC}"

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
    _install_watchdog || true

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

    echo -e "${DIM}These are needed only for specific samples. Skip any you don't need.${NC}"
    echo ""

    _install_ollama || true
    _install_gcloud || true
    _install_aws_cli || true
    _install_az_cli || true

fi  # end !KEYS_ONLY

# ── Phase 1.5: Ensure public registries ───────────────────────────────

_configure_public_registries

# ── Phase 2: Cloud authentication via CLI ─────────────────────────────

if ! $CHECK_ONLY; then

    _section "Cloud Authentication"

    echo -e "${DIM}Authenticate with cloud providers using their CLI tools.${NC}"
    echo -e "${DIM}This saves credentials to the standard CLI config locations.${NC}"

    _configure_gcloud_auth
    _configure_aws_auth
    _configure_az_auth

fi

# ── Phase 3: Configure API keys ──────────────────────────────────────

_section "API Key Configuration"

echo -e "Each provider needs an API key to run its sample."
echo -e "Type ${BOLD}skip${NC} or press ${BOLD}Enter${NC} to skip any key you don't need."
echo -e "Existing values are shown as defaults (masked for secrets)."
echo -e "All values are saved to ${CYAN}$ENV_FILE${NC}."

if ! [[ -t 0 ]] || ! [ -c /dev/tty ]; then
    echo ""
    echo -e "${YELLOW}Non-interactive mode detected. Skipping API key prompts.${NC}"
    echo "Run this script in an interactive terminal to configure keys."
else
    _configure_google_ai_keys
    _configure_anthropic_keys
    _configure_openai_keys
    _configure_deepseek_keys
    _configure_xai_keys
    _configure_mistral_keys
    _configure_huggingface_keys
    _configure_cohere_keys
    _configure_cloudflare_keys
    _configure_aws_keys
    _configure_azure_keys
    _configure_gcp_keys
    _configure_observability_keys
fi

# ── Phase 4: Shell integration ────────────────────────────────────────

_ensure_shell_sources_env

# ── Phase 5: Install Python dependencies ──────────────────────────────

if ! $KEYS_ONLY && ! $CHECK_ONLY; then
    _section "Python Dependencies"

    if _is_installed uv; then
        # We're in py/samples/ — sync the workspace root.
        _py_root="${SCRIPT_DIR}/.."
        if [[ -f "${_py_root}/pyproject.toml" ]]; then
            _venv_dir="${_py_root}/.venv"
            if [[ -d "$_venv_dir" ]]; then
                echo -e "Removing stale virtual environment..."
                rm -rf "$_venv_dir"
                echo -e "  ${GREEN}✓${NC} Removed ${DIM}${_venv_dir}${NC}"
            fi
            echo "Installing workspace dependencies (fresh .venv)..."
            echo ""
            (cd "$_py_root" && uv sync)
            echo -e "  ${GREEN}✓${NC} Workspace dependencies installed"
        else
            echo -e "  ${DIM}Not in monorepo — skipping workspace sync${NC}"
        fi
    fi
fi

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
    echo -e "  2. Pick a sample to run:     ${GREEN}cd provider-google-genai-hello${NC}"
    echo -e "  3. Start the sample:         ${GREEN}./run.sh${NC}"
    echo ""
    echo -e "  ${DIM}Tip: New terminal sessions auto-load ~/.environment${NC}"
    echo -e "  ${DIM}Tip: Re-run ./setup.sh any time to add more keys${NC}"
fi

echo ""
